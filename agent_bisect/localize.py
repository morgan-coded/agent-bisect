from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from .gates import GateResult, run_g1, run_g2, run_g3
from .model import Activity
from .shell_targets import ShellTargets, extract_shell_targets


Confidence = str


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    breaking_step: int
    breaking_gate: str
    failure_cascade: tuple[int, ...]
    confidence: Confidence
    coverage: str
    candidates: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "breaking_step": self.breaking_step,
            "breaking_gate": self.breaking_gate,
            "failure_cascade": list(self.failure_cascade),
            "confidence": self.confidence,
            "coverage": self.coverage,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True, slots=True)
class LocalizationReport:
    status: str
    failures: tuple[LocalizationResult, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "failures": [failure.to_dict() for failure in self.failures],
        }


@dataclass(frozen=True, slots=True)
class ShellTargetCoverage:
    shell_command_steps: int
    steps_with_targets: int
    added_edges: int

    def to_dict(self) -> dict[str, int]:
        return {
            "shell_command_steps": self.shell_command_steps,
            "steps_with_targets": self.steps_with_targets,
            "added_edges": self.added_edges,
        }


def localize_failures(activities: Iterable[Activity]) -> LocalizationReport:
    """Localize deterministic envelope failures without replaying the model.

    The graph is intentionally narrow: parent_step causality plus structured
    file-target edges. Opaque shell commands and unlinked steps are coverage
    gaps, so paths through them are downgraded to LOW confidence rather than
    reported as a false-precise single root cause.
    """

    ordered = sorted(list(activities), key=lambda activity: activity.step_index)
    if not ordered:
        return LocalizationReport(status="no_break")

    base_graph = _build_graph(ordered, include_shell_targets=False)
    graph = _build_graph(ordered, include_shell_targets=True)
    reverse_graph = _reverse_graph(graph)
    failures_by_step = _gate_failures_by_step(ordered)
    if not failures_by_step:
        return LocalizationReport(status="no_break")

    groups: dict[int, set[int]] = defaultdict(set)
    for observed_step in sorted(failures_by_step):
        upstream_failures = [
            step for step in _reachable_upstream(observed_step, reverse_graph) if step in failures_by_step
        ]
        breaking_step = min(upstream_failures)
        if observed_step != breaking_step:
            groups[breaking_step].add(observed_step)
        else:
            groups.setdefault(breaking_step, set())

    by_step = {activity.step_index: activity for activity in ordered}
    results: list[LocalizationResult] = []
    for breaking_step in sorted(groups):
        breaking_gate = failures_by_step[breaking_step][0].gate
        cascade = tuple(sorted(groups[breaking_step]))
        confidence, coverage, candidates = _confidence_for_group(
            breaking_step,
            cascade,
            graph,
            base_graph,
            by_step,
        )
        results.append(
            LocalizationResult(
                breaking_step=breaking_step,
                breaking_gate=breaking_gate,
                failure_cascade=cascade,
                confidence=confidence,
                coverage=coverage,
                candidates=candidates,
            )
        )

    return LocalizationReport(status="break", failures=tuple(results))


def shell_target_coverage(activities: Iterable[Activity]) -> ShellTargetCoverage:
    ordered = sorted(list(activities), key=lambda activity: activity.step_index)
    command_steps = [
        activity
        for activity in ordered
        if activity.kind in {"opaque_shell", "test_run"} and isinstance(activity.inputs.get("command"), str)
    ]
    steps_with_targets = sum(1 for activity in command_steps if _has_shell_targets(activity))
    base_edges = _edge_set(_build_graph(ordered, include_shell_targets=False))
    shell_edges = _edge_set(_build_graph(ordered, include_shell_targets=True))
    return ShellTargetCoverage(
        shell_command_steps=len(command_steps),
        steps_with_targets=steps_with_targets,
        added_edges=len(shell_edges - base_edges),
    )


def _gate_failures_by_step(activities: list[Activity]) -> dict[int, list[GateResult]]:
    failures: dict[int, list[GateResult]] = defaultdict(list)
    for result in [*run_g1(activities), *run_g2(activities), *run_g3(activities)]:
        if result.status == "FAIL" and result.step_index is not None:
            failures[result.step_index].append(result)
    return dict(failures)


def _build_graph(activities: list[Activity], *, include_shell_targets: bool = True) -> dict[int, set[int]]:
    by_step = {activity.step_index: activity for activity in activities}
    graph: dict[int, set[int]] = {activity.step_index: set() for activity in activities}

    for activity in activities:
        if activity.parent_step in by_step:
            graph[activity.parent_step].add(activity.step_index)

    prior_producers_by_file: dict[str, list[int]] = defaultdict(list)
    for activity in activities:
        refs = _structured_file_refs(activity, include_shell_targets=include_shell_targets)
        for file_path in sorted(refs):
            for producer_step in prior_producers_by_file[file_path]:
                if producer_step != activity.step_index:
                    graph[producer_step].add(activity.step_index)
        for file_path in sorted(_producer_file_refs(activity, refs, include_shell_targets=include_shell_targets)):
            prior_producers_by_file[file_path].append(activity.step_index)

    return graph


def _structured_file_refs(activity: Activity, *, include_shell_targets: bool = True) -> set[str]:
    if activity.kind == "unmapped":
        return set()
    if activity.kind == "opaque_shell" and not include_shell_targets:
        return set()

    refs: set[str] = set()
    if isinstance(activity.target, str) and activity.target and activity.target != "shell":
        refs.add(activity.target)

    file_path = activity.inputs.get("file_path")
    if isinstance(file_path, str) and file_path:
        refs.add(file_path)

    if include_shell_targets and activity.kind in {"opaque_shell", "test_run"}:
        shell_targets = _shell_targets_for_activity(activity)
        refs.update(shell_targets.reads)
        refs.update(shell_targets.writes)

    return refs


def _producer_file_refs(activity: Activity, refs: set[str], *, include_shell_targets: bool = True) -> set[str]:
    if activity.kind == "file_edit":
        return refs
    if include_shell_targets and activity.kind in {"opaque_shell", "test_run"}:
        return set(_shell_targets_for_activity(activity).writes)
    return set()


def _reverse_graph(graph: dict[int, set[int]]) -> dict[int, set[int]]:
    reverse: dict[int, set[int]] = {step: set() for step in graph}
    for source, targets in graph.items():
        for target in targets:
            reverse.setdefault(target, set()).add(source)
    return reverse


def _reachable_upstream(start: int, reverse_graph: dict[int, set[int]]) -> set[int]:
    seen: set[int] = set()
    queue: deque[int] = deque([start])
    while queue:
        step = queue.popleft()
        if step in seen:
            continue
        seen.add(step)
        for parent in sorted(reverse_graph.get(step, ())):
            queue.append(parent)
    return seen


def _confidence_for_group(
    breaking_step: int,
    cascade: tuple[int, ...],
    graph: dict[int, set[int]],
    base_graph: dict[int, set[int]],
    by_step: dict[int, Activity],
) -> tuple[Confidence, str, tuple[int, ...]]:
    opaque_steps: set[int] = set()
    unlinked_steps: set[int] = set()
    heuristic_edges: set[tuple[int, int]] = set()
    heuristic_steps: set[int] = set()

    for observed_step in cascade:
        path = _shortest_path(breaking_step, observed_step, graph)
        for parent, child in zip(path, path[1:]):
            if child not in base_graph.get(parent, set()):
                heuristic_edges.add((parent, child))
                if _has_shell_targets(by_step[parent]):
                    heuristic_steps.add(parent)
                if _has_shell_targets(by_step[child]):
                    heuristic_steps.add(child)
        for offset, step in enumerate(path):
            activity = by_step[step]
            if activity.kind in {"opaque_shell", "unmapped"}:
                opaque_steps.add(step)
            if offset > 0 and activity.parent_step is None:
                unlinked_steps.add(step)

    if opaque_steps or unlinked_steps or heuristic_edges:
        notes = []
        if opaque_steps:
            heuristic_opaque_steps = {step for step in opaque_steps if _has_shell_targets(by_step[step])}
            plain_opaque_steps = opaque_steps - heuristic_opaque_steps
            if heuristic_opaque_steps:
                notes.append(_count_note(len(heuristic_opaque_steps), "opaque_shell node (heuristic shell target)"))
            if plain_opaque_steps:
                notes.append(_count_note(len(plain_opaque_steps), "opaque_shell node"))
        if unlinked_steps:
            notes.append(_count_note(len(unlinked_steps), "unlinked step"))
        if heuristic_edges:
            notes.append(_count_note(len(heuristic_edges), "heuristic shell-target edge"))
        return "LOW", "; ".join(notes) + " on path", tuple(sorted(opaque_steps | unlinked_steps | heuristic_steps))

    return "HIGH", "structured path", ()


def _shortest_path(start: int, goal: int, graph: dict[int, set[int]]) -> tuple[int, ...]:
    queue: deque[tuple[int, ...]] = deque([(start,)])
    seen: set[int] = set()
    while queue:
        path = queue.popleft()
        step = path[-1]
        if step == goal:
            return path
        if step in seen:
            continue
        seen.add(step)
        for child in sorted(graph.get(step, ())):
            if child not in seen:
                queue.append((*path, child))
    return (start, goal)


def _count_note(count: int, noun: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"


def _shell_targets_for_activity(activity: Activity) -> ShellTargets:
    return extract_shell_targets(activity.inputs.get("command"))


def _has_shell_targets(activity: Activity) -> bool:
    targets = _shell_targets_for_activity(activity)
    return bool(targets.reads or targets.writes)


def _edge_set(graph: dict[int, set[int]]) -> set[tuple[int, int]]:
    return {(source, target) for source, targets in graph.items() for target in targets}
