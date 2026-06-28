from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from .gates import GateResult, run_g1, run_g2, run_g3
from .model import Activity


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

    graph = _build_graph(ordered)
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


def _gate_failures_by_step(activities: list[Activity]) -> dict[int, list[GateResult]]:
    failures: dict[int, list[GateResult]] = defaultdict(list)
    for result in [*run_g1(activities), *run_g2(activities), *run_g3(activities)]:
        if result.status == "FAIL" and result.step_index is not None:
            failures[result.step_index].append(result)
    return dict(failures)


def _build_graph(activities: list[Activity]) -> dict[int, set[int]]:
    by_step = {activity.step_index: activity for activity in activities}
    graph: dict[int, set[int]] = {activity.step_index: set() for activity in activities}

    for activity in activities:
        if activity.parent_step in by_step:
            graph[activity.parent_step].add(activity.step_index)

    prior_edits_by_file: dict[str, list[int]] = defaultdict(list)
    for activity in activities:
        refs = _structured_file_refs(activity)
        for file_path in sorted(refs):
            for edit_step in prior_edits_by_file[file_path]:
                if edit_step != activity.step_index:
                    graph[edit_step].add(activity.step_index)
        if activity.kind == "file_edit":
            for file_path in sorted(refs):
                prior_edits_by_file[file_path].append(activity.step_index)

    return graph


def _structured_file_refs(activity: Activity) -> set[str]:
    if activity.kind in {"opaque_shell", "unmapped"}:
        return set()

    refs: set[str] = set()
    if isinstance(activity.target, str) and activity.target and activity.target != "shell":
        refs.add(activity.target)

    file_path = activity.inputs.get("file_path")
    if isinstance(file_path, str) and file_path:
        refs.add(file_path)

    return refs


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
    by_step: dict[int, Activity],
) -> tuple[Confidence, str, tuple[int, ...]]:
    opaque_steps: set[int] = set()
    unlinked_steps: set[int] = set()

    for observed_step in cascade:
        path = _shortest_path(breaking_step, observed_step, graph)
        for offset, step in enumerate(path):
            activity = by_step[step]
            if activity.kind in {"opaque_shell", "unmapped"}:
                opaque_steps.add(step)
            if offset > 0 and activity.parent_step is None:
                unlinked_steps.add(step)

    if opaque_steps or unlinked_steps:
        notes = []
        if opaque_steps:
            notes.append(_count_note(len(opaque_steps), "opaque_shell node"))
        if unlinked_steps:
            notes.append(_count_note(len(unlinked_steps), "unlinked step"))
        return "LOW", "; ".join(notes) + " on path", tuple(sorted(opaque_steps | unlinked_steps))

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
