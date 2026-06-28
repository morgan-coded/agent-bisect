from __future__ import annotations

from collections import Counter
from typing import Iterable

from .gates import GateResult, run_g1, run_g2, run_g3
from .localize import localize_failures, shell_target_coverage
from .model import Activity


STRUCTURED_KINDS = {"file_edit", "test_run", "tool_call"}
STATUSES = ("PASS", "FAIL", "NA")


def explain_replay(activities: Iterable[Activity]) -> str:
    """Render a structural demo narrative over existing gate/localizer outputs."""

    ordered = sorted(list(activities), key=lambda activity: activity.step_index)
    run_id = ordered[0].run_id if ordered else ""
    kind_counts = Counter(activity.kind for activity in ordered)
    structured_count = sum(kind_counts.get(kind, 0) for kind in STRUCTURED_KINDS)
    gate_results = {
        "G1": run_g1(ordered),
        "G2": run_g2(ordered),
        "G3": run_g3(ordered),
    }
    localization = localize_failures(ordered)
    shell_coverage = shell_target_coverage(ordered)

    lines = [
        "agent-bisect replay --explain",
        f"run_id: {run_id}",
        f"activities: {len(ordered)}",
        f"kinds: {_format_counts(kind_counts)}",
        f"structured_fraction: {_format_fraction(structured_count, len(ordered))}",
        "shell_target_coverage: steps_with_targets={hits}/{total} added_edges={edges}".format(
            hits=shell_coverage.steps_with_targets,
            total=shell_coverage.shell_command_steps,
            edges=shell_coverage.added_edges,
        ),
        "gate_tallies:",
    ]
    for gate in ("G1", "G2", "G3"):
        tallies = _status_counts(gate_results[gate])
        lines.append(f"  {gate}: {_format_statuses(tallies)}")

    if localization.status == "no_break":
        lines.append("verdict: clean run")
        return "\n".join(lines) + "\n"

    confidence_counts = Counter(failure.confidence for failure in localization.failures)
    lines.append(
        "verdict: {count} break(s) localized (HIGH={high} LOW={low})".format(
            count=len(localization.failures),
            high=confidence_counts.get("HIGH", 0),
            low=confidence_counts.get("LOW", 0),
        )
    )
    lines.append("breaks:")
    by_step = {activity.step_index: activity for activity in ordered}
    for index, failure in enumerate(localization.failures, start=1):
        activity = by_step[failure.breaking_step]
        lines.extend(
            [
                f"  break {index}:",
                f"    breaking_step: {failure.breaking_step}",
                f"    gate: {failure.breaking_gate}",
                "    activity: kind={kind} tool={tool} target={target}".format(
                    kind=activity.kind,
                    tool=activity.tool_name or "",
                    target=activity.target or "",
                ),
                f"    cascade: {_format_steps(failure.failure_cascade)}",
                f"    confidence: {failure.confidence}",
                f"    coverage: {failure.coverage}",
            ]
        )
        if failure.candidates:
            lines.append(f"    candidates: {_format_steps(failure.candidates)}")

    return "\n".join(lines) + "\n"


def _status_counts(results: list[GateResult]) -> Counter[str]:
    return Counter(result.status for result in results)


def _format_statuses(counts: Counter[str]) -> str:
    return " ".join(f"{status}={counts.get(status, 0)}" for status in STATUSES)


def _format_counts(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return " ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _format_fraction(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0 (NA)"
    return f"{numerator}/{denominator} ({numerator / denominator:.3f})"


def _format_steps(steps: tuple[int, ...]) -> str:
    if not steps:
        return ""
    return ",".join(str(step) for step in steps)
