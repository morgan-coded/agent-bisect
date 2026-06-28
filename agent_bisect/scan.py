from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from .gates import run_g2, run_g3
from .io import load_activities
from .localize import localize_failures
from .model import Activity


def scan_paths(paths: list[Path]) -> dict[str, Any]:
    runs = []
    for path in _expand_paths(paths):
        activities = load_activities(path)
        runs.append(_scan_run(activities))
    return {
        "label": "generalization check (same Claude schema)",
        "limitation": "the deterministic gates validate within the Claude tool-call schema; use foreign-schema adapters for cross-platform coverage checks.",
        "runs": runs,
    }


def _scan_run(activities: list[Activity]) -> dict[str, Any]:
    run_id = activities[0].run_id if activities else ""
    localizations = {}
    for failure in localize_failures(activities).failures:
        localizations[failure.breaking_step] = failure
        for step in failure.failure_cascade:
            localizations[step] = failure
    failures = []
    for result in [*run_g2(activities), *run_g3(activities)]:
        if result.status != "FAIL" or result.step_index is None:
            continue
        activity = activities[result.step_index]
        localized = localizations.get(result.step_index)
        failures.append(
            {
                "run_id": run_id,
                "step_index": result.step_index,
                "kind": activity.kind,
                "tool_name": activity.tool_name,
                "gate": result.gate,
                "status": result.status,
                "content_hash": activity.content_hash,
                "actual_breaking_step": None if localized is None else localized.breaking_step,
                "confidence": None if localized is None else localized.confidence,
                "coverage": None if localized is None else localized.coverage,
            }
        )
    return {
        "run_id": run_id,
        "failure_count": len(failures),
        "failures": failures,
    }


def _expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        rendered = str(path)
        if any(char in rendered for char in "*?[]"):
            expanded.extend(Path(match) for match in glob.glob(rendered))
        elif path.is_dir():
            expanded.extend(sorted(path.glob("*.jsonl")))
        else:
            expanded.append(path)
    return sorted(expanded)
