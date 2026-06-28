from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

from .gates import run_g1, run_g2, run_g3
from .inject import FAULT_CLASSES, NON_TARGET_CLASSES, TARGET_CLASSES, eligible_steps_by_class, inject_fault
from .io import load_activities
from .localize import localize_failures
from .model import Activity, canonical_json, sha256_text


def evaluate_paths(paths: list[Path], *, seed: int = 1729, per_class: int = 3) -> dict[str, Any]:
    runs = [_load_run(path, index) for index, path in enumerate(paths)]
    clean_runs = [run for run in runs if not _has_gate_failures(run["activities"])]
    dirty_runs = [run for run in runs if _has_gate_failures(run["activities"])]

    eligible = _eligible_candidates(clean_runs)
    selected = _select_candidates(eligible, seed=seed, per_class=per_class)
    class_metrics = _empty_class_metrics()
    localization = _empty_localization()
    mutations: list[dict[str, Any]] = []

    for fault_class in FAULT_CLASSES:
        class_metrics[fault_class]["eligible_n"] = len(eligible[fault_class])
        class_metrics[fault_class]["injected_n"] = len(selected[fault_class])

    run_by_index = {run["index"]: run for run in clean_runs}
    for fault_class in FAULT_CLASSES:
        for run_index, step in selected[fault_class]:
            run = run_by_index[run_index]
            case = inject_fault(run["activities"], step, fault_class)
            outcome = _score_case(run["activities"], list(case.activities), case.truth.to_dict())
            _accumulate(class_metrics[fault_class], outcome)
            if fault_class in TARGET_CLASSES and outcome["target_caught"]:
                _accumulate_localization(localization, list(case.activities), case.truth.to_dict())
            mutations.append(case.truth.to_dict())

    for metrics in class_metrics.values():
        _finalize_metrics(metrics)

    source_counts = _source_counts(clean_runs, dirty_runs)
    report = {
        "schema_version": 1,
        "seed": seed,
        "per_class": per_class,
        "classes": class_metrics,
        "localization": _finalize_localization(localization),
        "source": source_counts,
        "mutations": sorted(mutations, key=lambda item: (item["fault_class"], item["run_id"], item["injected_step"])),
        "notes": {
            "g4": "G4 fault class deferred; g4_consistency_holds is not implemented.",
            "generalization_check": "generalization check (same Claude schema)",
            "limitation": "the deterministic gates validate within the Claude tool-call schema; use foreign-schema adapters for cross-platform coverage checks.",
            "localization_caveat": "Injected-fault localization is an upper bound on real-run localization because injection only targets structured eligible steps.",
        },
    }
    return report


def write_eval_reports(report: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "eval-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (reports_dir / "eval-report.md").write_text(_render_markdown(report), encoding="utf-8", newline="\n")


def _load_run(path: Path, index: int) -> dict[str, Any]:
    activities = load_activities(path)
    run_id = activities[0].run_id if activities else f"empty-{index}"
    return {"index": index, "run_id": run_id, "activities": activities}


def _eligible_candidates(runs: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    candidates: dict[str, list[tuple[int, int]]] = {fault_class: [] for fault_class in FAULT_CLASSES}
    for run in runs:
        eligible = eligible_steps_by_class(run["activities"])
        for fault_class, steps in eligible.items():
            for step in steps:
                candidates[fault_class].append((run["index"], step))
    return {fault_class: sorted(values) for fault_class, values in candidates.items()}


def _select_candidates(
    eligible: dict[str, list[tuple[int, int]]],
    *,
    seed: int,
    per_class: int,
) -> dict[str, list[tuple[int, int]]]:
    selected: dict[str, list[tuple[int, int]]] = {}
    for fault_class, candidates in eligible.items():
        if per_class <= 0 or not candidates:
            selected[fault_class] = []
            continue
        count = min(per_class, len(candidates))
        rng_seed = int(sha256_text(f"{seed}:{fault_class}:eval")[:16], 16)
        rng = random.Random(rng_seed)
        selected[fault_class] = sorted(rng.sample(candidates, count))
    return selected


def _score_case(
    base_activities: list[Activity],
    mutated_activities: list[Activity],
    truth: dict[str, Any],
) -> dict[str, Any]:
    before = _gate_statuses(base_activities)
    after = _gate_statuses(mutated_activities)
    any_fail = any(status == "FAIL" for step in after.values() for status in step.values())

    fault_class = truth["fault_class"]
    if fault_class in TARGET_CLASSES:
        expected_gate = truth["expected_gate"]
        injected_step = int(truth["injected_step"])
        expected_status = after.get(injected_step, {}).get(expected_gate, "NA")
        if expected_status == "FAIL":
            return {"bucket": "tp", "target_caught": True}
        if expected_status == "NA":
            return {"bucket": "na", "target_caught": False}
        if any_fail:
            return {"bucket": "fp", "target_caught": False}
        return {"bucket": "fn", "target_caught": False}

    changed = before != after
    if any_fail or changed:
        return {"bucket": "fp", "target_caught": False}
    return {"bucket": "clean", "target_caught": False}


def _gate_statuses(activities: list[Activity]) -> dict[int, dict[str, str]]:
    statuses: dict[int, dict[str, str]] = {}
    for result in [*run_g1(activities), *run_g2(activities), *run_g3(activities)]:
        if result.step_index is None:
            continue
        statuses.setdefault(result.step_index, {})[result.gate] = result.status
    return statuses


def _has_gate_failures(activities: list[Activity]) -> bool:
    return any(
        result.status == "FAIL"
        for result in [*run_g1(activities), *run_g2(activities), *run_g3(activities)]
    )


def _empty_class_metrics() -> dict[str, dict[str, Any]]:
    return {
        fault_class: {
            "eligible_n": 0,
            "injected_n": 0,
            "scored_n": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "na": 0,
            "precision": None,
            "recall": None,
            "false_positive_rate": None,
        }
        for fault_class in FAULT_CLASSES
    }


def _accumulate(metrics: dict[str, Any], outcome: dict[str, Any]) -> None:
    bucket = outcome["bucket"]
    if bucket == "tp":
        metrics["tp"] += 1
        metrics["scored_n"] += 1
    elif bucket == "fp":
        metrics["fp"] += 1
        metrics["scored_n"] += 1
    elif bucket == "fn":
        metrics["fn"] += 1
        metrics["scored_n"] += 1
    elif bucket == "na":
        metrics["na"] += 1
    elif bucket == "clean":
        metrics["scored_n"] += 1
    else:
        raise ValueError(f"unknown eval bucket: {bucket}")


def _finalize_metrics(metrics: dict[str, Any]) -> None:
    if metrics["tp"] + metrics["fp"] > 0:
        metrics["precision"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])
    if metrics["injected_n"] > 0 and (metrics["tp"] or metrics["fn"] or metrics["na"]):
        metrics["recall"] = metrics["tp"] / metrics["injected_n"]
    if metrics["scored_n"] > 0:
        metrics["false_positive_rate"] = metrics["fp"] / metrics["scored_n"]


def _empty_localization() -> dict[str, Any]:
    return {
        "HIGH": {"correct": 0, "total": 0},
        "LOW": {"correct": 0, "total": 0},
    }


def _accumulate_localization(localization: dict[str, Any], mutated_activities: list[Activity], truth: dict[str, Any]) -> None:
    report = localize_failures(mutated_activities)
    if report.status == "no_break":
        return
    expected_step = truth["expected_breaking_step"]
    for failure in report.failures:
        if failure.breaking_gate != truth["expected_gate"]:
            continue
        bucket = failure.confidence if failure.confidence in localization else "LOW"
        localization[bucket]["total"] += 1
        if failure.breaking_step == expected_step:
            localization[bucket]["correct"] += 1
        return


def _finalize_localization(localization: dict[str, Any]) -> dict[str, Any]:
    finalized = {}
    for confidence, counts in localization.items():
        total = counts["total"]
        finalized[confidence] = {
            "correct": counts["correct"],
            "total": total,
            "accuracy": None if total == 0 else counts["correct"] / total,
        }
    return finalized


def _source_counts(clean_runs: list[dict[str, Any]], dirty_runs: list[dict[str, Any]]) -> dict[str, Any]:
    activities = [activity for run in clean_runs for activity in run["activities"]]
    kind_counts = Counter(activity.kind for activity in activities)
    total_steps = len(activities)
    structured_steps = sum(kind_counts.get(kind, 0) for kind in ("file_edit", "test_run", "tool_call"))
    total_edit_steps = sum(1 for activity in activities if activity.kind == "file_edit" and activity.tool_name == "Edit")
    g2_eligible = sum(len(eligible_steps_by_class(run["activities"])["G2_TARGET"]) for run in clean_runs)
    return {
        "clean_runs": len(clean_runs),
        "excluded_dirty_runs": len(dirty_runs),
        "total_steps": total_steps,
        "structured_steps": structured_steps,
        "opaque_shell_steps": kind_counts.get("opaque_shell", 0),
        "structured_fraction": None if total_steps == 0 else structured_steps / total_steps,
        "g2_eligible_edits": g2_eligible,
        "g2_total_edits": total_edit_steps,
        "g2_eligible_ratio": None if total_edit_steps == 0 else g2_eligible / total_edit_steps,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# agent-bisect Eval Report",
        "",
        "## Confusion Matrix",
        "",
        "| fault_class | eligible-N | injected-n | scored-n | TP | FP | FN | NA | precision | recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fault_class, metrics in report["classes"].items():
        lines.append(
            "| {fault_class} | {eligible_n} | {injected_n} | {scored_n} | {tp} | {fp} | {fn} | {na} | {precision} | {recall} |".format(
                fault_class=fault_class,
                eligible_n=metrics["eligible_n"],
                injected_n=metrics["injected_n"],
                scored_n=metrics["scored_n"],
                tp=metrics["tp"],
                fp=metrics["fp"],
                fn=metrics["fn"],
                na=metrics["na"],
                precision=_rate(metrics["precision"]),
                recall=_rate(metrics["recall"]),
            )
        )

    source = report["source"]
    lines.extend(
        [
            "",
            "Recall is the false-negative axis for G1/G2/G3 target classes. CONTROL and BENIGN are false-positive probes; precision/recall are not their headline metric.",
            "",
            "## Coverage",
            "",
            f"G2 injectable on {source['g2_eligible_edits']}/{source['g2_total_edits']} Edit steps; the rest are anchor-NA coverage limits, not misses.",
            f"Structured fraction: {_rate(source['structured_fraction'])} ({source['structured_steps']}/{source['total_steps']} steps). Injected-fault localization is an upper bound on real-run localization because injection only targets structured eligible steps.",
            "",
            "## Localization",
            "",
            "| confidence | correct | total | accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for confidence, counts in report["localization"].items():
        lines.append(f"| {confidence} | {counts['correct']} | {counts['total']} | {_rate(counts['accuracy'])} |")

    lines.extend(
        [
            "",
            "## Generalization",
            "",
            "generalization check (same Claude schema)",
            "",
            "The deterministic gates validate within the Claude tool-call schema; use foreign-schema adapters for cross-platform coverage checks.",
            "",
            "Foreign-schema adapters provide coverage checks for non-Claude trajectory formats.",
            "",
        ]
    )
    return "\n".join(lines)


def _rate(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"
