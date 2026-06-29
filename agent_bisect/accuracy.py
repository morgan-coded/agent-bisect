from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .foreign import ingest_foreign_trajectory
from .gates import run_g1, run_g2, run_g3
from .inject import FAULT_CLASSES, NON_TARGET_CLASSES, TARGET_CLASSES, eligible_steps_by_class, inject_fault
from .io import load_activities
from .localize import LocalizationResult, localize_failures, shell_target_coverage
from .model import Activity, canonical_json, sha256_text


ACCURACY_REPORT_SCHEMA_VERSION = 1
ACCURACY_SCORER_VERSION = 1
DEFAULT_SEED = 260628
DEFAULT_PER_CLASS = 1000
DEFAULT_MAX_PER_RUN = 3


@dataclass(frozen=True, slots=True)
class ActivityRun:
    run_key: str
    source: str
    activities: tuple[Activity, ...]


@dataclass(frozen=True, slots=True)
class LabeledCase:
    case_id: str
    expected_breaking_step: int | None
    expected_gate: str | None
    activities: tuple[Activity, ...]
    exclude_reason: str = ""


def build_activity_runs(paths: Iterable[Path]) -> tuple[list[ActivityRun], dict[str, Any]]:
    runs: list[ActivityRun] = []
    errors: Counter[str] = Counter()
    for path in _expand_paths(paths):
        try:
            activities = tuple(_load_activity_path(path))
        except Exception as exc:
            errors[type(exc).__name__] += 1
            continue
        source = _source_for_path(path)
        runs.append(
            ActivityRun(
                run_key=_run_key(path, activities),
                source=source,
                activities=activities,
            )
        )
    return sorted(runs, key=lambda run: (run.source, run.run_key)), {"load_errors": dict(sorted(errors.items()))}


def evaluate_controlled_accuracy(
    runs: list[ActivityRun],
    *,
    seed: int = DEFAULT_SEED,
    per_class: int = DEFAULT_PER_CLASS,
    max_per_run: int = DEFAULT_MAX_PER_RUN,
) -> dict[str, Any]:
    clean_runs, dirty_runs = _split_clean_dirty(runs)
    census = census_runs(runs)
    eligible = _eligible_candidates(clean_runs)
    selected = _select_candidates(eligible, seed=seed, per_class=per_class, max_per_run=max_per_run)

    confusion = _empty_confusion()
    localization = _empty_localization()
    selected_by_source = _selected_by_source(selected, clean_runs)
    mutation_fingerprints: list[dict[str, Any]] = []

    run_by_index = {index: run for index, run in enumerate(clean_runs)}
    for fault_class in FAULT_CLASSES:
        confusion[fault_class]["eligible_n"] = len(eligible[fault_class])
        confusion[fault_class]["injected_n"] = len(selected[fault_class])
        for run_index, step in selected[fault_class]:
            run = run_by_index[run_index]
            case = inject_fault(list(run.activities), step, fault_class)
            outcome = _score_injection_case(list(case.activities), case.truth.expected_breaking_step, case.truth.expected_gate)
            _accumulate_confusion(confusion[fault_class], fault_class, outcome)
            if fault_class in TARGET_CLASSES:
                _accumulate_localization(localization, outcome)
            mutation_fingerprints.append(
                {
                    "source": run.source,
                    "run_key": run.run_key,
                    "fault_class": fault_class,
                    "injected_step": step,
                    "expected_gate": case.truth.expected_gate,
                    "mutation_hash": case.truth.mutation_hash,
                    "source_content_hash": case.truth.source_content_hash,
                }
            )

    for row in confusion.values():
        _finalize_confusion(row)

    finalized_localization = _finalize_localization(localization)
    target_injected = sum(confusion[fault_class]["injected_n"] for fault_class in TARGET_CLASSES)
    target_exact = finalized_localization["ALL"]["exact_step_correct"]
    target_cascade = finalized_localization["ALL"]["cascade_membership_correct"]
    return {
        "schema_version": ACCURACY_REPORT_SCHEMA_VERSION,
        "scorer_version": ACCURACY_SCORER_VERSION,
        "seed": seed,
        "per_class": per_class,
        "max_per_run": max_per_run,
        "source": {
            "run_count": len(runs),
            "clean_runs": len(clean_runs),
            "excluded_dirty_runs": len(dirty_runs),
            "census": census,
        },
        "scoring_rule": scoring_rule(),
        "controlled_accuracy": {
            "target_injections": target_injected,
            "exact_step_correct": target_exact,
            "exact_step_accuracy": None if target_injected == 0 else target_exact / target_injected,
            "cascade_membership_correct": target_cascade,
            "cascade_membership_accuracy": None if target_injected == 0 else target_cascade / target_injected,
            "high_exact_step_correct": finalized_localization["HIGH"]["exact_step_correct"],
            "high_total": finalized_localization["HIGH"]["total"],
            "high_exact_step_accuracy": finalized_localization["HIGH"]["exact_step_accuracy"],
            "high_share": None if target_injected == 0 else finalized_localization["HIGH"]["total"] / target_injected,
        },
        "confusion_matrix": confusion,
        "localization": finalized_localization,
        "selected_by_source": selected_by_source,
        "mutation_fingerprints": sorted(
            mutation_fingerprints,
            key=lambda item: (item["fault_class"], item["source"], item["run_key"], item["injected_step"]),
        ),
        "boundary_companion": who_when_boundary_summary(),
        "external_dataset_decision": external_dataset_decision(),
        "baseline_context": baseline_context(),
        "notes": {
            "controlled_ground_truth": "Controlled injected-fault accuracy uses known mutations over clean real ingested runs; it is not real-world attribution accuracy.",
            "real_corpus_privacy": "Raw Claude/Codex transcripts are read-only and uncommitted; reports contain aggregate counts and hashes only.",
            "g4": "G4 fault class deferred; g4_consistency_holds is not implemented.",
        },
    }


def score_labeled_cases(cases: list[LabeledCase]) -> dict[str, Any]:
    rows = [_score_labeled_case(case) for case in sorted(cases, key=lambda case: case.case_id)]
    included = [row for row in rows if not row["exclude_reason"]]
    excluded = [row for row in rows if row["exclude_reason"]]
    summary = _summary_for_labeled_rows(included)
    return {
        "schema_version": ACCURACY_REPORT_SCHEMA_VERSION,
        "scorer_version": ACCURACY_SCORER_VERSION,
        "scoring_rule": scoring_rule(),
        "summary": summary,
        "excluded": {
            "count": len(excluded),
            "reasons": dict(sorted(Counter(row["exclude_reason"] for row in excluded).items())),
        },
        "cases": rows,
    }


def census_runs(runs: list[ActivityRun]) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    totals = _empty_census_row()
    for run in sorted(runs, key=lambda item: (item.source, item.run_key)):
        row = by_source.setdefault(run.source, _empty_census_row())
        _accumulate_census(row, run.activities)
        _accumulate_census(totals, run.activities)
    for row in [totals, *by_source.values()]:
        _finalize_census(row)
    return {"all": totals, "by_source": dict(sorted(by_source.items()))}


def scoring_rule() -> dict[str, Any]:
    return {
        "primary_metric": "exact breaking-step accuracy over all controlled target injections",
        "secondary_metric": "cascade-membership accuracy over all controlled target injections",
        "prediction_set": "localize_failures(case.activities).failures sorted by breaking_step, gate, confidence, coverage",
        "exact_hit": "predicted breaking_step equals expected_breaking_step and predicted gate matches expected_gate when known",
        "cascade_hit": "expected_breaking_step appears as predicted breaking_step or inside failure_cascade with matching gate when known",
        "high_subset": "HIGH exact accuracy is reported with HIGH share; it is not the headline unless coverage share is shown",
        "low_handling": "LOW predictions count in all-confidence metrics but are reported as coverage-limited",
        "na_handling": "No prediction or unscorable matched prediction is a miss and a coverage gap for included cases",
        "control_handling": "CONTROL and BENIGN are non-target probes; any gate-status change is a false positive",
        "controlled_vs_real": "Controlled injection measures deterministic gate localization under known structured mutations; Who&When-style labels measure semantic failure attribution and are not apples-to-apples.",
    }


def who_when_boundary_summary() -> dict[str, Any]:
    return {
        "artifact": "BENCHMARK.md",
        "rows_processed": 184,
        "included_labels": 181,
        "excluded_labels": 3,
        "exact_step": "0/181",
        "cascade_membership": "0/181",
        "coverage_gaps": "181/181",
        "interpretation": "Who&When is a semantic multi-agent attribution benchmark; its histories do not expose deterministic file/test gate failures in agent-bisect's visible envelope.",
    }


def external_dataset_decision() -> dict[str, Any]:
    return {
        "chosen_primary_source": "controlled injected-fault campaign over real ingested Claude, Codex, and fixture corpora",
        "reason": "No public coding-agent dataset found in recon that is both step-localized and naturally aligned to deterministic file-edit/test gate failures.",
        "candidates": [
            {
                "name": "TraceElephant",
                "decision": "future adapter candidate, not primary",
                "reason": "step/deceptive-action labeling is useful, but the benchmark is full-observability multi-agent attribution rather than gate-visible coding/test failures.",
            },
            {
                "name": "AgentRx",
                "decision": "schema reference only",
                "reason": "has step_number and failure_category labels, but is not a SWE-style coding/test-failure corpus.",
            },
            {
                "name": "SWE-agent/SWE-smith/Open-SWE traces",
                "decision": "no-go as external real labels",
                "reason": "trajectory/final-outcome data lacks first-bad-step labels; useful as future substrate for planted labels.",
            },
        ],
    }


def baseline_context() -> list[dict[str, str]]:
    return [
        {
            "name": "TrajAudit / RootSE",
            "url": "https://arxiv.org/abs/2605.26563",
            "metric": "coding-agent failed-trajectory step localization; reported exact-step accuracy around 50-57% depending on reference availability",
            "caveat": "post-hoc LLM diagnosis over trajectories, not deterministic gate replay or controlled injected ground truth",
        },
        {
            "name": "AgentRx",
            "url": "https://github.com/microsoft/AgentRx",
            "metric": "critical step-index accuracy and accuracy within step tolerance",
            "caveat": "step-localized trajectory diagnosis, but not a SWE-style deterministic file-edit/test-failure benchmark",
        },
        {
            "name": "Who&When",
            "url": "https://github.com/ag2ai/Agents_Failure_Attribution",
            "metric": "semantic responsible-agent and step attribution over multi-agent histories",
            "caveat": "already reported in BENCHMARK.md as a visibility boundary for agent-bisect",
        },
    ]


def write_accuracy_reports(report: dict[str, Any], reports_dir: Path, accuracy_md: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "accuracy-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    accuracy_md.write_text(render_accuracy_markdown(report), encoding="utf-8", newline="\n")


def render_accuracy_markdown(report: dict[str, Any]) -> str:
    headline = report["controlled_accuracy"]
    source = report["source"]
    census = source["census"]["all"]
    boundary = report["boundary_companion"]
    lines = [
        "# Localization Accuracy",
        "",
        "## Headline",
        "",
        "This is a controlled injected-fault result over real ingested agent transcripts. It is a positive deterministic-envelope localization number, not a real-world attribution claim.",
        "",
        "| metric | result |",
        "| --- | ---: |",
        f"| controlled exact-step accuracy | {_fraction(headline['exact_step_correct'], headline['target_injections'])} |",
        f"| controlled cascade-membership accuracy | {_fraction(headline['cascade_membership_correct'], headline['target_injections'])} |",
        f"| HIGH exact-step accuracy | {_fraction(headline['high_exact_step_correct'], headline['high_total'])} |",
        f"| HIGH share of target injections | {_fraction(headline['high_total'], headline['target_injections'])} |",
        f"| clean runs used | {source['clean_runs']} |",
        f"| excluded dirty runs | {source['excluded_dirty_runs']} |",
        "",
        "## Pre-Registered Scoring Rule",
        "",
        "- Include clean runs only for controlled injection; dirty runs are counted and excluded before sampling.",
        "- Target classes are `G1_TARGET`, `G2_TARGET`, and `G3_TARGET`; non-target probes are `CONTROL` and `BENIGN`.",
        "- Exact hit means a localized failure has the expected breaking step and expected gate.",
        "- Cascade hit means the expected step is either the localized breaking step or a member of the failure cascade, with expected gate matching when known.",
        "- `LOW` predictions count in all-confidence accuracy but are reported separately as coverage-limited.",
        "- `NA` or no prediction is a miss for included cases, not a hidden pass.",
        "- `CONTROL` and `BENIGN` measure false positives: any gate-status change is a false positive.",
        "",
        "## Corpus Census",
        "",
        f"Runs considered: {source['run_count']}; clean: {source['clean_runs']}; dirty/excluded: {source['excluded_dirty_runs']}.",
        "",
        "| source | runs | clean | dirty | activities | G1 eligible | G2 eligible | G3 eligible | control eligible | benign eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["source"]["census"]["by_source"].items():
        lines.append(_census_row(name, row))
    lines.append(_census_row("all", census))

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| class | eligible-N | injected-N | scored-N | TP | FP | FN | NA | clean | precision | recall | false-positive rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fault_class, row in report["confusion_matrix"].items():
        lines.append(
            "| {fault_class} | {eligible_n} | {injected_n} | {scored_n} | {tp} | {fp} | {fn} | {na} | {clean} | {precision} | {recall} | {fpr} |".format(
                fault_class=fault_class,
                eligible_n=row["eligible_n"],
                injected_n=row["injected_n"],
                scored_n=row["scored_n"],
                tp=row["tp"],
                fp=row["fp"],
                fn=row["fn"],
                na=row["na"],
                clean=row["clean"],
                precision=_rate(row["precision"]),
                recall=_rate(row["recall"]),
                fpr=_rate(row["false_positive_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Localization Confidence",
            "",
            "| confidence | exact correct | cascade correct | total | exact accuracy | cascade accuracy | share of targets |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for bucket in ("HIGH", "LOW", "NA", "ALL"):
        row = report["localization"][bucket]
        lines.append(
            "| {bucket} | {exact} | {cascade} | {total} | {exact_rate} | {cascade_rate} | {share} |".format(
                bucket=bucket,
                exact=row["exact_step_correct"],
                cascade=row["cascade_membership_correct"],
                total=row["total"],
                exact_rate=_rate(row["exact_step_accuracy"]),
                cascade_rate=_rate(row["cascade_membership_accuracy"]),
                share=_rate(row["target_share"]),
            )
        )

    lines.extend(
        [
            "",
            "## Honest Two-Sided Result",
            "",
            f"The positive number above is controlled: target failures are injected into clean real ingested runs, so the breaking step is known by construction. It should be read as deterministic-envelope localization under known structured mutations.",
            "",
            f"The boundary companion remains `{boundary['artifact']}`: Who&When exact-step `{boundary['exact_step']}`, cascade `{boundary['cascade_membership']}`, coverage gaps `{boundary['coverage_gaps']}` over {boundary['included_labels']} included labels. That benchmark is semantic multi-agent failure attribution over natural-language histories, so the 0% result is a visibility boundary, not a contradiction.",
            "",
            "## Published Baseline Context",
            "",
            "These references are useful context, not direct apples-to-apples baselines for the controlled number above.",
            "",
            "| reference | metric | caveat |",
            "| --- | --- | --- |",
        ]
    )
    for item in report["baseline_context"]:
        lines.append(f"| [{item['name']}]({item['url']}) | {item['metric']} | {item['caveat']} |")

    lines.extend(
        [
            "",
            "## External Dataset Decision",
            "",
            report["external_dataset_decision"]["reason"],
            "",
            "| candidate | decision | reason |",
            "| --- | --- | --- |",
        ]
    )
    for candidate in report["external_dataset_decision"]["candidates"]:
        lines.append(f"| {candidate['name']} | {candidate['decision']} | {candidate['reason']} |")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is controlled ground truth, not real-world attribution accuracy.",
            "- Injection targets structured eligible steps; this can overstate performance on opaque or poorly linked transcripts.",
            "- Codex `apply_patch` activities are summarized as patches and are not G2 `Edit` anchors today, so G2 eligibility is mostly from Claude/foreign-style `Edit` records.",
            "- Reports contain aggregate counts and hashes only; raw local transcripts and generated journals stay uncommitted.",
            "",
            "## Determinism And Lineage",
            "",
            f"Seed: `{report['seed']}`; per-class cap: `{report['per_class']}`; max per run: `{report['max_per_run']}`.",
            "",
            "The JSON report is sorted and deterministic for the same inputs. Mutation fingerprints store source, hashed run key, class, step, expected gate, mutation hash, and source-content hash only.",
            "",
        ]
    )
    return "\n".join(lines)


def _score_labeled_case(case: LabeledCase) -> dict[str, Any]:
    if case.exclude_reason:
        return _excluded_labeled_row(case, case.exclude_reason)
    if case.expected_breaking_step is None:
        return _excluded_labeled_row(case, "missing_expected_breaking_step")
    if not (0 <= case.expected_breaking_step < len(case.activities)):
        return _excluded_labeled_row(case, "expected_breaking_step_out_of_range")

    outcome = _score_prediction(tuple(localize_failures(case.activities).failures), case.expected_breaking_step, case.expected_gate)
    return {
        "case_id": case.case_id,
        "expected_breaking_step": case.expected_breaking_step,
        "expected_gate": case.expected_gate or "",
        "exclude_reason": "",
        **outcome,
    }


def _excluded_labeled_row(case: LabeledCase, reason: str) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "expected_breaking_step": case.expected_breaking_step,
        "expected_gate": case.expected_gate or "",
        "exclude_reason": reason,
        "prediction_count": 0,
        "matched_confidence": "NA",
        "exact_step_hit": False,
        "cascade_membership_hit": False,
        "coverage_gap": True,
        "coverage_gap_reason": reason,
    }


def _score_injection_case(
    activities: list[Activity],
    expected_breaking_step: int | None,
    expected_gate: str,
) -> dict[str, Any]:
    after_statuses = _gate_statuses(activities)
    any_fail = any(status == "FAIL" for step in after_statuses.values() for status in step.values())
    if expected_gate == "NONE" or expected_breaking_step is None:
        return {"bucket": "clean", "gate_status_changed": False} if not any_fail else {"bucket": "fp", "gate_status_changed": True}

    expected_status = after_statuses.get(expected_breaking_step, {}).get(expected_gate, "NA")
    localization = localize_failures(activities)
    prediction = _score_prediction(tuple(localization.failures), expected_breaking_step, expected_gate)
    if expected_status == "FAIL":
        return {"bucket": "tp", "gate_status_changed": True, **prediction}
    if expected_status == "NA":
        return {"bucket": "na", "gate_status_changed": False, **prediction}
    if any_fail:
        return {"bucket": "fp", "gate_status_changed": True, **prediction}
    return {"bucket": "fn", "gate_status_changed": False, **prediction}


def _score_prediction(
    predictions: tuple[LocalizationResult, ...],
    expected_breaking_step: int,
    expected_gate: str | None,
) -> dict[str, Any]:
    sorted_predictions = sorted(predictions, key=lambda item: (item.breaking_step, item.breaking_gate, item.confidence, item.coverage))
    exact = next((item for item in sorted_predictions if _gate_matches(item, expected_gate) and item.breaking_step == expected_breaking_step), None)
    cascade = next(
        (
            item
            for item in sorted_predictions
            if _gate_matches(item, expected_gate)
            and expected_breaking_step in {item.breaking_step, *item.failure_cascade}
        ),
        None,
    )
    matched = exact or cascade or (sorted_predictions[0] if sorted_predictions else None)
    coverage_gap, coverage_gap_reason = _coverage_gap(matched)
    return {
        "prediction_count": len(sorted_predictions),
        "matched_breaking_step": None if matched is None else matched.breaking_step,
        "matched_gate": None if matched is None else matched.breaking_gate,
        "matched_confidence": "NA" if matched is None else matched.confidence,
        "exact_step_hit": exact is not None,
        "cascade_membership_hit": cascade is not None,
        "coverage_gap": coverage_gap,
        "coverage_gap_reason": coverage_gap_reason,
    }


def _gate_matches(prediction: LocalizationResult, expected_gate: str | None) -> bool:
    return not expected_gate or prediction.breaking_gate == expected_gate


def _coverage_gap(matched: LocalizationResult | None) -> tuple[bool, str]:
    if matched is None:
        return True, "no_prediction"
    if matched.confidence == "LOW":
        return True, matched.coverage
    lowered = matched.coverage.lower()
    for token in ("opaque", "unmapped", "unlinked", "heuristic"):
        if token in lowered:
            return True, matched.coverage
    return False, ""


def _split_clean_dirty(runs: list[ActivityRun]) -> tuple[list[ActivityRun], list[ActivityRun]]:
    clean: list[ActivityRun] = []
    dirty: list[ActivityRun] = []
    for run in runs:
        (dirty if _has_gate_failures(run.activities) else clean).append(run)
    return clean, dirty


def _has_gate_failures(activities: Iterable[Activity]) -> bool:
    return any(result.status == "FAIL" for result in [*run_g1(activities), *run_g2(activities), *run_g3(activities)])


def _gate_statuses(activities: list[Activity]) -> dict[int, dict[str, str]]:
    statuses: dict[int, dict[str, str]] = {}
    for result in [*run_g1(activities), *run_g2(activities), *run_g3(activities)]:
        if result.step_index is not None:
            statuses.setdefault(result.step_index, {})[result.gate] = result.status
    return statuses


def _eligible_candidates(runs: list[ActivityRun]) -> dict[str, list[tuple[int, int]]]:
    candidates: dict[str, list[tuple[int, int]]] = {fault_class: [] for fault_class in FAULT_CLASSES}
    for run_index, run in enumerate(runs):
        eligible = eligible_steps_by_class(list(run.activities))
        for fault_class, steps in eligible.items():
            candidates[fault_class].extend((run_index, step) for step in steps)
    return {fault_class: sorted(values) for fault_class, values in candidates.items()}


def _select_candidates(
    eligible: dict[str, list[tuple[int, int]]],
    *,
    seed: int,
    per_class: int,
    max_per_run: int,
) -> dict[str, list[tuple[int, int]]]:
    selected: dict[str, list[tuple[int, int]]] = {}
    for fault_class, candidates in eligible.items():
        if per_class <= 0 or not candidates:
            selected[fault_class] = []
            continue
        shuffled = list(candidates)
        rng_seed = int(sha256_text(f"{seed}:{fault_class}:accuracy")[:16], 16)
        random.Random(rng_seed).shuffle(shuffled)
        per_run_counts: Counter[int] = Counter()
        picked: list[tuple[int, int]] = []
        for run_index, step in shuffled:
            if max_per_run > 0 and per_run_counts[run_index] >= max_per_run:
                continue
            picked.append((run_index, step))
            per_run_counts[run_index] += 1
            if len(picked) >= min(per_class, len(candidates)):
                break
        selected[fault_class] = sorted(picked)
    return selected


def _selected_by_source(selected: dict[str, list[tuple[int, int]]], clean_runs: list[ActivityRun]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: {fault_class: 0 for fault_class in FAULT_CLASSES})
    for fault_class, items in selected.items():
        for run_index, _step in items:
            result[clean_runs[run_index].source][fault_class] += 1
    return {source: dict(sorted(counts.items())) for source, counts in sorted(result.items())}


def _empty_confusion() -> dict[str, dict[str, Any]]:
    return {
        fault_class: {
            "eligible_n": 0,
            "injected_n": 0,
            "scored_n": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "na": 0,
            "clean": 0,
            "precision": None,
            "recall": None,
            "false_positive_rate": None,
        }
        for fault_class in FAULT_CLASSES
    }


def _accumulate_confusion(row: dict[str, Any], fault_class: str, outcome: dict[str, Any]) -> None:
    bucket = outcome["bucket"]
    if bucket == "tp":
        row["tp"] += 1
        row["scored_n"] += 1
    elif bucket == "fp":
        row["fp"] += 1
        row["scored_n"] += 1
    elif bucket == "fn":
        row["fn"] += 1
        row["scored_n"] += 1
    elif bucket == "na":
        row["na"] += 1
    elif bucket == "clean":
        row["clean"] += 1
        row["scored_n"] += 1
    else:
        raise ValueError(f"unknown bucket: {bucket}")
    if fault_class in TARGET_CLASSES and bucket == "clean":
        row["fn"] += 1


def _finalize_confusion(row: dict[str, Any]) -> None:
    if row["tp"] + row["fp"] > 0:
        row["precision"] = row["tp"] / (row["tp"] + row["fp"])
    if row["injected_n"] > 0 and (row["tp"] or row["fn"] or row["na"]):
        row["recall"] = row["tp"] / row["injected_n"]
    if row["scored_n"] > 0:
        row["false_positive_rate"] = row["fp"] / row["scored_n"]


def _empty_localization() -> dict[str, dict[str, int]]:
    return {
        "HIGH": {"exact_step_correct": 0, "cascade_membership_correct": 0, "total": 0, "coverage_gap_count": 0},
        "LOW": {"exact_step_correct": 0, "cascade_membership_correct": 0, "total": 0, "coverage_gap_count": 0},
        "NA": {"exact_step_correct": 0, "cascade_membership_correct": 0, "total": 0, "coverage_gap_count": 0},
    }


def _accumulate_localization(localization: dict[str, dict[str, int]], outcome: dict[str, Any]) -> None:
    bucket = outcome.get("matched_confidence", "NA")
    if bucket not in localization:
        bucket = "LOW"
    localization[bucket]["total"] += 1
    if outcome.get("exact_step_hit"):
        localization[bucket]["exact_step_correct"] += 1
    if outcome.get("cascade_membership_hit"):
        localization[bucket]["cascade_membership_correct"] += 1
    if outcome.get("coverage_gap"):
        localization[bucket]["coverage_gap_count"] += 1


def _finalize_localization(localization: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    total = sum(row["total"] for row in localization.values())
    all_row = {
        "exact_step_correct": sum(row["exact_step_correct"] for row in localization.values()),
        "cascade_membership_correct": sum(row["cascade_membership_correct"] for row in localization.values()),
        "total": total,
        "coverage_gap_count": sum(row["coverage_gap_count"] for row in localization.values()),
    }
    rows = {**localization, "ALL": all_row}
    finalized: dict[str, dict[str, Any]] = {}
    for confidence, row in rows.items():
        row_total = row["total"]
        finalized[confidence] = {
            **row,
            "exact_step_accuracy": None if row_total == 0 else row["exact_step_correct"] / row_total,
            "cascade_membership_accuracy": None if row_total == 0 else row["cascade_membership_correct"] / row_total,
            "target_share": None if total == 0 else row_total / total,
            "coverage_gap_rate": None if row_total == 0 else row["coverage_gap_count"] / row_total,
        }
    return finalized


def _summary_for_labeled_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    localization = _empty_localization()
    for row in rows:
        _accumulate_localization(localization, row)
    return _finalize_localization(localization)


def _empty_census_row() -> dict[str, Any]:
    return {
        "runs": 0,
        "clean_runs": 0,
        "dirty_runs": 0,
        "activities": 0,
        "kind_counts": {},
        "eligible": {fault_class: 0 for fault_class in FAULT_CLASSES},
        "shell_command_steps": 0,
        "shell_steps_with_targets": 0,
        "shell_added_edges": 0,
        "g2_total_edits": 0,
        "g2_eligible_edits": 0,
    }


def _accumulate_census(row: dict[str, Any], activities: Iterable[Activity]) -> None:
    activity_list = list(activities)
    row["runs"] += 1
    if _has_gate_failures(activity_list):
        row["dirty_runs"] += 1
    else:
        row["clean_runs"] += 1
        eligible = eligible_steps_by_class(activity_list)
        for fault_class, steps in eligible.items():
            row["eligible"][fault_class] += len(steps)
    row["activities"] += len(activity_list)
    kind_counts = Counter(row["kind_counts"])
    kind_counts.update(activity.kind for activity in activity_list)
    row["kind_counts"] = dict(sorted(kind_counts.items()))
    shell_coverage = shell_target_coverage(activity_list)
    row["shell_command_steps"] += shell_coverage.shell_command_steps
    row["shell_steps_with_targets"] += shell_coverage.steps_with_targets
    row["shell_added_edges"] += shell_coverage.added_edges
    row["g2_total_edits"] += sum(1 for activity in activity_list if activity.kind == "file_edit" and activity.tool_name == "Edit")
    if not _has_gate_failures(activity_list):
        row["g2_eligible_edits"] += len(eligible_steps_by_class(activity_list)["G2_TARGET"])


def _finalize_census(row: dict[str, Any]) -> None:
    row["structured_steps"] = sum(row["kind_counts"].get(kind, 0) for kind in ("file_edit", "test_run", "tool_call"))
    row["structured_fraction"] = None if row["activities"] == 0 else row["structured_steps"] / row["activities"]
    row["g2_eligible_ratio"] = None if row["g2_total_edits"] == 0 else row["g2_eligible_edits"] / row["g2_total_edits"]


def _census_row(name: str, row: dict[str, Any]) -> str:
    eligible = row["eligible"]
    return (
        f"| {name} | {row['runs']} | {row['clean_runs']} | {row['dirty_runs']} | {row['activities']} | "
        f"{eligible['G1_TARGET']} | {eligible['G2_TARGET']} | {eligible['G3_TARGET']} | "
        f"{eligible['CONTROL']} | {eligible['BENIGN']} |"
    )


def _expand_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file() and _is_activity_candidate(candidate)
                )
            )
        elif path.is_file():
            expanded.append(path)
    return sorted({path.resolve() for path in expanded})


def _is_activity_candidate(path: Path) -> bool:
    if path.suffix == ".jsonl":
        return True
    return _is_fixture_path(path) and (path.suffix in {".json", ".traj"} or path.name.endswith(".traj"))


def _load_activity_path(path: Path) -> list[Activity]:
    schema = _foreign_schema_for_path(path)
    if schema is not None:
        return ingest_foreign_trajectory(path, schema=schema)
    return load_activities(path)


def _foreign_schema_for_path(path: Path) -> str | None:
    name = path.name.lower()
    if "swe_agent" in name or "swe-agent" in name:
        if "mini" in name:
            return "mini-swe-agent"
        return "swe-agent"
    if "openhands" in name:
        return "openhands"
    return None


def _source_for_path(path: Path) -> str:
    lowered = str(path).lower().replace("\\", "/")
    if "/.codex/" in lowered:
        return "codex"
    if "/.claude/" in lowered:
        return "claude"
    if _is_fixture_path(path):
        schema = _foreign_schema_for_path(path)
        if schema is not None:
            return f"foreign-{schema}"
        return "fixture"
    return "journal"


def _is_fixture_path(path: Path) -> bool:
    normalized = str(path).lower().replace("\\", "/")
    return "/tests/fixtures/" in normalized or normalized.startswith("tests/fixtures/")


def _run_key(path: Path, activities: tuple[Activity, ...]) -> str:
    first_hash = activities[0].content_hash if activities else ""
    return sha256_text(f"{_source_for_path(path)}:{path.stem}:{len(activities)}:{first_hash}")[:16]


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({_rate(None if denominator == 0 else numerator / denominator)})"


def _rate(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"
