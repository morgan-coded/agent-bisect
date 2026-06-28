from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
from typing import Any

from .localize import LocalizationResult, localize_failures
from .model import Activity, canonical_json, sha256_text


WHO_WHEN_DATASET = "Kevin355/Who_and_When"
WHO_WHEN_CONFIGS = ("Algorithm-Generated", "Hand-Crafted")
WHO_WHEN_SPLIT = "train"
WHO_WHEN_REPORT_SCHEMA_VERSION = 1
WHO_WHEN_SCORER_VERSION = 1
WHO_WHEN_BASELINE_STEP_ACCURACY = 0.141525
WHO_WHEN_BASELINE_STEP_CELLS = (25.51, 7.02, 15.31, 8.77)


@dataclass(frozen=True, slots=True)
class WhoWhenLabel:
    label_id: str
    config: str
    split: str
    row_idx: int
    question_id: str
    expected_breaking_step: int | None
    responsible_agent: str
    mistake_reason_hash: str
    exclude_reason: str = ""


@dataclass(frozen=True, slots=True)
class WhoWhenCase:
    label: WhoWhenLabel
    activities: tuple[Activity, ...]


def fetch_who_when_dataset(
    out_dir: Path,
    *,
    limit_per_config: int | None = None,
    page_size: int = 100,
) -> dict[str, Any]:
    """Fetch Who&When rows into ignored local cache with a pinned revision."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if limit_per_config is not None and limit_per_config <= 0:
        raise ValueError("limit_per_config must be positive when set")

    out_dir.mkdir(parents=True, exist_ok=True)
    revision = _huggingface_dataset_revision(WHO_WHEN_DATASET)
    sources: list[dict[str, Any]] = []
    config_totals: dict[str, int] = {}

    for config in WHO_WHEN_CONFIGS:
        fetched_for_config = 0
        offset = 0
        total_rows: int | None = None
        while True:
            if limit_per_config is not None and fetched_for_config >= limit_per_config:
                break
            length = page_size
            if limit_per_config is not None:
                length = min(length, limit_per_config - fetched_for_config)
            page = _huggingface_rows(
                WHO_WHEN_DATASET,
                config,
                WHO_WHEN_SPLIT,
                offset=offset,
                length=length,
                revision=revision,
            )
            rows = page.get("rows")
            if not isinstance(rows, list) or not rows:
                break
            if isinstance(page.get("num_rows_total"), int):
                total_rows = int(page["num_rows_total"])
                config_totals[config] = total_rows

            for wrapper in rows:
                if not isinstance(wrapper, dict) or not isinstance(wrapper.get("row"), dict):
                    continue
                row_idx = wrapper.get("row_idx")
                row_idx_int = int(row_idx) if isinstance(row_idx, int) else offset + fetched_for_config
                row = wrapper["row"]
                local_path = out_dir / _safe_filename(config) / f"{row_idx_int:04d}.json"
                local_path.parent.mkdir(parents=True, exist_ok=True)
                source_url = _huggingface_rows_url(
                    WHO_WHEN_DATASET,
                    config,
                    WHO_WHEN_SPLIT,
                    offset=row_idx_int,
                    length=1,
                    revision=revision,
                )
                payload = {
                    "schema_version": 1,
                    "dataset": WHO_WHEN_DATASET,
                    "dataset_revision": revision,
                    "config": config,
                    "split": WHO_WHEN_SPLIT,
                    "row_idx": row_idx_int,
                    "source_url": source_url,
                    "row": row,
                }
                local_path.write_text(
                    json.dumps(payload, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                sources.append(
                    {
                        "dataset": WHO_WHEN_DATASET,
                        "dataset_revision": revision,
                        "config": config,
                        "split": WHO_WHEN_SPLIT,
                        "row_idx": row_idx_int,
                        "local_path": local_path.resolve().as_posix(),
                        "source_url": source_url,
                        "question_id": str(row.get("question_ID") or row.get("question_id") or ""),
                    }
                )
                fetched_for_config += 1

            offset += len(rows)
            if total_rows is not None and offset >= total_rows:
                break

    manifest = {
        "schema_version": 1,
        "dataset": WHO_WHEN_DATASET,
        "dataset_revision": revision,
        "split": WHO_WHEN_SPLIT,
        "configs": list(WHO_WHEN_CONFIGS),
        "config_totals": config_totals,
        "sources": sorted(sources, key=lambda item: (item["config"], item["row_idx"])),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def load_who_when_cases(path: Path) -> tuple[list[WhoWhenCase], dict[str, Any]]:
    manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases: list[WhoWhenCase] = []
    for source in sorted(manifest.get("sources", []), key=lambda item: (item.get("config", ""), item.get("row_idx", 0))):
        local_path = Path(str(source["local_path"]))
        payload = json.loads(local_path.read_text(encoding="utf-8"))
        row = payload.get("row")
        if not isinstance(row, dict):
            row = {}
        config = str(payload.get("config") or source.get("config") or "")
        split = str(payload.get("split") or source.get("split") or WHO_WHEN_SPLIT)
        row_idx = int(payload.get("row_idx") if isinstance(payload.get("row_idx"), int) else source.get("row_idx", 0))
        source_url = str(payload.get("source_url") or source.get("source_url") or "")
        run_id = f"who-when:{config}:{row_idx}"
        activities = tuple(_activities_from_who_when_row(row, run_id=run_id, source_url=source_url))
        label = _label_from_who_when_row(row, config=config, split=split, row_idx=row_idx, activity_count=len(activities))
        cases.append(WhoWhenCase(label=label, activities=activities))
    return cases, manifest


def score_who_when_cases(cases: list[WhoWhenCase], manifest: dict[str, Any]) -> dict[str, Any]:
    scored = [_score_case(case) for case in sorted(cases, key=lambda case: (case.label.config, case.label.row_idx))]
    included = [row for row in scored if not row["exclude_reason"]]
    excluded = [row for row in scored if row["exclude_reason"]]
    by_config = {
        config: _summary_for_rows([row for row in included if row["config"] == config])
        for config in WHO_WHEN_CONFIGS
    }
    report = {
        "schema_version": WHO_WHEN_REPORT_SCHEMA_VERSION,
        "scorer_version": WHO_WHEN_SCORER_VERSION,
        "dataset": {
            "name": str(manifest.get("dataset", WHO_WHEN_DATASET)),
            "revision": str(manifest.get("dataset_revision", "")),
            "split": str(manifest.get("split", WHO_WHEN_SPLIT)),
            "configs": list(WHO_WHEN_CONFIGS),
            "manifest_hash": sha256_text(canonical_json(_manifest_fingerprint(manifest))),
            "source": "https://huggingface.co/datasets/Kevin355/Who_and_When",
            "paper": "https://arxiv.org/abs/2505.00212",
            "repo": "https://github.com/ag2ai/Agents_Failure_Attribution",
        },
        "scoring_rule": {
            "primary_denominator": "labels with integer mistake_step inside mapped Activity list",
            "step_index_base": "zero-based history index mapped one-to-one to Activity.step_index",
            "primary_metric": "exact_step_accuracy",
            "secondary_metric": "cascade_membership_accuracy",
            "agent_attribution": "unsupported by LocalizationReport schema",
        },
        "baseline_context": {
            "who_when_step_accuracy_context": WHO_WHEN_BASELINE_STEP_ACCURACY,
            "who_when_step_accuracy_cells_percent": list(WHO_WHEN_BASELINE_STEP_CELLS),
            "comparison_caveat": "Who&When is semantic multi-agent failure attribution; agent-bisect is deterministic transcript-localization over visible gates.",
        },
        "summary": _summary_for_rows(included),
        "by_config": by_config,
        "adapter_coverage": _adapter_coverage(cases),
        "excluded": _excluded_summary(excluded),
        "cases": scored,
    }
    return report


def write_who_when_reports(report: dict[str, Any], reports_dir: Path, benchmark_md: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "who-when-benchmark.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    benchmark_md.write_text(render_who_when_markdown(report), encoding="utf-8", newline="\n")


def render_who_when_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    dataset = report["dataset"]
    baseline = report["baseline_context"]
    coverage = report["adapter_coverage"]
    lines = [
        "# Who&When Localization Benchmark",
        "",
        "## Headline",
        "",
        "This is an honest coverage-limited benchmark of `agent-bisect` against Who&When labels. It is not an apples-to-apples comparison to the published Who&When LLM attribution baseline.",
        "",
        "| metric | result |",
        "| --- | ---: |",
        f"| exact-step accuracy | {_fraction(summary['exact_step_correct'], summary['included_label_count'])} |",
        f"| cascade-membership accuracy | {_fraction(summary['cascade_membership_correct'], summary['included_label_count'])} |",
        f"| coverage-gap rate | {_fraction(summary['coverage_gap_count'], summary['included_label_count'])} |",
        f"| rows processed | {coverage['case_count']} |",
        f"| included labels | {summary['included_label_count']} |",
        f"| excluded labels | {report['excluded']['count']} |",
        "",
        "## Confidence Split",
        "",
        "| confidence | exact correct | total | exact accuracy | share of labels |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for bucket in ("HIGH", "LOW", "NA"):
        row = summary["confidence"][bucket]
        lines.append(
            "| {bucket} | {correct} | {total} | {accuracy} | {share} |".format(
                bucket=bucket,
                correct=row["exact_step_correct"],
                total=row["total"],
                accuracy=_rate(row["exact_step_accuracy"]),
                share=_rate(row["all_label_share"]),
            )
        )

    lines.extend(
        [
            "",
            "## By Config",
            "",
            "| config | exact | cascade | coverage gaps | labels |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for config in WHO_WHEN_CONFIGS:
        row = report["by_config"][config]
        lines.append(
            "| {config} | {exact} | {cascade} | {gaps} | {labels} |".format(
                config=config,
                exact=_fraction(row["exact_step_correct"], row["included_label_count"]),
                cascade=_fraction(row["cascade_membership_correct"], row["included_label_count"]),
                gaps=_fraction(row["coverage_gap_count"], row["included_label_count"]),
                labels=row["included_label_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Excluded Labels",
            "",
            "| reason | count |",
            "| --- | ---: |",
        ]
    )
    if report["excluded"]["reasons"]:
        for reason, count in report["excluded"]["reasons"].items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## What Was Scored",
            "",
            f"Dataset: `{dataset['name']}` at revision `{dataset['revision']}`.",
            "",
            f"Manifest hash: `{dataset['manifest_hash']}`.",
            "",
            "Who&When labels `mistake_agent` and `mistake_step` over multi-agent conversation histories. This benchmark maps each history item one-to-one into an `Activity`, with `mistake_step` treated as a zero-based `Activity.step_index`.",
            "",
            "The adapter does not infer hidden file edits, commands, exit codes, or test failures from free text. Agent messages are preserved as `llm_call`/`user_msg`; terminal/computer records without recoverable commands become explicit `unmapped` activities.",
            "",
            "## Adapter Coverage",
            "",
            "| activity kind | count | share |",
            "| --- | ---: | ---: |",
        ]
    )
    for kind, count in coverage["kind_counts"].items():
        lines.append(f"| {kind} | {count} | {_rate(count / coverage['activity_count'] if coverage['activity_count'] else None)} |")

    lines.extend(
        [
            "",
            f"Rows: {coverage['case_count']}; activities: {coverage['activity_count']}; unmapped activities: {coverage['unmapped_activities']}.",
            "",
            "## Baseline Context",
            "",
            "The Who&When paper reports a best semantic step-level attribution result of about {baseline} for GPT-4o Step-by-Step judging, from the cells {cells}. That method judges multi-agent natural-language logs and sometimes has final-answer ground truth. `agent-bisect` instead localizes deterministic gate-visible breaks in normalized transcripts.".format(
                baseline=_percent(baseline["who_when_step_accuracy_context"]),
                cells=", ".join(f"{cell:.2f}%" for cell in baseline["who_when_step_accuracy_cells_percent"]),
            ),
            "",
            "Because the Who&When histories do not expose deterministic file/test gate failures in the format `agent-bisect` requires, the clean comparison is ill-posed. The result above is a standalone visibility result over the full labeled set, not an `Nx better` claim.",
            "",
            "## Lineage And License",
            "",
            "Sources: [Who&When on Hugging Face](https://huggingface.co/datasets/Kevin355/Who_and_When), [paper](https://arxiv.org/abs/2505.00212), and [GitHub repository](https://github.com/ag2ai/Agents_Failure_Attribution).",
            "",
            "Raw Who&When rows are fetched from source into ignored local `data/` only. They are not committed. The Hugging Face dataset card does not declare a dataset license; the GitHub repository is MIT-licensed, and the dataset is based on GAIA and AssistantBench tasks, so this repository ships only code, synthetic tests, and aggregate results.",
            "",
            "## Determinism",
            "",
            "The scorer sorts labels and predictions deterministically and writes canonical JSON to `reports/who-when-benchmark.json`. Re-running over the same manifest should produce byte-identical JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def _activities_from_who_when_row(row: dict[str, Any], *, run_id: str, source_url: str) -> list[Activity]:
    history = row.get("history")
    if not isinstance(history, list):
        return [_unmapped(run_id, 0, 0, source_url, "missing_history", row, None)]

    activities: list[Activity] = []
    for source_index, record in enumerate(history):
        parent_step = source_index - 1 if source_index > 0 else None
        activities.append(
            _activity_from_who_when_record(
                record,
                run_id=run_id,
                step_index=source_index,
                source_index=source_index,
                source_url=source_url,
                parent_step=parent_step,
            )
        )
    return activities


def _activity_from_who_when_record(
    record: Any,
    *,
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    parent_step: int | None,
) -> Activity:
    if not isinstance(record, dict):
        return _unmapped(run_id, step_index, source_index, source_url, "record_not_object", record, parent_step)

    role = str(record.get("role") or "")
    name = str(record.get("name") or "")
    content = record.get("content")
    inputs = {
        "source_index": source_index,
        "source_url": source_url,
        "role": role,
        "agent": _agent_name(role, name),
        "content_length": len(content) if isinstance(content, str) else 0,
        "content_hash": sha256_text(content) if isinstance(content, str) else "",
    }

    if _is_terminal_record(role, name):
        return _unmapped(
            run_id,
            step_index,
            source_index,
            source_url,
            "terminal_without_structured_command",
            record,
            parent_step,
        )

    if role.lower() in {"human", "user"} and not name:
        kind = "user_msg"
    elif _agent_name(role, name):
        kind = "llm_call"
    else:
        return _unmapped(run_id, step_index, source_index, source_url, "unknown_actor", record, parent_step)

    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts="",
        kind=kind,
        tool_name=inputs["agent"] or None,
        inputs=inputs,
        target=inputs["agent"] or None,
        parent_step=parent_step,
    )


def _label_from_who_when_row(
    row: dict[str, Any],
    *,
    config: str,
    split: str,
    row_idx: int,
    activity_count: int,
) -> WhoWhenLabel:
    question_id = str(row.get("question_ID") or row.get("question_id") or f"{config}:{row_idx}")
    label_id = f"{config}:{row_idx}:{question_id}"
    mistake_step_raw = row.get("mistake_step")
    expected_step: int | None = None
    exclude_reason = ""
    if isinstance(mistake_step_raw, bool):
        exclude_reason = "mistake_step_not_integer"
    elif isinstance(mistake_step_raw, int):
        expected_step = mistake_step_raw
    elif isinstance(mistake_step_raw, str) and re.fullmatch(r"\d+", mistake_step_raw.strip()):
        expected_step = int(mistake_step_raw.strip())
    else:
        exclude_reason = "mistake_step_not_integer"

    if expected_step is not None and not (0 <= expected_step < activity_count):
        exclude_reason = "mistake_step_out_of_range"

    mistake_reason = row.get("mistake_reason")
    return WhoWhenLabel(
        label_id=label_id,
        config=config,
        split=split,
        row_idx=row_idx,
        question_id=question_id,
        expected_breaking_step=expected_step,
        responsible_agent=str(row.get("mistake_agent") or ""),
        mistake_reason_hash=sha256_text(mistake_reason) if isinstance(mistake_reason, str) else "",
        exclude_reason=exclude_reason,
    )


def _score_case(case: WhoWhenCase) -> dict[str, Any]:
    label = case.label
    base = {
        "label_id": label.label_id,
        "config": label.config,
        "split": label.split,
        "row_idx": label.row_idx,
        "question_id": label.question_id,
        "expected_breaking_step": label.expected_breaking_step,
        "responsible_agent": label.responsible_agent,
        "activity_count": len(case.activities),
        "exclude_reason": label.exclude_reason,
    }
    if label.exclude_reason or label.expected_breaking_step is None:
        return {
            **base,
            "report_status": "excluded",
            "prediction_count": 0,
            "matched_confidence": "NA",
            "exact_step_hit": False,
            "cascade_membership_hit": False,
            "coverage_gap": True,
            "coverage_gap_reason": label.exclude_reason or "missing_label",
        }

    localization = localize_failures(case.activities)
    predictions = sorted(
        localization.failures,
        key=lambda failure: (failure.breaking_step, failure.breaking_gate, failure.confidence, failure.coverage),
    )
    exact = next((failure for failure in predictions if failure.breaking_step == label.expected_breaking_step), None)
    cascade = next(
        (
            failure
            for failure in predictions
            if label.expected_breaking_step in {failure.breaking_step, *failure.failure_cascade}
        ),
        None,
    )
    matched = exact or cascade or (predictions[0] if predictions else None)
    coverage_gap, coverage_gap_reason = _coverage_gap(matched)
    return {
        **base,
        "report_status": localization.status,
        "prediction_count": len(predictions),
        "matched_breaking_step": None if matched is None else matched.breaking_step,
        "matched_breaking_gate": None if matched is None else matched.breaking_gate,
        "matched_confidence": "NA" if matched is None else matched.confidence,
        "matched_coverage": None if matched is None else matched.coverage,
        "exact_step_hit": exact is not None,
        "cascade_membership_hit": cascade is not None,
        "coverage_gap": coverage_gap,
        "coverage_gap_reason": coverage_gap_reason,
    }


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


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = len(rows)
    exact = sum(1 for row in rows if row["exact_step_hit"])
    cascade = sum(1 for row in rows if row["cascade_membership_hit"])
    gaps = sum(1 for row in rows if row["coverage_gap"])
    confidence = {}
    for bucket in ("HIGH", "LOW", "NA"):
        bucket_rows = [row for row in rows if row["matched_confidence"] == bucket]
        bucket_exact = sum(1 for row in bucket_rows if row["exact_step_hit"])
        bucket_cascade = sum(1 for row in bucket_rows if row["cascade_membership_hit"])
        confidence[bucket] = {
            "total": len(bucket_rows),
            "exact_step_correct": bucket_exact,
            "cascade_membership_correct": bucket_cascade,
            "exact_step_accuracy": None if not bucket_rows else bucket_exact / len(bucket_rows),
            "cascade_membership_accuracy": None if not bucket_rows else bucket_cascade / len(bucket_rows),
            "all_label_share": None if denominator == 0 else len(bucket_rows) / denominator,
        }
    return {
        "included_label_count": denominator,
        "exact_step_correct": exact,
        "exact_step_accuracy": None if denominator == 0 else exact / denominator,
        "cascade_membership_correct": cascade,
        "cascade_membership_accuracy": None if denominator == 0 else cascade / denominator,
        "coverage_gap_count": gaps,
        "coverage_gap_rate": None if denominator == 0 else gaps / denominator,
        "confidence": confidence,
    }


def _adapter_coverage(cases: list[WhoWhenCase]) -> dict[str, Any]:
    kind_counts = Counter(activity.kind for case in cases for activity in case.activities)
    activity_count = sum(kind_counts.values())
    return {
        "case_count": len(cases),
        "activity_count": activity_count,
        "kind_counts": dict(sorted(kind_counts.items())),
        "unmapped_activities": kind_counts.get("unmapped", 0),
        "unmapped_activity_rate": None if activity_count == 0 else kind_counts.get("unmapped", 0) / activity_count,
    }


def _excluded_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(row["exclude_reason"] for row in rows if row["exclude_reason"])
    return {
        "count": len(rows),
        "reasons": dict(sorted(reasons.items())),
    }


def _manifest_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": manifest.get("dataset"),
        "dataset_revision": manifest.get("dataset_revision"),
        "split": manifest.get("split"),
        "sources": [
            {
                "config": source.get("config"),
                "row_idx": source.get("row_idx"),
                "question_id": source.get("question_id"),
            }
            for source in sorted(manifest.get("sources", []), key=lambda item: (item.get("config", ""), item.get("row_idx", 0)))
        ],
    }


def _unmapped(
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    reason: str,
    record: Any,
    parent_step: int | None,
) -> Activity:
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts="",
        kind="unmapped",
        tool_name="who_when_unmapped",
        inputs={
            "source_index": source_index,
            "source_url": source_url,
            "reason": reason,
            "record_kind": type(record).__name__,
            "top_level_keys": sorted(str(key) for key in record.keys()) if isinstance(record, dict) else [],
            "record_hash": sha256_text(canonical_json(record)),
        },
        parent_step=parent_step,
    )


def _agent_name(role: str, name: str) -> str:
    if name:
        return name
    stripped = role.strip()
    if not stripped:
        return ""
    if stripped.lower() in {"human", "user", "assistant"}:
        return ""
    return stripped.split("(", 1)[0].strip()


def _is_terminal_record(role: str, name: str) -> bool:
    value = f"{role} {name}".lower().replace(" ", "_")
    return "computer_terminal" in value or value in {"terminal", "computer"}


def _huggingface_dataset_revision(dataset: str) -> str:
    data = _download_json(f"https://huggingface.co/api/datasets/{dataset}")
    return str(data["sha"])


def _huggingface_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    offset: int,
    length: int,
    revision: str,
) -> dict[str, Any]:
    data = _download_json(_huggingface_rows_url(dataset, config, split, offset=offset, length=length, revision=revision))
    if not isinstance(data, dict):
        raise ValueError("Hugging Face rows response was not an object")
    return data


def _huggingface_rows_url(
    dataset: str,
    config: str,
    split: str,
    *,
    offset: int,
    length: int,
    revision: str,
) -> str:
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
        "revision": revision,
    }
    return "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(params)


def _download_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "agent-bisect"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return safe.strip("._") or "unknown"


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({_rate(None if denominator == 0 else numerator / denominator)})"


def _percent(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.1f}%"


def _rate(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"
