from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any

from .foreign import ingest_foreign_trajectory
from .gates import run_g1, run_g2, run_g3
from .ingest_claude import ingest_transcript
from .ingest_codex import ingest_codex_transcript
from .localize import localize_failures, shell_target_coverage
from .model import Activity, canonical_json


ACTION_KINDS = {"file_edit", "test_run", "tool_call", "opaque_shell", "unmapped"}
RUN_BUCKETS = ("no_break", "HIGH", "LOW")
CONFIDENCE_BUCKETS = ("HIGH", "LOW")
KNOWN_SOURCES = ("claude", "codex", "foreign")
STUDY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class StudyInput:
    source: str
    path: Path
    schema: str = ""


def run_corpus_study(inputs: list[StudyInput]) -> dict[str, Any]:
    source_aggregates = {source: _empty_source_aggregate(source) for source in KNOWN_SOURCES}
    for study_input in inputs:
        if study_input.source not in source_aggregates:
            raise ValueError(f"unsupported study source: {study_input.source}")
        aggregate = source_aggregates[study_input.source]
        for path in _expand_input(study_input):
            aggregate["runs_considered"] += 1
            try:
                source_records = _source_record_count(path, study_input.source, study_input.schema)
                activities = _load_input_activities(study_input.source, path, study_input.schema)
            except Exception:
                aggregate["parse_errors"] += 1
                continue
            _accumulate_run(aggregate, activities, source_records)

    overall = _finalize_overall_aggregate(source_aggregates)
    by_source = {
        source: _finalize_source_aggregate(aggregate)
        for source, aggregate in source_aggregates.items()
    }
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "privacy": "aggregate_only_no_paths_no_run_ids_no_raw_content",
        "summary": overall,
        "by_source": by_source,
        "notes": {
            "run_localization_status": "no_break means no deterministic gate-visible break; HIGH means all localized failures in the run used structured paths; LOW means at least one localized failure used a coverage-limited path.",
            "gate_detectable_break": "a run with at least one G1/G2/G3 FAIL step in the normalized transcript.",
            "shell_target_lift": "recorded shell/test steps for which conservative literal file-target extraction added graph evidence.",
            "private_corpus": "raw transcripts are read only and are not copied into reports.",
        },
    }


def write_study_reports(report: dict[str, Any], reports_dir: Path, study_md: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "corpus-study.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    study_md.write_text(render_study_markdown(report), encoding="utf-8", newline="\n")


def render_study_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Corpus Study",
        "",
        "Aggregate-only empirical profile over local real agent-run transcripts plus shipped foreign fixtures. This report contains no raw transcript content, commands, file paths, credentials, usernames, or per-run identifiers.",
        "",
        "## Headline",
        "",
        "| metric | result |",
        "| --- | ---: |",
        f"| runs processed | {summary['parsed_runs']}/{summary['runs_considered']} |",
        f"| source records processed | {summary['source_records']} |",
        f"| normalized activities | {summary['activities']} |",
        f"| runs with gate-detectable break | {_fraction(summary['gate_failure_runs'], summary['parsed_runs'])} |",
        f"| no-break runs | {_fraction(summary['localization_runs']['no_break'], summary['parsed_runs'])} |",
        f"| HIGH localized runs | {_fraction(summary['localization_runs']['HIGH'], summary['parsed_runs'])} |",
        f"| LOW localized runs | {_fraction(summary['localization_runs']['LOW'], summary['parsed_runs'])} |",
        f"| linked action activities | {_fraction(summary['linked_action_activities'], summary['action_activities'])} |",
        f"| opaque or unmapped activities | {_fraction(summary['opaque_or_unmapped_activities'], summary['activities'])} |",
        f"| median per-run opaque/unmapped action rate | {_rate(summary['median_opaque_unmapped_action_rate'])} |",
        f"| shell-target lift | {summary['shell_steps_with_targets']}/{summary['shell_command_steps']} steps; {summary['shell_added_edges']} added edges |",
        "",
        "## By Source",
        "",
        "| source | runs | records | activities | break runs | no_break | HIGH | LOW | linked actions | opaque/unmapped | shell targets | parse errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in KNOWN_SOURCES:
        row = report["by_source"][source]
        lines.append(
            "| {source} | {runs} | {records} | {activities} | {breaks} | {no_break} | {high} | {low} | {linked} | {opaque} | {targets} | {errors} |".format(
                source=source,
                runs=row["parsed_runs"],
                records=row["source_records"],
                activities=row["activities"],
                breaks=_fraction(row["gate_failure_runs"], row["parsed_runs"]),
                no_break=_fraction(row["localization_runs"]["no_break"], row["parsed_runs"]),
                high=_fraction(row["localization_runs"]["HIGH"], row["parsed_runs"]),
                low=_fraction(row["localization_runs"]["LOW"], row["parsed_runs"]),
                linked=_fraction(row["linked_action_activities"], row["action_activities"]),
                opaque=_fraction(row["opaque_or_unmapped_activities"], row["activities"]),
                targets=f"{row['shell_steps_with_targets']}/{row['shell_command_steps']}",
                errors=row["parse_errors"],
            )
        )

    lines.extend(
        [
            "",
            "## Gate Failures",
            "",
            "| scope | G1 fail steps | G2 fail steps | G3 fail steps | total fail steps |",
            "| --- | ---: | ---: | ---: | ---: |",
            _gate_row("overall", summary),
        ]
    )
    for source in KNOWN_SOURCES:
        lines.append(_gate_row(source, report["by_source"][source]))

    lines.extend(
        [
            "",
            "## Activity Mix",
            "",
            "| scope | file_edit | test_run | tool_call | opaque_shell | unmapped | user_msg | llm_call | verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _kind_row("overall", summary),
        ]
    )
    for source in KNOWN_SOURCES:
        lines.append(_kind_row(source, report["by_source"][source]))

    lines.extend(
        [
            "",
            "## What This Means",
            "",
            "The corpus profile is a visibility measurement, not a claim that every no-break run was successful. A no-break result means the transcript did not expose a deterministic G1/G2/G3 failure for `agent-bisect` to localize.",
            "",
            "HIGH localized runs are the most inspectable slice: the gate-visible break is connected through structured transcript evidence. LOW localized runs are real gate-visible breaks whose causal path contains opaque, unmapped, unlinked, or heuristic shell-target evidence.",
            "",
            "Opaque/unmapped and action-linkage rates show where coverage is lost in real transcripts. Shell-target lift reports how often conservative literal command parsing adds graph evidence; it is reported as lift, not ground truth.",
            "",
            "## Privacy",
            "",
            "Only aggregate counters and rates are committed. Raw corpus files are read-only, and generated JSON stays under ignored reports.",
            "",
        ]
    )
    return "\n".join(lines)


def _empty_source_aggregate(source: str) -> dict[str, Any]:
    return {
        "source": source,
        "runs_considered": 0,
        "parsed_runs": 0,
        "parse_errors": 0,
        "source_records": 0,
        "activities": 0,
        "action_activities": 0,
        "linked_action_activities": 0,
        "opaque_or_unmapped_activities": 0,
        "gate_failure_runs": 0,
        "gate_failure_steps": 0,
        "gate_failures_by_gate": Counter(),
        "gate_status_counts": Counter(),
        "localization_runs": Counter(),
        "localization_results": Counter(),
        "kind_counts": Counter(),
        "shell_command_steps": 0,
        "shell_steps_with_targets": 0,
        "shell_added_edges": 0,
        "opaque_unmapped_action_rates": [],
    }


def _accumulate_run(aggregate: dict[str, Any], activities: list[Activity], source_records: int) -> None:
    aggregate["parsed_runs"] += 1
    aggregate["source_records"] += source_records
    aggregate["activities"] += len(activities)

    kind_counts = Counter(activity.kind for activity in activities)
    aggregate["kind_counts"].update(kind_counts)
    aggregate["opaque_or_unmapped_activities"] += kind_counts.get("opaque_shell", 0) + kind_counts.get("unmapped", 0)

    action_activities = [activity for activity in activities if activity.kind in ACTION_KINDS]
    linked_actions = sum(1 for activity in action_activities if activity.parent_step is not None)
    opaque_unmapped_actions = sum(1 for activity in action_activities if activity.kind in {"opaque_shell", "unmapped"})
    aggregate["action_activities"] += len(action_activities)
    aggregate["linked_action_activities"] += linked_actions
    if action_activities:
        aggregate["opaque_unmapped_action_rates"].append(opaque_unmapped_actions / len(action_activities))

    gate_results = [*run_g1(activities), *run_g2(activities), *run_g3(activities)]
    failure_steps: set[int] = set()
    for result in gate_results:
        aggregate["gate_status_counts"][f"{result.gate}:{result.status}"] += 1
        if result.status == "FAIL" and result.step_index is not None:
            failure_steps.add(result.step_index)
            aggregate["gate_failures_by_gate"][result.gate] += 1
    aggregate["gate_failure_steps"] += len(failure_steps)

    localization = localize_failures(activities)
    if localization.status == "break":
        aggregate["gate_failure_runs"] += 1
    # A parsed run with zero normalized activities is conservatively counted as
    # no_break: the transcript exposed no deterministic gate-visible failure.
    if localization.status == "no_break":
        aggregate["localization_runs"]["no_break"] += 1
    else:
        result_confidences = [failure.confidence for failure in localization.failures]
        run_bucket = "LOW" if any(confidence == "LOW" for confidence in result_confidences) else "HIGH"
        aggregate["localization_runs"][run_bucket] += 1
        aggregate["localization_results"].update(confidence for confidence in result_confidences if confidence in CONFIDENCE_BUCKETS)

    shell_coverage = shell_target_coverage(activities)
    aggregate["shell_command_steps"] += shell_coverage.shell_command_steps
    aggregate["shell_steps_with_targets"] += shell_coverage.steps_with_targets
    aggregate["shell_added_edges"] += shell_coverage.added_edges


def _finalize_source_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    rates = aggregate.pop("opaque_unmapped_action_rates")
    finalized = {
        key: value
        for key, value in aggregate.items()
        if key not in {"gate_failures_by_gate", "gate_status_counts", "localization_runs", "localization_results", "kind_counts"}
    }
    finalized["gate_failures_by_gate"] = _counter_dict(aggregate["gate_failures_by_gate"], ("G1", "G2", "G3"))
    finalized["gate_status_counts"] = dict(sorted(aggregate["gate_status_counts"].items()))
    finalized["localization_runs"] = _counter_dict(aggregate["localization_runs"], RUN_BUCKETS)
    finalized["localization_results"] = _counter_dict(aggregate["localization_results"], CONFIDENCE_BUCKETS)
    finalized["kind_counts"] = _counter_dict(
        aggregate["kind_counts"],
        ("file_edit", "test_run", "tool_call", "opaque_shell", "unmapped", "user_msg", "llm_call", "verdict"),
    )
    finalized["median_opaque_unmapped_action_rate"] = None if not rates else statistics.median(rates)
    return finalized


def _finalize_overall_aggregate(source_aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = _empty_source_aggregate("overall")
    for source in KNOWN_SOURCES:
        row = source_aggregates[source]
        for key in (
            "runs_considered",
            "parsed_runs",
            "parse_errors",
            "source_records",
            "activities",
            "action_activities",
            "linked_action_activities",
            "opaque_or_unmapped_activities",
            "gate_failure_runs",
            "gate_failure_steps",
            "shell_command_steps",
            "shell_steps_with_targets",
            "shell_added_edges",
        ):
            merged[key] += row[key]
        merged["gate_failures_by_gate"].update(row["gate_failures_by_gate"])
        merged["gate_status_counts"].update(row["gate_status_counts"])
        merged["localization_runs"].update(row["localization_runs"])
        merged["localization_results"].update(row["localization_results"])
        merged["kind_counts"].update(row["kind_counts"])
        merged["opaque_unmapped_action_rates"].extend(row["opaque_unmapped_action_rates"])
    return _finalize_source_aggregate(merged)


def _expand_input(study_input: StudyInput) -> list[Path]:
    path = study_input.path
    if study_input.source in {"claude", "codex"}:
        return _expand_jsonl(path)
    if study_input.source == "foreign":
        return _expand_foreign(path, study_input.schema)
    raise ValueError(f"unsupported source: {study_input.source}")


def _load_input_activities(source: str, path: Path, schema: str) -> list[Activity]:
    if source == "claude":
        return ingest_transcript(path)
    if source == "codex":
        return ingest_codex_transcript(path)
    if source == "foreign":
        return ingest_foreign_trajectory(path, schema=schema)
    raise ValueError(f"unsupported source: {source}")


def _expand_jsonl(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(set(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file()))
    if path.is_file() and path.suffix == ".jsonl":
        return [path]
    return []


def _expand_foreign(path: Path, schema: str) -> list[Path]:
    suffixes = {
        "swe-agent": {".traj"},
        "mini-swe-agent": {".json", ".jsonl"},
        "openhands": {".json"},
    }.get(schema)
    if suffixes is None:
        raise ValueError(f"unsupported foreign schema: {schema}")
    if path.is_file():
        return [path] if path.suffix in suffixes or path.name.endswith(".traj.json") else []
    if not path.is_dir():
        return []
    expanded: list[Path] = []
    for suffix in sorted(suffixes):
        expanded.extend(sorted(candidate for candidate in path.rglob(f"*{suffix}") if candidate.is_file()))
    return sorted(set(expanded))


def _source_record_count(path: Path, source: str, schema: str) -> int:
    if source in {"claude", "codex"}:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if source == "foreign":
        data = json.loads(path.read_text(encoding="utf-8"))
        return _foreign_record_count(data, schema)
    return 0


def _foreign_record_count(data: Any, schema: str) -> int:
    if schema == "openhands" and isinstance(data, dict):
        row = data.get("row") if isinstance(data.get("row"), dict) else data
        trajectory = row.get("trajectory") if isinstance(row, dict) else None
        return len(trajectory) if isinstance(trajectory, list) else 1
    if isinstance(data, dict):
        for key in ("history", "messages", "trajectory"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
    if isinstance(data, list):
        return len(data)
    return 1


def _counter_dict(counter: Counter | dict[str, int], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in keys}


def _gate_row(scope: str, row: dict[str, Any]) -> str:
    gates = row["gate_failures_by_gate"]
    return "| {scope} | {g1} | {g2} | {g3} | {total} |".format(
        scope=scope,
        g1=gates.get("G1", 0),
        g2=gates.get("G2", 0),
        g3=gates.get("G3", 0),
        total=row["gate_failure_steps"],
    )


def _kind_row(scope: str, row: dict[str, Any]) -> str:
    kinds = row["kind_counts"]
    return "| {scope} | {file_edit} | {test_run} | {tool_call} | {opaque_shell} | {unmapped} | {user_msg} | {llm_call} | {verdict} |".format(
        scope=scope,
        file_edit=kinds.get("file_edit", 0),
        test_run=kinds.get("test_run", 0),
        tool_call=kinds.get("tool_call", 0),
        opaque_shell=kinds.get("opaque_shell", 0),
        unmapped=kinds.get("unmapped", 0),
        user_msg=kinds.get("user_msg", 0),
        llm_call=kinds.get("llm_call", 0),
        verdict=kinds.get("verdict", 0),
    )


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({_rate(None if denominator == 0 else numerator / denominator)})"


def _rate(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"
