from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request
from typing import Any, Iterable

from .gates import run_g1, run_g2, run_g3
from .localize import localize_failures
from .model import Activity, Journal, canonical_json, sha256_text


SWE_AGENT_REPO = "SWE-agent/SWE-agent"
SWE_AGENT_BRANCH = "main"
OPENHANDS_REALTASK_DATASET = "nebius/SWE-rebench-openhands-trajectories"
OPENHANDS_REALTASK_CONFIG = "default"
OPENHANDS_REALTASK_SPLIT = "train"
FOREIGN_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ForeignSource:
    schema: str
    local_path: str
    source_url: str
    raw_url: str
    repo: str
    commit: str
    instance_id: str = ""
    trajectory_id: str = ""
    row_idx: int | None = None
    dataset_config: str = ""
    split: str = ""


def ingest_foreign_trajectory(path: Path, *, schema: str, source_url: str = "") -> list[Activity]:
    if schema == "swe-agent":
        return _ingest_swe_agent(path, source_url=source_url)
    if schema == "mini-swe-agent":
        return _ingest_mini_swe_agent(path, source_url=source_url)
    if schema == "openhands":
        return _ingest_openhands(path, source_url=source_url)
    raise ValueError(f"unsupported foreign schema: {schema}")


def fetch_swe_agent_trajectories(out_dir: Path, *, limit: int | None = None) -> list[ForeignSource]:
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _github_head_commit(SWE_AGENT_REPO, SWE_AGENT_BRANCH)
    paths = _github_tree_paths(SWE_AGENT_REPO, commit)
    traj_paths = sorted(path for path in paths if path.endswith(".traj"))
    if limit is not None:
        traj_paths = traj_paths[:limit]

    sources: list[ForeignSource] = []
    for repo_path in traj_paths:
        raw_url = f"https://raw.githubusercontent.com/{SWE_AGENT_REPO}/{commit}/{repo_path}"
        source_url = f"https://github.com/{SWE_AGENT_REPO}/blob/{commit}/{repo_path}"
        local_path = out_dir / "swe-agent" / repo_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        text = _download_text(raw_url)
        local_path.write_text(text, encoding="utf-8", newline="\n")
        sources.append(
            ForeignSource(
                schema="swe-agent",
                local_path=local_path.resolve().as_posix(),
                source_url=source_url,
                raw_url=raw_url,
                repo=SWE_AGENT_REPO,
                commit=commit,
            )
        )

    manifest = {
        "schema_version": 1,
        "sources": [asdict(source) for source in sources],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sources


def fetch_openhands_realtask_trajectories(
    out_dir: Path,
    *,
    limit: int = 50,
    page_size: int = 100,
) -> list[ForeignSource]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _huggingface_dataset_commit(OPENHANDS_REALTASK_DATASET)
    seen_instance_ids: set[str] = set()
    sources: list[ForeignSource] = []
    offset = 0
    total_rows: int | None = None

    while len(sources) < limit:
        page = _huggingface_rows(
            OPENHANDS_REALTASK_DATASET,
            OPENHANDS_REALTASK_CONFIG,
            OPENHANDS_REALTASK_SPLIT,
            offset=offset,
            length=page_size,
            revision=commit,
        )
        rows = page.get("rows")
        if not isinstance(rows, list) or not rows:
            break
        if isinstance(page.get("num_rows_total"), int):
            total_rows = int(page["num_rows_total"])

        for wrapper in rows:
            if len(sources) >= limit:
                break
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("row"), dict):
                continue
            row = wrapper["row"]
            instance_id = _as_nonempty_str(row.get("instance_id"))
            trajectory = row.get("trajectory")
            if instance_id is None or not isinstance(trajectory, list):
                continue
            if instance_id in seen_instance_ids:
                continue
            seen_instance_ids.add(instance_id)

            row_idx = wrapper.get("row_idx")
            row_idx_int = int(row_idx) if isinstance(row_idx, int) else offset
            trajectory_id = _as_nonempty_str(row.get("trajectory_id")) or f"row-{row_idx_int}"
            local_path = out_dir / "openhands" / f"{_safe_filename(instance_id)}__{_safe_filename(trajectory_id)}.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "schema": "openhands",
                "dataset": OPENHANDS_REALTASK_DATASET,
                "dataset_commit": commit,
                "dataset_config": OPENHANDS_REALTASK_CONFIG,
                "split": OPENHANDS_REALTASK_SPLIT,
                "row_idx": row_idx_int,
                "row": row,
            }
            local_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
            row_url = _huggingface_rows_url(
                OPENHANDS_REALTASK_DATASET,
                OPENHANDS_REALTASK_CONFIG,
                OPENHANDS_REALTASK_SPLIT,
                offset=row_idx_int,
                length=1,
                revision=commit,
            )
            sources.append(
                ForeignSource(
                    schema="openhands",
                    local_path=local_path.resolve().as_posix(),
                    source_url=row_url,
                    raw_url=row_url,
                    repo=OPENHANDS_REALTASK_DATASET,
                    commit=commit,
                    instance_id=instance_id,
                    trajectory_id=trajectory_id,
                    row_idx=row_idx_int,
                    dataset_config=OPENHANDS_REALTASK_CONFIG,
                    split=OPENHANDS_REALTASK_SPLIT,
                )
            )

        offset += len(rows)
        if total_rows is not None and offset >= total_rows:
            break

    manifest = {
        "schema_version": 1,
        "deduplication_key": "instance_id",
        "dataset": OPENHANDS_REALTASK_DATASET,
        "dataset_commit": commit,
        "sources": [asdict(source) for source in sources],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sources


def sweep_foreign_trajectories(
    paths: list[Path],
    *,
    schema: str,
    reports_dir: Path,
    report_stem: str = "foreign-coverage-report",
) -> dict[str, Any]:
    source_index = _load_source_index(paths)
    trajectory_paths = _expand_foreign_paths(paths, schema=schema)
    runs = []
    for path in trajectory_paths:
        source = source_index.get(path.resolve().as_posix(), {})
        activities = ingest_foreign_trajectory(path, schema=schema, source_url=str(source.get("source_url", "")))
        journal_hash = Journal.from_activities(activities).records[-1].record_hash if activities else ""
        runs.append(_sweep_run(path, activities, source, journal_hash))

    report = _build_foreign_report(schema=schema, runs=runs)
    rendered_json = json.dumps(report, sort_keys=True, indent=2) + "\n"
    rendered_md = render_foreign_markdown(report)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{report_stem}.json").write_text(rendered_json, encoding="utf-8", newline="\n")
    (reports_dir / f"{report_stem}.md").write_text(rendered_md, encoding="utf-8", newline="\n")
    return report


def render_foreign_markdown(report: dict[str, Any]) -> str:
    coverage = report["localization_coverage"]
    gaps = report["coverage_gaps"]
    lines = [
        "# Foreign-Schema Coverage Sweep",
        "",
        "Scope: schema normalization plus deterministic gate/localizer coverage on public non-Claude foreign trajectories.",
        "",
        f"Schema: `{report['schema']}`",
        f"Trajectories swept: {report['trajectory_count']}",
        f"Distinct instance_ids: {report['distinct_instance_id_count']}",
        f"Deduplication key: `{report['deduplication_key']}`",
        "",
        "## Coverage Profile",
        "",
        "| bucket | count | denominator | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for bucket in ("HIGH", "LOW", "NA"):
        row = coverage[bucket]
        lines.append(f"| {bucket} | {row['count']} | {row['denominator']} | {_rate(row['rate'])} |")

    lines.extend(
        [
            "",
            "## Coverage Gaps",
            "",
            "| metric | count | denominator | rate |",
            "| --- | ---: | ---: | ---: |",
            f"| opaque_shell_action_steps | {gaps['opaque_shell_action_steps']} | {gaps['action_activity_count']} | {_rate(gaps['opaque_shell_action_fraction'])} |",
            f"| unmapped_action_steps | {gaps['unmapped_action_steps']} | {gaps['action_activity_count']} | {_rate(gaps['unmapped_action_fraction'])} |",
            f"| opaque_or_unmapped_action_steps | {gaps['opaque_or_unmapped_action_steps']} | {gaps['action_activity_count']} | {_rate(gaps['opaque_or_unmapped_action_fraction'])} |",
            f"| unlinked_action_steps_excluding_first_action | {gaps['unlinked_action_steps_excluding_first_action']} | {gaps['action_activity_count']} | {_rate(gaps['unlinked_action_fraction_excluding_first_action'])} |",
            f"| all_preserved_record_opaque_or_unmapped_steps | {gaps['all_activity_opaque_or_unmapped_steps']} | {gaps['activity_count']} | {_rate(gaps['all_activity_opaque_or_unmapped_fraction'])} |",
            "",
            "Opaque shell actions and unmapped actions are reported separately so coverage gaps stay visible.",
            "",
            "## Source Summary",
            "",
            "| repo | commit | count |",
            "| --- | --- | ---: |",
        ]
    )
    for source in report["source_summary"]:
        lines.append(f"| {source['repo']} | `{source['commit']}` | {source['count']} |")

    lines.extend(
        [
            "",
            "## Per-Instance Breakdown",
            "",
            "| instance_id | trajectories | activities | actions | gate_failures | HIGH | LOW | NA | opaque_shell_actions | unmapped_actions | unlinked_actions_excluding_first |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for instance in report["per_instance_breakdown"]:
        row = instance["localization_coverage"]
        lines.append(
            "| {instance_id} | {trajectory_count} | {activity_count} | {action_activity_count} | {gate_failure_steps} | {high} | {low} | {na} | {opaque} | {unmapped} | {unlinked} |".format(
                instance_id=instance["instance_id"],
                trajectory_count=instance["trajectory_count"],
                activity_count=instance["activity_count"],
                action_activity_count=instance["action_activity_count"],
                gate_failure_steps=instance["gate_failure_steps"],
                high=row["HIGH"]["count"],
                low=row["LOW"]["count"],
                na=row["NA"]["count"],
                opaque=instance["opaque_shell_action_steps"],
                unmapped=instance["unmapped_action_steps"],
                unlinked=instance["unlinked_action_steps_excluding_first_action"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-Trajectory Breakdown",
            "",
            "| instance_id | trajectory_id | local_path | source_url | activities | actions | gate_failures | HIGH | LOW | NA | opaque_shell_actions | unmapped_actions | unlinked_actions_excluding_first |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in report["runs"]:
        row = run["localization_coverage"]
        lines.append(
            "| {instance_id} | {trajectory_id} | {local_path} | {source_url} | {activity_count} | {action_activity_count} | {gate_failure_steps} | {high} | {low} | {na} | {opaque} | {unmapped} | {unlinked} |".format(
                instance_id=run["instance_id"],
                trajectory_id=run["trajectory_id"],
                local_path=run["local_path"],
                source_url=run["source_url"],
                activity_count=run["activity_count"],
                action_activity_count=run["action_activity_count"],
                gate_failure_steps=run["gate_failure_steps"],
                high=row["HIGH"]["count"],
                low=row["LOW"]["count"],
                na=row["NA"]["count"],
                opaque=run["opaque_shell_action_steps"],
                unmapped=run["unmapped_action_steps"],
                unlinked=run["unlinked_action_steps_excluding_first_action"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _ingest_swe_agent(path: Path, *, source_url: str) -> list[Activity]:
    data = json.loads(path.read_text(encoding="utf-8"))
    run_id = path.stem
    history = _history_records(data)
    activities: list[Activity] = []
    call_to_step: dict[str, int] = {}
    pending_action_step: int | None = None
    current_open_file: str | None = None
    created_files: set[str] = set()

    for source_index, record in enumerate(history):
        force_observation = _record_role(record) == "user" and pending_action_step is not None
        activity = _activity_from_swe_agent_record(
            record,
            run_id=run_id,
            step_index=len(activities),
            source_index=source_index,
            source_url=source_url,
            parent_step=pending_action_step,
            current_open_file=current_open_file,
            force_observation=force_observation,
            created_files=created_files,
        )
        activities.append(activity)

        for call_id in _tool_call_ids(record):
            call_to_step[call_id] = activity.step_index

        if _is_observation(record) or force_observation:
            linked_step = _linked_action_step(record, call_to_step, pending_action_step)
            if linked_step is not None:
                activities[linked_step].outputs.update(_observation_outputs(record))
                activities[linked_step].refresh_hash()
                activity.parent_step = linked_step
                activity.refresh_hash()
            current_open_file = _current_open_file_from_observation(record, current_open_file)
        elif activity.kind in {"file_edit", "test_run", "tool_call", "opaque_shell", "unmapped"} and _record_role(record) == "assistant":
            pending_action_step = activity.step_index
            current_open_file = _current_open_file_from_action(record, current_open_file)
            if activity.tool_name == "create" and activity.target:
                created_files.add(activity.target)
            if activity.tool_name == "Write" and activity.target in created_files:
                created_files.remove(activity.target)

    for index, activity in enumerate(activities):
        activity.step_index = index
        activity.refresh_hash()
    return activities


def _ingest_openhands(path: Path, *, source_url: str) -> list[Activity]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row = _openhands_row(data)
    run_id = _openhands_run_id(row, path)
    history = row.get("trajectory") if isinstance(row, dict) else None
    if not isinstance(history, list):
        return [_unmapped(run_id, 0, 0, source_url, "openhands_missing_trajectory", data, None)]

    activities: list[Activity] = []
    call_to_step: dict[str, int] = {}
    pending_action_step: int | None = None

    for source_index, record in enumerate(history):
        activity = _activity_from_openhands_record(
            record,
            run_id=run_id,
            step_index=len(activities),
            source_index=source_index,
            source_url=source_url,
            parent_step=pending_action_step,
        )
        activities.append(activity)

        for call_id in _tool_call_ids(record):
            call_to_step[call_id] = activity.step_index

        if isinstance(record, dict) and _is_observation(record):
            linked_step = _linked_action_step(record, call_to_step, pending_action_step)
            if linked_step is not None:
                activities[linked_step].outputs.update(_observation_outputs(record))
                activities[linked_step].refresh_hash()
                activity.parent_step = linked_step
                activity.refresh_hash()
        elif activity.kind in {"file_edit", "test_run", "tool_call", "opaque_shell", "unmapped"} and isinstance(record, dict) and _record_role(record) == "assistant":
            pending_action_step = activity.step_index

    for index, activity in enumerate(activities):
        activity.step_index = index
        activity.refresh_hash()
    return activities


def _ingest_mini_swe_agent(path: Path, *, source_url: str) -> list[Activity]:
    data = json.loads(path.read_text(encoding="utf-8"))
    run_id = _mini_run_id(data, path)
    messages = _mini_messages(data)
    activities: list[Activity] = []
    pending_action_step: int | None = None

    for source_index, record in enumerate(messages):
        activity = _activity_from_mini_record(
            record,
            run_id=run_id,
            step_index=len(activities),
            source_index=source_index,
            source_url=source_url,
            parent_step=pending_action_step,
        )
        activities.append(activity)
        if activity.kind == "verdict" and pending_action_step is not None:
            activities[pending_action_step].outputs.update(activity.outputs)
            activities[pending_action_step].refresh_hash()
            activity.parent_step = pending_action_step
            activity.refresh_hash()
            pending_action_step = None
        elif activity.kind in {"file_edit", "test_run", "tool_call", "opaque_shell", "unmapped"}:
            pending_action_step = activity.step_index

    for index, activity in enumerate(activities):
        activity.step_index = index
        activity.refresh_hash()
    return activities


def _mini_messages(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        messages = data.get("messages")
        if isinstance(messages, list):
            return messages
        trajectory = data.get("trajectory")
        if isinstance(trajectory, list):
            return trajectory
    return [data]


def _mini_run_id(data: Any, path: Path) -> str:
    if isinstance(data, dict):
        instance_id = _as_nonempty_str(data.get("instance_id"))
        if instance_id:
            return instance_id
        info = data.get("info")
        if isinstance(info, dict):
            instance_id = _as_nonempty_str(info.get("instance_id"))
            if instance_id:
                return instance_id
    return path.stem


def _activity_from_mini_record(
    record: Any,
    *,
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    parent_step: int | None,
) -> Activity:
    if not isinstance(record, dict):
        return _unmapped(run_id, step_index, source_index, source_url, "record_not_object", {}, parent_step)

    role = _record_role(record)
    base = _base_inputs(record, source_index, source_url)
    if role == "user" and parent_step is not None and _looks_like_mini_observation(record):
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="verdict",
            tool_name="mini_swe_agent_observation",
            inputs=base,
            outputs=_mini_observation_outputs(record),
            parent_step=parent_step,
        )
    if role in {"system", "user"}:
        return Activity(run_id=run_id, step_index=step_index, ts="", kind="user_msg", inputs=base, parent_step=parent_step)
    if role == "tool" or _is_observation(record):
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="verdict",
            tool_name=_as_nonempty_str(record.get("name")) or "mini_swe_agent_observation",
            inputs=base,
            outputs=_mini_observation_outputs(record),
            parent_step=parent_step,
        )
    if role == "exit":
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="verdict",
            tool_name="mini_swe_agent_exit",
            inputs=base,
            outputs=_mini_observation_outputs(record),
            parent_step=parent_step,
        )
    if role == "assistant":
        command = _mini_command(record)
        if command is not None:
            return _mini_command_activity(
                command,
                run_id=run_id,
                step_index=step_index,
                source_index=source_index,
                source_url=source_url,
                record=record,
                parent_step=parent_step,
            )
        tool_calls = record.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) == 1:
            return _activity_from_openhands_function_call(
                record,
                tool_calls[0],
                run_id,
                step_index,
                source_index,
                source_url,
                parent_step,
            )
        if isinstance(tool_calls, list) and len(tool_calls) > 1:
            return _unmapped(run_id, step_index, source_index, source_url, "multiple_tool_calls", record, parent_step)
        if _as_nonempty_str(record.get("content")):
            return Activity(run_id=run_id, step_index=step_index, ts="", kind="llm_call", inputs=base, parent_step=parent_step)
    return _unmapped(run_id, step_index, source_index, source_url, f"unknown_role:{role or '<missing>'}", record, parent_step)


def _mini_command(record: dict[str, Any]) -> str | None:
    extra = record.get("extra")
    if isinstance(extra, dict):
        actions = extra.get("actions")
        if isinstance(actions, list) and len(actions) == 1 and isinstance(actions[0], dict):
            command = _as_nonempty_str(actions[0].get("command"))
            if command:
                return command
    content = record.get("content")
    if not isinstance(content, str):
        return None
    match = re.search(r"```(?:mswea_bash_command|bash|sh)?\s*\n(.*?)\n```", content, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _mini_command_activity(
    command: str,
    *,
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    record: dict[str, Any],
    parent_step: int | None,
) -> Activity:
    inputs = {**_base_inputs(record, source_index, source_url), "command": command}
    submit_match = re.search(r"\becho\s+COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\b", command)
    if submit_match:
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="tool_call",
            tool_name="submit",
            inputs=inputs,
            target="shell",
            parent_step=parent_step,
        )

    sed_match = re.search(r"sed\s+-i\s+['\"]s/(.*?)/(.*?)/(?:g)?['\"]\s+(.+)$", command.strip())
    if sed_match:
        target = sed_match.group(3).strip().strip("\"'")
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="file_edit",
            tool_name="Edit",
            inputs={**inputs, "file_path": target, "old_string": sed_match.group(1), "new_string": sed_match.group(2)},
            target=target,
            parent_step=parent_step,
        )
    write_match = re.search(r"(?:cat|tee)\s+>?\s*([A-Za-z0-9_./-]+)\s*<<", command)
    if write_match:
        target = write_match.group(1)
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="file_edit",
            tool_name="Write",
            inputs={**inputs, "write_mode": True},
            target=target,
            parent_step=parent_step,
        )
    kind = "test_run" if _looks_like_foreign_test(command) else "opaque_shell"
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts="",
        kind=kind,
        tool_name="Bash",
        inputs=inputs,
        target="shell",
        parent_step=parent_step,
    )


def _looks_like_mini_observation(record: dict[str, Any]) -> bool:
    content = record.get("content")
    if not isinstance(content, str):
        return False
    return "<returncode>" in content or "<output>" in content or content.strip().startswith("{")


def _mini_observation_outputs(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content")
    text = content if isinstance(content, str) else canonical_json(content)
    returncode_match = re.search(r"<returncode>(-?\d+)</returncode>", text)
    output_match = re.search(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL)
    output_text = output_match.group(1) if output_match else text
    outputs = {
        "result_text": output_text,
        "content_length": len(text),
        "content_hash": sha256_text(text),
    }
    if returncode_match:
        outputs["exit_code"] = int(returncode_match.group(1))
    return outputs


def _openhands_row(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("row"), dict):
        return data["row"]
    return data if isinstance(data, dict) else {}


def _openhands_run_id(row: dict[str, Any], path: Path) -> str:
    instance_id = _as_nonempty_str(row.get("instance_id"))
    trajectory_id = _as_nonempty_str(row.get("trajectory_id"))
    if instance_id and trajectory_id:
        return f"{instance_id}::{trajectory_id}"
    if instance_id:
        return instance_id
    return path.stem


def _activity_from_openhands_record(
    record: Any,
    *,
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    parent_step: int | None,
) -> Activity:
    if not isinstance(record, dict):
        return _unmapped(run_id, step_index, source_index, source_url, "record_not_object", {}, parent_step)

    role = _record_role(record)
    base = _base_inputs(record, source_index, source_url)

    if role in {"system", "user"}:
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="user_msg",
            inputs=base,
            parent_step=parent_step,
        )

    if _is_observation(record):
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="verdict",
            tool_name=_as_nonempty_str(record.get("name")) or "openhands_observation",
            inputs=base,
            outputs=_observation_outputs(record),
            parent_step=parent_step,
        )

    if role == "assistant":
        tool_calls = record.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) == 1:
            return _activity_from_openhands_function_call(
                record,
                tool_calls[0],
                run_id,
                step_index,
                source_index,
                source_url,
                parent_step,
            )
        if isinstance(tool_calls, list) and len(tool_calls) > 1:
            return _unmapped(run_id, step_index, source_index, source_url, "multiple_tool_calls", record, parent_step)
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="llm_call",
            outputs=_content_summary(record),
            parent_step=parent_step,
        )

    return _unmapped(run_id, step_index, source_index, source_url, f"unknown_role:{role or '<missing>'}", record, parent_step)


def _activity_from_openhands_function_call(
    record: dict[str, Any],
    tool_call: Any,
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    parent_step: int | None,
) -> Activity:
    if not isinstance(tool_call, dict):
        return _unmapped(run_id, step_index, source_index, source_url, "tool_call_not_object", record, parent_step)
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return _unmapped(run_id, step_index, source_index, source_url, "missing_function_call", record, parent_step)
    name = str(function.get("name") or "")
    args = _parse_arguments(function.get("arguments"))
    inputs = {**_base_inputs(record, source_index, source_url), **args, "foreign_tool_name": name}
    if tool_call.get("id"):
        inputs["tool_call_id"] = str(tool_call["id"])

    if name == "execute_bash":
        command = _as_nonempty_str(args.get("command"))
        if command is None:
            return _unmapped(run_id, step_index, source_index, source_url, "execute_bash_missing_command", record, parent_step)
        return _shell_activity(run_id, step_index, parent_step, command, inputs)

    if name == "str_replace_editor":
        return _activity_from_openhands_editor(
            record,
            args,
            inputs,
            run_id,
            step_index,
            source_index,
            source_url,
            parent_step,
        )

    if name == "think":
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="llm_call",
            tool_name="think",
            inputs=inputs,
            parent_step=parent_step,
        )

    if name in {"finish", "task_tracker"}:
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="tool_call",
            tool_name=name,
            inputs=inputs,
            target=name,
            parent_step=parent_step,
        )

    return _unmapped(run_id, step_index, source_index, source_url, f"unknown_function:{name or '<missing>'}", record, parent_step)


def _activity_from_openhands_editor(
    record: dict[str, Any],
    args: dict[str, Any],
    inputs: dict[str, Any],
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    parent_step: int | None,
) -> Activity:
    command = _as_nonempty_str(args.get("command"))
    file_path = _as_nonempty_str(args.get("path"))
    if command is None:
        return _unmapped(run_id, step_index, source_index, source_url, "str_replace_editor_missing_command", record, parent_step)

    if command == "view":
        if file_path is None:
            return _unmapped(run_id, step_index, source_index, source_url, "view_missing_path", record, parent_step)
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="tool_call",
            tool_name="Read",
            inputs={**inputs, "file_path": file_path},
            target=file_path,
            parent_step=parent_step,
        )

    if command == "str_replace":
        old_string = _as_nonempty_str(args.get("old_str"))
        new_string = args.get("new_str")
        if file_path is None or old_string is None or not isinstance(new_string, str):
            return _unmapped(run_id, step_index, source_index, source_url, "str_replace_missing_path_or_fragments", record, parent_step)
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="file_edit",
            tool_name="Edit",
            inputs={**inputs, "file_path": file_path, "old_string": old_string, "new_string": new_string},
            target=file_path,
            parent_step=parent_step,
        )

    if command == "create":
        file_text = args.get("file_text")
        if file_path is None or not isinstance(file_text, str):
            return _unmapped(run_id, step_index, source_index, source_url, "create_missing_path_or_text", record, parent_step)
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="file_edit",
            tool_name="Write",
            inputs={**inputs, "file_path": file_path, "old_string": "", "new_string": file_text, "write_mode": True},
            target=file_path,
            parent_step=parent_step,
        )

    return _unmapped(run_id, step_index, source_index, source_url, f"unknown_editor_command:{command}", record, parent_step)


def _activity_from_swe_agent_record(
    record: Any,
    *,
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    parent_step: int | None,
    current_open_file: str | None,
    force_observation: bool = False,
    created_files: set[str] | None = None,
) -> Activity:
    if not isinstance(record, dict):
        return _unmapped(run_id, step_index, source_index, source_url, "record_not_object", {}, parent_step)

    role = _record_role(record)
    message_type = str(record.get("message_type") or "")
    base = _base_inputs(record, source_index, source_url)

    if role in {"system", "user"} and not (_is_observation(record) or force_observation):
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="user_msg",
            inputs=base,
            parent_step=parent_step,
        )

    if _is_observation(record) or force_observation:
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="verdict",
            tool_name="swe_agent_observation",
            inputs=base,
            outputs=_observation_outputs(record),
            parent_step=parent_step,
        )

    if role == "assistant":
        tool_calls = record.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) == 1:
            return _activity_from_function_call(record, tool_calls[0], run_id, step_index, source_index, source_url, parent_step, current_open_file, created_files or set())
        if isinstance(tool_calls, list) and len(tool_calls) > 1:
            return _unmapped(run_id, step_index, source_index, source_url, "multiple_tool_calls", record, parent_step)
        action = record.get("action")
        if isinstance(action, str) and action.strip():
            return _activity_from_action_string(record, action, run_id, step_index, source_index, source_url, parent_step, current_open_file, created_files or set())
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="llm_call",
            outputs=_content_summary(record),
            parent_step=parent_step,
        )

    return _unmapped(run_id, step_index, source_index, source_url, f"unknown_role:{role or '<missing>'}", record, parent_step)


def _activity_from_function_call(
    record: dict[str, Any],
    tool_call: Any,
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    parent_step: int | None,
    current_open_file: str | None,
    created_files: set[str],
) -> Activity:
    if not isinstance(tool_call, dict):
        return _unmapped(run_id, step_index, source_index, source_url, "tool_call_not_object", record, parent_step)
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return _unmapped(run_id, step_index, source_index, source_url, "missing_function_call", record, parent_step)
    name = str(function.get("name") or "")
    args = _parse_arguments(function.get("arguments"))
    inputs = {**_base_inputs(record, source_index, source_url), **args, "foreign_tool_name": name}
    if tool_call.get("id"):
        inputs["tool_call_id"] = str(tool_call["id"])

    if name == "edit":
        file_path = _as_nonempty_str(args.get("path")) or current_open_file
        search = _as_nonempty_str(args.get("search"))
        replace = _as_nonempty_str(args.get("replace"))
        if file_path and search is not None and replace is not None:
            return Activity(
                run_id=run_id,
                step_index=step_index,
                ts="",
                kind="file_edit",
                tool_name="Edit",
                inputs={**inputs, "file_path": file_path, "old_string": search, "new_string": replace},
                target=file_path,
                parent_step=parent_step,
            )
        return _unmapped(run_id, step_index, source_index, source_url, "edit_missing_path_or_fragments", record, parent_step)

    if name in {"open", "find_file", "search_file", "search_dir", "goto", "scroll_down"}:
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="tool_call",
            tool_name=name,
            inputs=inputs,
            target=_target_from_args(args),
            parent_step=parent_step,
        )

    if name in {"bash", "python"} or "command" in args:
        command = str(args.get("command") or record.get("action") or "")
        return _shell_activity(run_id, step_index, parent_step, command, inputs)

    return _unmapped(run_id, step_index, source_index, source_url, f"unknown_function:{name or '<missing>'}", record, parent_step)


def _activity_from_action_string(
    record: dict[str, Any],
    action: str,
    run_id: str,
    step_index: int,
    source_index: int,
    source_url: str,
    parent_step: int | None,
    current_open_file: str | None,
    created_files: set[str],
) -> Activity:
    command = action.strip()
    inputs = {**_base_inputs(record, source_index, source_url), "command": command}
    first = command.splitlines()[0] if command else ""

    if first.startswith("edit "):
        payload = _line_edit_payload(command)
        if current_open_file is not None and payload is not None and first.startswith("edit 1:1") and current_open_file in created_files:
            return Activity(
                run_id=run_id,
                step_index=step_index,
                ts="",
                kind="file_edit",
                tool_name="Write",
                inputs={**inputs, "file_path": current_open_file, "old_string": "", "new_string": payload, "write_mode": True},
                target=current_open_file,
                parent_step=parent_step,
            )
        if current_open_file is None:
            return _unmapped(run_id, step_index, source_index, source_url, "line_edit_without_open_file", record, parent_step)
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="unmapped",
            tool_name="swe_agent_line_edit",
            inputs={**inputs, "file_path": current_open_file, "reason": "line_range_edit_has_no_old_string"},
            target=current_open_file,
            parent_step=parent_step,
        )

    opened = _open_target_from_action(command)
    if opened:
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="tool_call",
            tool_name="open",
            inputs={**inputs, "file_path": opened},
            target=opened,
            parent_step=parent_step,
        )

    created = _create_target_from_action(command)
    if created:
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="tool_call",
            tool_name="create",
            inputs={**inputs, "file_path": created},
            target=created,
            parent_step=parent_step,
        )

    if first.startswith(("find_file ", "search_file ", "search_dir ", "goto ", "scroll_down")):
        return Activity(
            run_id=run_id,
            step_index=step_index,
            ts="",
            kind="tool_call",
            tool_name=first.split(" ", 1)[0],
            inputs=inputs,
            target=None,
            parent_step=parent_step,
        )

    return _shell_activity(run_id, step_index, parent_step, command, inputs)


def _shell_activity(run_id: str, step_index: int, parent_step: int | None, command: str, inputs: dict[str, Any]) -> Activity:
    kind = "test_run" if _looks_like_foreign_test(command) else "opaque_shell"
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts="",
        kind=kind,
        tool_name="Bash",
        inputs={**inputs, "command": command},
        target="shell",
        parent_step=parent_step,
    )


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
        tool_name="foreign_unmapped",
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


def _sweep_run(path: Path, activities: list[Activity], source: dict[str, Any], journal_hash: str) -> dict[str, Any]:
    g1 = run_g1(activities)
    g2 = run_g2(activities)
    g3 = run_g3(activities)
    gate_results = [*g1, *g2, *g3]
    failure_steps = sorted({result.step_index for result in gate_results if result.status == "FAIL" and result.step_index is not None})
    localization_by_step = _localization_by_step(activities)
    coverage = _coverage_counts(failure_steps, localization_by_step)
    kind_counts = Counter(activity.kind for activity in activities)
    action_kinds = {"file_edit", "test_run", "tool_call", "opaque_shell", "unmapped"}
    action_activities = [activity for activity in activities if activity.kind in action_kinds]
    first_action_step = min((activity.step_index for activity in action_activities), default=None)
    unlinked_actions = sum(
        1
        for activity in action_activities
        if activity.parent_step is None and activity.step_index != first_action_step
    )
    opaque_shell_actions = sum(1 for activity in action_activities if activity.kind == "opaque_shell")
    unmapped_actions = sum(1 for activity in action_activities if activity.kind == "unmapped")
    return {
        "local_path": path.resolve().as_posix(),
        "source_url": str(source.get("source_url", "")),
        "raw_url": str(source.get("raw_url", "")),
        "repo": str(source.get("repo", "")),
        "commit": str(source.get("commit", "")),
        "instance_id": str(source.get("instance_id", "")),
        "trajectory_id": str(source.get("trajectory_id", "")),
        "journal_tail_hash": journal_hash,
        "activity_count": len(activities),
        "action_activity_count": len(action_activities),
        "kind_counts": dict(sorted(kind_counts.items())),
        "gate_failure_steps": len(failure_steps),
        "gate_status_counts": _gate_status_counts(gate_results),
        "localization_coverage": coverage,
        "opaque_shell_action_steps": opaque_shell_actions,
        "unmapped_action_steps": unmapped_actions,
        "opaque_or_unmapped_action_steps": opaque_shell_actions + unmapped_actions,
        "unlinked_action_steps_excluding_first_action": unlinked_actions,
        "all_activity_opaque_or_unmapped_steps": kind_counts.get("opaque_shell", 0) + kind_counts.get("unmapped", 0),
    }


def _build_foreign_report(*, schema: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    total_activities = sum(run["activity_count"] for run in runs)
    total_action_activities = sum(run["action_activity_count"] for run in runs)
    total_failures = sum(run["gate_failure_steps"] for run in runs)
    coverage = {
        bucket: {
            "count": sum(run["localization_coverage"][bucket]["count"] for run in runs),
            "denominator": total_failures,
            "rate": None,
        }
        for bucket in ("HIGH", "LOW", "NA")
    }
    for row in coverage.values():
        row["rate"] = None if row["denominator"] == 0 else row["count"] / row["denominator"]

    opaque_shell_actions = sum(run["opaque_shell_action_steps"] for run in runs)
    unmapped_actions = sum(run["unmapped_action_steps"] for run in runs)
    opaque_or_unmapped_actions = sum(run["opaque_or_unmapped_action_steps"] for run in runs)
    unlinked_actions = sum(run["unlinked_action_steps_excluding_first_action"] for run in runs)
    all_activity_opaque_or_unmapped = sum(run["all_activity_opaque_or_unmapped_steps"] for run in runs)
    return {
        "schema_version": FOREIGN_REPORT_SCHEMA_VERSION,
        "schema": schema,
        "trajectory_count": len(runs),
        "deduplication_key": "instance_id",
        "distinct_instance_id_count": len({run["instance_id"] for run in runs if run["instance_id"]}),
        "activity_count": total_activities,
        "action_activity_count": total_action_activities,
        "localization_denominator": "gate_failure_steps",
        "localization_coverage": coverage,
        "coverage_gaps": {
            "activity_count": total_activities,
            "action_activity_count": total_action_activities,
            "opaque_shell_action_steps": opaque_shell_actions,
            "opaque_shell_action_fraction": None if total_action_activities == 0 else opaque_shell_actions / total_action_activities,
            "unmapped_action_steps": unmapped_actions,
            "unmapped_action_fraction": None if total_action_activities == 0 else unmapped_actions / total_action_activities,
            "opaque_or_unmapped_action_steps": opaque_or_unmapped_actions,
            "opaque_or_unmapped_action_fraction": None if total_action_activities == 0 else opaque_or_unmapped_actions / total_action_activities,
            "unlinked_action_steps_excluding_first_action": unlinked_actions,
            "unlinked_action_fraction_excluding_first_action": None if total_action_activities == 0 else unlinked_actions / total_action_activities,
            "all_activity_opaque_or_unmapped_steps": all_activity_opaque_or_unmapped,
            "all_activity_opaque_or_unmapped_fraction": None if total_activities == 0 else all_activity_opaque_or_unmapped / total_activities,
        },
        "source_summary": _source_summary(runs),
        "per_instance_breakdown": _per_instance_breakdown(runs),
        "runs": sorted(runs, key=lambda run: run["local_path"]),
    }


def _per_instance_breakdown(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        instance_id = run["instance_id"] or f"local:{run['local_path']}"
        grouped.setdefault(instance_id, []).append(run)

    breakdown = []
    for instance_id, instance_runs in sorted(grouped.items()):
        failures = sum(run["gate_failure_steps"] for run in instance_runs)
        coverage = {
            bucket: {
                "count": sum(run["localization_coverage"][bucket]["count"] for run in instance_runs),
                "denominator": failures,
                "rate": None,
            }
            for bucket in ("HIGH", "LOW", "NA")
        }
        for row in coverage.values():
            row["rate"] = None if row["denominator"] == 0 else row["count"] / row["denominator"]
        breakdown.append(
            {
                "instance_id": instance_id,
                "trajectory_count": len(instance_runs),
                "activity_count": sum(run["activity_count"] for run in instance_runs),
                "action_activity_count": sum(run["action_activity_count"] for run in instance_runs),
                "gate_failure_steps": failures,
                "localization_coverage": coverage,
                "opaque_shell_action_steps": sum(run["opaque_shell_action_steps"] for run in instance_runs),
                "unmapped_action_steps": sum(run["unmapped_action_steps"] for run in instance_runs),
                "opaque_or_unmapped_action_steps": sum(run["opaque_or_unmapped_action_steps"] for run in instance_runs),
                "unlinked_action_steps_excluding_first_action": sum(run["unlinked_action_steps_excluding_first_action"] for run in instance_runs),
            }
        )
    return breakdown


def _localization_by_step(activities: list[Activity]) -> dict[int, str]:
    by_step: dict[int, str] = {}
    for failure in localize_failures(activities).failures:
        by_step[failure.breaking_step] = failure.confidence
        for step in failure.failure_cascade:
            by_step[step] = failure.confidence
    return by_step


def _coverage_counts(failure_steps: list[int], localization_by_step: dict[int, str]) -> dict[str, dict[str, Any]]:
    counts = {bucket: 0 for bucket in ("HIGH", "LOW", "NA")}
    for step in failure_steps:
        confidence = localization_by_step.get(step, "NA")
        counts[confidence if confidence in {"HIGH", "LOW"} else "NA"] += 1
    return {
        bucket: {
            "count": counts[bucket],
            "denominator": len(failure_steps),
            "rate": None if not failure_steps else counts[bucket] / len(failure_steps),
        }
        for bucket in ("HIGH", "LOW", "NA")
    }


def _source_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for run in runs:
        key = (run["repo"], run["commit"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"repo": repo, "commit": commit, "count": count}
        for (repo, commit), count in sorted(counts.items())
    ]


def _gate_status_counts(results: list[Any]) -> dict[str, int]:
    counter = Counter(f"{result.gate}:{result.status}" for result in results)
    return dict(sorted(counter.items()))


def _load_source_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    manifests = []
    for path in paths:
        if path.is_dir():
            manifest = path / "manifest.json"
            if manifest.exists():
                manifests.append(manifest)
        elif path.name == "manifest.json":
            manifests.append(path)
        else:
            manifest = path.parent / "manifest.json"
            if manifest.exists():
                manifests.append(manifest)
    index: dict[str, dict[str, Any]] = {}
    for manifest in sorted(set(manifests)):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for source in data.get("sources", []):
            if isinstance(source, dict) and source.get("local_path"):
                index[str(source["local_path"])] = source
    return index


def _expand_foreign_paths(paths: list[Path], *, schema: str) -> list[Path]:
    suffixes = {
        "swe-agent": {".traj"},
        "openhands": {".json"},
        "mini-swe-agent": {".json", ".jsonl"},
    }.get(schema)
    if suffixes is None:
        raise ValueError(f"unsupported foreign schema: {schema}")

    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            for suffix in sorted(suffixes):
                expanded.extend(
                    sorted(candidate for candidate in path.rglob(f"*{suffix}") if candidate.name != "manifest.json")
                )
        elif path.suffix in suffixes and path.name != "manifest.json":
            expanded.append(path)
    return sorted({path.resolve() for path in expanded})


def _history_records(data: Any) -> list[Any]:
    if isinstance(data, dict):
        history = data.get("history")
        if isinstance(history, list):
            return history
        trajectory = data.get("trajectory")
        if isinstance(trajectory, list):
            return trajectory
        return [data]
    if isinstance(data, list):
        return data
    return [data]


def _record_role(record: dict[str, Any]) -> str:
    return str(record.get("role") or "").lower()


def _is_observation(record: dict[str, Any]) -> bool:
    role = _record_role(record)
    message_type = str(record.get("message_type") or "").lower()
    return role == "tool" or message_type == "observation"


def _linked_action_step(record: dict[str, Any], call_to_step: dict[str, int], pending_action_step: int | None) -> int | None:
    single_id = record.get("tool_call_id")
    if single_id is not None and str(single_id) in call_to_step:
        return call_to_step[str(single_id)]
    ids = record.get("tool_call_ids")
    if isinstance(ids, list):
        for call_id in ids:
            if str(call_id) in call_to_step:
                return call_to_step[str(call_id)]
    return pending_action_step


def _tool_call_ids(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return []
    ids = []
    for tool_call in record.get("tool_calls") or []:
        if isinstance(tool_call, dict) and tool_call.get("id"):
            ids.append(str(tool_call["id"]))
    return ids


def _base_inputs(record: dict[str, Any], source_index: int, source_url: str) -> dict[str, Any]:
    summary = _content_summary(record)
    return {
        "source_index": source_index,
        "source_url": source_url,
        "role": str(record.get("role") or ""),
        "message_type": str(record.get("message_type") or ""),
        "agent": str(record.get("agent") or ""),
        **summary,
    }


def _content_summary(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content")
    thought = record.get("thought")
    action = record.get("action")
    return {
        "content_length": len(content) if isinstance(content, str) else 0,
        "content_hash": sha256_text(content) if isinstance(content, str) else "",
        "thought_length": len(thought) if isinstance(thought, str) else 0,
        "thought_hash": sha256_text(thought) if isinstance(thought, str) else "",
        "action_length": len(action) if isinstance(action, str) else 0,
        "action_hash": sha256_text(action) if isinstance(action, str) else "",
    }


def _observation_outputs(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content")
    text = content if isinstance(content, str) else canonical_json(content)
    return {
        "result_text": text,
        "content_length": len(text),
        "content_hash": sha256_text(text),
    }


def _current_open_file_from_action(record: dict[str, Any], current: str | None) -> str | None:
    action = record.get("action")
    if isinstance(action, str):
        return _open_target_from_action(action) or _create_target_from_action(action) or current
    tool_calls = record.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) == 1:
        function = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
        if isinstance(function, dict):
            args = _parse_arguments(function.get("arguments"))
            name = str(function.get("name") or "")
            if name in {"open", "create"}:
                return _target_from_args(args) or current
    return current


def _current_open_file_from_observation(record: dict[str, Any], current: str | None) -> str | None:
    content = record.get("content")
    if not isinstance(content, str):
        return current
    match = re.search(r"\(Open file:\s*([^)]+)\)", content)
    if match:
        value = match.group(1).strip()
        if value and value.lower() != "n/a":
            return value
    file_match = re.search(r"\[File:\s*([^\]\r\n]+)", content)
    if file_match:
        return file_match.group(1).split("(", 1)[0].strip()
    return current


def _open_target_from_action(action: str) -> str | None:
    match = re.match(r"open\s+(.+?)(?:\s+\d+)?$", action.strip())
    if not match:
        return None
    return match.group(1).strip().strip("\"'")


def _create_target_from_action(action: str) -> str | None:
    match = re.match(r"create\s+(.+?)$", action.strip())
    if not match:
        return None
    return match.group(1).strip().strip("\"'")


def _line_edit_payload(action: str) -> str | None:
    lines = action.splitlines()
    if len(lines) < 3 or not lines[0].startswith("edit "):
        return None
    if lines[-1].strip() != "end_of_edit":
        return None
    return "\n".join(lines[1:-1]) + "\n"


def _target_from_args(args: dict[str, Any]) -> str | None:
    for key in ("path", "file_path", "file_name"):
        value = _as_nonempty_str(args.get(key))
        if value:
            return value
    return None


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw_arguments": value}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {}


def _looks_like_foreign_test(command: str) -> bool:
    normalized = command.lower().strip()
    patterns = [
        r"\bpytest\b",
        r"\bpython(?:3)?\s+[\w./-]+\.py\b",
        r"\bnpm\s+(test|run\s+test|run\s+build)\b",
        r"\btox\b",
        r"\bgo\s+test\b",
        r"\bcargo\s+test\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _github_head_commit(repo: str, branch: str) -> str:
    data = _download_json(f"https://api.github.com/repos/{repo}/git/refs/heads/{branch}")
    return str(data["object"]["sha"])


def _github_tree_paths(repo: str, commit: str) -> list[str]:
    data = _download_json(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")
    return sorted(str(item["path"]) for item in data.get("tree", []) if item.get("type") == "blob")


def _huggingface_dataset_commit(dataset: str) -> str:
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
    return json.loads(_download_text(url))


def _download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "agent-bisect"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _rate(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def _as_nonempty_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return safe.strip("._") or "unknown"
