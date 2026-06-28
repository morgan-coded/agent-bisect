from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .localize import shell_target_coverage
from .model import Activity, Journal, canonical_json, sha256_text


CODEX_TOP_LEVEL_TYPES = {
    "session_meta",
    "turn_context",
    "response_item",
    "event_msg",
    "compacted",
}

SHELL_TOOL_NAMES = {"shell_command", "exec_command", "local_shell_call"}
PATCH_TOOL_NAMES = {"apply_patch"}
MESSAGE_PAYLOAD_TYPES = {"message"}
REASONING_PAYLOAD_TYPES = {"reasoning"}
USER_EVENT_TYPES = {"user_message"}
AGENT_EVENT_TYPES = {"agent_message"}
VERDICT_EVENT_TYPES = {
    "task_complete",
    "turn_aborted",
    "thread_goal_updated",
    "thread_rolled_back",
    "patch_apply_end",
}
OUTPUT_PAYLOAD_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "tool_search_output",
}
CALL_PAYLOAD_TYPES = {
    "function_call",
    "custom_tool_call",
    "web_search_call",
    "tool_search_call",
    "image_generation_call",
}
ACTION_KINDS = {"file_edit", "test_run", "tool_call", "opaque_shell", "unmapped"}
TEXT_KEYS = ("text", "message", "summary", "content", "output", "stdout", "stderr")


def ingest_codex_transcript(path: str | Path) -> list[Activity]:
    transcript_path = Path(path)
    run_id = _run_id_for_path(transcript_path)
    state = _IngestState(run_id=run_id)

    for source_index, record in enumerate(_iter_jsonl(transcript_path)):
        state.ingest_record(record, source_index)

    for index, activity in enumerate(state.activities):
        activity.step_index = index
        activity.refresh_hash()
    return state.activities


def looks_like_codex(path: str | Path) -> bool:
    for record in _iter_jsonl(Path(path), strict=False):
        if not isinstance(record, dict):
            return False
        record_type = _as_optional_str(record.get("type"))
        payload = record.get("payload")
        return record_type in CODEX_TOP_LEVEL_TYPES and isinstance(payload, dict)
    return False


def codex_coverage_report(paths: Iterable[str | Path]) -> dict[str, Any]:
    transcript_paths = _expand_paths([Path(path) for path in paths])
    runs = []
    aggregate_unmapped = Counter()
    aggregate_kinds = Counter()
    aggregate_record_types = Counter()
    total_records = 0
    total_activities = 0
    total_action_activities = 0
    total_linked_actions = 0
    total_opaque = 0
    total_shell_steps = 0
    total_shell_steps_with_targets = 0
    total_shell_added_edges = 0

    for transcript_path in transcript_paths:
        records = _iter_jsonl(transcript_path)
        activities = ingest_codex_transcript(transcript_path)
        kind_counts = Counter(activity.kind for activity in activities)
        record_type_counts = Counter(_record_family(record) for record in records if isinstance(record, dict))
        unmapped_counts = Counter(
            str(activity.inputs.get("payload_type") or activity.inputs.get("record_type") or "<missing>")
            for activity in activities
            if activity.kind == "unmapped"
        )
        action_activities = [activity for activity in activities if activity.kind in ACTION_KINDS]
        linked_actions = sum(1 for activity in action_activities if activity.parent_step is not None)
        shell_coverage = shell_target_coverage(activities)

        runs.append(
            {
                "run_key": _run_id_for_path(transcript_path),
                "records": len(records),
                "activities": len(activities),
                "kind_counts": dict(sorted(kind_counts.items())),
                "record_type_counts": dict(sorted(record_type_counts.items())),
                "unmapped_counts": dict(sorted(unmapped_counts.items())),
                "action_activities": len(action_activities),
                "linked_action_activities": linked_actions,
                "opaque_shell_activities": kind_counts.get("opaque_shell", 0),
                "shell_command_steps": shell_coverage.shell_command_steps,
                "steps_with_targets": shell_coverage.steps_with_targets,
                "shell_added_edges": shell_coverage.added_edges,
            }
        )

        total_records += len(records)
        total_activities += len(activities)
        total_action_activities += len(action_activities)
        total_linked_actions += linked_actions
        total_opaque += kind_counts.get("opaque_shell", 0)
        total_shell_steps += shell_coverage.shell_command_steps
        total_shell_steps_with_targets += shell_coverage.steps_with_targets
        total_shell_added_edges += shell_coverage.added_edges
        aggregate_unmapped.update(unmapped_counts)
        aggregate_kinds.update(kind_counts)
        aggregate_record_types.update(record_type_counts)

    return {
        "schema_version": 1,
        "transcripts": len(transcript_paths),
        "total_records": total_records,
        "total_activities": total_activities,
        "mapped_activities": total_activities - aggregate_kinds.get("unmapped", 0),
        "unmapped_activities": aggregate_kinds.get("unmapped", 0),
        "kind_counts": dict(sorted(aggregate_kinds.items())),
        "record_type_counts": dict(sorted(aggregate_record_types.items())),
        "action_activities": total_action_activities,
        "linked_action_activities": total_linked_actions,
        "opaque_shell_activities": total_opaque,
        "shell_command_steps": total_shell_steps,
        "steps_with_targets": total_shell_steps_with_targets,
        "shell_added_edges": total_shell_added_edges,
        "unmapped_counts": dict(sorted(aggregate_unmapped.items())),
        "runs": runs,
    }


def render_codex_coverage_markdown(report: dict[str, Any]) -> str:
    total_activities = int(report["total_activities"])
    action_activities = int(report["action_activities"])
    shell_steps = int(report["shell_command_steps"])
    lines = [
        "# Codex Transcript Coverage",
        "",
        f"Transcripts: {report['transcripts']}",
        f"Source records: {report['total_records']}",
        f"Activities: {report['total_activities']}",
        f"Mapped activities: {report['mapped_activities']}/{total_activities} ({_rate(report['mapped_activities'], total_activities)})",
        f"Unmapped activities: {report['unmapped_activities']}/{total_activities} ({_rate(report['unmapped_activities'], total_activities)})",
        f"Opaque shell activities: {report['opaque_shell_activities']}/{total_activities} ({_rate(report['opaque_shell_activities'], total_activities)})",
        f"Linked action activities: {report['linked_action_activities']}/{action_activities} ({_rate(report['linked_action_activities'], action_activities)})",
        f"Shell-target lift: steps_with_targets {report['steps_with_targets']}/{shell_steps}; added_edges {report['shell_added_edges']}",
        "",
        "## Kind Distribution",
        "",
        "| kind | count |",
        "| --- | ---: |",
    ]
    for kind, count in report["kind_counts"].items():
        lines.append(f"| {kind} | {count} |")

    lines.extend(["", "## Record Families", "", "| family | count |", "| --- | ---: |"])
    for family, count in report["record_type_counts"].items():
        lines.append(f"| {family} | {count} |")

    lines.extend(["", "## Unmapped Payload Types", "", "| payload_type | count |", "| --- | ---: |"])
    if report["unmapped_counts"]:
        for payload_type, count in report["unmapped_counts"].items():
            lines.append(f"| {payload_type} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Per-Transcript Summary",
            "",
            "| run_key | records | activities | action_activities | linked_actions | shell_targets | unmapped |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in report["runs"]:
        lines.append(
            "| {run_id} | {records} | {activities} | {actions} | {linked} | {targets}/{shells} | {unmapped} |".format(
                run_id=run["run_key"],
                records=run["records"],
                activities=run["activities"],
                actions=run["action_activities"],
                linked=run["linked_action_activities"],
                targets=run["steps_with_targets"],
                shells=run["shell_command_steps"],
                unmapped=run["kind_counts"].get("unmapped", 0),
            )
        )
    lines.append("")
    return "\n".join(lines)


class _IngestState:
    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        self.activities: list[Activity] = []
        self.call_to_step: dict[str, int] = {}
        self.id_to_step: dict[str, int] = {}
        self.last_step_by_turn: dict[str, int] = {}
        self.last_step: int | None = None

    def ingest_record(self, record: Any, source_index: int) -> None:
        if not isinstance(record, dict):
            self.add(_unmapped(self.run_id, self.next_step, source_index, "non_object_record", record, None))
            return

        record_type = _as_optional_str(record.get("type")) or "<missing>"
        payload = record.get("payload")
        if not isinstance(payload, dict):
            self.add(_unmapped(self.run_id, self.next_step, source_index, "missing_payload", record, None))
            return

        payload_type = _as_optional_str(payload.get("type")) or "<missing>"
        turn_id = _turn_id(payload)
        parent_step = self._parent_for(payload)
        before_count = len(self.activities)

        if record_type == "response_item":
            self._ingest_response_item(record, payload, payload_type, source_index, parent_step)
        elif record_type == "event_msg":
            self._ingest_event(record, payload, payload_type, source_index, parent_step)
        elif record_type in {"session_meta", "turn_context", "compacted"}:
            self.add(_unmapped(self.run_id, self.next_step, source_index, payload_type, record, parent_step))
        else:
            self.add(_unmapped(self.run_id, self.next_step, source_index, payload_type, record, parent_step))

        if len(self.activities) > before_count:
            first_created = self.activities[before_count].step_index
            item_id = _as_optional_str(payload.get("id"))
            if item_id:
                self.id_to_step[item_id] = first_created
            if turn_id:
                self.last_step_by_turn[turn_id] = self.activities[-1].step_index

    @property
    def next_step(self) -> int:
        return len(self.activities)

    def add(self, activity: Activity) -> None:
        self.activities.append(activity)
        self.last_step = activity.step_index
        call_id = _as_optional_str(activity.inputs.get("call_id"))
        if call_id:
            self.call_to_step[call_id] = activity.step_index

    def _parent_for(self, payload: dict[str, Any]) -> int | None:
        call_id = _as_optional_str(payload.get("call_id"))
        if call_id and payload.get("type") in OUTPUT_PAYLOAD_TYPES and call_id in self.call_to_step:
            return self.call_to_step[call_id]
        turn_id = _turn_id(payload)
        if turn_id and turn_id in self.last_step_by_turn:
            return self.last_step_by_turn[turn_id]
        return self.last_step

    def _ingest_response_item(
        self,
        record: dict[str, Any],
        payload: dict[str, Any],
        payload_type: str,
        source_index: int,
        parent_step: int | None,
    ) -> None:
        if payload_type in MESSAGE_PAYLOAD_TYPES:
            activity = _message_activity(self.run_id, self.next_step, source_index, record, payload, parent_step)
            self.add(activity)
            return
        if payload_type in REASONING_PAYLOAD_TYPES:
            self.add(_reasoning_activity(self.run_id, self.next_step, source_index, record, payload, parent_step))
            return
        if payload_type in CALL_PAYLOAD_TYPES:
            self.add(_call_activity(self.run_id, self.next_step, source_index, record, payload, parent_step))
            return
        if payload_type in OUTPUT_PAYLOAD_TYPES:
            if not self._apply_output(payload):
                self.add(_unmapped(self.run_id, self.next_step, source_index, "unlinked_" + payload_type, record, parent_step))
            return
        self.add(_unmapped(self.run_id, self.next_step, source_index, payload_type, record, parent_step))

    def _ingest_event(
        self,
        record: dict[str, Any],
        payload: dict[str, Any],
        payload_type: str,
        source_index: int,
        parent_step: int | None,
    ) -> None:
        if payload_type in USER_EVENT_TYPES:
            self.add(_event_text_activity(self.run_id, self.next_step, source_index, record, payload, "user_msg", parent_step))
            return
        if payload_type in AGENT_EVENT_TYPES:
            self.add(_event_text_activity(self.run_id, self.next_step, source_index, record, payload, "llm_call", parent_step))
            return
        if payload_type in VERDICT_EVENT_TYPES:
            self.add(_verdict_activity(self.run_id, self.next_step, source_index, record, payload, parent_step))
            return
        if payload.get("call_id") and self._apply_output(payload):
            return
        self.add(_unmapped(self.run_id, self.next_step, source_index, payload_type, record, parent_step))

    def _apply_output(self, payload: dict[str, Any]) -> bool:
        call_id = _as_optional_str(payload.get("call_id"))
        if not call_id or call_id not in self.call_to_step:
            return False
        activity = self.activities[self.call_to_step[call_id]]
        activity.outputs.update(_output_summary(payload, activity.kind))
        activity.refresh_hash()
        turn_id = _turn_id(payload)
        if turn_id:
            self.last_step_by_turn[turn_id] = activity.step_index
        self.last_step = activity.step_index
        return True


def _iter_jsonl(path: Path, *, strict: bool = True) -> list[Any]:
    records: list[Any] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        if strict:
            raise
        return records
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
                return []
    return records


def _message_activity(
    run_id: str,
    step_index: int,
    source_index: int,
    record: dict[str, Any],
    payload: dict[str, Any],
    parent_step: int | None,
) -> Activity:
    role = str(payload.get("role") or "")
    kind = "llm_call" if role == "assistant" else "user_msg"
    summary = _content_summary(payload.get("content"))
    summary.update(_base_inputs(source_index, record, payload))
    summary["role"] = role
    kwargs = {"outputs": summary} if kind == "llm_call" else {"inputs": summary}
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=_timestamp(record, payload),
        kind=kind,
        parent_step=parent_step,
        **kwargs,
    )


def _reasoning_activity(
    run_id: str,
    step_index: int,
    source_index: int,
    record: dict[str, Any],
    payload: dict[str, Any],
    parent_step: int | None,
) -> Activity:
    outputs = _base_inputs(source_index, record, payload)
    outputs.update(
        {
            "summary_count": len(payload.get("summary")) if isinstance(payload.get("summary"), list) else 0,
            "encrypted_content_length": len(payload.get("encrypted_content") or "")
            if isinstance(payload.get("encrypted_content"), str)
            else 0,
            "encrypted_content_hash": sha256_text(payload.get("encrypted_content"))
            if isinstance(payload.get("encrypted_content"), str)
            else "",
        }
    )
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=_timestamp(record, payload),
        kind="llm_call",
        outputs=outputs,
        parent_step=parent_step,
    )


def _event_text_activity(
    run_id: str,
    step_index: int,
    source_index: int,
    record: dict[str, Any],
    payload: dict[str, Any],
    kind: str,
    parent_step: int | None,
) -> Activity:
    summary = _text_value_summary(payload.get("message"))
    summary.update(_base_inputs(source_index, record, payload))
    if isinstance(payload.get("text_elements"), list):
        summary["text_element_count"] = len(payload["text_elements"])
    summary["image_count"] = len(payload.get("images")) if isinstance(payload.get("images"), list) else 0
    summary["local_image_count"] = len(payload.get("local_images")) if isinstance(payload.get("local_images"), list) else 0
    kwargs = {"outputs": summary} if kind == "llm_call" else {"inputs": summary}
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=_timestamp(record, payload),
        kind=kind,
        parent_step=parent_step,
        **kwargs,
    )


def _call_activity(
    run_id: str,
    step_index: int,
    source_index: int,
    record: dict[str, Any],
    payload: dict[str, Any],
    parent_step: int | None,
) -> Activity:
    tool_name = _tool_name(payload)
    arguments = _arguments(payload)
    call_id = _as_optional_str(payload.get("call_id"))
    if tool_name in SHELL_TOOL_NAMES:
        return _shell_activity(run_id, step_index, source_index, record, payload, arguments, parent_step)
    if tool_name in PATCH_TOOL_NAMES:
        return _patch_activity(run_id, step_index, source_index, record, payload, arguments, parent_step)

    inputs = _base_inputs(source_index, record, payload)
    inputs.update(_argument_summary(arguments))
    if call_id:
        inputs["call_id"] = call_id
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=_timestamp(record, payload),
        kind="tool_call",
        tool_name=tool_name,
        inputs=inputs,
        target=_target_from_arguments(arguments),
        parent_step=parent_step,
    )


def _shell_activity(
    run_id: str,
    step_index: int,
    source_index: int,
    record: dict[str, Any],
    payload: dict[str, Any],
    arguments: Any,
    parent_step: int | None,
) -> Activity:
    command = _shell_command(arguments)
    inputs = _base_inputs(source_index, record, payload)
    inputs.update(_argument_summary(arguments))
    inputs["command"] = command or ""
    inputs["command_length"] = len(command or "")
    inputs["command_hash"] = sha256_text(command or "")
    call_id = _as_optional_str(payload.get("call_id"))
    if call_id:
        inputs["call_id"] = call_id
    if isinstance(arguments, dict) and "workdir" in arguments:
        inputs["workdir_summary"] = _value_summary(arguments.get("workdir"))

    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=_timestamp(record, payload),
        kind="test_run" if _looks_like_test_or_build(command) else "opaque_shell",
        tool_name="PowerShell",
        inputs=inputs,
        target="shell",
        parent_step=parent_step,
    )


def _patch_activity(
    run_id: str,
    step_index: int,
    source_index: int,
    record: dict[str, Any],
    payload: dict[str, Any],
    arguments: Any,
    parent_step: int | None,
) -> Activity:
    patch_text = arguments if isinstance(arguments, str) else canonical_json(arguments)
    targets = _patch_targets(patch_text)
    target = targets[0] if len(targets) == 1 else None
    patch_hash = sha256_text(patch_text)
    inputs = _base_inputs(source_index, record, payload)
    inputs.update(
        {
            "file_path": target or "<multiple-or-unknown>",
            "old_string": f"<patch-before-unavailable:{patch_hash[:12]}>",
            "new_string": f"<patch-summary:{len(patch_text)}:{patch_hash[:12]}>",
            "patch_length": len(patch_text),
            "patch_hash": patch_hash,
            "patch_targets": targets,
        }
    )
    call_id = _as_optional_str(payload.get("call_id"))
    if call_id:
        inputs["call_id"] = call_id
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=_timestamp(record, payload),
        kind="file_edit",
        tool_name="apply_patch",
        inputs=inputs,
        target=target,
        parent_step=parent_step,
    )


def _verdict_activity(
    run_id: str,
    step_index: int,
    source_index: int,
    record: dict[str, Any],
    payload: dict[str, Any],
    parent_step: int | None,
) -> Activity:
    outputs = _base_inputs(source_index, record, payload)
    outputs.update(_payload_structural_summary(payload))
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=_timestamp(record, payload),
        kind="verdict",
        tool_name=_as_optional_str(payload.get("type")),
        outputs=outputs,
        parent_step=parent_step,
    )


def _unmapped(
    run_id: str,
    step_index: int,
    source_index: int,
    reason: str,
    record: Any,
    parent_step: int | None,
) -> Activity:
    payload = record.get("payload") if isinstance(record, dict) else None
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=_timestamp(record if isinstance(record, dict) else {}, payload if isinstance(payload, dict) else {}),
        kind="unmapped",
        tool_name="codex_unmapped",
        inputs={
            "source_index": source_index,
            "reason": reason,
            "record_type": _as_optional_str(record.get("type")) if isinstance(record, dict) else type(record).__name__,
            "payload_type": _as_optional_str(payload.get("type")) if isinstance(payload, dict) else "<missing>",
            "top_level_keys": sorted(str(key) for key in record.keys()) if isinstance(record, dict) else [],
            "payload_keys": sorted(str(key) for key in payload.keys()) if isinstance(payload, dict) else [],
            "record_hash": sha256_text(canonical_json(record)),
        },
        parent_step=parent_step,
    )


def _output_summary(payload: dict[str, Any], activity_kind: str) -> dict[str, Any]:
    output = payload.get("output")
    summary = {
        "output": _value_summary(output),
        "payload_type": _as_optional_str(payload.get("type")) or "<missing>",
        "call_id": _as_optional_str(payload.get("call_id")) or "",
    }
    exit_code = _extract_exit_code(output)
    if exit_code is not None:
        summary["exit_code"] = exit_code
    if activity_kind == "test_run":
        signal = _test_result_signal(output, exit_code)
        if signal:
            summary["result_text"] = signal
    return summary


def _base_inputs(source_index: int, record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "record_type": _as_optional_str(record.get("type")) or "<missing>",
        "payload_type": _as_optional_str(payload.get("type")) or "<missing>",
        "payload_keys": sorted(str(key) for key in payload.keys()),
    }


def _payload_structural_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in sorted(payload.items()):
        if key in TEXT_KEYS:
            summary[f"{key}_summary"] = _value_summary(value)
        elif key == "changes" and isinstance(value, dict):
            summary["change_count"] = len(value)
            summary["change_path_hashes"] = [sha256_text(str(path)) for path in sorted(value)]
        elif key not in {"type", "call_id", "turn_id"}:
            summary[f"{key}_kind"] = type(value).__name__
    return summary


def _content_summary(content: Any) -> dict[str, Any]:
    blocks = content if isinstance(content, list) else [content] if content is not None else []
    texts = []
    content_types = []
    image_count = 0
    for block in blocks:
        if isinstance(block, dict):
            content_type = _as_optional_str(block.get("type")) or "<missing>"
            content_types.append(content_type)
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
            if content_type == "input_image":
                image_count += 1
        elif isinstance(block, str):
            content_types.append("str")
            texts.append(block)
        else:
            content_types.append(type(block).__name__)
    text = "\n".join(texts)
    return {
        "content_blocks": len(blocks),
        "content_types": sorted(Counter(content_types).items()),
        "text_length": len(text),
        "text_hash": sha256_text(text),
        "image_count": image_count,
    }


def _text_value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "text_length": len(value),
            "text_hash": sha256_text(value),
        }
    return {
        "text_length": 0,
        "text_hash": "",
        "text_kind": type(value).__name__,
    }


def _value_summary(value: Any) -> dict[str, Any]:
    rendered = canonical_json(value)
    keys = sorted(str(key) for key in value.keys()) if isinstance(value, dict) else []
    return {
        "kind": type(value).__name__,
        "keys": keys,
        "length": len(rendered),
        "hash": sha256_text(rendered),
    }


def _argument_summary(arguments: Any) -> dict[str, Any]:
    summary = _value_summary(arguments)
    return {
        "arguments_kind": summary["kind"],
        "argument_keys": summary["keys"],
        "arguments_length": summary["length"],
        "arguments_hash": summary["hash"],
    }


def _arguments(payload: dict[str, Any]) -> Any:
    if "input" in payload:
        return payload.get("input")
    raw = payload.get("arguments")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return parsed
    return raw


def _tool_name(payload: dict[str, Any]) -> str:
    name = _as_optional_str(payload.get("name"))
    if name:
        return name
    return _as_optional_str(payload.get("type")) or "<missing>"


def _shell_command(arguments: Any) -> str | None:
    if isinstance(arguments, dict):
        for key in ("command", "cmd"):
            value = arguments.get(key)
            if isinstance(value, str):
                return value
        parameters = arguments.get("parameters")
        if isinstance(parameters, dict):
            value = parameters.get("command")
            if isinstance(value, str):
                return value
    if isinstance(arguments, str):
        return arguments
    return None


def _target_from_arguments(arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return None
    for key in ("file_path", "path", "target", "filename"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _patch_targets(patch_text: str) -> list[str]:
    targets: set[str] = set()
    for line in patch_text.splitlines():
        for prefix in ("*** Add File: ", "*** Update File: ", "*** Delete File: ", "*** Move to: "):
            if line.startswith(prefix):
                target = line[len(prefix) :].strip()
                if target:
                    targets.add(target)
    return sorted(targets)


def _extract_exit_code(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("exit_code", "exitCode", "code", "status"):
            candidate = value.get(key)
            parsed = _parse_int(candidate)
            if parsed is not None:
                return parsed
        for nested in value.values():
            parsed = _extract_exit_code(nested)
            if parsed is not None:
                return parsed
    if isinstance(value, str):
        match = re.search(r"\bExit code:\s*(-?\d+)\b", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _test_result_signal(value: Any, exit_code: int | None) -> str:
    text = _extract_text(value).lower()
    pass_match = re.search(r"\b(\d+)\s+passed\b", text)
    if pass_match:
        return f"{pass_match.group(1)} passed"
    fail_match = re.search(r"\b(\d+)\s+(failed|failures|errors?)\b", text)
    if fail_match:
        return f"{fail_match.group(1)} failed"
    if exit_code is not None and exit_code != 0:
        return "1 failed"
    return ""


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_extract_text(item) for item in value)
    if isinstance(value, dict):
        parts = []
        for key in TEXT_KEYS:
            if key in value:
                parts.append(_extract_text(value[key]))
        return "\n".join(part for part in parts if part)
    return ""


def _looks_like_test_or_build(command: Any) -> bool:
    if not isinstance(command, str):
        return False
    normalized = command.lower()
    patterns = [
        r"\bpytest\b",
        r"\bpython(?:\.exe)?\s+-m\s+pytest\b",
        r"\bpython(?:\.exe)?\s+-m\s+unittest\b",
        r"\bpython(?:\.exe)?\s+-m\s+[\w.]*verify\b",
        r"\bnpm\s+test\b",
        r"\bnpm\s+run\s+(test|build)\b",
        r"\bpnpm\s+(test|build)\b",
        r"\byarn\s+(test|build)\b",
        r"\bgo\s+test\b",
        r"\bcargo\s+test\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _turn_id(payload: dict[str, Any]) -> str | None:
    direct = _as_optional_str(payload.get("turn_id"))
    if direct:
        return direct
    for key in ("metadata", "internal_chat_message_metadata_passthrough"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            value = _as_optional_str(nested.get("turn_id"))
            if value:
                return value
    return None


def _timestamp(record: dict[str, Any], payload: dict[str, Any]) -> str:
    return _as_optional_str(record.get("timestamp")) or _as_optional_str(payload.get("timestamp")) or ""


def _record_family(record: dict[str, Any]) -> str:
    payload = record.get("payload")
    payload_type = payload.get("type") if isinstance(payload, dict) else "<missing>"
    return f"{record.get('type') or '<missing>'}:{payload_type or '<missing>'}"


def _expand_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(candidate for candidate in path.rglob("*.jsonl") if candidate.is_file()))
        elif path.is_file():
            expanded.append(path)
    return sorted({path.resolve() for path in expanded})


def _run_id_for_path(path: Path) -> str:
    return "codex-" + sha256_text(path.stem)[:16]


def _rate(numerator: Any, denominator: int) -> str:
    if denominator == 0:
        return "NA"
    return f"{int(numerator) / denominator:.3f}"


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    return None


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
