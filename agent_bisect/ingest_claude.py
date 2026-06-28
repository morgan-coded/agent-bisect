from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .model import Activity, canonical_json, sha256_text


FILE_EDIT_TOOLS = {"Edit", "MultiEdit", "Write"}
STRUCTURED_TOOL_CALLS = {"Read", "Grep", "Glob"}
SHELL_TOOLS = {"Bash", "PowerShell"}

# TODO(agent CLI ingest): add adapters for additional agent transcript formats
# once their command and tool-call schemas are normalized.
# TODO(subagents): link sidecar transcripts through tool_use_id/agent ids in a
# later slice. This adapter uses only top-level main-agent transcripts.


def ingest_transcript(path: str | Path) -> list[Activity]:
    transcript_path = Path(path)
    run_id = transcript_path.stem
    activities: list[Activity] = []
    uuid_to_step: dict[str, int] = {}
    tool_use_to_step: dict[str, int] = {}

    for record in _iter_jsonl(transcript_path):
        record_uuid = _as_optional_str(record.get("uuid"))
        parent_step = _resolve_parent(record, uuid_to_step, tool_use_to_step)
        created_steps: list[int] = []

        for activity in _activities_from_record(record, run_id, parent_step, len(activities)):
            activities.append(activity)
            created_steps.append(activity.step_index)
            tool_use_id = activity.inputs.get("tool_use_id")
            if isinstance(tool_use_id, str) and tool_use_id:
                tool_use_to_step[tool_use_id] = activity.step_index

        result_steps = _apply_tool_results(record, activities, tool_use_to_step)
        if record_uuid:
            if created_steps:
                uuid_to_step[record_uuid] = created_steps[0]
            elif result_steps:
                uuid_to_step[record_uuid] = result_steps[0]

    for index, activity in enumerate(activities):
        activity.step_index = index
        activity.refresh_hash()

    return activities


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record") from exc
    return records


def _activities_from_record(
    record: dict[str, Any],
    run_id: str,
    parent_step: int | None,
    next_step: int,
) -> list[Activity]:
    record_type = _as_optional_str(record.get("type")) or ""
    ts = _as_optional_str(record.get("timestamp")) or ""
    blocks = _content_blocks(record)
    activities: list[Activity] = []

    text_blocks = [block for block in blocks if block.get("type") in (None, "text") and _block_text(block)]
    tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
    tool_results = [block for block in blocks if block.get("type") == "tool_result"]

    if record_type == "user" and text_blocks and not tool_results:
        activities.append(
            Activity(
                run_id=run_id,
                step_index=next_step + len(activities),
                ts=ts,
                kind="user_msg",
                inputs=_text_summary(text_blocks),
                parent_step=parent_step,
            )
        )

    if record_type == "assistant" and text_blocks:
        activities.append(
            Activity(
                run_id=run_id,
                step_index=next_step + len(activities),
                ts=ts,
                kind="llm_call",
                outputs=_text_summary(text_blocks),
                parent_step=parent_step,
            )
        )

    for block in tool_uses:
        activities.append(_activity_from_tool_use(record, block, run_id, parent_step, next_step + len(activities)))

    if not activities and not tool_results and record_type not in {"user", "assistant"}:
        activities.append(
            Activity(
                run_id=run_id,
                step_index=next_step,
                ts=ts,
                kind="opaque_shell",
                inputs={
                    "record_type": record_type or "<missing>",
                    "top_level_keys": sorted(record.keys()),
                },
                parent_step=parent_step,
            )
        )

    return activities


def _activity_from_tool_use(
    record: dict[str, Any],
    block: dict[str, Any],
    run_id: str,
    parent_step: int | None,
    step_index: int,
) -> Activity:
    ts = _as_optional_str(record.get("timestamp")) or ""
    tool_name = _as_optional_str(block.get("name"))
    raw_inputs = _json_object(block.get("input"))
    tool_use_id = _as_optional_str(block.get("id"))
    inputs = _normalize_tool_inputs(tool_name, raw_inputs)
    if tool_use_id:
        inputs["tool_use_id"] = tool_use_id

    kind = _classify_tool(tool_name, inputs)
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts=ts,
        kind=kind,
        tool_name=tool_name,
        inputs=inputs,
        target=_target_for(tool_name, inputs),
        parent_step=parent_step,
    )


def _apply_tool_results(
    record: dict[str, Any],
    activities: list[Activity],
    tool_use_to_step: dict[str, int],
) -> list[int]:
    linked_steps: list[int] = []
    top_level_result = record.get("toolUseResult")
    for block in _content_blocks(record):
        if block.get("type") != "tool_result":
            continue
        tool_use_id = _as_optional_str(block.get("tool_use_id"))
        if not tool_use_id or tool_use_id not in tool_use_to_step:
            continue
        step = tool_use_to_step[tool_use_id]
        activity = activities[step]
        activity.outputs = _summarize_activity_result(activity, block, top_level_result)
        activity.refresh_hash()
        linked_steps.append(step)
    return linked_steps


def _resolve_parent(
    record: dict[str, Any],
    uuid_to_step: dict[str, int],
    tool_use_to_step: dict[str, int],
) -> int | None:
    parent_uuid = _as_optional_str(record.get("parentUuid"))
    if parent_uuid and parent_uuid in uuid_to_step:
        return uuid_to_step[parent_uuid]

    for block in _content_blocks(record):
        tool_use_id = _as_optional_str(block.get("tool_use_id"))
        if tool_use_id and tool_use_id in tool_use_to_step:
            return tool_use_to_step[tool_use_id]
    return None


def _classify_tool(tool_name: str | None, inputs: dict[str, Any]) -> str:
    if tool_name in FILE_EDIT_TOOLS:
        return "file_edit"
    if tool_name in STRUCTURED_TOOL_CALLS:
        return "tool_call"
    if tool_name in SHELL_TOOLS:
        return "test_run" if _looks_like_test_or_build(inputs.get("command")) else "opaque_shell"
    return "opaque_shell"


def _normalize_tool_inputs(tool_name: str | None, raw_inputs: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "Write":
        return {
            "file_path": raw_inputs.get("file_path"),
            "old_string": "",
            "new_string": raw_inputs.get("content", ""),
            "write_mode": True,
        }
    if tool_name == "MultiEdit":
        edits = raw_inputs.get("edits")
        if isinstance(edits, list):
            old_string = "\n".join(str(edit.get("old_string", "")) for edit in edits if isinstance(edit, dict))
            new_string = "\n".join(str(edit.get("new_string", "")) for edit in edits if isinstance(edit, dict))
        else:
            old_string = raw_inputs.get("old_string", "")
            new_string = raw_inputs.get("new_string", "")
        normalized = dict(raw_inputs)
        normalized.setdefault("old_string", old_string)
        normalized.setdefault("new_string", new_string)
        return normalized
    return dict(raw_inputs)


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


def _target_for(tool_name: str | None, inputs: dict[str, Any]) -> str | None:
    if tool_name in {"Edit", "MultiEdit", "Write", "Read"}:
        return _as_optional_str(inputs.get("file_path"))
    if tool_name == "Grep":
        return _as_optional_str(inputs.get("path")) or _as_optional_str(inputs.get("pattern"))
    if tool_name == "Glob":
        return _as_optional_str(inputs.get("path")) or _as_optional_str(inputs.get("pattern"))
    if tool_name in SHELL_TOOLS:
        return "shell"
    return _as_optional_str(inputs.get("file_path"))


def _content_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = record.get("message")
    content: Any
    if isinstance(message, dict) and "content" in message:
        content = message["content"]
    else:
        content = record.get("content")

    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _block_text(block: dict[str, Any]) -> str:
    text = block.get("text")
    if isinstance(text, str):
        return text
    return ""


def _text_summary(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(_block_text(block) for block in blocks)
    return {
        "text_blocks": len(blocks),
        "text_length": len(text),
        "text_hash": sha256_text(text),
    }


def _summarize_tool_result_block(block: dict[str, Any]) -> dict[str, Any]:
    content = block.get("content")
    return {
        "is_error": bool(block.get("is_error", False)),
        "content_kind": type(content).__name__,
        "content_length": len(canonical_json(content)),
        "content_hash": sha256_text(canonical_json(content)),
    }


def _summarize_activity_result(
    activity: Activity,
    block: dict[str, Any],
    top_level_result: Any,
) -> dict[str, Any]:
    tool_result = _summarize_tool_result_block(block)
    tool_use_result = _summarize_payload(top_level_result) if top_level_result is not None else None
    outputs: dict[str, Any] = {
        "tool_result": tool_result,
        "tool_use_result": tool_use_result,
    }

    if activity.kind == "test_run":
        result_text = _extract_text_payload(block.get("content"))
        top_level_text = _extract_text_payload(top_level_result)
        combined_text = "\n".join(text for text in (result_text, top_level_text) if text)
        if combined_text:
            outputs["result_text"] = combined_text
        exit_code = _extract_exit_code(block) if top_level_result is None else _extract_exit_code(top_level_result)
        if exit_code is None:
            exit_code = _extract_exit_code(block)
        if exit_code is not None:
            outputs["exit_code"] = exit_code

    return outputs


def _summarize_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        keys = sorted(str(key) for key in payload.keys())
    else:
        keys = []
    rendered = canonical_json(payload)
    return {
        "payload_kind": type(payload).__name__,
        "keys": keys,
        "payload_length": len(rendered),
        "payload_hash": sha256_text(rendered),
    }


def _extract_text_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        text_parts = []
        for item in payload:
            extracted = _extract_text_payload(item)
            if extracted:
                text_parts.append(extracted)
        return "\n".join(text_parts)
    if isinstance(payload, dict):
        text_parts: list[str] = []
        for key in ("stdout", "stderr", "output", "text", "content"):
            value = payload.get(key)
            extracted = _extract_text_payload(value)
            if extracted:
                text_parts.append(extracted)
        return "\n".join(text_parts)
    return ""


def _extract_exit_code(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("exit_code", "exitCode", "code", "status"):
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
    for value in payload.values():
        nested = _extract_exit_code(value)
        if nested is not None:
            return nested
    return None


def _json_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
