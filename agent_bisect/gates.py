from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from .model import Activity


Status = str


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    status: Status
    evidence: str
    step_index: int | None = None


def g1_schema(activity: Activity) -> GateResult:
    if activity.kind in {"opaque_shell", "unmapped", "user_msg", "llm_call"}:
        return GateResult("G1", "NA", "not a structured activity", activity.step_index)

    if activity.kind == "file_edit":
        return _validate_file_edit(activity)
    if activity.kind == "tool_call":
        return _validate_tool_call(activity)
    if activity.kind == "test_run":
        return _validate_test_run(activity)
    if activity.kind == "verdict":
        return GateResult("G1", "PASS", "verdict activity schema ok", activity.step_index)

    return GateResult("G1", "FAIL", f"unknown activity kind: {activity.kind}", activity.step_index)


def run_g1(activities: Iterable[Activity]) -> list[GateResult]:
    return [g1_schema(activity) for activity in activities]


def run_g2(activities: Iterable[Activity]) -> list[GateResult]:
    """Run stateful fold-forward diff checks without reading live files."""

    from .foldforward import run_fold_forward

    return run_fold_forward(list(activities))


def g2_diff_applies(activity: Activity, prior_activities: Iterable[Activity] = ()) -> GateResult:
    """Check one file_edit against prior journal-provided full-content anchors.

    This is a convenience wrapper. CLI and batch callers should prefer run_g2 so
    the fold-forward state advances exactly once across the complete run.
    """

    return run_g2([*prior_activities, activity])[-1]


def g3_tests_pass(activity: Activity) -> GateResult:
    """Parse recorded test/build results deterministically and fail closed.

    agent-bisect checks the deterministic envelope: recorded command output and exit
    status captured at ingest time. It does not re-run tests or replay the model.
    Ambiguous output returns NA rather than a false PASS.
    """

    if activity.kind != "test_run":
        return GateResult("G3", "NA", "not a test_run", activity.step_index)

    exit_code = _extract_exit_code(activity.outputs)
    if exit_code is not None and exit_code != 0:
        return GateResult("G3", "FAIL", "non-zero exit code", activity.step_index)

    output = _normalized_test_output(_extract_result_text(activity.outputs))
    if output == "":
        return GateResult("G3", "NA", "unparseable test output", activity.step_index)

    if _has_fail_signal(output):
        return GateResult("G3", "FAIL", "test failure signal", activity.step_index)
    if _has_pass_signal(output):
        return GateResult("G3", "PASS", "test pass signal", activity.step_index)

    return GateResult("G3", "NA", "unparseable test output", activity.step_index)


def g4_consistency_holds(*_args: Any, **_kwargs: Any) -> GateResult:
    raise NotImplementedError("G4 internal-consistency is not yet implemented.")


def run_g3(activities: Iterable[Activity]) -> list[GateResult]:
    return [g3_tests_pass(activity) for activity in activities]


def _validate_file_edit(activity: Activity) -> GateResult:
    inputs = activity.inputs
    missing_or_bad = _required_strs(inputs, ["file_path", "old_string", "new_string"])
    if missing_or_bad:
        return GateResult("G1", "FAIL", f"file_edit malformed: {missing_or_bad}", activity.step_index)
    return GateResult("G1", "PASS", "file_edit schema ok", activity.step_index)


def _validate_tool_call(activity: Activity) -> GateResult:
    if not isinstance(activity.tool_name, str) or not activity.tool_name:
        return GateResult("G1", "FAIL", "tool_call malformed: missing tool_name", activity.step_index)
    inputs = activity.inputs
    if activity.tool_name == "Read":
        missing_or_bad = _required_strs(inputs, ["file_path"])
    elif activity.tool_name in {"Grep", "Glob"}:
        missing_or_bad = _required_strs(inputs, ["pattern"])
    else:
        missing_or_bad = ""
    if missing_or_bad:
        return GateResult("G1", "FAIL", f"tool_call malformed: {missing_or_bad}", activity.step_index)
    return GateResult("G1", "PASS", "tool_call schema ok", activity.step_index)


def _validate_test_run(activity: Activity) -> GateResult:
    if activity.tool_name not in {"Bash", "PowerShell"}:
        return GateResult("G1", "FAIL", "test_run malformed: tool_name must be Bash or PowerShell", activity.step_index)
    missing_or_bad = _required_strs(activity.inputs, ["command"])
    if missing_or_bad:
        return GateResult("G1", "FAIL", f"test_run malformed: {missing_or_bad}", activity.step_index)
    return GateResult("G1", "PASS", "test_run schema ok", activity.step_index)


def _required_strs(inputs: dict[str, Any], keys: list[str]) -> str:
    failures = []
    for key in keys:
        value = inputs.get(key)
        if not isinstance(value, str):
            failures.append(f"{key} missing/not str")
    return "; ".join(failures)


def _extract_exit_code(outputs: dict[str, Any]) -> int | None:
    for key in ("exit_code", "exitCode", "code"):
        value = outputs.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
    for nested_key in ("tool_result", "tool_use_result"):
        nested = outputs.get(nested_key)
        if isinstance(nested, dict):
            exit_code = _extract_exit_code(nested)
            if exit_code is not None:
                return exit_code
    return None


def _extract_result_text(outputs: dict[str, Any]) -> str:
    texts: list[str] = []
    for key in ("result_text", "stdout", "stderr", "text", "output"):
        value = outputs.get(key)
        if isinstance(value, str) and value:
            texts.append(value)
    for nested_key in ("tool_result", "tool_use_result"):
        nested = outputs.get(nested_key)
        if isinstance(nested, dict):
            nested_text = _extract_result_text(nested)
            if nested_text:
                texts.append(nested_text)
    return "\n".join(texts)


def _normalized_test_output(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\b(?:in|after)\s+\d+(?:\.\d+)?s\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bchunk(?:-|_)?id[:=]\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\btokens?[:=]\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip().lower()


def _has_fail_signal(output: str) -> bool:
    fail_patterns = [
        r"\b[1-9]\d*\s+failed\b",
        r"\b[1-9]\d*\s+failures?\b",
        r"\b[1-9]\d*\s+errors?\b",
        r"^=+\s*failures?\s*=+$",
        r"^failed\b",
        r"^fail\s+\S+",
        r"^errors?\b",
        r"\berror:",
        r"\bnpm err!\b",
        r"\btraceback\b",
        r"\bexception\b",
        r"\bassertionerror\b",
        r"\bpanic\b",
        r"\bcompilation failed\b",
        r"test result:\s*failed",
        r"---\s*fail:",
    ]
    return any(re.search(pattern, output, flags=re.MULTILINE) for pattern in fail_patterns)


def _has_pass_signal(output: str) -> bool:
    pass_patterns = [
        r"\b\d+\s+passed\b",
        r"\btests?:\s*\d+\s+passed\b",
        r"\btest files?\s+\d+\s+passed\b",
        r"\btest result:\s*ok\b",
        r"^ok\s+\S+",
        r"^ok$",
        r"\bpass\b",
        r"\bcompiled successfully\b",
        r"\bbuild succeeded\b",
    ]
    return any(re.search(pattern, output, flags=re.MULTILINE) for pattern in pass_patterns)
