from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from .gates import run_g1, run_g2, run_g3
from .model import Activity, canonical_json, sha256_text


TARGET_CLASSES = ("G1_TARGET", "G2_TARGET", "G3_TARGET")
NON_TARGET_CLASSES = ("CONTROL", "BENIGN")
FAULT_CLASSES = (*TARGET_CLASSES, *NON_TARGET_CLASSES)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    run_id: str
    injected_step: int
    fault_class: str
    expected_gate: str
    expected_breaking_step: int | None
    mutation_field: str
    mutation_hash: str
    source_content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "injected_step": self.injected_step,
            "fault_class": self.fault_class,
            "expected_gate": self.expected_gate,
            "expected_breaking_step": self.expected_breaking_step,
            "mutation_field": self.mutation_field,
            "mutation_hash": self.mutation_hash,
            "source_content_hash": self.source_content_hash,
        }


@dataclass(frozen=True, slots=True)
class InjectionCase:
    truth: GroundTruth
    activities: tuple[Activity, ...]


def eligible_steps_by_class(activities: list[Activity]) -> dict[str, list[int]]:
    g1_results = {result.step_index: result for result in run_g1(activities)}
    g2_results = {result.step_index: result for result in run_g2(activities)}
    g3_results = {result.step_index: result for result in run_g3(activities)}

    eligible: dict[str, list[int]] = {fault_class: [] for fault_class in FAULT_CLASSES}
    for activity in activities:
        step = activity.step_index
        if _is_g1_eligible(activity, g1_results):
            eligible["G1_TARGET"].append(step)
        if _is_g2_eligible(activity, g2_results):
            eligible["G2_TARGET"].append(step)
        if _is_g3_eligible(activity, g3_results):
            eligible["G3_TARGET"].append(step)
            eligible["BENIGN"].append(step)
        if activity.kind == "llm_call":
            eligible["CONTROL"].append(step)

    return {fault_class: sorted(steps) for fault_class, steps in eligible.items()}


def inject_faults(
    activities: list[Activity],
    *,
    seed: int,
    per_class: int,
) -> list[InjectionCase]:
    """Create world-defined injected faults from real recorded run effects.

    Fault classes are defined by the external run effect they corrupt: malformed
    structured calls (G1), stale patches that cannot apply to the prior file
    state (G2), recorded OS/test failures (G3), non-effect control changes, and
    documented nondeterministic metadata changes. G4 is intentionally absent
    because consistency replay is deferred.
    """

    eligible = eligible_steps_by_class(activities)
    cases: list[InjectionCase] = []
    for fault_class in FAULT_CLASSES:
        selected_steps = _select_steps(eligible[fault_class], seed=seed, fault_class=fault_class, per_class=per_class)
        for step in selected_steps:
            cases.append(inject_fault(activities, step, fault_class))
    return cases


def inject_fault(activities: list[Activity], step: int, fault_class: str) -> InjectionCase:
    return _inject_one(activities, step, fault_class)


def _is_g1_eligible(activity: Activity, g1_results: dict[int | None, Any]) -> bool:
    if activity.kind not in {"file_edit", "test_run", "tool_call"}:
        return False
    result = g1_results.get(activity.step_index)
    return result is not None and result.status == "PASS"


def _is_g2_eligible(activity: Activity, g2_results: dict[int | None, Any]) -> bool:
    if activity.kind != "file_edit" or activity.tool_name != "Edit":
        return False
    if not isinstance(activity.inputs.get("old_string"), str) or activity.inputs.get("old_string") == "":
        return False
    result = g2_results.get(activity.step_index)
    return result is not None and result.status == "PASS"


def _is_g3_eligible(activity: Activity, g3_results: dict[int | None, Any]) -> bool:
    if activity.kind != "test_run":
        return False
    if not isinstance(activity.outputs.get("result_text"), str):
        return False
    result = g3_results.get(activity.step_index)
    return result is not None and result.status == "PASS"


def _select_steps(steps: list[int], *, seed: int, fault_class: str, per_class: int) -> list[int]:
    if per_class <= 0 or not steps:
        return []
    count = min(per_class, len(steps))
    rng_seed = int(sha256_text(f"{seed}:{fault_class}")[:16], 16)
    rng = random.Random(rng_seed)
    return sorted(rng.sample(steps, count))


def _inject_one(activities: list[Activity], step: int, fault_class: str) -> InjectionCase:
    mutated = [_clone_activity(activity) for activity in activities]
    activity = mutated[step]

    if fault_class == "G1_TARGET":
        expected_gate = "G1"
        mutation_field, mutation_value = _mutate_g1(activity)
        expected_breaking_step = step
    elif fault_class == "G2_TARGET":
        expected_gate = "G2"
        mutation_field = "inputs.old_string"
        mutation_value = f"__agent_bisect_stale_patch_{sha256_text(activity.content_hash)[:12]}__"
        activity.inputs["old_string"] = mutation_value
        expected_breaking_step = step
    elif fault_class == "G3_TARGET":
        expected_gate = "G3"
        mutation_field = "outputs.exit_code"
        mutation_value = 1
        activity.outputs["exit_code"] = mutation_value
        activity.outputs["result_text"] = "FAILED sanitized_injected_test.py::test_injected - AssertionError"
        expected_breaking_step = step
    elif fault_class == "CONTROL":
        expected_gate = "NONE"
        mutation_field = "outputs.control_marker_hash"
        mutation_value = sha256_text(f"control:{activity.content_hash}")
        activity.outputs["control_marker_hash"] = mutation_value
        expected_breaking_step = None
    elif fault_class == "BENIGN":
        expected_gate = "NONE"
        mutation_field = "outputs.result_text.duration"
        mutation_value = _benign_test_text(str(activity.outputs["result_text"]))
        activity.outputs["result_text"] = mutation_value
        expected_breaking_step = None
    else:
        raise ValueError(f"unknown fault class: {fault_class}")

    activity.refresh_hash()
    truth = GroundTruth(
        run_id=activity.run_id,
        injected_step=step,
        fault_class=fault_class,
        expected_gate=expected_gate,
        expected_breaking_step=expected_breaking_step,
        mutation_field=mutation_field,
        mutation_hash=sha256_text(canonical_json(mutation_value)),
        source_content_hash=activities[step].content_hash,
    )
    return InjectionCase(truth=truth, activities=tuple(mutated))


def _mutate_g1(activity: Activity) -> tuple[str, Any]:
    if activity.kind == "file_edit":
        activity.inputs["file_path"] = None
        return "inputs.file_path", None
    if activity.kind == "test_run":
        activity.inputs["command"] = None
        return "inputs.command", None
    if activity.tool_name == "Read":
        activity.inputs["file_path"] = None
        return "inputs.file_path", None
    activity.inputs["pattern"] = None
    return "inputs.pattern", None


def _benign_test_text(text: str) -> str:
    if " in " in text:
        prefix = text.rsplit(" in ", 1)[0]
        return f"{prefix} in 9.99s"
    return f"{text} in 9.99s"


def _clone_activity(activity: Activity) -> Activity:
    return Activity.from_dict(activity.to_dict())
