import json

from agent_bisect.accuracy import ActivityRun, LabeledCase, evaluate_controlled_accuracy, score_labeled_cases
from agent_bisect.model import Activity, canonical_json


def test_labeled_scoring_exact_high_cascade_low_na_and_exclusions():
    report = score_labeled_cases(
        [
            LabeledCase("exact-high", 1, "G2", tuple([_write(0), _bad_edit(1, parent_step=0)])),
            LabeledCase(
                "cascade-low",
                3,
                None,
                tuple([_write(0), _bad_edit(1, parent_step=0), _opaque(2, parent_step=1), _test_fail(3, parent_step=2)]),
            ),
            LabeledCase("na", 0, "G1", tuple([_llm(0)])),
            LabeledCase("missing", None, "G1", tuple([_llm(0)])),
            LabeledCase("out-of-range", 7, "G1", tuple([_llm(0)])),
        ]
    )

    summary = report["summary"]
    assert summary["ALL"]["total"] == 3
    assert summary["ALL"]["exact_step_correct"] == 1
    assert summary["ALL"]["cascade_membership_correct"] == 2
    assert summary["HIGH"]["total"] == 1
    assert summary["HIGH"]["exact_step_accuracy"] == 1.0
    assert summary["LOW"]["total"] == 1
    assert summary["LOW"]["exact_step_accuracy"] == 0.0
    assert summary["LOW"]["cascade_membership_accuracy"] == 1.0
    assert summary["NA"]["total"] == 1
    assert report["excluded"]["reasons"] == {
        "expected_breaking_step_out_of_range": 1,
        "missing_expected_breaking_step": 1,
    }


def test_controlled_accuracy_matrix_and_localization_deterministic():
    runs = [ActivityRun("run-a", "synthetic", tuple(_clean_campaign_activities(extra=True)))]

    first = evaluate_controlled_accuracy(runs, seed=42, per_class=2, max_per_run=5)
    second = evaluate_controlled_accuracy(runs, seed=42, per_class=2, max_per_run=5)

    assert canonical_json(first) == canonical_json(second)
    for fault_class in ("G1_TARGET", "G2_TARGET", "G3_TARGET"):
        row = first["confusion_matrix"][fault_class]
        assert row["injected_n"] >= 1
        assert row["tp"] == row["injected_n"]
        assert row["fp"] == 0
        assert row["fn"] == 0
        assert row["recall"] == 1.0

    assert first["confusion_matrix"]["CONTROL"]["fp"] == 0
    assert first["confusion_matrix"]["BENIGN"]["fp"] == 0
    assert first["controlled_accuracy"]["target_injections"] >= 3
    assert first["controlled_accuracy"]["exact_step_correct"] == first["controlled_accuracy"]["target_injections"]
    assert "controlled_vs_real" in first["scoring_rule"]


def test_accuracy_report_is_structural_only():
    runs = [ActivityRun("run-a", "synthetic", tuple(_clean_campaign_activities()))]
    report = evaluate_controlled_accuracy(runs, seed=100, per_class=1, max_per_run=5)
    rendered = json.dumps(report, sort_keys=True)

    for sentinel in ("RAW_COMMAND_SENTINEL", "RAW_OLD_SENTINEL", "RAW_NEW_SENTINEL", "RAW_FILE_SENTINEL"):
        assert sentinel not in rendered


def _clean_campaign_activities(extra: bool = False) -> list[Activity]:
    activities = [
        _llm(0),
        _write(1, "RAW_FILE_SENTINEL_A.py", "RAW_OLD_SENTINEL\nbody\n", parent_step=0),
        _edit(2, "RAW_FILE_SENTINEL_A.py", "RAW_OLD_SENTINEL", "RAW_NEW_SENTINEL", parent_step=1),
        _test_pass(3, parent_step=2),
    ]
    if extra:
        activities.extend(
            [
                _llm(4),
                _write(5, "RAW_FILE_SENTINEL_B.py", "RAW_B_OLD_SENTINEL\nbody\n", parent_step=4),
                _edit(6, "RAW_FILE_SENTINEL_B.py", "RAW_B_OLD_SENTINEL", "RAW_B_NEW_SENTINEL", parent_step=5),
                _test_pass(7, parent_step=6),
            ]
        )
    for activity in activities:
        activity.refresh_hash()
    return activities


def _llm(step_index: int) -> Activity:
    return Activity(run_id="synthetic", step_index=step_index, ts="", kind="llm_call", outputs={"text_hash": "abc"})


def _write(step_index: int, file_path: str = "demo.py", content: str = "alpha", parent_step: int | None = None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="file_edit",
        tool_name="Write",
        inputs={"file_path": file_path, "old_string": "", "new_string": content, "write_mode": True},
        target=file_path,
        parent_step=parent_step,
    )


def _edit(step_index: int, file_path: str, old_string: str, new_string: str, parent_step: int | None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="file_edit",
        tool_name="Edit",
        inputs={"file_path": file_path, "old_string": old_string, "new_string": new_string},
        target=file_path,
        parent_step=parent_step,
    )


def _bad_edit(step_index: int, parent_step: int | None) -> Activity:
    return _edit(step_index, "demo.py", "missing", "beta", parent_step)


def _opaque(step_index: int, parent_step: int | None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="opaque_shell",
        tool_name="PowerShell",
        inputs={"command": "sanitized opaque command"},
        target="shell",
        parent_step=parent_step,
    )


def _test_pass(step_index: int, parent_step: int | None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="test_run",
        tool_name="PowerShell",
        inputs={"command": "RAW_COMMAND_SENTINEL pytest"},
        outputs={"result_text": "2 passed in 0.04s", "exit_code": 0},
        target="shell",
        parent_step=parent_step,
    )


def _test_fail(step_index: int, parent_step: int | None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="test_run",
        tool_name="PowerShell",
        inputs={"command": "python -m pytest"},
        outputs={"result_text": "FAILED sanitized_test.py::test_demo", "exit_code": 1},
        target="shell",
        parent_step=parent_step,
    )
