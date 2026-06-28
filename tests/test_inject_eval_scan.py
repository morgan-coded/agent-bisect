import json

from agent_bisect.cli import main
from agent_bisect.eval import evaluate_paths, write_eval_reports
from agent_bisect.gates import run_g1, run_g2, run_g3
from agent_bisect.inject import eligible_steps_by_class, inject_faults
from agent_bisect.model import Activity, Journal, canonical_json
from agent_bisect.scan import scan_paths


def test_inject_classes_produce_expected_ground_truth_records():
    activities = _eval_activities()
    cases = inject_faults(activities, seed=100, per_class=1)
    by_class = {case.truth.fault_class: case for case in cases}

    assert set(by_class) == {"G1_TARGET", "G2_TARGET", "G3_TARGET", "CONTROL", "BENIGN"}
    assert by_class["G1_TARGET"].truth.expected_gate == "G1"
    assert by_class["G2_TARGET"].truth.expected_gate == "G2"
    assert by_class["G3_TARGET"].truth.expected_gate == "G3"
    assert by_class["CONTROL"].truth.expected_gate == "NONE"
    assert by_class["BENIGN"].truth.expected_gate == "NONE"
    assert by_class["G2_TARGET"].truth.mutation_field == "inputs.old_string"
    assert by_class["G3_TARGET"].truth.mutation_field == "outputs.exit_code"


def test_inject_eligibility_excludes_g2_anchor_na_and_g3_non_pass():
    activities = _eval_activities()
    eligible = eligible_steps_by_class(activities)

    assert 2 in eligible["G2_TARGET"]
    assert 3 not in eligible["G2_TARGET"]
    assert 4 in eligible["G3_TARGET"]
    assert 5 not in eligible["G3_TARGET"]
    assert 6 not in eligible["G3_TARGET"]


def test_injected_eval_matrix_counts_and_localization(tmp_path):
    journal_path = tmp_path / "clean.journal.jsonl"
    Journal.from_activities(_eval_activities()).write_jsonl(journal_path)

    report = evaluate_paths([journal_path], seed=100, per_class=1)

    for fault_class in ("G1_TARGET", "G2_TARGET", "G3_TARGET"):
        metrics = report["classes"][fault_class]
        assert metrics["eligible_n"] >= 1
        assert metrics["injected_n"] == 1
        assert metrics["scored_n"] == 1
        assert metrics["tp"] == 1
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
        assert metrics["na"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    assert report["classes"]["CONTROL"]["fp"] == 0
    assert report["classes"]["BENIGN"]["fp"] == 0
    assert report["source"]["g2_eligible_edits"] == 1
    assert report["source"]["g2_total_edits"] == 2
    assert report["localization"]["HIGH"]["correct"] >= 3
    assert report["localization"]["HIGH"]["total"] >= 3


def test_eval_determinism_same_seed_and_different_seed_changes_selection(tmp_path):
    journal_path = tmp_path / "clean.journal.jsonl"
    Journal.from_activities(_eval_activities(extra=True)).write_jsonl(journal_path)

    first = evaluate_paths([journal_path], seed=100, per_class=1)
    second = evaluate_paths([journal_path], seed=100, per_class=1)
    third = evaluate_paths([journal_path], seed=101, per_class=1)

    assert canonical_json(first) == canonical_json(second)
    assert [item["injected_step"] for item in first["mutations"]] != [
        item["injected_step"] for item in third["mutations"]
    ]


def test_eval_reports_do_not_emit_raw_inputs(tmp_path):
    journal_path = tmp_path / "clean.journal.jsonl"
    activities = _eval_activities()
    Journal.from_activities(activities).write_jsonl(journal_path)
    report = evaluate_paths([journal_path], seed=100, per_class=1)

    reports_dir = tmp_path / "reports"
    write_eval_reports(report, reports_dir)
    rendered = (reports_dir / "eval-report.json").read_text(encoding="utf-8")
    rendered += (reports_dir / "eval-report.md").read_text(encoding="utf-8")

    for raw in _raw_input_literals(activities):
        assert raw not in rendered


def test_scan_surfaces_real_failures_structurally(tmp_path, capsys):
    clean_path = tmp_path / "clean.jsonl"
    fail_path = tmp_path / "fail.jsonl"
    Journal.from_activities(_eval_activities()).write_jsonl(clean_path)
    Journal.from_activities(_scan_failure_activities()).write_jsonl(fail_path)

    report = scan_paths([tmp_path])
    assert report["label"] == "generalization check (same Claude schema)"
    assert sum(run["failure_count"] for run in report["runs"]) >= 1
    rendered = json.dumps(report, sort_keys=True)
    for raw in _raw_input_literals(_scan_failure_activities()):
        assert raw not in rendered

    assert main(["scan", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "generalization check (same Claude schema)" in output
    assert "cross-platform coverage checks" in output
    for raw in _raw_input_literals(_scan_failure_activities()):
        assert raw not in output


def _eval_activities(extra: bool = False) -> list[Activity]:
    activities = [
        _llm(0, 0),
        _write(1, "eval_a.py", "SECRET_ALPHA_OLD\nSAFE_BODY\n", parent_step=0),
        _edit(2, "eval_a.py", "SECRET_ALPHA_OLD", "SECRET_ALPHA_NEW", parent_step=1),
        _edit(3, "preexisting.py", "SECRET_NO_ANCHOR_OLD", "SECRET_NO_ANCHOR_NEW", parent_step=2),
        _test(4, "SECRET_TEST_COMMAND_ALPHA", "============================== 2 passed in 0.04s ==============================", 0, parent_step=2),
        _test(5, "SECRET_TEST_COMMAND_BETA", "completed without parseable summary", 0, parent_step=4),
        _test(6, "SECRET_TEST_COMMAND_GAMMA", "completed without parseable summary", 0, parent_step=5),
    ]
    if extra:
        activities.extend(
            [
                _llm(7, 1),
                _write(8, "eval_b.py", "SECRET_BETA_OLD\nSAFE_BODY\n", parent_step=7),
                _edit(9, "eval_b.py", "SECRET_BETA_OLD", "SECRET_BETA_NEW", parent_step=8),
                _test(10, "SECRET_TEST_COMMAND_DELTA", "============================== 3 passed in 0.03s ==============================", 0, parent_step=9),
            ]
        )
    return _refresh(activities)


def _scan_failure_activities() -> list[Activity]:
    return _refresh(
        [
            _llm(0, 0, run_id="scan-fail"),
            _write(1, "scan_demo.py", "SECRET_SCAN_OLD\n", parent_step=0, run_id="scan-fail"),
            _edit(2, "scan_demo.py", "SECRET_SCAN_MISSING", "SECRET_SCAN_NEW", parent_step=1, run_id="scan-fail"),
            _test(3, "SECRET_SCAN_COMMAND", "FAILED sanitized.py::test_scan - AssertionError", 1, parent_step=2, run_id="scan-fail"),
        ],
        clean=False,
    )


def _llm(step_index: int, marker: int, run_id: str = "eval-run") -> Activity:
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="llm_call",
        outputs={"text_length": 12 + marker, "text_hash": f"hash-{marker}"},
    )


def _write(step_index: int, file_path: str, content: str, parent_step: int, run_id: str = "eval-run") -> Activity:
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="file_edit",
        tool_name="Write",
        inputs={"file_path": file_path, "old_string": "", "new_string": content, "write_mode": True},
        target=file_path,
        parent_step=parent_step,
    )


def _edit(
    step_index: int,
    file_path: str,
    old_string: str,
    new_string: str,
    parent_step: int,
    run_id: str = "eval-run",
) -> Activity:
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="file_edit",
        tool_name="Edit",
        inputs={"file_path": file_path, "old_string": old_string, "new_string": new_string},
        target=file_path,
        parent_step=parent_step,
    )


def _test(
    step_index: int,
    command: str,
    result_text: str,
    exit_code: int,
    parent_step: int,
    run_id: str = "eval-run",
) -> Activity:
    return Activity(
        run_id=run_id,
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="test_run",
        tool_name="PowerShell",
        inputs={"command": command},
        outputs={"result_text": result_text, "exit_code": exit_code},
        target="shell",
        parent_step=parent_step,
    )


def _refresh(activities: list[Activity], clean: bool = True) -> list[Activity]:
    for activity in activities:
        activity.refresh_hash()
    if clean:
        assert not any(result.status == "FAIL" for result in [*run_g1(activities), *run_g2(activities), *run_g3(activities)])
    return activities


def _raw_input_literals(activities: list[Activity]) -> list[str]:
    raw: list[str] = []
    for activity in activities:
        for key in ("old_string", "new_string", "command"):
            value = activity.inputs.get(key)
            if isinstance(value, str) and len(value) > 8:
                raw.append(value)
    return raw
