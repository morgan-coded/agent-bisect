from agent_bisect.gates import g1_schema, g3_tests_pass, run_g2, run_g3
from agent_bisect.ingest_claude import ingest_transcript
from agent_bisect.model import Activity, Journal


def test_g1_pass_fail_na_on_structured_and_unstructured(fixture_path):
    activities = ingest_transcript(fixture_path)
    by_kind = {activity.kind: activity for activity in activities}

    assert g1_schema(by_kind["file_edit"]).status == "PASS"
    assert g1_schema(by_kind["tool_call"]).status == "PASS"
    assert g1_schema(by_kind["test_run"]).status == "PASS"
    assert g1_schema(by_kind["user_msg"]).status == "NA"
    assert g1_schema(by_kind["llm_call"]).status == "NA"
    assert g1_schema(by_kind["opaque_shell"]).status == "NA"

    malformed = Activity(
        run_id="malformed",
        step_index=0,
        ts="2026-06-27T00:00:00Z",
        kind="file_edit",
        tool_name="Edit",
        inputs={"file_path": "x.py", "new_string": "after"},
    )
    result = g1_schema(malformed)
    assert result.status == "FAIL"
    assert "old_string" in result.evidence


def test_g3_parses_pass_fail_and_unparseable_outputs():
    passing = _test_run("pass", "============================== 3 passed in 0.11s ==============================")
    cargo_passing = _test_run("cargo-pass", "test result: ok. 4 passed; 0 failed; 0 ignored")
    failing = _test_run("fail", "FAILED tests/test_demo.py::test_demo - AssertionError")
    erroring = _test_run("error", "tests/test_demo.py E\nERROR tests/test_demo.py::test_demo")
    nonzero = _test_run("nonzero", "test command exited", exit_code=1)
    unparseable = _test_run("unknown", "command completed")

    assert g3_tests_pass(passing).status == "PASS"
    assert g3_tests_pass(cargo_passing).status == "PASS"
    assert g3_tests_pass(failing).status == "FAIL"
    assert g3_tests_pass(erroring).status == "FAIL"
    assert g3_tests_pass(nonzero).status == "FAIL"

    result = g3_tests_pass(unparseable)
    assert result.status == "NA"
    assert result.evidence == "unparseable test output"


def test_g2_write_then_edit_chain_passes():
    results = run_g2(
        [
            _write("demo.py", "alpha before omega"),
            _edit("demo.py", "before", "after", step_index=1),
        ]
    )

    assert [result.status for result in results] == ["PASS", "PASS"]
    assert results[1].evidence == "old_string matched uniquely"


def test_g2_fails_when_old_string_not_found():
    results = run_g2(
        [
            _write("demo.py", "alpha before omega"),
            _edit("demo.py", "missing", "after", step_index=1),
        ]
    )

    assert results[1].status == "FAIL"
    assert "not found" in results[1].evidence


def test_g2_fails_when_old_string_is_ambiguous():
    results = run_g2(
        [
            _write("demo.py", "repeat repeat"),
            _edit("demo.py", "repeat", "once", step_index=1),
        ]
    )

    assert results[1].status == "FAIL"
    assert "ambiguous" in results[1].evidence


def test_g2_returns_na_without_full_content_anchor():
    result = run_g2([_edit("demo.py", "before", "after")])[0]

    assert result.status == "NA"
    assert result.evidence == "no full-content anchor"


def test_g2_multiedit_checks_fragments_in_sequence():
    results = run_g2(
        [
            _write("demo.py", "one two three"),
            _multiedit(
                "demo.py",
                [
                    {"old_string": "one", "new_string": "1"},
                    {"old_string": "1 two", "new_string": "1 2"},
                ],
                step_index=1,
            ),
            _edit("demo.py", "1 2 three", "done", step_index=2),
        ]
    )

    assert [result.status for result in results] == ["PASS", "PASS", "PASS"]
    assert results[1].evidence == "2 edits matched uniquely"


def test_ingest_and_gate_results_are_deterministic(slice2_fixture_path):
    first_activities = ingest_transcript(slice2_fixture_path)
    second_activities = ingest_transcript(slice2_fixture_path)

    assert Journal.from_activities(first_activities).to_jsonl() == Journal.from_activities(second_activities).to_jsonl()
    assert _gate_signature(first_activities) == _gate_signature(second_activities)


def test_cli_gate_columns_remain_structural(slice2_fixture_path, capsys):
    from agent_bisect.cli import main

    assert main(["show", str(slice2_fixture_path), "--gates"]) == 0
    output = capsys.readouterr().out

    assert "G1\tG1 evidence\tG2\tG2 evidence\tG3\tG3 evidence" in output
    assert "full-content anchor established" in output
    assert "test pass signal" in output
    assert "2 passed" not in output


def _test_run(run_id: str, result_text: str, exit_code: int | None = None) -> Activity:
    outputs = {"result_text": result_text}
    if exit_code is not None:
        outputs["exit_code"] = exit_code
    return Activity(
        run_id=run_id,
        step_index=0,
        ts="2026-06-27T00:00:00Z",
        kind="test_run",
        tool_name="PowerShell",
        inputs={"command": "python -m pytest"},
        outputs=outputs,
        target="shell",
    )


def _write(file_path: str, content: str, step_index: int = 0) -> Activity:
    return Activity(
        run_id="g2",
        step_index=step_index,
        ts="2026-06-27T00:00:00Z",
        kind="file_edit",
        tool_name="Write",
        inputs={"file_path": file_path, "old_string": "", "new_string": content, "write_mode": True},
        target=file_path,
    )


def _edit(file_path: str, old_string: str, new_string: str, step_index: int = 0) -> Activity:
    return Activity(
        run_id="g2",
        step_index=step_index,
        ts="2026-06-27T00:00:00Z",
        kind="file_edit",
        tool_name="Edit",
        inputs={"file_path": file_path, "old_string": old_string, "new_string": new_string},
        target=file_path,
    )


def _multiedit(file_path: str, edits: list[dict[str, str]], step_index: int = 0) -> Activity:
    return Activity(
        run_id="g2",
        step_index=step_index,
        ts="2026-06-27T00:00:00Z",
        kind="file_edit",
        tool_name="MultiEdit",
        inputs={"file_path": file_path, "edits": edits},
        target=file_path,
    )


def _gate_signature(activities: list[Activity]) -> list[tuple[str, str, str, int | None]]:
    results = [g1_schema(activity) for activity in activities]
    results.extend(run_g2(activities))
    results.extend(run_g3(activities))
    return [(result.gate, result.status, result.evidence, result.step_index) for result in results]
