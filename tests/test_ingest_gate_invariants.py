import random

import pytest

from agent_bisect.gates import g1_schema, g3_tests_pass, run_g1, run_g2, run_g3
from agent_bisect.ingest_claude import ingest_transcript
from agent_bisect.model import Activity, Journal


def test_ingesting_same_fixture_twice_yields_byte_identical_activities(fixture_path):
    first = ingest_transcript(fixture_path)
    second = ingest_transcript(fixture_path)

    assert _activity_dicts(first) == _activity_dicts(second)
    assert [(activity.step_index, activity.kind, activity.content_hash) for activity in first] == [
        (activity.step_index, activity.kind, activity.content_hash) for activity in second
    ]
    assert Journal.from_activities(first).to_jsonl() == Journal.from_activities(second).to_jsonl()


def test_tool_result_linkage_and_parent_resolution_are_stable(fixture_path):
    first = ingest_transcript(fixture_path)
    second = ingest_transcript(fixture_path)

    assert [activity.parent_step for activity in first] == [None, 0, 1, 2, 3, 4]
    assert [activity.parent_step for activity in second] == [None, 0, 1, 2, 3, 4]

    read_call = first[2]
    edit_call = first[3]
    test_call = first[4]
    assert read_call.inputs["tool_use_id"] == "toolu_read"
    assert edit_call.inputs["tool_use_id"] == "toolu_edit"
    assert test_call.inputs["tool_use_id"] == "toolu_test"
    assert read_call.outputs["tool_result"]["content_hash"] == second[2].outputs["tool_result"]["content_hash"]
    assert edit_call.outputs["tool_use_result"]["keys"] == ["filePath", "type"]
    assert test_call.outputs["exit_code"] == 0
    assert test_call.outputs["result_text"] == second[4].outputs["result_text"]


def test_gate_results_are_deterministic_for_same_activities(slice2_fixture_path):
    first = ingest_transcript(slice2_fixture_path)
    second = ingest_transcript(slice2_fixture_path)

    assert _gate_signature(first) == _gate_signature(first)
    assert _gate_signature(first) == _gate_signature(second)


@pytest.mark.parametrize(
    ("outputs", "expected_status", "expected_evidence"),
    [
        ({"result_text": "command completed", "exit_code": 0}, "NA", "unparseable test output"),
        ({"stdout": "3 passed in 0.02s", "exitCode": 1}, "FAIL", "non-zero exit code"),
        ({"stdout": "3 passed in 0.02s"}, "PASS", "test pass signal"),
        ({"stderr": "FAILED repo/test_example.py::test_case - AssertionError"}, "FAIL", "test failure signal"),
        ({"tool_result": {"stdout": "ok repo/test_example.py", "exit_code": "0"}}, "PASS", "test pass signal"),
    ],
)
def test_g3_fail_closed_table(outputs, expected_status, expected_evidence):
    result = g3_tests_pass(_test_run(outputs=outputs))

    assert result.status == expected_status
    assert result.evidence == expected_evidence


@pytest.mark.parametrize(
    ("activity", "expected_status", "expected_evidence"),
    [
        (
            Activity(
                run_id="g1",
                step_index=0,
                ts="2026-06-28T00:00:00Z",
                kind="file_edit",
                tool_name="Edit",
                inputs={"file_path": "repo/example.py", "new_string": "after"},
            ),
            "FAIL",
            "old_string missing/not str",
        ),
        (
            Activity(
                run_id="g1",
                step_index=1,
                ts="2026-06-28T00:00:01Z",
                kind="tool_call",
                inputs={"file_path": "repo/example.py"},
            ),
            "FAIL",
            "missing tool_name",
        ),
        (
            Activity(
                run_id="g1",
                step_index=2,
                ts="2026-06-28T00:00:02Z",
                kind="test_run",
                tool_name="PowerShell",
                inputs={},
            ),
            "FAIL",
            "command missing/not str",
        ),
        (
            Activity(
                run_id="g1",
                step_index=3,
                ts="2026-06-28T00:00:03Z",
                kind="opaque_shell",
                tool_name="PowerShell",
                inputs={"command": "Get-ChildItem"},
            ),
            "NA",
            "not a structured activity",
        ),
        (
            Activity(
                run_id="g1",
                step_index=4,
                ts="2026-06-28T00:00:04Z",
                kind="user_msg",
                inputs={"text": "request"},
            ),
            "NA",
            "not a structured activity",
        ),
        (
            Activity(
                run_id="g1",
                step_index=5,
                ts="2026-06-28T00:00:05Z",
                kind="llm_call",
                outputs={"text": "response"},
            ),
            "NA",
            "not a structured activity",
        ),
    ],
)
def test_g1_schema_table_for_malformed_and_unstructured_activities(
    activity,
    expected_status,
    expected_evidence,
):
    result = g1_schema(activity)

    assert result.status == expected_status
    assert expected_evidence in result.evidence


def test_seeded_synthetic_sequences_round_trip_and_gate_determinism(tmp_path):
    rng = random.Random(20260628)

    for case_index in range(12):
        tokens = [f"case{case_index}_token{token_index}" for token_index in range(6)]
        remaining_tokens = list(tokens)
        content = " ".join(tokens)
        activities = [_write(case_index, 0, content)]

        for step_index in range(1, 5):
            if remaining_tokens and rng.choice([True, False]):
                token_index = rng.randrange(len(remaining_tokens))
                old_string = remaining_tokens.pop(token_index)
                new_string = f"{old_string}_edited"
                content = content.replace(old_string, new_string, 1)
                activities.append(_edit(case_index, step_index, old_string, new_string, parent_step=step_index - 1))
            else:
                activities.append(_test_run(case_index, step_index, parent_step=step_index - 1))

        journal_path = tmp_path / f"synthetic-{case_index}.jsonl"
        journal = Journal.from_activities(activities)
        journal.write_jsonl(journal_path)
        loaded = Journal.read_jsonl(journal_path).activities

        assert _activity_dicts(loaded) == _activity_dicts(journal.activities)
        assert _gate_signature(loaded) == _gate_signature(loaded)


def _activity_dicts(activities: list[Activity]) -> list[dict]:
    return [activity.to_dict() for activity in activities]


def _gate_signature(activities: list[Activity]) -> list[tuple[str, str, str, int | None]]:
    results = [*run_g1(activities), *run_g2(activities), *run_g3(activities)]
    return [(result.gate, result.status, result.evidence, result.step_index) for result in results]


def _test_run(
    run_id: str = "g3",
    step_index: int = 0,
    parent_step: int | None = None,
    outputs: dict | None = None,
) -> Activity:
    return Activity(
        run_id=str(run_id),
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="test_run",
        tool_name="PowerShell",
        inputs={"command": "python -m pytest"},
        outputs=outputs or {"result_text": "1 passed in 0.01s", "exit_code": 0},
        target="shell",
        parent_step=parent_step,
    )


def _write(run_id: int, step_index: int, content: str) -> Activity:
    return Activity(
        run_id=f"synthetic-{run_id}",
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="file_edit",
        tool_name="Write",
        inputs={
            "file_path": f"repo/synthetic_{run_id}.py",
            "old_string": "",
            "new_string": content,
            "write_mode": True,
        },
        target=f"repo/synthetic_{run_id}.py",
    )


def _edit(
    run_id: int,
    step_index: int,
    old_string: str,
    new_string: str,
    parent_step: int,
) -> Activity:
    return Activity(
        run_id=f"synthetic-{run_id}",
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="file_edit",
        tool_name="Edit",
        inputs={
            "file_path": f"repo/synthetic_{run_id}.py",
            "old_string": old_string,
            "new_string": new_string,
        },
        target=f"repo/synthetic_{run_id}.py",
        parent_step=parent_step,
    )
