from agent_bisect.cli import main
from agent_bisect.ingest_claude import ingest_transcript
from agent_bisect.localize import localize_failures, shell_target_coverage
from agent_bisect.model import Activity, Journal


def test_planted_fault_localizes_breaking_edit_high_confidence(localize_planted_fault_path):
    report = localize_failures(ingest_transcript(localize_planted_fault_path))

    assert report.status == "break"
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.breaking_step == 2
    assert failure.breaking_gate == "G2"
    assert failure.failure_cascade == (3,)
    assert failure.confidence == "HIGH"
    assert failure.coverage == "structured path"
    assert failure.candidates == ()


def test_clean_run_reports_no_break(slice2_fixture_path):
    report = localize_failures(ingest_transcript(slice2_fixture_path))

    assert report.status == "no_break"
    assert report.failures == ()


def test_opaque_path_downgrades_confidence_and_surfaces_candidate():
    report = localize_failures(
        [
            _write(0, "demo.py", "alpha"),
            _edit(1, "demo.py", "missing", "beta", parent_step=0),
            _opaque(2, parent_step=1),
            _test_fail(3, parent_step=2),
        ]
    )

    assert report.status == "break"
    failure = report.failures[0]
    assert failure.breaking_step == 1
    assert failure.failure_cascade == (3,)
    assert failure.confidence == "LOW"
    assert failure.candidates == (2,)
    assert failure.coverage == "1 opaque_shell node on path"


def test_multiple_independent_failures_get_separate_cascades():
    report = localize_failures(
        [
            _write(0, "a.py", "alpha"),
            _edit(1, "a.py", "missing-a", "beta", parent_step=0),
            _test_fail(2, parent_step=1),
            _write(3, "b.py", "gamma"),
            _edit(4, "b.py", "missing-b", "delta", parent_step=3),
            _test_fail(5, parent_step=4),
        ]
    )

    assert report.status == "break"
    assert [failure.breaking_step for failure in report.failures] == [1, 4]
    assert [failure.failure_cascade for failure in report.failures] == [(2,), (5,)]
    assert [failure.confidence for failure in report.failures] == ["HIGH", "HIGH"]


def test_file_dependency_edge_links_unparented_structured_failure():
    report = localize_failures(
        [
            _write(0, "demo.py", "alpha"),
            _edit(1, "demo.py", "missing", "beta", parent_step=0),
            _test_fail(2, parent_step=None, file_path="demo.py"),
        ]
    )

    failure = report.failures[0]
    assert failure.breaking_step == 1
    assert failure.failure_cascade == (2,)
    assert failure.confidence == "LOW"
    assert failure.candidates == (2,)
    assert failure.coverage == "1 unlinked step on path"


def test_localization_is_deterministic_for_same_journal(localize_planted_fault_path, tmp_path):
    activities = ingest_transcript(localize_planted_fault_path)
    journal_path = tmp_path / "planted.journal.jsonl"
    Journal.from_activities(activities).write_jsonl(journal_path)

    first = localize_failures(Journal.read_jsonl(journal_path).activities).to_dict()
    second = localize_failures(Journal.read_jsonl(journal_path).activities).to_dict()

    assert first == second


def test_shell_target_fixture_links_writer_to_later_failure(shell_target_coverage_path):
    activities = ingest_transcript(shell_target_coverage_path)

    report = localize_failures(activities)

    assert report.status == "break"
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.breaking_step == 0
    assert failure.failure_cascade == (1,)
    assert failure.confidence == "LOW"
    assert "heuristic shell-target edge" in failure.coverage
    assert failure.candidates == (0, 1)

    coverage = shell_target_coverage(activities)
    assert coverage.to_dict() == {
        "shell_command_steps": 2,
        "steps_with_targets": 2,
        "added_edges": 1,
    }


def test_shell_target_localization_is_deterministic(shell_target_coverage_path):
    activities = ingest_transcript(shell_target_coverage_path)

    first = localize_failures(activities).to_dict()
    second = localize_failures(activities).to_dict()

    assert first == second


def test_ambiguous_shell_target_does_not_create_false_consumer_link():
    activities = [
        _test_fail(
            0,
            parent_step=None,
            command="cat $LOG | grep x > repo/out.txt",
        ),
        _test_fail(
            1,
            parent_step=None,
            command="pytest repo/out.txt",
        ),
    ]

    report = localize_failures(activities)

    assert report.status == "break"
    assert [failure.breaking_step for failure in report.failures] == [0, 1]
    assert all(failure.failure_cascade == () for failure in report.failures)
    assert not any(
        failure.breaking_step == 0 and failure.failure_cascade == (1,) and failure.confidence == "HIGH"
        for failure in report.failures
    )
    assert shell_target_coverage(activities).to_dict() == {
        "shell_command_steps": 2,
        "steps_with_targets": 1,
        "added_edges": 0,
    }


def test_localize_cli_prints_structural_sample_only(localize_planted_fault_path, capsys):
    assert main(["localize", str(localize_planted_fault_path)]) == 0
    output = capsys.readouterr().out

    assert "breaking_step\tgate\tcascade\tconfidence\tcoverage\tcandidates" in output
    assert "2\tG2\t3\tHIGH\tstructured path" in output
    assert "FAILED" not in output
    assert "AssertionError" not in output


def _write(step_index: int, file_path: str, content: str) -> Activity:
    return Activity(
        run_id="localize",
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="file_edit",
        tool_name="Write",
        inputs={"file_path": file_path, "old_string": "", "new_string": content, "write_mode": True},
        target=file_path,
    )


def _edit(
    step_index: int,
    file_path: str,
    old_string: str,
    new_string: str,
    parent_step: int | None,
) -> Activity:
    return Activity(
        run_id="localize",
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="file_edit",
        tool_name="Edit",
        inputs={"file_path": file_path, "old_string": old_string, "new_string": new_string},
        target=file_path,
        parent_step=parent_step,
    )


def _opaque(step_index: int, parent_step: int | None) -> Activity:
    return Activity(
        run_id="localize",
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="opaque_shell",
        tool_name="PowerShell",
        inputs={"command": "sanitized opaque command"},
        target="shell",
        parent_step=parent_step,
    )


def _test_fail(
    step_index: int,
    parent_step: int | None,
    file_path: str | None = None,
    command: str = "python -m pytest",
) -> Activity:
    inputs = {"command": command}
    if file_path is not None:
        inputs["file_path"] = file_path
    return Activity(
        run_id="localize",
        step_index=step_index,
        ts="2026-06-28T00:00:00Z",
        kind="test_run",
        tool_name="PowerShell",
        inputs=inputs,
        outputs={"result_text": "FAILED sanitized_test.py::test_demo", "exit_code": 1},
        target="shell",
        parent_step=parent_step,
    )
