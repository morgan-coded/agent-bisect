import json
from pathlib import Path

from agent_bisect import foreign
from agent_bisect.cli import main
from agent_bisect.foreign import fetch_openhands_realtask_trajectories, ingest_foreign_trajectory, sweep_foreign_trajectories
from agent_bisect.gates import run_g1, run_g3
from agent_bisect.model import Journal


def test_swe_agent_function_call_fixture_maps_to_expected_activities():
    path = _fixture_path()

    activities = ingest_foreign_trajectory(path, schema="swe-agent", source_url="https://example.test/traj")

    assert [activity.kind for activity in activities] == [
        "user_msg",
        "user_msg",
        "tool_call",
        "verdict",
        "file_edit",
        "verdict",
        "test_run",
        "verdict",
    ]
    assert activities[2].tool_name == "open"
    assert activities[2].target == "tests/missing_colon.py"
    assert activities[4].tool_name == "Edit"
    assert activities[4].target == "tests/missing_colon.py"
    assert activities[4].inputs["old_string"] == "def division(a: float, b: float) -> float"
    assert activities[4].inputs["new_string"] == "def division(a: float, b: float) -> float:"
    assert activities[6].tool_name == "Bash"
    assert activities[6].outputs["result_text"] == "8.2\n(Open file: tests/missing_colon.py)\nbash-$"
    assert all(result.status in {"PASS", "NA"} for result in run_g1(activities))


def test_swe_agent_ingest_journal_is_deterministic(tmp_path):
    path = _fixture_path()

    first = Journal.from_activities(ingest_foreign_trajectory(path, schema="swe-agent")).to_jsonl()
    second = Journal.from_activities(ingest_foreign_trajectory(path, schema="swe-agent")).to_jsonl()

    assert first == second
    journal_path = tmp_path / "foreign.journal.jsonl"
    journal_path.write_text(first, encoding="utf-8", newline="\n")
    assert Journal.read_jsonl(journal_path).to_jsonl() == first


def test_unknown_foreign_record_is_unmapped_not_dropped(tmp_path):
    path = tmp_path / "unknown.traj"
    path.write_text(json.dumps({"history": [{"role": "alien", "payload": {"x": 1}}]}), encoding="utf-8")

    activities = ingest_foreign_trajectory(path, schema="swe-agent")

    assert len(activities) == 1
    assert activities[0].kind == "unmapped"
    assert activities[0].inputs["reason"] == "unknown_role:alien"
    assert run_g1(activities)[0].status == "NA"


def test_classic_swe_agent_user_observation_attaches_to_action_and_drives_g3(tmp_path):
    path = tmp_path / "classic.traj"
    path.write_text(
        json.dumps(
            {
                "history": [
                    {"role": "system", "content": "shell"},
                    {"role": "user", "content": "fix it"},
                    {"role": "assistant", "content": "run repro", "thought": "check", "action": "python reproduce.py\n"},
                    {"role": "user", "content": "Traceback (most recent call last):\nAssertionError\nbash-$"},
                ]
            }
        ),
        encoding="utf-8",
    )

    activities = ingest_foreign_trajectory(path, schema="swe-agent")

    assert [activity.kind for activity in activities] == ["user_msg", "user_msg", "test_run", "verdict"]
    assert activities[3].parent_step == 2
    assert "Traceback" in activities[2].outputs["result_text"]
    assert run_g3(activities)[2].status == "FAIL"


def test_ingest_foreign_cli_writes_hash_chained_journal(tmp_path, capsys):
    out = tmp_path / "foreign.journal.jsonl"

    assert main(["ingest-foreign", "--schema", "swe-agent", str(_fixture_path()), "--out", str(out)]) == 0
    output = capsys.readouterr().out

    assert "wrote 8 activities" in output
    assert len(Journal.read_jsonl(out).records) == 8


def test_sweep_foreign_report_has_coverage_denominators(tmp_path):
    report = sweep_foreign_trajectories([_fixture_path()], schema="swe-agent", reports_dir=tmp_path)

    assert report["trajectory_count"] == 1
    assert report["activity_count"] == 8
    assert report["localization_denominator"] == "gate_failure_steps"
    assert set(report["localization_coverage"]) == {"HIGH", "LOW", "NA"}
    assert (tmp_path / "foreign-coverage-report.json").exists()
    assert (tmp_path / "foreign-coverage-report.md").exists()


def test_openhands_fixture_maps_to_expected_activities():
    activities = ingest_foreign_trajectory(_openhands_fixture_path(), schema="openhands", source_url="https://example.test/row")

    assert [activity.kind for activity in activities] == [
        "user_msg",
        "user_msg",
        "tool_call",
        "verdict",
        "file_edit",
        "verdict",
        "file_edit",
        "verdict",
        "test_run",
        "verdict",
    ]
    assert activities[2].tool_name == "Read"
    assert activities[2].target == "/workspace/example/repo.py"
    assert activities[3].parent_step == 2
    assert activities[4].tool_name == "Write"
    assert activities[4].inputs["write_mode"] is True
    assert activities[6].tool_name == "Edit"
    assert activities[6].inputs["old_string"] == "assert True"
    assert activities[8].tool_name == "Bash"
    assert activities[8].outputs["result_text"] == "1 passed"
    assert run_g3(activities)[8].status == "PASS"


def test_mini_swe_agent_fixture_maps_actions_and_observations():
    activities = ingest_foreign_trajectory(_mini_fixture_path(), schema="mini-swe-agent", source_url="https://example.test/mini")

    kinds = [activity.kind for activity in activities]
    assert kinds.count("file_edit") == 1
    assert kinds.count("test_run") == 1
    assert "verdict" in kinds
    edit = next(activity for activity in activities if activity.kind == "file_edit")
    assert edit.tool_name == "Edit"
    assert edit.target == "tests/missing_colon.py"
    assert edit.inputs["old_string"] == "def division(a: float, b: float) -> float"
    assert edit.inputs["new_string"] == "def division(a: float, b: float) -> float:"
    test_run = next(activity for activity in activities if activity.kind == "test_run")
    assert test_run.outputs["exit_code"] == 0
    assert "8.2" in test_run.outputs["result_text"]
    assert all(result.status in {"PASS", "NA"} for result in run_g1(activities))


def test_mini_swe_agent_unknown_record_is_unmapped_not_dropped(tmp_path):
    path = tmp_path / "unknown-mini.json"
    path.write_text(json.dumps({"trajectory_format": "mini-swe-agent-1.1", "messages": [{"role": "alien", "payload": {"x": 1}}]}), encoding="utf-8")

    activities = ingest_foreign_trajectory(path, schema="mini-swe-agent")

    assert len(activities) == 1
    assert activities[0].kind == "unmapped"
    assert activities[0].inputs["reason"] == "unknown_role:alien"
    assert run_g1(activities)[0].status == "NA"


def test_openhands_unknown_function_is_unmapped_not_dropped(tmp_path):
    path = tmp_path / "unknown-openhands.json"
    path.write_text(
        json.dumps(
            {
                "row": {
                    "instance_id": "example__repo-2",
                    "trajectory_id": "traj-2",
                    "trajectory": [
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-unknown",
                                    "type": "function",
                                    "function": {"name": "mystery_tool", "arguments": "{}"},
                                }
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    activities = ingest_foreign_trajectory(path, schema="openhands")

    assert len(activities) == 1
    assert activities[0].kind == "unmapped"
    assert activities[0].inputs["reason"] == "unknown_function:mystery_tool"
    assert run_g1(activities)[0].status == "NA"


def test_fetch_openhands_realtask_dedups_by_instance_id(monkeypatch, tmp_path):
    rows = [
        _openhands_fetch_row(0, "example__repo-1", "traj-1"),
        _openhands_fetch_row(1, "example__repo-1", "traj-duplicate"),
        _openhands_fetch_row(2, "example__repo-2", "traj-2"),
    ]

    monkeypatch.setattr(foreign, "_huggingface_dataset_commit", lambda dataset: "test-sha")
    monkeypatch.setattr(
        foreign,
        "_huggingface_rows",
        lambda *args, **kwargs: {"rows": rows, "num_rows_total": len(rows)},
    )

    sources = fetch_openhands_realtask_trajectories(tmp_path, limit=2)

    assert [source.instance_id for source in sources] == ["example__repo-1", "example__repo-2"]
    assert len({source.instance_id for source in sources}) == 2
    assert (tmp_path / "manifest.json").exists()
    assert all(Path(source.local_path).exists() for source in sources)


def test_openhands_sweep_report_has_real_task_denominators(monkeypatch, tmp_path):
    rows = [
        _openhands_fetch_row(0, "example__repo-1", "traj-1"),
        _openhands_fetch_row(1, "example__repo-2", "traj-2"),
    ]
    monkeypatch.setattr(foreign, "_huggingface_dataset_commit", lambda dataset: "test-sha")
    monkeypatch.setattr(
        foreign,
        "_huggingface_rows",
        lambda *args, **kwargs: {"rows": rows, "num_rows_total": len(rows)},
    )
    fetch_dir = tmp_path / "foreign-trajectories"
    fetch_openhands_realtask_trajectories(fetch_dir, limit=2)

    report = sweep_foreign_trajectories(
        [fetch_dir],
        schema="openhands",
        reports_dir=tmp_path,
        report_stem="foreign-coverage-realtask-report",
    )

    assert report["trajectory_count"] == 2
    assert report["distinct_instance_id_count"] == 2
    assert report["deduplication_key"] == "instance_id"
    assert len(report["per_instance_breakdown"]) == 2
    assert (tmp_path / "foreign-coverage-realtask-report.json").exists()
    assert (tmp_path / "foreign-coverage-realtask-report.md").exists()


def _fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "swe_agent_function_call.traj"


def _openhands_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "openhands_realtask.json"


def _mini_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "mini-swe-agent-github-issue.traj.json"


def _openhands_fetch_row(row_idx: int, instance_id: str, trajectory_id: str) -> dict[str, object]:
    return {
        "row_idx": row_idx,
        "row": {
            "instance_id": instance_id,
            "repo": "example/repo",
            "trajectory_id": trajectory_id,
            "trajectory": [
                {"role": "user", "content": "fix it", "tool_calls": None},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{row_idx}",
                            "type": "function",
                            "function": {
                                "name": "execute_bash",
                                "arguments": json.dumps({"command": "pytest -q"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "execute_bash",
                    "tool_call_id": f"call-{row_idx}",
                    "content": "1 passed",
                    "tool_calls": None,
                },
            ],
        },
    }
