from pathlib import Path

from agent_bisect.cli import main
from agent_bisect.ingest_codex import ingest_codex_transcript
from agent_bisect.io import load_activities, looks_like_codex
from agent_bisect.localize import shell_target_coverage
from agent_bisect.model import Journal


FIXTURE = Path(__file__).parent / "fixtures" / "codex_sanitized.jsonl"


def test_codex_fixture_ingest_kinds_and_linkage():
    activities = ingest_codex_transcript(FIXTURE)

    assert [activity.kind for activity in activities] == [
        "unmapped",
        "user_msg",
        "llm_call",
        "opaque_shell",
        "file_edit",
        "test_run",
        "tool_call",
        "verdict",
        "unmapped",
    ]

    shell = activities[3]
    assert shell.tool_name == "PowerShell"
    assert shell.target == "shell"
    assert shell.inputs["command"] == "cat repo/input.txt > repo/out.txt"
    assert shell.outputs["exit_code"] == 0
    assert shell.parent_step == 2

    patch = activities[4]
    assert patch.tool_name == "apply_patch"
    assert patch.target == "repo/app.py"
    assert patch.inputs["file_path"] == "repo/app.py"
    assert patch.inputs["patch_targets"] == ["repo/app.py"]
    assert "Success. Updated files" not in str(patch.outputs)

    test_run = activities[5]
    assert test_run.kind == "test_run"
    assert test_run.inputs["command"] == "python -m pytest repo/out.txt"
    assert test_run.outputs["exit_code"] == 0
    assert test_run.outputs["result_text"] == "1 passed"

    assert activities[-1].kind == "unmapped"
    assert activities[-1].inputs["payload_type"] == "mystery_record"
    assert "record_hash" in activities[-1].inputs


def test_codex_auto_detect_routes_through_io():
    assert looks_like_codex(FIXTURE)
    assert [activity.kind for activity in load_activities(FIXTURE)][:3] == ["unmapped", "user_msg", "llm_call"]


def test_codex_ingest_is_deterministic():
    first = Journal.from_activities(ingest_codex_transcript(FIXTURE)).to_jsonl()
    second = Journal.from_activities(ingest_codex_transcript(FIXTURE)).to_jsonl()

    assert first == second


def test_codex_shell_targets_lift_from_command_inputs():
    coverage = shell_target_coverage(ingest_codex_transcript(FIXTURE))

    assert coverage.shell_command_steps == 2
    assert coverage.steps_with_targets == 2
    assert coverage.added_edges >= 1


def test_ingest_codex_cli_round_trip(tmp_path):
    out = tmp_path / "codex.journal.jsonl"

    assert main(["ingest-codex", str(FIXTURE), "--out", str(out)]) == 0
    loaded = Journal.read_jsonl(out).activities

    assert len(loaded) == 9
    assert loaded[5].kind == "test_run"
