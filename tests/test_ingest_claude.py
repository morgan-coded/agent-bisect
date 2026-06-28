from agent_bisect.ingest_claude import ingest_transcript
from agent_bisect.model import Journal


def test_fixture_ingest_kinds_and_parent_threading(fixture_path):
    activities = ingest_transcript(fixture_path)

    assert [activity.kind for activity in activities] == [
        "user_msg",
        "llm_call",
        "tool_call",
        "file_edit",
        "test_run",
        "opaque_shell",
    ]

    assert activities[2].tool_name == "Read"
    assert activities[2].target == "repo/example.py"
    assert activities[2].parent_step == 1

    assert activities[3].tool_name == "Edit"
    assert activities[3].inputs["old_string"] == "before"
    assert activities[3].inputs["new_string"] == "after"
    assert activities[3].parent_step == 2

    assert activities[4].tool_name == "PowerShell"
    assert activities[4].target == "shell"
    assert activities[4].parent_step == 3
    assert activities[4].outputs["exit_code"] == 0
    assert "1 passed" in activities[4].outputs["result_text"]


def test_journal_hashes_are_deterministic(fixture_path):
    first = Journal.from_activities(ingest_transcript(fixture_path)).to_jsonl()
    second = Journal.from_activities(ingest_transcript(fixture_path)).to_jsonl()

    assert first == second
    assert '"prev_hash"' in first
    assert '"record_hash"' in first
