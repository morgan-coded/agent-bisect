from agent_bisect.cli import main
from agent_bisect.ingest_claude import ingest_transcript
from agent_bisect.model import Journal
from agent_bisect.replay import explain_replay


def test_replay_explain_planted_fault_is_structural(localize_planted_fault_path, capsys):
    assert main(["replay", str(localize_planted_fault_path), "--explain"]) == 0
    output = capsys.readouterr().out

    assert "agent-bisect replay --explain" in output
    assert "activities: 4" in output
    assert "gate_tallies:" in output
    assert "verdict: 1 break(s) localized (HIGH=1 LOW=0)" in output
    assert "breaking_step: 2" in output
    assert "gate: G2" in output
    assert "cascade: 3" in output
    assert "confidence: HIGH" in output
    assert "coverage: structured path" in output

    for raw in _raw_input_literals(ingest_transcript(localize_planted_fault_path)):
        assert raw not in output


def test_replay_explain_clean_run(slice2_fixture_path, capsys):
    assert main(["replay", str(slice2_fixture_path), "--explain"]) == 0
    output = capsys.readouterr().out

    assert "verdict: clean run" in output
    assert "breaks:" not in output


def test_replay_explain_is_deterministic_for_same_journal(localize_planted_fault_path, tmp_path):
    activities = ingest_transcript(localize_planted_fault_path)
    journal_path = tmp_path / "planted.journal.jsonl"
    Journal.from_activities(activities).write_jsonl(journal_path)

    first = explain_replay(Journal.read_jsonl(journal_path).activities)
    second = explain_replay(Journal.read_jsonl(journal_path).activities)

    assert first == second


def test_replay_without_explain_returns_error(localize_planted_fault_path, capsys):
    assert main(["replay", str(localize_planted_fault_path)]) == 2
    assert "replay requires --explain" in capsys.readouterr().out


def _raw_input_literals(activities):
    raw = []
    for activity in activities:
        for key in ("old_string", "new_string", "command"):
            value = activity.inputs.get(key)
            if isinstance(value, str) and len(value) > 4:
                raw.append(value)
    return raw
