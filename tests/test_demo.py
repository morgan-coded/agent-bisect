from agent_bisect.cli import main


def test_packaged_demo_localizes_then_abstains(capsys):
    assert main(["demo"]) == 0
    output = capsys.readouterr().out

    assert "[1/2] WIN: HIGH-confidence localization" in output
    assert "breaking_step\tgate\tcascade\tconfidence\tcoverage\tcandidates" in output
    assert "2\tG2\t3\tHIGH\tstructured path\t-" in output
    assert "verdict: 1 break(s) localized (HIGH=1 LOW=0)" in output

    assert "[2/2] CONTROL: clean fixture abstains" in output
    assert "status\tno_break" in output
    assert "verdict: clean run" in output
    assert "done: deterministic replay completed" in output
