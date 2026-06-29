from pathlib import Path

import pytest

from agent_bisect.cli import main
from agent_bisect.oracle import RegexPatternSpec, load_pattern_set, render_cli_report, run_regex_oracle


@pytest.mark.parametrize(
    ("pattern", "input_text", "reference", "candidate", "agreed"),
    [
        ("cat", "cat", True, True, True),
        ("cat", "concatenate", True, False, False),
        ("cat", "dog", False, False, True),
        ("^cat$", "concatenate", False, False, True),
    ],
)
def test_search_vs_fullmatch_table(pattern, input_text, reference, candidate, agreed):
    report = run_regex_oracle([RegexPatternSpec("case", pattern, (input_text,))], max_generated=0)

    observation = report.pattern_results[0].observations[0]

    assert observation.reference is reference
    assert observation.candidate is candidate
    assert observation.agreed is agreed


def test_planted_divergence_localizes_exact_first_input():
    report = run_regex_oracle(
        [
            RegexPatternSpec(
                "planted-search-vs-fullmatch",
                "cat",
                ("cat", "concatenate", "dog"),
            )
        ],
        max_generated=0,
    )

    result = report.pattern_results[0]
    divergence = result.first_divergence

    assert divergence is not None
    assert divergence.input_index == 1
    assert divergence.input_text == "concatenate"
    assert divergence.reference is True
    assert divergence.candidate is False
    assert result.localization.status == "break"
    assert result.localization.failures[0].breaking_step == 1
    assert result.localization.failures[0].breaking_gate == "G3"
    assert result.localization.failures[0].confidence == "HIGH"


def test_regex_oracle_is_deterministic_for_same_inputs():
    specs = [
        RegexPatternSpec("planted-search-vs-fullmatch", "cat", ("cat", "concatenate", "dog")),
        RegexPatternSpec("anchored-agreement", "^cat$", ("cat", "concatenate", "dog")),
    ]

    first = run_regex_oracle(specs, max_generated=0)
    second = run_regex_oracle(specs, max_generated=0)

    assert first.to_dict() == second.to_dict()
    assert render_cli_report(first) == render_cli_report(second)


def test_pattern_set_fixture_and_cli_output(capsys):
    fixture = Path(__file__).parent / "fixtures" / "regex_oracle_patterns.json"

    assert main(["regex-oracle", str(fixture), "--max-generated", "0"]) == 0

    output = capsys.readouterr().out
    assert "first_divergence\tpattern=planted-search-vs-fullmatch\tinput_index=1\tinput=concatenate" in output
    assert "no_divergence\tpattern=anchored-agreement\tinputs=3\tagreement=3/3" in output
    assert "agreement_summary\tagree=5/6\tdiverge=1/6\tunsupported=0/2" in output


def test_invalid_pattern_fails_closed():
    report = run_regex_oracle([RegexPatternSpec("bad", "[", ("x",))], max_generated=0)
    result = report.pattern_results[0]

    assert result.supported is False
    assert result.compile_error
    assert result.observations == ()
    assert "unsupported\tpattern=bad\terror=" in render_cli_report(report)


def test_loader_generates_when_inputs_absent(tmp_path):
    pattern_set = tmp_path / "patterns.json"
    pattern_set.write_text('{"patterns":[{"id":"generated","pattern":"ab"}]}', encoding="utf-8")

    specs = load_pattern_set(pattern_set)
    report = run_regex_oracle(specs, max_generated=4)

    assert specs[0].generate is True
    assert report.pattern_results[0].input_count == 4
