import json
from pathlib import Path

from agent_bisect.cli import main
from agent_bisect.study import StudyInput, render_study_markdown, run_corpus_study, write_study_reports


FIXTURES = Path(__file__).parent / "fixtures"


def test_corpus_study_fixture_aggregates_have_denominators():
    report = run_corpus_study(
        [
            StudyInput("claude", FIXTURES / "localize_planted_fault.jsonl"),
            StudyInput("codex", FIXTURES / "codex_sanitized.jsonl"),
            StudyInput("foreign", FIXTURES / "swe_agent_function_call.traj", schema="swe-agent"),
        ]
    )

    summary = report["summary"]
    assert summary["parsed_runs"] == 3
    assert summary["runs_considered"] == 3
    assert summary["source_records"] > 0
    assert summary["activities"] > 0
    assert summary["gate_failure_runs"] >= 1
    assert set(summary["localization_runs"]) == {"no_break", "HIGH", "LOW"}
    assert report["by_source"]["claude"]["parsed_runs"] == 1
    assert report["by_source"]["codex"]["parsed_runs"] == 1
    assert report["by_source"]["foreign"]["parsed_runs"] == 1


def test_corpus_study_break_run_count_matches_localization_status():
    report = run_corpus_study(
        [
            StudyInput("claude", FIXTURES / "localize_planted_fault.jsonl"),
            StudyInput("claude", FIXTURES / "shell_target_coverage.jsonl"),
            StudyInput("claude", FIXTURES / "slice2_sanitized.jsonl"),
        ]
    )

    summary = report["summary"]
    localized_break_runs = summary["localization_runs"]["HIGH"] + summary["localization_runs"]["LOW"]
    assert summary["gate_failure_runs"] == localized_break_runs
    assert summary["localization_runs"]["no_break"] == 1


def test_corpus_study_empty_parsed_run_counts_as_no_break(tmp_path):
    empty_transcript = tmp_path / "empty.jsonl"
    empty_transcript.write_text("", encoding="utf-8")

    report = run_corpus_study([StudyInput("claude", empty_transcript)])
    summary = report["summary"]

    assert summary["parsed_runs"] == 1
    assert summary["activities"] == 0
    assert summary["gate_failure_runs"] == 0
    assert summary["localization_runs"]["no_break"] == 1


def test_corpus_study_report_is_aggregate_only(tmp_path):
    report = run_corpus_study(
        [
            StudyInput("claude", FIXTURES / "localize_planted_fault.jsonl"),
            StudyInput("codex", FIXTURES / "codex_sanitized.jsonl"),
            StudyInput("foreign", FIXTURES / "mini-swe-agent-github-issue.traj.json", schema="mini-swe-agent"),
        ]
    )
    reports_dir = tmp_path / "reports"
    study_md = tmp_path / "STUDY.md"

    write_study_reports(report, reports_dir, study_md)

    rendered = study_md.read_text(encoding="utf-8")
    rendered += (reports_dir / "corpus-study.json").read_text(encoding="utf-8")
    for forbidden in (
        _sentinel(114, 101, 112, 111, 47, 108, 111, 99, 97, 108, 105, 122, 101, 100, 95, 100, 101, 109, 111, 46, 112, 121),
        _sentinel(114, 101, 112, 111, 47, 105, 110, 112, 117, 116, 46, 116, 120, 116),
        _sentinel(114, 101, 112, 111, 47, 111, 117, 116, 46, 116, 120, 116),
        _sentinel(109, 105, 115, 115, 105, 110, 103, 95, 99, 111, 108, 111, 110, 46, 112, 121),
        _sentinel(112, 121, 116, 104, 111, 110, 32, 45, 109, 32, 112, 121, 116, 101, 115, 116),
        _sentinel(65, 115, 115, 101, 114, 116, 105, 111, 110, 69, 114, 114, 111, 114),
        str(FIXTURES),
    ):
        assert forbidden not in rendered


def test_corpus_study_cli_writes_reports(tmp_path, capsys):
    reports_dir = tmp_path / "reports"
    study_md = tmp_path / "STUDY.md"

    assert main(
        [
            "corpus-study",
            "--claude",
            str(FIXTURES / "localize_planted_fault.jsonl"),
            "--codex",
            str(FIXTURES / "codex_sanitized.jsonl"),
            "--foreign",
            "openhands",
            str(FIXTURES / "openhands_realtask.json"),
            "--reports-dir",
            str(reports_dir),
            "--study-md",
            str(study_md),
        ]
    ) == 0
    output = capsys.readouterr().out

    assert "runs\t3/3" in output
    assert (reports_dir / "corpus-study.json").exists()
    assert study_md.exists()


def test_render_study_markdown_has_no_per_run_section():
    report = run_corpus_study([StudyInput("claude", FIXTURES / "slice2_sanitized.jsonl")])

    markdown = render_study_markdown(report)

    assert "Per-Run" not in markdown
    assert "run_id" not in markdown
    assert "## By Source" in markdown


def _sentinel(*codes: int) -> str:
    return "".join(chr(code) for code in codes)
