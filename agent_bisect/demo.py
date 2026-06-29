from __future__ import annotations

from pathlib import Path
import sys
from time import sleep
from typing import TextIO

from .ingest_claude import ingest_transcript
from .io import load_activities
from .localize import localize_failures
from .model import Journal
from .replay import explain_replay


def run_packaging_demo(root: Path | None = None, *, out: TextIO | None = None, paced: bool = False) -> int:
    root = root or Path.cwd()
    out = out or sys.stdout
    replay_dir = root / "demo" / "_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    high_fixture_rel = Path("tests") / "fixtures" / "localize_planted_fault.jsonl"
    control_fixture_rel = Path("tests") / "fixtures" / "claude_sanitized.jsonl"
    high_journal_rel = Path("demo") / "_replay" / "high.journal.jsonl"
    control_journal_rel = Path("demo") / "_replay" / "control.journal.jsonl"
    high_fixture = root / high_fixture_rel
    control_fixture = root / control_fixture_rel
    high_journal = root / high_journal_rel
    control_journal = root / control_journal_rel

    _line(out, "agent-bisect demo")
    _line(out, "git-bisect for agent runs: localize the first visible break, or abstain.")
    _line(out, "")

    _line(out, "[1/2] WIN: HIGH-confidence localization on a shipped fixture")
    _ingest_and_show(high_fixture, high_fixture_rel, high_journal, high_journal_rel, out=out, paced=paced)
    _localize_and_show(high_journal, high_journal_rel, out=out, paced=paced)
    _replay_and_show(high_journal, high_journal_rel, out=out, paced=paced)

    _line(out, "")
    _line(out, "[2/2] CONTROL: clean fixture abstains instead of guessing")
    _ingest_and_show(control_fixture, control_fixture_rel, control_journal, control_journal_rel, out=out, paced=paced)
    _localize_and_show(control_journal, control_journal_rel, out=out, paced=paced)
    _replay_and_show(control_journal, control_journal_rel, out=out, paced=paced)

    _line(out, "")
    _line(out, "done: deterministic replay completed")
    return 0


def _ingest_and_show(source: Path, source_label: Path, target: Path, target_label: Path, *, out: TextIO, paced: bool) -> None:
    _command(out, f"agent-bisect ingest {_rel(source_label)} --out {_rel(target_label)}", paced)
    activities = ingest_transcript(source)
    journal = Journal.from_activities(activities)
    journal.write_jsonl(target)
    _line(out, f"wrote {len(journal.records)} activities to {_rel(target_label)}")


def _localize_and_show(path: Path, path_label: Path, *, out: TextIO, paced: bool) -> None:
    _command(out, f"agent-bisect localize {_rel(path_label)}", paced)
    report = localize_failures(load_activities(path))
    if report.status == "no_break":
        _line(out, "status\tno_break")
        return
    _line(out, "breaking_step\tgate\tcascade\tconfidence\tcoverage\tcandidates")
    for failure in report.failures:
        row = [
            str(failure.breaking_step),
            failure.breaking_gate,
            ",".join(str(step) for step in failure.failure_cascade),
            failure.confidence,
            failure.coverage,
            ",".join(str(step) for step in failure.candidates) or "-",
        ]
        _line(out, "\t".join(_safe_cell(cell) for cell in row))


def _replay_and_show(path: Path, path_label: Path, *, out: TextIO, paced: bool) -> None:
    _command(out, f"agent-bisect replay {_rel(path_label)} --explain", paced)
    _line(out, explain_replay(load_activities(path)).rstrip())


def _command(out: TextIO, command: str, paced: bool) -> None:
    if paced:
        sleep(1.5)
    _line(out, f"$ {command}")


def _line(out: TextIO, text: str) -> None:
    print(text, file=out)


def _rel(path: Path) -> str:
    return path.as_posix()


def _safe_cell(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")
