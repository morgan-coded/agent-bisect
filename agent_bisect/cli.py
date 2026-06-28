from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .eval import evaluate_paths, write_eval_reports
from .foreign import fetch_openhands_realtask_trajectories, fetch_swe_agent_trajectories, ingest_foreign_trajectory, sweep_foreign_trajectories
from .gates import g1_schema, run_g2, run_g3
from .ingest_claude import ingest_transcript
from .ingest_codex import codex_coverage_report, ingest_codex_transcript, render_codex_coverage_markdown
from .io import load_activities
from .localize import localize_failures
from .model import Journal
from .replay import explain_replay
from .scan import scan_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-bisect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="ingest a Claude transcript into a journal")
    ingest_parser.add_argument("transcript", type=Path)
    ingest_parser.add_argument("--out", type=Path, help="output journal path")

    ingest_codex_parser = subparsers.add_parser("ingest-codex", help="ingest a Codex transcript into a journal")
    ingest_codex_parser.add_argument("transcript", type=Path)
    ingest_codex_parser.add_argument("--out", type=Path, help="output journal path")

    ingest_foreign_parser = subparsers.add_parser("ingest-foreign", help="ingest a foreign trajectory into a journal")
    ingest_foreign_parser.add_argument("--schema", required=True, choices=["swe-agent", "mini-swe-agent", "openhands"])
    ingest_foreign_parser.add_argument("trajectory", type=Path)
    ingest_foreign_parser.add_argument("--out", type=Path, help="output journal path")
    ingest_foreign_parser.add_argument("--source-url", default="")

    show_parser = subparsers.add_parser("show", help="show a transcript or journal structural timeline")
    show_parser.add_argument("path", type=Path)
    show_parser.add_argument("--gates", action="store_true", help="include G1/G2/G3 gate verdicts")

    localize_parser = subparsers.add_parser("localize", help="localize deterministic gate failures")
    localize_parser.add_argument("path", type=Path)

    replay_parser = subparsers.add_parser("replay", help="render a replay demo narrative")
    replay_parser.add_argument("path", type=Path)
    replay_parser.add_argument("--explain", action="store_true", help="show the structural explain view")

    eval_parser = subparsers.add_parser("eval", help="run injected-fault eval")
    eval_parser.add_argument("paths", nargs="+", type=Path)
    eval_parser.add_argument("--seed", type=int, default=1729)
    eval_parser.add_argument("--per-class", type=int, default=3)
    eval_parser.add_argument("--reports-dir", type=Path, default=Path("reports"))

    scan_parser = subparsers.add_parser("scan", help="scan real transcripts or journals for gate failures")
    scan_parser.add_argument("paths", nargs="+", type=Path)

    sweep_foreign_parser = subparsers.add_parser("sweep-foreign", help="sweep foreign trajectories through the deterministic gates")
    sweep_foreign_parser.add_argument("--schema", required=True, choices=["swe-agent", "mini-swe-agent", "openhands"])
    sweep_foreign_parser.add_argument("paths", nargs="+", type=Path)
    sweep_foreign_parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    sweep_foreign_parser.add_argument("--report-stem", default="foreign-coverage-report")

    coverage_codex_parser = subparsers.add_parser("coverage-codex", help="summarize Codex transcript ingest coverage")
    coverage_codex_parser.add_argument("paths", nargs="+", type=Path)
    coverage_codex_parser.add_argument("--out", type=Path, help="optional markdown report path")

    fetch_foreign_parser = subparsers.add_parser("fetch-swe-agent-trajectories", help="fetch public SWE-agent .traj files")
    fetch_foreign_parser.add_argument("--out-dir", type=Path, default=Path("data/foreign-trajectories"))
    fetch_foreign_parser.add_argument("--limit", type=int)

    fetch_openhands_parser = subparsers.add_parser("fetch-openhands-realtask-trajectories", help="fetch public OpenHands real-task trajectories")
    fetch_openhands_parser.add_argument("--out-dir", type=Path, default=Path("data/foreign-trajectories/openhands-realtask"))
    fetch_openhands_parser.add_argument("--limit", type=int, default=50)
    fetch_openhands_parser.add_argument("--page-size", type=int, default=100)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args.transcript, args.out)
    if args.command == "ingest-codex":
        return _cmd_ingest_codex(args.transcript, args.out)
    if args.command == "show":
        return _cmd_show(args.path, args.gates)
    if args.command == "localize":
        return _cmd_localize(args.path)
    if args.command == "replay":
        return _cmd_replay(args.path, args.explain)
    if args.command == "eval":
        return _cmd_eval(args.paths, args.seed, args.per_class, args.reports_dir)
    if args.command == "scan":
        return _cmd_scan(args.paths)
    if args.command == "ingest-foreign":
        return _cmd_ingest_foreign(args.schema, args.trajectory, args.out, args.source_url)
    if args.command == "fetch-swe-agent-trajectories":
        return _cmd_fetch_swe_agent_trajectories(args.out_dir, args.limit)
    if args.command == "fetch-openhands-realtask-trajectories":
        return _cmd_fetch_openhands_realtask_trajectories(args.out_dir, args.limit, args.page_size)
    if args.command == "sweep-foreign":
        return _cmd_sweep_foreign(args.schema, args.paths, args.reports_dir, args.report_stem)
    if args.command == "coverage-codex":
        return _cmd_coverage_codex(args.paths, args.out)
    parser.error("unknown command")
    return 2


def _cmd_ingest(transcript: Path, out: Path | None) -> int:
    activities = ingest_transcript(transcript)
    journal = Journal.from_activities(activities)
    out_path = out or transcript.with_suffix(".journal.jsonl")
    journal.write_jsonl(out_path)
    print(f"wrote {len(journal.records)} activities to {out_path}")
    return 0


def _cmd_ingest_codex(transcript: Path, out: Path | None) -> int:
    activities = ingest_codex_transcript(transcript)
    journal = Journal.from_activities(activities)
    out_path = out or transcript.with_suffix(".journal.jsonl")
    journal.write_jsonl(out_path)
    print(f"wrote {len(journal.records)} activities to {out_path}")
    return 0


def _cmd_show(path: Path, include_gates: bool) -> int:
    activities = load_activities(path)
    header = ["step", "kind", "tool", "target", "parent"]
    g2_results = run_g2(activities) if include_gates else []
    g3_results = run_g3(activities) if include_gates else []
    if include_gates:
        header.extend(["G1", "G1 evidence", "G2", "G2 evidence", "G3", "G3 evidence"])
    print("\t".join(header))
    for index, activity in enumerate(activities):
        row = [
            str(activity.step_index),
            activity.kind,
            activity.tool_name or "",
            activity.target or "",
            "" if activity.parent_step is None else str(activity.parent_step),
        ]
        if include_gates:
            g1_result = g1_schema(activity)
            g2_result = g2_results[index]
            g3_result = g3_results[index]
            row.extend(
                [
                    g1_result.status,
                    g1_result.evidence,
                    g2_result.status,
                    g2_result.evidence,
                    g3_result.status,
                    g3_result.evidence,
                ]
            )
        print("\t".join(_safe_cell(cell) for cell in row))
    return 0


def _cmd_localize(path: Path) -> int:
    report = localize_failures(load_activities(path))
    if report.status == "no_break":
        print("status\tno_break")
        return 0

    print("breaking_step\tgate\tcascade\tconfidence\tcoverage\tcandidates")
    for failure in report.failures:
        row = [
            str(failure.breaking_step),
            failure.breaking_gate,
            ",".join(str(step) for step in failure.failure_cascade),
            failure.confidence,
            failure.coverage,
            ",".join(str(step) for step in failure.candidates),
        ]
        print("\t".join(_safe_cell(cell) for cell in row))
    return 0


def _cmd_replay(path: Path, explain: bool) -> int:
    if not explain:
        print("replay requires --explain")
        return 2
    print(explain_replay(load_activities(path)), end="")
    return 0


def _cmd_eval(paths: list[Path], seed: int, per_class: int, reports_dir: Path) -> int:
    report = evaluate_paths(paths, seed=seed, per_class=per_class)
    write_eval_reports(report, reports_dir)
    print("fault_class\teligible\tinjected\tscored\tTP\tFP\tFN\tNA\tprecision\trecall")
    for fault_class, metrics in report["classes"].items():
        row = [
            fault_class,
            metrics["eligible_n"],
            metrics["injected_n"],
            metrics["scored_n"],
            metrics["tp"],
            metrics["fp"],
            metrics["fn"],
            metrics["na"],
            _format_rate(metrics["precision"]),
            _format_rate(metrics["recall"]),
        ]
        print("\t".join(str(value) for value in row))
    return 0


def _cmd_scan(paths: list[Path]) -> int:
    report = scan_paths(paths)
    print("label\tgeneralization check (same Claude schema)")
    print("limitation\tthe deterministic gates validate within the Claude tool-call schema; use foreign-schema adapters for cross-platform coverage checks.")
    print("runs\tfailures")
    for run in report["runs"]:
        print(f"{run['run_id']}\t{run['failure_count']}")
    return 0


def _cmd_ingest_foreign(schema: str, trajectory: Path, out: Path | None, source_url: str) -> int:
    activities = ingest_foreign_trajectory(trajectory, schema=schema, source_url=source_url)
    journal = Journal.from_activities(activities)
    out_path = out or trajectory.with_suffix(".journal.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    journal.write_jsonl(out_path)
    print(f"wrote {len(journal.records)} activities to {out_path}")
    return 0


def _cmd_fetch_swe_agent_trajectories(out_dir: Path, limit: int | None) -> int:
    sources = fetch_swe_agent_trajectories(out_dir, limit=limit)
    print(f"fetched {len(sources)} SWE-agent trajectories to {out_dir}")
    print(f"manifest\t{out_dir / 'manifest.json'}")
    return 0


def _cmd_fetch_openhands_realtask_trajectories(out_dir: Path, limit: int, page_size: int) -> int:
    sources = fetch_openhands_realtask_trajectories(out_dir, limit=limit, page_size=page_size)
    distinct_instance_ids = {source.instance_id for source in sources if source.instance_id}
    print(f"fetched {len(sources)} OpenHands trajectories to {out_dir}")
    print(f"distinct_instance_ids\t{len(distinct_instance_ids)}")
    print(f"manifest\t{out_dir / 'manifest.json'}")
    return 0


def _cmd_sweep_foreign(schema: str, paths: list[Path], reports_dir: Path, report_stem: str) -> int:
    report = sweep_foreign_trajectories(paths, schema=schema, reports_dir=reports_dir, report_stem=report_stem)
    print(f"wrote {reports_dir / (report_stem + '.json')}")
    print(f"wrote {reports_dir / (report_stem + '.md')}")
    print(f"trajectories\t{report['trajectory_count']}")
    print(f"distinct_instance_ids\t{report['distinct_instance_id_count']}")
    print(f"activities\t{report['activity_count']}")
    coverage = report["localization_coverage"]
    print(
        "localization\tHIGH={high}/{denom} LOW={low}/{denom} NA={na}/{denom}".format(
            high=coverage["HIGH"]["count"],
            low=coverage["LOW"]["count"],
            na=coverage["NA"]["count"],
            denom=coverage["HIGH"]["denominator"],
        )
    )
    gaps = report["coverage_gaps"]
    print(
        "coverage_gaps\topaque_shell_actions={opaque}/{total} unmapped_actions={unmapped}/{total} unlinked_actions_excluding_first={unlinked}/{total}".format(
            opaque=gaps["opaque_shell_action_steps"],
            unmapped=gaps["unmapped_action_steps"],
            unlinked=gaps["unlinked_action_steps_excluding_first_action"],
            total=gaps["action_activity_count"],
        )
    )
    return 0


def _cmd_coverage_codex(paths: list[Path], out: Path | None) -> int:
    report = codex_coverage_report(paths)
    rendered = render_codex_coverage_markdown(report)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {out}")
    print(rendered, end="")
    return 0


def _safe_cell(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _format_rate(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
