import json
from pathlib import Path

from agent_bisect.benchmark import (
    WhoWhenCase,
    WhoWhenLabel,
    load_who_when_cases,
    score_who_when_cases,
    write_who_when_reports,
)
from agent_bisect.cli import main
from agent_bisect.model import Activity


def test_who_when_adapter_preserves_label_index_and_fails_closed(tmp_path):
    data_dir = _write_cache(
        tmp_path,
        {
            "Algorithm-Generated": [
                {
                    "question_ID": "synthetic-question",
                    "history": [
                        {"role": "human", "content": "SECRET_PROMPT"},
                        {"role": "assistant", "name": "Planner", "content": "SECRET_PLAN"},
                        {"role": "user", "name": "Computer_terminal", "content": "SECRET_TERMINAL"},
                    ],
                    "mistake_agent": "Planner",
                    "mistake_step": "1",
                    "mistake_reason": "SECRET_REASON",
                }
            ]
        },
    )

    cases, _manifest = load_who_when_cases(data_dir)

    assert len(cases) == 1
    case = cases[0]
    assert case.label.expected_breaking_step == 1
    assert case.label.responsible_agent == "Planner"
    assert [activity.kind for activity in case.activities] == ["user_msg", "llm_call", "unmapped"]
    assert case.activities[2].inputs["reason"] == "terminal_without_structured_command"
    rendered_inputs = json.dumps([activity.inputs for activity in case.activities], sort_keys=True)
    assert "SECRET_PROMPT" not in rendered_inputs
    assert "SECRET_PLAN" not in rendered_inputs
    assert "SECRET_TERMINAL" not in rendered_inputs
    assert "SECRET_REASON" not in rendered_inputs


def test_who_when_scoring_reports_exact_cascade_confidence_and_gaps():
    report = score_who_when_cases(_synthetic_cases(), _synthetic_manifest())
    summary = report["summary"]

    assert summary["included_label_count"] == 3
    assert summary["exact_step_correct"] == 2
    assert summary["cascade_membership_correct"] == 2
    assert summary["coverage_gap_count"] == 2
    assert summary["confidence"]["HIGH"]["total"] == 1
    assert summary["confidence"]["HIGH"]["exact_step_accuracy"] == 1.0
    assert summary["confidence"]["LOW"]["total"] == 1
    assert summary["confidence"]["LOW"]["exact_step_accuracy"] == 1.0
    assert summary["confidence"]["NA"]["total"] == 1
    assert summary["confidence"]["NA"]["exact_step_accuracy"] == 0.0
    assert report["scoring_rule"]["agent_attribution"] == "unsupported by LocalizationReport schema"


def test_who_when_scoring_is_deterministic_for_same_inputs():
    first = json.dumps(score_who_when_cases(_synthetic_cases(), _synthetic_manifest()), sort_keys=True, indent=2)
    second = json.dumps(score_who_when_cases(_synthetic_cases(), _synthetic_manifest()), sort_keys=True, indent=2)

    assert first == second


def test_who_when_cli_scores_existing_cache_and_writes_reports(tmp_path, capsys):
    data_dir = _write_cache(
        tmp_path,
        {
            "Algorithm-Generated": [
                {
                    "question_ID": "synthetic-question",
                    "history": [{"role": "assistant", "name": "Planner", "content": "SECRET_PLAN"}],
                    "mistake_agent": "Planner",
                    "mistake_step": "0",
                    "mistake_reason": "SECRET_REASON",
                }
            ]
        },
    )
    reports_dir = tmp_path / "reports"
    benchmark_md = tmp_path / "BENCHMARK.md"

    assert main(
        [
            "benchmark-who-when",
            "--data-dir",
            str(data_dir),
            "--reports-dir",
            str(reports_dir),
            "--benchmark-md",
            str(benchmark_md),
        ]
    ) == 0
    output = capsys.readouterr().out

    assert "labels\t1" in output
    assert (reports_dir / "who-when-benchmark.json").exists()
    assert benchmark_md.exists()
    assert "SECRET_PLAN" not in (reports_dir / "who-when-benchmark.json").read_text(encoding="utf-8")
    assert "SECRET_REASON" not in benchmark_md.read_text(encoding="utf-8")


def test_who_when_report_writer_is_structural_only(tmp_path):
    report = score_who_when_cases(_synthetic_cases(), _synthetic_manifest())
    reports_dir = tmp_path / "reports"
    benchmark_md = tmp_path / "BENCHMARK.md"

    write_who_when_reports(report, reports_dir, benchmark_md)

    assert "Who&When Localization Benchmark" in benchmark_md.read_text(encoding="utf-8")
    assert (reports_dir / "who-when-benchmark.json").read_text(encoding="utf-8") == json.dumps(
        report, sort_keys=True, indent=2
    ) + "\n"


def _synthetic_cases() -> list[WhoWhenCase]:
    return [
        WhoWhenCase(
            label=_label("Algorithm-Generated", 0, 1),
            activities=tuple(
                [
                    _write(0, "demo.py", "alpha", parent_step=None),
                    _edit(1, "demo.py", "missing", "beta", parent_step=0),
                    _test_fail(2, parent_step=1),
                ]
            ),
        ),
        WhoWhenCase(
            label=_label("Algorithm-Generated", 1, 1),
            activities=tuple(
                [
                    _write(0, "demo.py", "alpha", parent_step=None),
                    _edit(1, "demo.py", "missing", "beta", parent_step=0),
                    _opaque(2, parent_step=1),
                    _test_fail(3, parent_step=2),
                ]
            ),
        ),
        WhoWhenCase(
            label=_label("Hand-Crafted", 0, 0),
            activities=(
                Activity(
                    run_id="synthetic-na",
                    step_index=0,
                    ts="",
                    kind="llm_call",
                    tool_name="Planner",
                    inputs={"agent": "Planner", "content_hash": "abc", "content_length": 3},
                    target="Planner",
                ),
            ),
        ),
    ]


def _synthetic_manifest() -> dict[str, object]:
    return {
        "dataset": "synthetic",
        "dataset_revision": "test-revision",
        "split": "train",
        "sources": [
            {"config": "Algorithm-Generated", "row_idx": 0, "question_id": "q0"},
            {"config": "Algorithm-Generated", "row_idx": 1, "question_id": "q1"},
            {"config": "Hand-Crafted", "row_idx": 0, "question_id": "q2"},
        ],
    }


def _label(config: str, row_idx: int, step: int) -> WhoWhenLabel:
    return WhoWhenLabel(
        label_id=f"{config}:{row_idx}:q{row_idx}",
        config=config,
        split="train",
        row_idx=row_idx,
        question_id=f"q{row_idx}",
        expected_breaking_step=step,
        responsible_agent="Planner",
        mistake_reason_hash="reason-hash",
    )


def _write_cache(tmp_path: Path, rows_by_config: dict[str, list[dict[str, object]]]) -> Path:
    data_dir = tmp_path / "who-when"
    sources = []
    for config, rows in rows_by_config.items():
        for row_idx, row in enumerate(rows):
            local_path = data_dir / config / f"{row_idx:04d}.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "dataset": "synthetic",
                "dataset_revision": "test-revision",
                "config": config,
                "split": "train",
                "row_idx": row_idx,
                "source_url": f"https://example.test/{config}/{row_idx}",
                "row": row,
            }
            local_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
            sources.append(
                {
                    "dataset": "synthetic",
                    "dataset_revision": "test-revision",
                    "config": config,
                    "split": "train",
                    "row_idx": row_idx,
                    "local_path": local_path.resolve().as_posix(),
                    "source_url": f"https://example.test/{config}/{row_idx}",
                    "question_id": str(row.get("question_ID", "")),
                }
            )
    manifest = {
        "schema_version": 1,
        "dataset": "synthetic",
        "dataset_revision": "test-revision",
        "split": "train",
        "sources": sources,
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return data_dir


def _write(step_index: int, file_path: str, content: str, parent_step: int | None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="file_edit",
        tool_name="Write",
        inputs={"file_path": file_path, "old_string": "", "new_string": content, "write_mode": True},
        target=file_path,
        parent_step=parent_step,
    )


def _edit(step_index: int, file_path: str, old_string: str, new_string: str, parent_step: int | None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="file_edit",
        tool_name="Edit",
        inputs={"file_path": file_path, "old_string": old_string, "new_string": new_string},
        target=file_path,
        parent_step=parent_step,
    )


def _opaque(step_index: int, parent_step: int | None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="opaque_shell",
        tool_name="PowerShell",
        inputs={"command": "sanitized opaque command"},
        target="shell",
        parent_step=parent_step,
    )


def _test_fail(step_index: int, parent_step: int | None) -> Activity:
    return Activity(
        run_id="synthetic",
        step_index=step_index,
        ts="",
        kind="test_run",
        tool_name="PowerShell",
        inputs={"command": "python -m pytest"},
        outputs={"result_text": "FAILED sanitized_test.py::test_demo", "exit_code": 1},
        target="shell",
        parent_step=parent_step,
    )
