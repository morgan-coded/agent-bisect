import json

import pytest

from agent_bisect.model import Activity, GENESIS_HASH, Journal, canonical_json


def test_journal_round_trip_preserves_hash_chain_and_activities(tmp_path):
    journal = Journal.from_activities(_activities())
    journal_path = tmp_path / "journal.jsonl"

    assert journal.records[0].prev_hash == GENESIS_HASH
    for previous, current in zip(journal.records, journal.records[1:]):
        assert current.prev_hash == previous.record_hash

    jsonl = journal.to_jsonl()
    journal.write_jsonl(journal_path)
    loaded = Journal.read_jsonl(journal_path)

    assert journal_path.read_text(encoding="utf-8") == jsonl
    assert _record_dicts(loaded) == _record_dicts(journal)
    assert _activity_dicts(loaded.activities) == _activity_dicts(journal.activities)
    assert [record.record_hash for record in loaded.records] == [
        record.record_hash for record in journal.records
    ]
    assert [activity.content_hash for activity in loaded.activities] == [
        activity.content_hash for activity in journal.activities
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("prev_hash", "record 1: prev_hash mismatch"),
        ("activity_field", "record 1: record_hash mismatch"),
        ("record_hash", "record 1: record_hash mismatch"),
    ],
)
def test_journal_read_jsonl_rejects_tampered_records_at_the_right_index(
    tmp_path,
    mutation,
    expected_error,
):
    records = _record_dicts(Journal.from_activities(_activities()))
    tampered_path = tmp_path / f"tampered-{mutation}.jsonl"

    if mutation == "prev_hash":
        records[1]["prev_hash"] = "f" * 64
    elif mutation == "activity_field":
        records[1]["activity"]["inputs"]["new_string"] = "tampered"
    elif mutation == "record_hash":
        records[1]["record_hash"] = "f" * 64
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    _write_records(tampered_path, records)

    with pytest.raises(ValueError, match=expected_error):
        Journal.read_jsonl(tampered_path)


def _activities() -> list[Activity]:
    return [
        Activity(
            run_id="journal-invariants",
            step_index=0,
            ts="2026-06-28T00:00:00Z",
            kind="user_msg",
            inputs={"text": "verify deterministic journal"},
        ),
        Activity(
            run_id="journal-invariants",
            step_index=1,
            ts="2026-06-28T00:00:01Z",
            kind="file_edit",
            tool_name="Write",
            inputs={
                "file_path": "repo/invariants.py",
                "old_string": "",
                "new_string": "value = 'before'\n",
                "write_mode": True,
            },
            target="repo/invariants.py",
            parent_step=0,
        ),
        Activity(
            run_id="journal-invariants",
            step_index=2,
            ts="2026-06-28T00:00:02Z",
            kind="file_edit",
            tool_name="Edit",
            inputs={
                "file_path": "repo/invariants.py",
                "old_string": "before",
                "new_string": "after",
            },
            target="repo/invariants.py",
            parent_step=1,
        ),
        Activity(
            run_id="journal-invariants",
            step_index=3,
            ts="2026-06-28T00:00:03Z",
            kind="test_run",
            tool_name="PowerShell",
            inputs={"command": "python -m pytest tests/test_invariants.py"},
            outputs={"result_text": "1 passed in 0.01s", "exit_code": 0},
            target="shell",
            parent_step=2,
        ),
    ]


def _activity_dicts(activities: list[Activity]) -> list[dict]:
    return [activity.to_dict() for activity in activities]


def _record_dicts(journal: Journal) -> list[dict]:
    return [record.to_dict() for record in journal.records]


def _write_records(path, records: list[dict]) -> None:
    path.write_text(
        "".join(f"{canonical_json(record)}\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
