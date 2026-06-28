import json

from agent_bisect.model import Activity, canonical_json


def test_content_hash_is_independent_of_dict_insertion_order():
    first = Activity(
        run_id="canonical-json",
        step_index=0,
        ts="2026-06-28T00:00:00Z",
        kind="tool_call",
        tool_name="Read",
        inputs={
            "file_path": "repo/example.py",
            "options": {"z": 3, "a": [2, 1], "m": {"right": True, "left": False}},
        },
        target="repo/example.py",
    )
    second = Activity(
        run_id="canonical-json",
        step_index=0,
        ts="2026-06-28T00:00:00Z",
        kind="tool_call",
        tool_name="Read",
        inputs={
            "options": {"m": {"left": False, "right": True}, "a": [2, 1], "z": 3},
            "file_path": "repo/example.py",
        },
        target="repo/example.py",
    )

    assert canonical_json(first.hash_payload()) == canonical_json(second.hash_payload())
    assert first.compute_content_hash() == second.compute_content_hash()
    assert first.content_hash == second.content_hash


def test_content_hash_excludes_itself_and_refresh_is_stable():
    activity = Activity(
        run_id="canonical-json",
        step_index=1,
        ts="2026-06-28T00:00:01Z",
        kind="file_edit",
        tool_name="Edit",
        inputs={
            "file_path": "repo/example.py",
            "old_string": "before",
            "new_string": "after",
        },
        target="repo/example.py",
    )
    expected_hash = activity.content_hash

    activity.content_hash = "not-the-recorded-hash"

    assert activity.compute_content_hash() == expected_hash
    activity.refresh_hash()
    assert activity.content_hash == expected_hash
    activity.refresh_hash()
    assert activity.content_hash == expected_hash


def test_unicode_canonical_json_round_trips_without_ascii_escaping():
    payload = {
        "message": "\u2603 caf\u00e9",
        "nested": {"word": "\u65e5\u672c", "symbols": ["\u03bb", "\u2713"]},
    }

    rendered = canonical_json(payload)
    assert json.loads(rendered) == payload
    assert "\\u2603" not in rendered
    assert "\\u00e9" not in rendered
    assert "\\u65e5" not in rendered

    activity = Activity(
        run_id="unicode",
        step_index=0,
        ts="2026-06-28T00:00:00Z",
        kind="tool_call",
        tool_name="Read",
        inputs={"file_path": "repo/unicode.txt", "payload": payload},
        outputs={"echo": payload},
        target="repo/unicode.txt",
    )

    encoded = canonical_json(activity.to_dict())
    decoded = Activity.from_dict(json.loads(encoded))

    assert decoded.to_dict() == activity.to_dict()
