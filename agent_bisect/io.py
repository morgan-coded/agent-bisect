from __future__ import annotations

import json
from pathlib import Path

from .ingest_claude import ingest_transcript
from .model import Activity, Journal


def load_activities(path: Path) -> list[Activity]:
    if looks_like_journal(path):
        return Journal.read_jsonl(path).activities
    return ingest_transcript(path)


def looks_like_journal(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return False
            return isinstance(data, dict) and {"activity", "prev_hash", "record_hash"}.issubset(data)
    return False
