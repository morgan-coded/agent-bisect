from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ActivityKind = str

VALID_KINDS: set[ActivityKind] = {
    "user_msg",
    "llm_call",
    "tool_call",
    "file_edit",
    "test_run",
    "opaque_shell",
    "unmapped",
    "verdict",
}

ACTIVITY_HASH_FIELDS = (
    "run_id",
    "step_index",
    "ts",
    "kind",
    "tool_name",
    "inputs",
    "outputs",
    "target",
    "parent_step",
)

GENESIS_HASH = "0" * 64


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for hashing and JSONL output."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Activity:
    run_id: str
    step_index: int
    ts: str
    kind: ActivityKind
    tool_name: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    target: str | None = None
    parent_step: int | None = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"invalid activity kind: {self.kind}")
        if self.content_hash == "":
            self.content_hash = self.compute_content_hash()

    def hash_payload(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in ACTIVITY_HASH_FIELDS}

    def compute_content_hash(self) -> str:
        return sha256_text(canonical_json(self.hash_payload()))

    def refresh_hash(self) -> None:
        self.content_hash = self.compute_content_hash()

    def to_dict(self) -> dict[str, Any]:
        data = self.hash_payload()
        data["content_hash"] = self.content_hash
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Activity":
        return cls(
            run_id=str(data["run_id"]),
            step_index=int(data["step_index"]),
            ts=str(data.get("ts", "")),
            kind=str(data["kind"]),
            tool_name=data.get("tool_name"),
            inputs=dict(data.get("inputs") or {}),
            outputs=dict(data.get("outputs") or {}),
            target=data.get("target"),
            parent_step=data.get("parent_step"),
            content_hash=str(data.get("content_hash", "")),
        )


@dataclass(frozen=True, slots=True)
class JournalRecord:
    activity: Activity
    prev_hash: str
    record_hash: str

    @classmethod
    def build(cls, activity: Activity, prev_hash: str) -> "JournalRecord":
        activity.refresh_hash()
        payload = {"activity": activity.to_dict(), "prev_hash": prev_hash}
        record_hash = sha256_text(canonical_json(payload))
        return cls(activity=activity, prev_hash=prev_hash, record_hash=record_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "activity": self.activity.to_dict(),
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JournalRecord":
        return cls(
            activity=Activity.from_dict(data["activity"]),
            prev_hash=str(data["prev_hash"]),
            record_hash=str(data["record_hash"]),
        )


@dataclass(slots=True)
class Journal:
    records: list[JournalRecord]

    @classmethod
    def from_activities(cls, activities: Iterable[Activity]) -> "Journal":
        records: list[JournalRecord] = []
        prev_hash = GENESIS_HASH
        for activity in activities:
            record = JournalRecord.build(activity, prev_hash)
            records.append(record)
            prev_hash = record.record_hash
        return cls(records)

    @property
    def activities(self) -> list[Activity]:
        return [record.activity for record in self.records]

    def to_jsonl(self) -> str:
        lines = [canonical_json(record.to_dict()) for record in self.records]
        return "\n".join(lines) + ("\n" if lines else "")

    def write_jsonl(self, path: str | Path) -> None:
        Path(path).write_text(self.to_jsonl(), encoding="utf-8", newline="\n")

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "Journal":
        records: list[JournalRecord] = []
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                records.append(JournalRecord.from_dict(data))
            except KeyError as exc:
                raise ValueError(f"line {line_no} is not a journal record") from exc
        cls._validate_chain(records)
        return cls(records)

    @staticmethod
    def _validate_chain(records: list[JournalRecord]) -> None:
        prev_hash = GENESIS_HASH
        for index, record in enumerate(records):
            if record.prev_hash != prev_hash:
                raise ValueError(f"journal chain break at record {index}: prev_hash mismatch")
            expected = JournalRecord.build(record.activity, record.prev_hash).record_hash
            if record.record_hash != expected:
                raise ValueError(f"journal chain break at record {index}: record_hash mismatch")
            prev_hash = record.record_hash
