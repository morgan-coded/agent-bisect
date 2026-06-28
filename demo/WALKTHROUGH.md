# Walkthrough

This synthetic walkthrough shows the shape of a small run and the kind of evidence `agent-bisect` reports. It uses generic paths and fixture-style activity, not a real private project transcript.

## Input

- run_id: `synthetic-demo-run`
- target: `repo/example.py`
- transcript: `tests/fixtures/claude_sanitized.jsonl`

The run contains a user message, a model step, a file read, a file edit, a recorded test command, and one opaque shell command. The structured steps are enough to show gate output and replay counts, while the opaque command remains a visible coverage gap.

## Ingest

```powershell
agent-bisect ingest tests/fixtures/claude_sanitized.jsonl --out ./demo.journal.jsonl
```

Expected shape:

```text
wrote 6 activities to demo.journal.jsonl
```

## Localize

```powershell
agent-bisect localize ./demo.journal.jsonl
```

For this clean fixture, the expected verdict is:

```text
status  no_break
```

## Replay

```powershell
agent-bisect replay ./demo.journal.jsonl --explain
```

The replay view reports the run id, activity counts, gate tallies, structured fraction, and localized break summary. It does not print raw prompt text, tool input bodies, or full command output.

