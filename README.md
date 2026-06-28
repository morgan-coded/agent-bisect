# agent-bisect

`agent-bisect` is a small CLI for finding the first deterministic break in an agent run. It ingests an agent transcript, normalizes tool activity into a journal, runs deterministic gates, localizes the earliest breaking step, and reports the coverage gaps it cannot inspect.

The core loop is:

```powershell
agent-bisect ingest tests/fixtures/claude_sanitized.jsonl --out ./example.journal.jsonl
agent-bisect localize ./example.journal.jsonl
agent-bisect replay ./example.journal.jsonl --explain
```

The Claude transcript adapter supports JSONL transcripts such as:

```text
~/.claude/projects/<project>/<session>.jsonl
```

Foreign-schema adapters are included for SWE-agent, mini-swe-agent, and OpenHands trajectories:

```powershell
agent-bisect ingest-foreign --schema swe-agent tests/fixtures/swe_agent_function_call.traj --out ./swe-agent.journal.jsonl
agent-bisect sweep-foreign --schema mini-swe-agent tests/fixtures/mini-swe-agent-github-issue.traj.json
```

## What It Checks

- G1 validates that normalized activities have the required structural fields.
- G2 checks edit causality against earlier full-content anchors in the same run.
- G3 parses recorded test/build results deterministically.
- Localization reports the first breaking step, confidence, cascade, and coverage.

The tool verifies the deterministic envelope: recorded file/tool/test effects. It does not replay the model, infer hidden intent, or claim visibility into opaque shell commands that the transcript did not structure.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\agent-bisect --help
```

Runtime code uses only the Python standard library. The development extra installs pytest for tests.

## Commands

- `ingest`
- `ingest-foreign`
- `show`
- `localize`
- `replay`
- `eval`
- `scan`
- `sweep-foreign`
- `fetch-swe-agent-trajectories`
- `fetch-openhands-realtask-trajectories`

## Test

```powershell
python -m pytest -q
```

