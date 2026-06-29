# Walkthrough

This walkthrough uses shipped sanitized fixtures. The examples are constructed, not taken from a local real transcript, so they are reproducible and contain only neutral `repo/...` paths and structural summaries.

## Caught Failure: HIGH Confidence

The fixture `tests/fixtures/localize_planted_fault.jsonl` represents a short run with a user step, a clean write, a bad edit, and a recorded failing test. The bad edit has a stale full-content anchor, so G2 catches the breaking step; the recorded test failure becomes the downstream cascade.

### Ingest

```powershell
agent-bisect ingest tests/fixtures/localize_planted_fault.jsonl --out demo.high.journal.jsonl
```

Output:

```text
wrote 4 activities to demo.high.journal.jsonl
```

### Localize

```powershell
agent-bisect localize demo.high.journal.jsonl
```

Output:

```text
breaking_step	gate	cascade	confidence	coverage	candidates
2	G2	3	HIGH	structured path
```

What this means:

- Breaking step `2` is the first visible deterministic break.
- Gate `G2` failed because the edit cannot be justified against the prior full-content anchor.
- Step `3` is the cascade: the later recorded test failure is downstream of the bad edit.
- Confidence is `HIGH` because the path is fully structured.

### Replay

```powershell
agent-bisect replay demo.high.journal.jsonl --explain
```

Output:

```text
agent-bisect replay --explain
run_id: localize_planted_fault
activities: 4
kinds: file_edit=2 test_run=1 user_msg=1
structured_fraction: 3/4 (0.750)
shell_target_coverage: steps_with_targets=1/1 added_edges=0
gate_tallies:
  G1: PASS=3 FAIL=0 NA=1
  G2: PASS=1 FAIL=1 NA=2
  G3: PASS=0 FAIL=1 NA=3
verdict: 1 break(s) localized (HIGH=1 LOW=0)
breaks:
  break 1:
    breaking_step: 2
    gate: G2
    activity: kind=file_edit tool=Edit target=repo/localized_demo.py
    cascade: 3
    confidence: HIGH
    coverage: structured path
```

The replay view stays structural. It does not print prompt text, edit bodies, commands, or full test output.

## Uncertainty Example: LOW Confidence

The fixture `tests/fixtures/shell_target_coverage.jsonl` shows a gate-visible break whose causal link depends on a conservative shell-target heuristic. The tool still reports the break, but it marks the path as `LOW` instead of claiming full structure.

```powershell
agent-bisect localize tests/fixtures/shell_target_coverage.jsonl
```

Output:

```text
breaking_step	gate	cascade	confidence	coverage	candidates
0	G3	1	LOW	1 unlinked step; 1 heuristic shell-target edge on path	0,1
```

```powershell
agent-bisect replay tests/fixtures/shell_target_coverage.jsonl --explain
```

Output:

```text
agent-bisect replay --explain
run_id: shell_target_coverage
activities: 2
kinds: test_run=2
structured_fraction: 2/2 (1.000)
shell_target_coverage: steps_with_targets=2/2 added_edges=1
gate_tallies:
  G1: PASS=2 FAIL=0 NA=0
  G2: PASS=0 FAIL=0 NA=2
  G3: PASS=0 FAIL=2 NA=0
verdict: 1 break(s) localized (HIGH=0 LOW=1)
breaks:
  break 1:
    breaking_step: 0
    gate: G3
    activity: kind=test_run tool=PowerShell target=shell
    cascade: 1
    confidence: LOW
    coverage: 1 unlinked step; 1 heuristic shell-target edge on path
    candidates: 0,1
```

That LOW label is part of the contract: `agent-bisect` keeps uncertainty visible instead of turning a heuristic path into a false-precise root cause.
