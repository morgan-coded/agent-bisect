# agent-bisect demo transcript

Replay it yourself:

```powershell
pip install -e .
agent-bisect demo
```

asciinema was attempted first, but the Windows Python environment lacks POSIX PTY support. This transcript is the fallback recording artifact; `agent-bisect demo` is the source of truth.

```text
agent-bisect demo
git-bisect for agent runs: localize the first visible break, or abstain.

[1/2] WIN: HIGH-confidence localization on a shipped fixture
$ agent-bisect ingest tests/fixtures/localize_planted_fault.jsonl --out demo/_replay/high.journal.jsonl
wrote 4 activities to demo/_replay/high.journal.jsonl
$ agent-bisect localize demo/_replay/high.journal.jsonl
breaking_step	gate	cascade	confidence	coverage	candidates
2	G2	3	HIGH	structured path	-
$ agent-bisect replay demo/_replay/high.journal.jsonl --explain
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

[2/2] CONTROL: clean fixture abstains instead of guessing
$ agent-bisect ingest tests/fixtures/claude_sanitized.jsonl --out demo/_replay/control.journal.jsonl
wrote 6 activities to demo/_replay/control.journal.jsonl
$ agent-bisect localize demo/_replay/control.journal.jsonl
status	no_break
$ agent-bisect replay demo/_replay/control.journal.jsonl --explain
agent-bisect replay --explain
run_id: claude_sanitized
activities: 6
kinds: file_edit=1 llm_call=1 opaque_shell=1 test_run=1 tool_call=1 user_msg=1
structured_fraction: 3/6 (0.500)
shell_target_coverage: steps_with_targets=0/2 added_edges=0
gate_tallies:
  G1: PASS=3 FAIL=0 NA=3
  G2: PASS=0 FAIL=0 NA=6
  G3: PASS=1 FAIL=0 NA=5
verdict: clean run

done: deterministic replay completed
```
