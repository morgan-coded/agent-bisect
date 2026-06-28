# Design

`agent-bisect` treats an agent run as an ordered stream of activities. The useful public claim is narrow: given a recorded transcript, the tool checks deterministic evidence in that transcript and localizes the first visible break.

## Honest Determinism

The CLI verifies the deterministic envelope: structured file edits, tool calls, recorded command exits, and recorded test output. It does not replay model reasoning or re-run the original agent. If a transcript contains an opaque shell command, an unmapped record, or a step that cannot be linked to its cause, the report keeps that gap visible instead of pretending the step was inspectable.

## Gates

G1 validates activity shape. G2 checks edit causality against earlier full-content anchors from the same run. G3 parses recorded test/build output. Each gate returns `PASS`, `FAIL`, or `NA`; `NA` is a coverage result, not a hidden pass.

## Localization

The localizer walks gate failures back through parent-step links and reports the earliest breaking step it can justify. `HIGH` confidence means the break is directly tied to a structured target step. `LOW` confidence means the break is visible but the causal chain has gaps.

## Evaluation

The eval path uses injected faults over clean runs because most real agent transcripts do not come with step-level labels. Mutation testing gives known ground truth, repeatable denominators, and an honest way to measure precision, recall, and localization confidence.

## Boundaries

This project is not a durable-execution runtime, scheduler, or long-term state system. It is a transcript analysis tool. Its output is strongest when the source transcript records structured actions and weakest when the source format collapses work into opaque text.

