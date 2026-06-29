# Design

`agent-bisect` treats an agent run as an ordered stream of recorded activities. Its public claim is intentionally narrow: given a transcript, check the deterministic evidence inside that transcript and localize the first visible break.

## Deterministic Envelope

The deterministic envelope is the part of a run the transcript records in a checkable form:

- structured file edits and tool calls
- parent-step links and file-target links
- recorded command exits
- recorded test or build output

The tool does not replay model reasoning, re-run the original agent, or infer hidden work from prose. It verifies the deterministic envelope and reports the gaps around it. That lets the non-deterministic core be tested indirectly: compare what the agent claimed or attempted against recorded, deterministic effects, without pretending the model itself can be replayed.

## Gates

- G1 validates normalized activity shape.
- G2 checks edit causality against earlier full-content anchors in the same run.
- G3 parses recorded test/build results.

Each gate returns `PASS`, `FAIL`, or `NA`. `NA` is a coverage result, not a hidden pass.

## Localization

The localizer walks gate failures backward through parent-step links and conservative file-target edges. It reports:

- the earliest visible breaking step
- the gate that failed
- downstream cascade steps
- `HIGH` confidence for structured paths
- `LOW` confidence when the path contains opaque, unmapped, unlinked, or heuristic evidence

The design is precision-first: report the smallest justified claim, and keep uncertainty visible.

## Evaluation Ground Truth

Most real agent transcripts do not include reliable step-level labels for the first deterministic break. The accuracy report therefore uses injected faults over clean real runs: the mutation supplies known ground truth while the surrounding transcript remains realistic.

That is why `ACCURACY.md` is a controlled result, not a real-world attribution claim. It measures how precisely `agent-bisect` localizes known structured breaks once they are visible to G1/G2/G3. `STUDY.md` supplies the coverage denominator for how often real runs expose such breaks at all, and `BENCHMARK.md` shows the semantic boundary where the tool abstains.

## Boundaries

This project is a transcript analysis tool. It is not a runtime, scheduler, state store, model replay system, or semantic judge. Its output is strongest when the transcript records structured actions and weakest when work is collapsed into opaque text.
