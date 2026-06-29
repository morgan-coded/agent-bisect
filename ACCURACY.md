# Localization Accuracy

## Headline

This is a controlled injected-fault result over real ingested agent transcripts. It is a positive deterministic-envelope localization number, not a real-world attribution claim.

**Read the headline with the class mix in mind:** 87% of target injections (1000/1151) are the trivially-localizable `G1` schema class; only 90 are `G2` and 61 `G3`. So `0.943` is a G1-weighted average. Per class it is **G1 93.4% / G2 100% / G3 100%** (see the Confusion Matrix) — the realistic-fault classes (G2/G3) are perfect but small-n, and the overall number is dominated by the easiest class.

| metric | result |
| --- | ---: |
| controlled exact-step accuracy | 1085/1151 (0.943) |
| controlled cascade-membership accuracy | 1085/1151 (0.943) |
| HIGH exact-step accuracy | 1083/1083 (1.000) |
| HIGH share of target injections | 1083/1151 (0.941) |
| clean runs used | 6626 |
| excluded dirty runs | 115 |

## Pre-Registered Scoring Rule

- Include clean runs only for controlled injection; dirty runs are counted and excluded before sampling.
- Target classes are `G1_TARGET`, `G2_TARGET`, and `G3_TARGET`; non-target probes are `CONTROL` and `BENIGN`.
- Exact hit means a localized failure has the expected breaking step and expected gate.
- Cascade hit means the expected step is either the localized breaking step or a member of the failure cascade, with expected gate matching when known.
- `LOW` predictions count in all-confidence accuracy but are reported separately as coverage-limited.
- `NA` or no prediction is a miss for included cases, not a hidden pass.
- `CONTROL` and `BENIGN` measure false positives: any gate-status change is a false positive.

## Corpus Census

Runs considered: 6741; clean: 6626; dirty/excluded: 115.

| source | runs | clean | dirty | activities | G1 eligible | G2 eligible | G3 eligible | control eligible | benign eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| claude | 6286 | 6226 | 60 | 153228 | 17902 | 158 | 46 | 29869 | 46 |
| codex | 447 | 394 | 53 | 82302 | 2317 | 0 | 34 | 28471 | 34 |
| fixture | 5 | 3 | 2 | 25 | 9 | 1 | 3 | 2 | 3 |
| foreign-mini-swe-agent | 1 | 1 | 0 | 14 | 3 | 0 | 0 | 0 | 0 |
| foreign-openhands | 1 | 1 | 0 | 10 | 4 | 1 | 1 | 0 | 1 |
| foreign-swe-agent | 1 | 1 | 0 | 8 | 3 | 0 | 0 | 0 | 0 |
| all | 6741 | 6626 | 115 | 235587 | 20238 | 160 | 84 | 58342 | 84 |

## Confusion Matrix

| class | eligible-N | injected-N | scored-N | TP | FP | FN | NA | clean | precision | recall | false-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1_TARGET | 20238 | 1000 | 1000 | 934 | 0 | 66 | 0 | 0 | 1.000 | 0.934 | 0.000 |
| G2_TARGET | 160 | 90 | 90 | 90 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 0.000 |
| G3_TARGET | 84 | 61 | 61 | 61 | 0 | 0 | 0 | 0 | 1.000 | 1.000 | 0.000 |
| CONTROL | 58342 | 1000 | 1000 | 0 | 0 | 0 | 0 | 1000 | NA | NA | 0.000 |
| BENIGN | 84 | 61 | 61 | 0 | 0 | 0 | 0 | 61 | NA | NA | 0.000 |

## Localization Confidence

| confidence | exact correct | cascade correct | total | exact accuracy | cascade accuracy | share of targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HIGH | 1083 | 1083 | 1083 | 1.000 | 1.000 | 0.941 |
| LOW | 2 | 2 | 2 | 1.000 | 1.000 | 0.002 |
| NA | 0 | 0 | 66 | 0.000 | 0.000 | 0.057 |
| ALL | 1085 | 1085 | 1151 | 0.943 | 0.943 | 1.000 |

## Honest Two-Sided Result

The positive number above is controlled: target failures are injected into clean real ingested runs, so the breaking step is known by construction. It should be read as deterministic-envelope localization under known structured mutations.

The coverage companion is `STUDY.md`: across 6,735 real runs, only ~1.7% expose a deterministic gate-visible break at all. So this accuracy is measured on that thin gate-detectable slice — it says "when a break is visible to the gates, here is how precisely it is localized," not "agent-bisect localizes agent failures broadly." The three documents together are the honest picture: coverage (~1.7%, STUDY.md), accuracy on the covered slice (this), and the semantic boundary (0%, BENCHMARK.md).

The boundary companion remains `BENCHMARK.md`: Who&When exact-step `0/181`, cascade `0/181`, coverage gaps `181/181` over 181 included labels. That benchmark is semantic multi-agent failure attribution over natural-language histories, so the 0% result is a visibility boundary, not a contradiction.

## Published Baseline Context

These references are useful context, not direct apples-to-apples baselines for the controlled number above.

| reference | metric | caveat |
| --- | --- | --- |
| [TrajAudit / RootSE](https://arxiv.org/abs/2605.26563) | coding-agent failed-trajectory step localization; reported exact-step accuracy around 50-57% depending on reference availability | post-hoc LLM diagnosis over trajectories, not deterministic gate replay or controlled injected ground truth |
| [AgentRx](https://github.com/microsoft/AgentRx) | critical step-index accuracy and accuracy within step tolerance | step-localized trajectory diagnosis, but not a SWE-style deterministic file-edit/test-failure benchmark |
| [Who&When](https://github.com/ag2ai/Agents_Failure_Attribution) | semantic responsible-agent and step attribution over multi-agent histories | already reported in BENCHMARK.md as a visibility boundary for agent-bisect |

## External Dataset Decision

No public coding-agent dataset found in recon that is both step-localized and naturally aligned to deterministic file-edit/test gate failures.

| candidate | decision | reason |
| --- | --- | --- |
| TraceElephant | future adapter candidate, not primary | step/deceptive-action labeling is useful, but the benchmark is full-observability multi-agent attribution rather than gate-visible coding/test failures. |
| AgentRx | schema reference only | has step_number and failure_category labels, but is not a SWE-style coding/test-failure corpus. |
| SWE-agent/SWE-smith/Open-SWE traces | no-go as external real labels | trajectory/final-outcome data lacks first-bad-step labels; useful as future substrate for planted labels. |

## Caveats

- This is controlled ground truth, not real-world attribution accuracy.
- Injection targets structured eligible steps; this can overstate performance on opaque or poorly linked transcripts.
- Codex `apply_patch` activities are summarized as patches and are not G2 `Edit` anchors today, so G2 eligibility is mostly from Claude/foreign-style `Edit` records.
- Reports contain aggregate counts and hashes only; raw local transcripts and generated journals stay uncommitted.

## Determinism And Lineage

Seed: `260628`; per-class cap: `1000`; max per run: `3`.

The JSON report is sorted and deterministic for the same inputs. Mutation fingerprints store source, hashed run key, class, step, expected gate, mutation hash, and source-content hash only.
