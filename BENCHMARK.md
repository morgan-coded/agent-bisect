# Who&When Localization Benchmark

## Headline

This is an honest coverage-limited benchmark of `agent-bisect` against Who&When labels. It is not an apples-to-apples comparison to the published Who&When LLM attribution baseline.

| metric | result |
| --- | ---: |
| exact-step accuracy | 0/181 (0.000) |
| cascade-membership accuracy | 0/181 (0.000) |
| coverage-gap rate | 181/181 (1.000) |
| rows processed | 184 |
| included labels | 181 |
| excluded labels | 3 |

## Confidence Split

| confidence | exact correct | total | exact accuracy | share of labels |
| --- | ---: | ---: | ---: | ---: |
| HIGH | 0 | 0 | NA | 0.000 |
| LOW | 0 | 0 | NA | 0.000 |
| NA | 0 | 181 | 0.000 | 1.000 |

## By Config

| config | exact | cascade | coverage gaps | labels |
| --- | ---: | ---: | ---: | ---: |
| Algorithm-Generated | 0/126 (0.000) | 0/126 (0.000) | 126/126 (1.000) | 126 |
| Hand-Crafted | 0/55 (0.000) | 0/55 (0.000) | 55/55 (1.000) | 55 |

## Excluded Labels

| reason | count |
| --- | ---: |
| mistake_step_out_of_range | 3 |

## What Was Scored

Dataset: `Kevin355/Who_and_When` at revision `59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c`.

Manifest hash: `84bb9d16f34f36d16e145854aeb27c65d7a7a732848ea5a0bcd8933959c8e2af`.

Who&When labels `mistake_agent` and `mistake_step` over multi-agent conversation histories. This benchmark maps each history item one-to-one into an `Activity`, with `mistake_step` treated as a zero-based `Activity.step_index`.

The adapter does not infer hidden file edits, commands, exit codes, or test failures from free text. Agent messages are preserved as `llm_call`/`user_msg`; terminal/computer records without recoverable commands become explicit `unmapped` activities.

## Adapter Coverage

| activity kind | count | share |
| --- | ---: | ---: |
| llm_call | 3692 | 0.902 |
| unmapped | 342 | 0.084 |
| user_msg | 58 | 0.014 |

Rows: 184; activities: 4092; unmapped activities: 342.

## Baseline Context

The Who&When paper reports a best semantic step-level attribution result of about 14.2% for GPT-4o Step-by-Step judging, from the cells 25.51%, 7.02%, 15.31%, 8.77%. That method judges multi-agent natural-language logs and sometimes has final-answer ground truth. `agent-bisect` instead localizes deterministic gate-visible breaks in normalized transcripts.

Because the Who&When histories do not expose deterministic file/test gate failures in the format `agent-bisect` requires, the clean comparison is ill-posed. The result above is a standalone visibility result over the full labeled set, not an `Nx better` claim.

## Lineage And License

Sources: [Who&When on Hugging Face](https://huggingface.co/datasets/Kevin355/Who_and_When), [paper](https://arxiv.org/abs/2505.00212), and [GitHub repository](https://github.com/ag2ai/Agents_Failure_Attribution).

Raw Who&When rows are fetched from source into ignored local `data/` only. They are not committed. The Hugging Face dataset card does not declare a dataset license; the GitHub repository is MIT-licensed, and the dataset is based on GAIA and AssistantBench tasks, so this repository ships only code, synthetic tests, and aggregate results.

## Determinism

The scorer sorts labels and predictions deterministically and writes canonical JSON to `reports/who-when-benchmark.json`. Re-running over the same manifest should produce byte-identical JSON.
