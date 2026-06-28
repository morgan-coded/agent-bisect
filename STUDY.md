# Corpus Study

Aggregate-only empirical profile over local real agent-run transcripts plus shipped foreign fixtures. This report contains no raw transcript content, commands, file paths, credentials, usernames, or per-run identifiers.

## Headline

| metric | result |
| --- | ---: |
| runs processed | 6735/6735 |
| source records processed | 323845 |
| normalized activities | 235360 |
| runs with gate-detectable break | 113/6735 (0.017) |
| no-break runs | 6622/6735 (0.983) |
| HIGH localized runs | 78/6735 (0.012) |
| LOW localized runs | 35/6735 (0.005) |
| linked action activities | 118824/151259 (0.786) |
| opaque or unmapped activities | 124942/235360 (0.531) |
| median per-run opaque/unmapped action rate | 1.000 |
| shell-target lift | 205/32505 steps; 30 added edges |

## By Source

| source | runs | records | activities | break runs | no_break | HIGH | LOW | linked actions | opaque/unmapped | shell targets | parse errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| claude | 6285 | 217644 | 153192 | 60/6285 (0.010) | 6225/6285 (0.990) | 44/6285 (0.007) | 16/6285 (0.003) | 78768/110748 (0.711) | 88222/153192 (0.576) | 127/11392 | 0 |
| codex | 447 | 106169 | 82136 | 53/447 (0.119) | 394/447 (0.881) | 34/447 (0.076) | 19/447 (0.043) | 40051/40498 (0.989) | 36717/82136 (0.447) | 74/21107 | 0 |
| foreign | 3 | 32 | 32 | 0/3 (0.000) | 3/3 (1.000) | 0/3 (0.000) | 0/3 (0.000) | 5/13 (0.385) | 3/32 (0.094) | 4/6 | 0 |

## Gate Failures

| scope | G1 fail steps | G2 fail steps | G3 fail steps | total fail steps |
| --- | ---: | ---: | ---: | ---: |
| overall | 0 | 203 | 316 | 519 |
| claude | 0 | 203 | 177 | 380 |
| codex | 0 | 0 | 139 | 139 |
| foreign | 0 | 0 | 0 | 0 |

## Activity Mix

| scope | file_edit | test_run | tool_call | opaque_shell | unmapped | user_msg | llm_call | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 4631 | 1112 | 20574 | 108752 | 16190 | 10985 | 70926 | 2190 |
| claude | 3501 | 529 | 18496 | 88222 | 0 | 7616 | 34828 | 0 |
| codex | 1126 | 580 | 2075 | 20527 | 16190 | 3363 | 36098 | 2177 |
| foreign | 4 | 3 | 3 | 3 | 0 | 6 | 0 | 13 |

## What This Means

The corpus profile is a visibility measurement, not a claim that every no-break run was successful. A no-break result means the transcript did not expose a deterministic G1/G2/G3 failure for `agent-bisect` to localize.

HIGH localized runs are the most inspectable slice: the gate-visible break is connected through structured transcript evidence. LOW localized runs are real gate-visible breaks whose causal path contains opaque, unmapped, unlinked, or heuristic shell-target evidence.

Opaque/unmapped and action-linkage rates show where coverage is lost in real transcripts. Shell-target lift reports how often conservative literal command parsing adds graph evidence; it is reported as lift, not ground truth.

## Privacy

Only aggregate counters and rates are committed. Raw corpus files are read-only, and generated JSON stays under ignored reports.
