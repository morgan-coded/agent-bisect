# Regex Oracle

This is a methodology-generalization demo: one deterministic-spec instance showing that the verification pattern travels outside agent transcripts, not the project's headline.

## Pairing

The harness treats Python `re.search` as the reference oracle and Python `re.fullmatch` as the candidate matcher configuration by default. Both sides compile the same pattern with Python `re`; the differential question is whether the candidate's match-mode semantics agree with the reference for each ordered input.

That pairing is intentionally small and deterministic. It creates real semantic divergences such as `pattern="cat"` on `input="concatenate"`: the reference finds the substring while the candidate requires the whole input to match.

## Input Order

Pattern sets are JSON files with a `patterns` list. Each entry has an `id`, a `pattern`, and optionally an ordered `inputs` list. Explicit inputs run first and are never shuffled. If inputs are omitted, or `generate` is set to `true`, the harness adds a bounded deterministic corpus from literals extracted from the pattern plus fixed seed strings. There is no randomness at import or runtime.

## Localization Mapping

Each input becomes a synthetic `test_run` activity. Agreement records `1 passed`; divergence records `1 failed`. The ordered inputs are parent-linked, so existing G3 gate parsing and `localize_failures` can report the earliest diverging input as the breaking step. The planted fixture localizes `planted-search-vs-fullmatch` to input index `1`, input `concatenate`, with G3/HIGH confidence.

## Limits

Python `re` is the oracle only for Python regex semantics. This harness does not prove POSIX, PCRE, Rust, JavaScript, or third-party `regex` engine behavior. Invalid patterns fail closed as unsupported. Catastrophic backtracking and ReDoS behavior are disclosed limits: the standard library does not provide a per-match timeout here, so pattern sets should avoid adversarial exponential cases. Agreement means the two configured match modes agreed on the tested finite input set, not on every possible string.

## Demo Summary

The checked fixture compares 2 patterns across 6 ordered inputs. It reports 5/6 agreements and 1/6 divergences. The first divergence is exact: `planted-search-vs-fullmatch`, input index `1`, input `concatenate`.
