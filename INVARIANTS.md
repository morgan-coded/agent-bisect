# Agent-Bisect Invariants

- Journal genesis: the first record starts from `GENESIS_HASH`, giving every run a deterministic chain anchor.
- Journal chain integrity: each record commits to the previous record hash, making serialized journal tampering detectable.
- Activity content hash: activity hashes cover the deterministic activity payload and exclude the stored `content_hash`, so refreshes are stable.
- Canonical JSON: sorted keys and compact separators make equivalent dict payloads hash the same way across runs.
- Unicode JSON: canonical serialization preserves non-ASCII transcript content without corrupting the deterministic envelope.
- Ingest determinism: ingesting the same transcript twice yields identical activities, step indexes, parent links, and hashes.
- Tool-result linkage: `tool_use_id` resolution attaches recorded results to the same tool activities on repeated ingests.
- Gate determinism: G1, G2, and G3 return identical `GateResult` values for identical activity sequences.
- G1 fail-closed schema checks: malformed structured activities fail, while unstructured activity kinds stay `NA`.
- G3 fail-closed parsing: ambiguous test output stays `NA`, non-zero recorded exits fail, and only clear pass signals pass.
- Seeded synthetic coverage: fixed-seed activity sequences round-trip through journals and gates without nondeterminism.
