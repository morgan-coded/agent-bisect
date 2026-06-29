# Contributing

Keep changes focused on transcript analysis: ingest adapters, deterministic gates, localization, coverage reporting, and tests.

Before opening a change:

```powershell
python -m pytest -q
```

Do not commit generated reports, local datasets, virtual environments, or journals. Keep fixtures synthetic or from clearly public third-party sources, and avoid machine-local paths in committed test data.

When changing docs, keep measurement claims traceable to `STUDY.md`, `ACCURACY.md`, or `BENCHMARK.md`. If the tool cannot see a failure through G1/G2/G3, describe that as a coverage gap rather than guessing.

