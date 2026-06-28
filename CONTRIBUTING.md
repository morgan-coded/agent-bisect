# Contributing

Keep changes focused on public transcript analysis: ingest adapters, deterministic gates, localization, coverage reporting, and tests.

Before opening a change:

```powershell
python -m pytest -q
```

Do not commit generated reports, local datasets, virtual environments, or journals. Keep fixtures synthetic or from clearly public third-party sources, and avoid machine-local paths in committed test data.

