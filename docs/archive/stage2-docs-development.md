# Development

From the repository root:

```bash
uv sync --extra dev --python 3.12
uv run pre-commit install
uv run pytest
uv run pre-commit run --all-files
```

Pre-commit runs Ruff and file hygiene checks; it may fix files in place.
CI runs the same hooks and pytest. Most unit tests use model doubles;
live container/harness tests require their documented opt-in settings.

For example import and documentation coverage:

```bash
uv run pytest tests/test_examples.py
```

Operational references remain at [host setup](../guides/host-setup.md),
[checkpoint configuration](../guides/checkpoints.md), and
[raw trace viewer setup](../guides/profiling.md#raw-trace-viewer). Benchmark instructions stay in
their respective `benchmarks/` directories.

Previous documentation and validation reports are preserved in
[the documentation archive](README.md). Treat archived commands and
implementation descriptions as historical; use the current code when they differ.
