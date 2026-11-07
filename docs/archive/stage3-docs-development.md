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

## Documentation build

Use a separate environment so the build installs only documentation dependencies:

```bash
UV_PROJECT_ENVIRONMENT=.venv-docs uv sync --locked --only-group docs --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --no-sync --only-group docs python scripts/check_docs.py
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --no-sync --only-group docs mkdocs build --strict
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --no-sync --only-group docs python scripts/check_docs.py --site site
```

For local preview, run `UV_PROJECT_ENVIRONMENT=.venv-docs uv run --no-sync --only-group docs mkdocs serve`.
The build uses static mkdocstrings extraction; no Agency import, provider credentials,
containers, GPU setup or profiler UI build is required. `site/` is generated and ignored.
Docs CI runs separately from runtime tests.

When adding a public export, method, namespace or example import, update
[docs/api/public-api.json](../api/public-api.json) and its reference destination.
Use explicit member selections in Markdown. The source coverage check detects missing
exports/methods/imports; rendered links and anchors are checked after a strict build.
Namespace field tables are generated from dataclass ASTs. Runtime behavior descriptions
remain hand-written and must be traced through implementation/tests. Archive replaced
prose under `docs/old/` and update its migration index; it stays outside site search.
