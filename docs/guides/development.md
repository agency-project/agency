# Development

From the repository root:

```bash
uv sync --locked --extra dev --python 3.12
uv run pre-commit install
uv run pytest
uv run pre-commit run --all-files
```

Pre-commit runs Ruff and file hygiene checks; it may fix files in place.
CI runs the same hooks and pytest. Most unit tests use model doubles;
live container/harness tests require their documented opt-in settings.

The full `uv run pytest` command discovers the release suite under `tests/`,
as configured in `pyproject.toml`. Benchmark sources are maintained outside this
repository (locally in `~/Documents/benchmarks/`). Tests that import them skip
when the external package is unavailable. To include them, add the benchmark
folder's parent to `PYTHONPATH`; benchmark READMEs describe their separate tests.
The tool-annotation experiment tests are stored with the external benchmarks;
Agency's annotation unit tests remain under `tests/tool_annotation_effect/`.
Report existing skips and environment-dependent failures with the result;
passing unit tests does not validate opted-out live model or checkpoint tests.

For example import and documentation coverage:

```bash
uv run pytest tests/test_examples.py
```

Operational references remain at [host setup](host-setup.md),
[checkpoint configuration](checkpoints.md), and
[profiling and trace artifacts](profiling.md#raw-trace-artifacts). Benchmark instructions stay in
their external directories. `scripts/agency-bench` uses `~/Documents/benchmarks/`
by default; set `AGENCY_BENCHMARKS_DIR` to use a different location.

## JavaScript tests

With Node 24, run the profiler UI tests without installing npm dependencies:

```bash
node --test tests/agwebui/*.test.mjs tests/agwebui/test_live_follow.cjs
```

CI runs these tests separately from pytest.

## Documentation

Read the Markdown files directly on GitHub or in your editor. There is no
separate documentation site or Python documentation build environment.

When adding a public export, method, namespace or example import, update
the [API index](../api/index.md) and its reference page.
Keep source links, configuration field tables, and behavioral explanations in
sync with the Python implementation. Keep reader pages under guides, API, or
architecture, and documentation assets under `docs/assets/`.

## Architecture diagrams

Mermaid fences in architecture Markdown are the canonical editable diagrams.
They render directly on GitHub without local Node dependencies. Preview changes
on GitHub and review event arrows against source/tests. No architecture raster
image is maintained independently; any future image export should be regenerated
from the Markdown when the diagram changes.
