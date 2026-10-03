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

Operational references remain at [host setup](host-setup.md),
[checkpoint configuration](checkpoints.md), and
[profiling and trace artifacts](profiling.md#raw-trace-artifacts). Benchmark instructions stay in
their respective `benchmarks/` directories.

Previous documentation and validation reports are preserved in
[the documentation archive](../archive/README.md). Treat archived commands and
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
prose under `docs/archive/` and update its migration index; it stays outside site search.
Keep reader pages under guides, API, or architecture; site assets belong under
`docs/assets/`, and npm tooling belongs under `tools/docs/`. Moved site pages
have URL mappings in `tools/docs/redirects.json`; the build generates redirects
without retaining duplicate source pages.

## Architecture diagrams

Mermaid fences in architecture Markdown are the canonical editable diagrams.
They render on GitHub; the documentation site uses the same pinned Mermaid
version checked by the parser. The site renderer loads Mermaid 12.1.0 from
jsDelivr, so browser rendering needs network access; the static build and parser
use no remote renderer or Agency runtime. If assets are unavailable, the
Markdown diagram source remains readable. No architecture raster image is
maintained independently.

With Node 24, install only the locked diagram-check dependencies:

```bash
npm ci --prefix tools/docs --ignore-scripts
npm run --prefix tools/docs check:diagrams
```

The checker parses every active Markdown Mermaid fence using Mermaid plus
jsdom (DOM support for parsing), excluding archives. CI runs it alongside the
isolated strict documentation build. Review event arrows against source/tests;
syntax validation alone cannot establish that a diagram describes the code.
Any future SVG/PNG export must be derived from these fences and regenerated
when the source changes. Integration follows [Mermaid's official usage API](https://mermaid.js.org/config/usage.html).
