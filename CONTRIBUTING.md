# Contributing to Agency

Start with the [quickstart](docs/getting-started.md) to understand an agent,
a skill, and a pending result. Documentation corrections, small runnable
examples, and focused bug reproductions are useful starting points.

## Set up

Use Python 3.12+ and uv. Live agent runs need Linux x86-64 and a local Docker
or Podman engine; ordinary unit tests use doubles and can run without API keys.
From your checkout:

```bash
uv sync --locked --extra dev --python 3.12
uv run pre-commit install
```

## Check your change

Run the tests closest to what you changed. For example:

```bash
uv run pytest tests/test_agdata.py tests/test_agconfig.py
uv run pytest tests/test_examples.py -k 'example_imports or imports_every_public_api'
uv run ruff check examples/quickstart.py
uv run ruff format --check examples/quickstart.py
```

Before submitting, run the full checks used by [CI](.github/workflows/tests.yml):

```bash
uv run pytest
uv run pre-commit run --all-files
```

Pre-commit can modify files; review the resulting diff. Some tests require
Linux, container images, or installed harness binaries. Live tests have
explicit environment opt-ins in their test files; report skips and setup
limitations. Model smoke tests make paid API calls. Optional checkpoint host
provisioning is unnecessary for unit tests or the native quickstart.

## Submit a contribution

1. Describe the problem and check for related work in the repository's issues
   and pull requests. Discuss broad API or architecture changes before building them.
2. Work on a branch or fork. Keep the change focused and add a pytest regression
   test for runtime fixes; update examples and docs when behavior changes.
3. Open a pull request explaining the resulting behavior, relevant tests, and
   any checks you could not run. Include a small reproduction for bug fixes.

Use the current implementation when older notes disagree. Keep code readable,
avoid unrelated formatting changes, and never include credentials in examples
or test fixtures. When replacing documentation, preserve the prior page under
[`docs/archive/`](docs/archive/README.md) and update its path mapping.

## Where to look

- `agency/agent.py`, `agskill.py`, and `agdata.py`: task submission and results.
- `agency/native_harness/`, `harness/`, and `llm/`: execution and model backends.
- `agency/sandbox/` and `orchestrator/`: containers and coordination.
- `agency/observability/`: logs, profiler, and browser UI.
- `tests/`: corresponding unit and integration coverage.
- [Examples](examples/README.md) and [documentation](docs/index.md): user workflows.

Benchmarks remain under `benchmarks/`; their READMEs describe their own setup
and measurement requirements. See [development notes](docs/guides/development.md)
and the [Stage 1 verification record](docs/archive/documentation-refactor.md) for
current documentation follow-ups.

## Documentation

Run the [isolated documentation checks](docs/guides/development.md#documentation-build)
before changing the API reference. The strict MkDocs build uses static source
extraction and runs without installing Agency's runtime dependencies.
