# Documentation refactor: Stage 1 record

Stage 1 adds a native first run, contributor setup, documentation navigation,
and profiler startup instructions. It changes documentation and the new
`examples/quickstart.py` only. Existing runtime, APIs, licensing, packaging
metadata, tests, and earlier archives remain unchanged by this stage.
The [archive mapping](README.md#stage-1-path-mapping) preserves the
pre-stage landing pages and observability instructions, including existing
working-tree edits.

## Brief repository review

Reviewed official repositories during this implementation (October 2, 2026):

| Source | Pattern applied |
| --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | Define the project concretely, then route installation, quickstart, and contributions to dedicated pages. |
| [uv](https://github.com/astral-sh/uv) | Put runnable commands next to their expected outcome; keep detailed setup in linked documentation. |
| [Browser Use](https://github.com/browser-use/browser-use) | Show credentials alongside a complete program and distinguish execution paths. Agency presents native first. |
| [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell), observed on [GitHub Trending](https://github.com/trending) | State host/runtime prerequisites before commands and provide task-oriented next steps. |

Contribution entry points were also reviewed: vLLM links its developer guide,
uv links its contributor guide, and OpenShell separates first-contributor
workflow from setup. Agency supplies its own short setup and verification
workflow. No popularity, performance, or release claims were adopted.

## Verification

- Inspected packaging, CI, examples, installation/host tooling, benchmark and
  test structure. Traced the first-run behavior through config namespaces,
  `Agent.run`, skill submission, pending `agdata`, native harness execution,
  container runtime detection, GPU flags, dependency bootstrap, and cleanup.
- Inspected `agwebui.run`, standalone server arguments/bind, profiler
  environment handling/output paths, trace reload controls, and Perfetto builder.
- The README is 582 words including code, with one complete Python example.
  AST comparison confirms that its workload matches `quickstart.main()`.
- Quickstart imports without credentials; Python syntax and Ruff lint/format
  checks pass. Synthetic execution of both CLI branches verifies native/CPU
  configuration and explicit waiting before output access or workload return.
- All local Markdown links and heading anchors in the root README,
  CONTRIBUTING, example index, and `docs/` were checked, including snapshots.
- `uv sync --locked --python 3.12 --dry-run --offline` resolves the source
  installation. The contributor sync command also passes an offline dry run.
  This is not a fresh installation on Linux. Command help was executed for
  quickstart, the standalone server, Perfetto builder, tutorial runner, model
  smoke tool, and SWE-bench example. Checks used the existing environment
  with `uv run --no-sync` and a writable temporary uv cache.
- Existing focused checks: **161 passed, 8 skipped, 1 pre-existing failure**
  across `test_agdata`, `test_agconfig`, `test_agent`, attribution smoke, and
  example tests. Web UI/build checks: **81 passed** across
  `test_build_perfetto`, `test_agwebui_hooks`, and `test_agwebui_server`.

## Follow-ups

- `tests/test_examples.py::test_tutorial_is_numbered_and_documented` assumes
  only lessons 01–10; this checkout already includes 11 and 12. Update the
  numbering expectation and review the unfiltered `run_all.py` live test,
  because lesson 12 needs explicit arguments. The new index lists all examples.
- The numbered helper defaults to Codex and leaves GPU passthrough enabled.
  Decide its CPU policy separately; Stage 1's native quickstart disables
  passthrough explicitly and sets an initial one-CPU sandbox limit.
- This verification host is **macOS ARM64**. A live native container run,
  provider/model access, fresh Linux dependency installation, and actual
  Perfetto download/build and completed-trace display remain unverified.
  No provider calls, privileged host provisioning, or remote host runs were
  performed. Web UI tests use fixtures; their success does not prove a live
  model run or upstream asset build.
