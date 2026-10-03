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

# Documentation refactor: Stage 2 record

Stage 2 adds an eight-group [library reference](../api/index.md), source-backed API
inventory, generated namespace tables, and a static MkDocs/mkdocstrings build.
It corrects pending-error/input-validation docstrings and Stage 1's error-access
claim without changing runtime behavior. The README stays short and links onward.

## Stage 2 migration

[Stage 2 snapshots and path mappings](README.md#stage-2-path-mapping) preserve
the pre-edit checkout prose. Thin pointers at the invocation/redirect/execution
loop/orchestrator paths keep old links useful. Archives are excluded from site
output and search; source Markdown links are translated to repository links
when the destination is outside the built site. Detailed profiler interpretation
is retained but remains separate from the application reference.

## Stage 2 verification

Executed on macOS ARM64 using the existing application environment and a separate docs-only environment:

- Static inventory covers all **31 current root exports**, **53 source objects**, selected public methods/properties, all 12 configuration namespaces, supported example imports and explicit subpackage aliases/exclusions. Counts are derived, not acceptance constants.
- MkDocs **strict build passes** with mkdocstrings static inspection disabled. The isolated docs environment has neither OpenAI nor MCP installed; the build and check assert that Agency was never imported. No credentials, container startup, host provisioning or viewer build is required.
- **55 Python Markdown snippets parse** (including preserved snapshots); repository-local targets exist. **1,318 rendered local links/anchors** pass, including all selected generated object/member anchors. The archive is absent from site output.
- **713 passed, 95 skipped** in the focused existing result/config/agent/skill/tool/schema/type/workflow/context/resource/sandbox/checkpoint/orchestration/redirect/profiler/Web UI tests. These test fixture or mocked infrastructure contracts, not live provider behavior.
- Example tests: **15 passed, 1 skipped, 1 pre-existing failure**, the 01–10 numbering assertion against this checkout's existing 11/12 lessons. No runtime/test expectation was changed to hide it.
- Seven complete model-free reference examples executed successfully (validation, tools/policy, mapping/team composition and pending errors). A separate synthetic cancellation reproduced wrapper-class retention and AttributeError versus direct AgError behavior.
- Ruff check/format pass for new docs tooling; runtime ASTs in all four edited Python files are unchanged after removing docstrings. README remains under 600 words. Earlier dirty runtime/test/example changes were preserved.

Source-only inspection covered missing automatic input validation, initial resource-config propagation, executor resizing boundaries, tool/flush deadline omissions, host versus sandbox dispatch, config ownership, save/load credentials and backend portability restrictions. Their evidence and priorities are in [known issues](../api/contracts.md).

Live native Linux x86-64/model execution, real privileged ZFS/CRIU checkpoints, cross-host restores and actual upstream Perfetto download/build remain unverified on this host. Documentation CI uses only the locked docs group and runs independently of expensive runtime integration tests.

Tooling configuration was checked against the official [MkDocs configuration guide](https://www.mkdocs.org/user-guide/configuration/) and [mkdocstrings-python options](https://mkdocstrings.github.io/python/usage/configuration/), plus the installed locked versions. Documentation dependencies and lockfile were updated through uv.
