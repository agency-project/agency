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

# Documentation refactor: Stage 3 record

Stage 3 adds an [architecture overview](../architecture/index.md), five focused
mechanism pages, six short engineering decision records and scoped measurement
evidence. The overview is 869 whitespace-delimited words including its diagram
(810 excluding it). Reconstructed rationale is explicitly engineering
interpretation; historical results and proposed contract changes remain labeled.

## Stage 3 migration and corrections

[Pre-edit snapshots and replacement mappings](README.md#stage-3-path-mapping)
preserve the checkpoint guide, earlier design notes, API claims and stage/index
records. Legacy design paths now point to current architecture; archives remain
outside site/search. The missing architecture PNG named in repository
instructions was not recreated: editable Mermaid fences are canonical and
verified against source/tests. Four diagrams show ownership, settlement,
restore fallback and observability boundaries, each with its guarantee/evidence.

The consistency pass corrected cancellation's rollback-notice claim, separated
missing-output repair from other validation failures, and documented native/PTY
inactivity as accepted partial output. Checkpoint documentation now separates
portable image export from private latest-only cow_zfs restore and optional
process acceleration. Profiler/session finalization, callback retirement,
capacity release and current pause/redirect limits are source-backed rather
than inferred from older diagrams.

The [measurement snapshot](../archive/measurement-evidence.json) records
existing artifact hashes/configuration/source identities and recomputed cohort
medians. Ignored checkpoint/trajectory raw artifacts were inspected locally;
their absence from clean checkouts and site output is explicit. Historical
checkpoint timings, trajectory summaries, blocked stress execution and a narrow
MCP factory benchmark establish no new performance or isolation claim.

## Stage 3 verification

Executed October 3, 2026 on macOS ARM64:

- Strict static MkDocs build passes in the isolated docs environment, without
  Agency imports or model/runtime dependencies. Source inventory still covers
  **31 root exports and 53 objects**. **62 Python snippets** parse, including
  archive snapshots; local source/Markdown destinations exist. **2,201 rendered
  links and anchors** pass, including this stage record. Archives and npm dependencies are
  excluded from site output.
- All **four canonical Mermaid diagrams** parse with locked Mermaid 12.1.0;
  the validator checks that the site's renderer uses the same version. All four
  also rendered as SVGs in the built local site using the in-app browser.
  GitHub-compatible fences remain editable in the repository. Site rendering
  loads a pinned CDN module and requires browser network access; the build is
  static. Documentation CI installs the locked npm dependencies and validates
  diagrams alongside existing strict build/link checks.
- **168 focused existing tests pass** across transaction fences, engine
  lifecycle, host services, PTY/native boundaries, fast resume, ZFS fallback and
  profiler lifecycle/accuracy. The first sandboxed run had 167 passes and one
  Unix socket bind permission failure; that test passed when rerun with socket
  access. These fixture/mocked tests resolve ordering and boundary questions,
  not privileged Linux runtime correctness.
- Ruff lint/format and JavaScript syntax checks pass for edited/new tooling.
  All 121 `agency/**/*.py` files match their Stage 3 starting SHA-256; Stage 3
  changes no runtime behavior and preserves earlier working-tree changes.

No provider calls, container benchmarks, Linux ptrace integration, ZFS/CRIU
host provisioning or external evaluator runs were performed. Earlier Stage 1/2
test limitations remain recorded above, including the pre-existing example
numbering failure.

## Maintainer decisions

[Open contracts](../api/contracts.md#stage-3-findings-requiring-a-code-decision)
record whether false commit returns should fail, whether adapter inactivity
should have a distinct status, whether native pause should suspend its progress
deadline, and what bounds to impose on abandoned RPC readers/callback drain.
Those changes need code decisions and targeted tests; this stage documents
current behavior without silently changing it.

# Structure consolidation: October 3, 2026

The [current migration map](README.md#current-layout-and-consolidation) records
the simpler reader/tooling layout. The root has only `index.md` and
`getting-started.md`, plus guides, API, architecture, assets and archive
directories. Profiling has one launch guide and linked detail pages; earlier
launch/viewer pages are preserved under `pre-consolidation/`. npm manifests
and generated dependencies live under `tools/docs/`.

Verification: locked npm dependencies installed from the existing local cache;
strict static build passes; all 31 root exports/53 source objects remain
covered; 64 Python snippets parse; 1,603 rendered links/anchors pass, including
generated redirects. All four diagrams parse and render as SVGs after the
asset move. Browser checks confirmed the four-section navigation and redirects
from the old observability, configuration and PTY pages. Ruff lint/format,
JavaScript syntax and diff whitespace checks pass. All 121 runtime Python
files match this consolidation's starting hashes; no runtime tests or live
provider/infrastructure runs were needed for the file moves and prose edits.
