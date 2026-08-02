# Benchmark Integration Plan

## Goal

Build a portable runner that lets Agency agents execute tasks from SWE-bench and
Terminal-bench through a common interface, with clean extension points for adding
more benchmarks (and, within a benchmark, more environment shapes) later.

Branch: `sunga/benchmark-integration`

---

## Directory structure

```
benchmarks/
├── __init__.py
├── base.py                    # core abstractions
├── cli.py                     # CLI implementation (importable; run-benchmark is a shim)
├── runner.py                  # connects provider → backend, collects results
├── backends/
│   ├── __init__.py
│   └── agency.py              # AgencyBackend — the one agent backend
├── environments/
│   ├── __init__.py
│   ├── host_workspace.py       # bind-mounted host workspace (SWE-bench)
│   └── container_image.py      # task-defined container image (Terminal-Bench)
└── providers/
    ├── __init__.py
    ├── swe_bench.py            # SWE-bench adapter
    └── terminal_bench.py       # Terminal-Bench (Harbor format) adapter
run-benchmark                   # CLI entry point (shim over benchmarks/cli.py)
```

**Design revision (post-review):** the first pass modeled Terminal-Bench's
task-owned-container requirement as a second `AgentBackend`
(`AgencyContainerBackend`), alongside `AgencyBackend` for SWE-bench's
bind-mounted-workspace shape. Both ran the exact same Agency agent — the only
real difference was *how a task's files reach the sandbox and how artifacts
are pulled back out*, which isn't a backend concern. Collapsed to one
`AgencyBackend` plus an `ExecutionEnvironment` strategy (`HostWorkspaceEnvironment`
/ `ContainerImageEnvironment`), so adding a benchmark with yet another
environment shape doesn't mean adding another agent backend. See Step 2.

---

## Step 1 — `benchmarks/base.py` (core abstractions)

Two dataclasses:

- **`BenchmarkTask`** — `task_id`, `benchmark`, `instructions`, `workspace: Path | None`, `metadata: dict`
- **`BenchmarkResult`** — `task_id`, `completed: bool` (did the agent finish without crashing),
  `summary`, `patch: str | None`, `metadata: dict`, `elapsed_s: float`
- **`PreparedEnvironment`** — `prompt_hint: str`, `context: Any`. What an `ExecutionEnvironment`
  hands back to the backend for one task run: text to splice into the agent's system prompt,
  plus whatever private state the environment's own `collect_artifacts()`/`cleanup()` need.

  Note: `completed` is separate from pass/fail. Pass/fail is determined by `evaluate()`, not by the agent.

Three ABCs:

- **`BenchmarkProvider`** — `name`, `load_tasks(limit=None)`, `evaluate(task, result) -> dict`
- **`AgentBackend`** — `name`, `run_task(task) -> BenchmarkResult`
- **`ExecutionEnvironment`** — `prepare(cfg, task) -> PreparedEnvironment`,
  `collect_artifacts(prepared, sandbox) -> dict`, `cleanup(prepared)` (default no-op).
  Framework-agnostic like the rest of this file — no `agency` import here; `cfg`/`sandbox`
  are typed `Any` and only given real types by implementations.

---

## Step 2 — `benchmarks/backends/agency.py` + `benchmarks/environments/`

**`AgencyBackend(AgentBackend)`** (`backends/agency.py`) knows how to run an Agency agent and
nothing about how a task's environment is shaped:

1. Takes an `agConfig` and an `ExecutionEnvironment` at construction. Note: `agConfig` itself
   has no flat fields — sandbox/LLM/resource settings go through `agSandboxConfig`/
   `agLLMBackendConfig`/`agResourcePoolConfig` views passed into it.
2. Per task: clones the `agConfig`, calls `environment.prepare(cfg, task)` — the environment
   mutates `cfg`'s sandbox config (mount a workspace / set a base image, whichever it is) and
   returns a `PreparedEnvironment`.
3. Builds an `agskill` with an `output_schema` (e.g. `agdata(status=str, summary=str)`) and a
   system prompt that names the task, splices in `prepared.prompt_hint` (the environment's own
   text on where the agent's files live), and tells it to report its outcome when done (the
   framework auto-generates `return_<field>` tools from `output_schema` — no manual tool needed).
4. Runs `agent.run(skill, agdata(instructions=...))`. `run()` is non-blocking — it returns
   a pending `agdata` immediately. Timing must wrap `.wait()` (or a field access) on the
   result, not just the `run()` call itself, or elapsed time reads ~0.
5. In a `finally`: calls `environment.collect_artifacts(prepared, ag.sandbox)` **before**
   destroying the sandbox (some environments need the live sandbox — e.g. to retag a
   checkpoint image — before it's gone), then `ag.sandbox.destroy()`, then
   `environment.cleanup(prepared)`. `destroy()` is explicit here, not left to `__del__`/`atexit`
   — those don't run on SIGKILL, and podman/Docker containers share a session-keyring quota
   (`Design_resource_control.md`) that a crashed sweep would leave zombie containers eating
   into, over hundreds of sequential tasks.
6. Merges `artifacts["patch"]`/`artifacts["metadata"]` into the returned `BenchmarkResult`.

**`ExecutionEnvironment` implementations** (`environments/`) — each owns mounting, base-image
selection, checkpoint preservation, and artifact collection for one environment shape;
`AgencyBackend` never branches on which one it has:

- **`HostWorkspaceEnvironment`** (`host_workspace.py`) — used by SWE-bench. Copies
  `task.workspace` to a temp dir (isolation — never mutates the original), bind-mounts it at
  `/workspace` via `agSandboxConfig.add_mount()` (must happen before the sandbox is first
  provisioned — mounts resolve once on first use, can't be added to a live sandbox). After the
  run, `collect_artifacts()` shells out `git diff` on that temp copy to produce the patch (the
  container's own overlay is discarded, not committed, so extraction is host-side).
  `cleanup()` removes the temp dir.
- **`ContainerImageEnvironment`** (`container_image.py`) — used by Terminal-Bench. Points the
  sandbox at `task.metadata["image"]` via `agSandboxConfig.set_base_image()` instead of the
  default sandbox image — confirmed mechanically sound because Agency's container backend
  always runs `<runtime> run ... <image> tail -f /dev/null` regardless of the image's own CMD,
  so any task image stays alive for exec the same way. (Gap: an image with its own `ENTRYPOINT`
  would wrap `tail -f /dev/null` rather than replace it — not handled.) No `/workspace` mount —
  the task's own image IS the workspace, so the prompt hint instead tells the agent its real
  starting directory is `task.metadata["workdir"]` and to pass `workdir=` explicitly.
  `collect_artifacts()` retags the sandbox's live checkpoint to a stable, predictable name
  (`type(sandbox._backend).tag_image(...)` — dispatched by the sandbox's own backend, the same
  pattern `agSandbox.fork()` uses, not the bare container-only `agSandbox.tag_image()`
  forwarder) *before* the caller destroys the sandbox, and returns it via
  `metadata["checkpoint_image"]` — a plain string, not the live `agSandbox` object, which would
  leak a lock/backend-state object into a dataclass the runner needs to JSON-serialize.

Constraint (`HostWorkspaceEnvironment`): the sandbox image has no per-repo deps installed.
The agent edits source files only — it does not install dependencies or run tests.
Test-suite evaluation is delegated to the official SWE-bench harness.

**Both fully validated end-to-end with real live LLM calls (2026-07-29, under the prior
two-backend shape; the 2026-08 refactor preserved this exact runtime behavior — same
mount/base-image calls, same checkpoint-retag-before-destroy ordering — just moved which
class owns each piece):**

- *Host workspace path:* ran against a tiny local git repo with a real one-line bug, using
  claude-sonnet-5 via `agAnthropicBackendConfig`, from inside WSL with the sandbox image built
  via podman. The agent found the bug, fixed it, verified it, and reported success; the
  collected patch was exactly the correct one-line diff. `completed=True`, `elapsed_s=34.6`
  (confirms the `.wait()` timing fix is real, not ~0), `[DESTROYED]` cleanup log confirmed. One
  environment-only snag, not a code bug: this machine's podman tried to attach an NVIDIA GPU via
  CDI and isn't configured for it — worked around with `CUDA_VISIBLE_DEVICES=-1` to disable
  `agresources.detect_gpus()` (see `dev-environment-container-runtime` memory), no
  `benchmarks/` code involved.
- *Container-image path:* ran `TerminalBenchProvider.load_tasks()` → agent run →
  `TerminalBenchProvider.evaluate()` chained together for real, against a tiny Harbor-format
  task (write a specific file), using claude-sonnet-5. Task image built, agent correctly created
  the file and reported success (`completed=True`, `elapsed_s=22.8`), checkpoint image correctly
  retagged and passed through `BenchmarkResult.metadata["checkpoint_image"]`, `evaluate()`
  correctly spun up a grading container from it, ran `test.sh`, and reported
  `{"passed": true, "reward": 1.0, "test_exit_code": 0}`. No leftover containers or checkpoint
  images after the run — only the task's own cached build image remained, which is intentional.

**Re-confirmed live against the actual post-refactor classes (2026-08-02)**, not just inferred
from "the calls didn't change": ran both scenarios again through the real `AgencyBackend` +
`HostWorkspaceEnvironment`/`ContainerImageEnvironment` (throwaway script, not committed), same
WSL/podman setup. Host workspace: `completed=True`, `elapsed_s=23.3`, correct one-line patch.
Container image: `completed=True`, `elapsed_s=17.5`, checkpoint retagged and graded correctly
(`passed=True, reward=1.0`). `podman ps -a`/`podman images` confirmed no leaked eval containers
or checkpoint images afterward.

---

## Step 3 — `benchmarks/providers/swe_bench.py`

**`SWEBenchProvider(BenchmarkProvider)`**:

- Loads tasks from a local JSONL file (standard SWE-bench export). No hard `datasets` dep —
  user exports from HuggingFace separately.
- Each record: `instance_id`, `repo`, `base_commit`, `problem_statement`, `patch` (gold).
- `load_tasks()`: reads JSONL, clones repo at `base_commit` into a local cache dir,
  returns `BenchmarkTask` with that clone as `workspace`.

**Implemented scope, stated explicitly (post-review):** task loading, Agency execution
(via `HostWorkspaceEnvironment`), patch generation, and prediction export in the official
harness's format. **Not implemented:** SWE-bench correctness evaluation. Whether an issue
was actually resolved requires applying the patch and running the instance's real test suite
— that's the official harness's job, not this provider's.

- `evaluate()`: returns `{"passed": None, "note": "...", "patch_generated": bool,
  "lines_changed": int}`. `passed` is always `None` — deliberately never inferred from patch
  size or similarity to the gold patch, since neither is a resolution signal. `patch_generated`/
  `lines_changed` are diagnostic only (did the agent touch anything, roughly how much), not a
  pass/fail proxy.
- `SWEBenchProvider.write_predictions(records, path, model_name_or_path)`: writes the Runner's
  result records as `predictions.jsonl` in the shape the official SWE-bench harness
  (`python -m swebench.harness.run_evaluation --predictions_path ...`) expects — one JSON object
  per line with `instance_id`, `model_name_or_path`, `model_patch`. Run that harness against this
  file to get an actual resolved/unresolved verdict; this provider stops at producing the patch.

---

## Step 4 — `benchmarks/providers/terminal_bench.py`

**Verified against the real repo (2026-07-20) — an earlier format assumption was wrong.**
Terminal-Bench tasks are in the **Harbor** task format (the project's task spec now
lives at harborframework.com, linked from the terminal-bench README):

```
<org>/<name>/
├── instruction.md   # task instructions (plain markdown, not a task.toml field)
├── task.toml         # [task]/[environment]/[agent]/[verifier]/[solution]/[metadata]
├── environment/       # Dockerfile defining the task's OWN container
│                      #   (or a bare `docker_image` ref in [environment], no Dockerfile)
└── tests/             # copied into the container and run at grading time
```

This is a deeper mismatch than a field-name difference: each task defines its **own**
container environment, not a plain directory of files to bind-mount — handled by
`ContainerImageEnvironment` (Step 2), not by branching `AgencyBackend` or adding a second one.

**`TerminalBenchProvider(BenchmarkProvider)`**:

- `load_tasks()`: walks the tree for every `task.toml` (task_id = its path relative to
  the root, joined with `__`); builds `environment/Dockerfile` if present, else pulls
  `[environment].docker_image`; reads `instruction.md` for `instructions`; resolves the
  image's `WorkingDir` for `ContainerImageEnvironment`'s prompt hint.
- `evaluate()`: spins up a container from `result.metadata["checkpoint_image"]`,
  `docker cp`s the task's `tests/` in, runs `bash /tests/test.sh`, reads
  `/logs/verifier/reward.txt` or `.json`, then removes both the eval container and the
  checkpoint image (final cleanup ownership sits here, not in the environment/backend, since
  grading needs that image to still exist after `run_task()` returns).

**Explicitly scoped down** from Harbor's full spec (same spirit as SWE-bench's
patch-only eval — document the gap rather than chase full harness fidelity):
- Single-container tasks only — no `docker-compose.yaml` multi-service environments.
- Only `[verifier].environment_mode = "shared"` (tests run in the agent's own container,
  the default) — `"separate"` grading containers are not supported.
- `task.toml`'s resource limits (`cpus`, `memory_mb`, `gpus`, ...) and `network_mode`
  are read but not enforced — the agent runs under Agency's own sandbox defaults.

**Bug caught by testing against real podman (2026-07-20):** every subprocess call in
this file was hardcoded to `"docker"`. On a podman-only machine (no docker at all —
this dev box, via WSL2 Ubuntu + `apt install podman`) that's a hard failure. Fixed by
calling `agency.agsandbox.get_container_runtime()` (prefers podman when both are
present) instead of hardcoding a binary name.

**Mechanically validated end-to-end against real podman** (build → `inspect
WorkingDir` → `run ... tail -f /dev/null` keep-alive → simulated edit → `commit` +
`tag` + `rm`/`rmi` (the checkpoint-retag mechanism) → fresh eval container → `cp` tests
in → run `test.sh` → read `reward.txt` → cleanup), and **fully validated end-to-end with a
real live LLM call** chained through `ContainerImageEnvironment` (2026-07-29, details in
Step 2). Not yet validated: multi-container/`docker-compose` tasks (out of scope per above).

---

## Step 5 — `benchmarks/runner.py`

**`Runner`**:

- Takes a `BenchmarkProvider` + `AgentBackend`.
- `run(tasks)`: iterates tasks, calls `backend.run_task(task)`, then `provider.evaluate(task, result)`.
- Flushes each result to disk immediately after completion (crash safety) to
  `output_dir/results.jsonl`. A single task raising (backend or evaluate()) is caught
  and recorded as a failed task rather than aborting the rest of the sweep.
- Returns a report dict: metadata, per-task results, aggregate stats (pass rate, mean/p95 elapsed).
- `save_report(report, path)`: writes JSON + human-readable text summary.
- `run(tasks, on_result=callback)`: optional per-task callback, called right after each
  result is flushed — added so a CLI (or any caller) can print progress during a sweep
  that may run for hours, without `Runner` itself doing any printing.

**Cross-provider `evaluate()` contract**: neither provider's originally-planned metrics
(`patch_generated`/`lines_changed` for SWE-bench; `exit_code`/`output` for
Terminal-Bench) give the Runner a uniform signal to compute a "pass rate" from. Settled
on requiring every `evaluate()` to include a `"passed": bool | None` key alongside its
own metrics — `None` where genuinely undeterminable (SWE-bench's own `evaluate()` never
knows if the issue was actually fixed — see Step 3) rather than faking a boolean. `pass_rate`
is computed only over tasks with a known (`non-None`) `"passed"` value; for a pure SWE-bench
sweep this correctly reads `n/a`, not `0%`.

---

## Step 6 — `benchmarks/cli.py` + `run-benchmark` (CLI)

`run-benchmark` is a 3-line shim importing `benchmarks/cli.py`'s `main()` — the logic lives in
the importable module because the shim's hyphenated filename can't be imported by pytest, and
this also makes `_select_environment()` directly unit-testable.

There is one agent backend (`AgencyBackend`) — no `--backend` flag. Which `ExecutionEnvironment`
it runs a task inside is chosen from `--provider` via `_select_environment()`:
`ENVIRONMENTS = {"swe-bench": HostWorkspaceEnvironment, "terminal-bench": ContainerImageEnvironment}`.

Flags:
- `--provider swe-bench | terminal-bench`
- `--tasks` path to JSONL or task directory
- `--limit N`
- `--output path/to/report.json`
- `--cache-dir` (default `.benchmark_cache`) — passed through to `SWEBenchProvider`
  (its repo-mirror cache; unused by `terminal-bench`)
- `--predictions` — where to write the official-SWE-bench-harness-format predictions file
  (swe-bench only; default `<output_dir>/predictions.jsonl`)
- `--model-name-or-path` — identifier recorded in exported predictions (default: derived
  from the LLM env vars below)
- LLM config from env vars: if `ANTHROPIC_API_KEY` is set and `LLM_BASE_URL` is not,
  builds `agAnthropicBackendConfig` (talks to api.anthropic.com directly); otherwise
  builds `agVLLMBackendConfig` from `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` (any
  OpenAI-compatible endpoint) — same env vars `examples/base_example.py` already uses.

For `swe-bench` runs, after the report is written the CLI also calls
`SWEBenchProvider.write_predictions()` and prints where the predictions file went, along with
an explicit reminder that resolution status is unknown until the official harness runs against
it — the run's own console/report output never implies pass/fail for SWE-bench.

---

## Out of scope (explicit)

- Full SWE-bench test-suite evaluation (this integration produces patches + harness-format
  predictions; running the official harness against them is a separate, external step)
- Parallel task execution — sequential first. When added, use `agent.fork()`/`agteam`,
  Agency's existing fan-out primitive (`Design_parallelization.md`: fan-out is a list
  comprehension over forked agents) — not a bespoke thread pool in the runner.
- Claude Code / Aider backends (the `AgentBackend` ABC is the extension point)
- Per-repo dependency installation inside the Agency sandbox
- Multi-container/`docker-compose` Terminal-Bench tasks and `"separate"`-mode verifiers
