# Benchmark Integration

`benchmarks/` runs Agency agents against third-party agentic-coding benchmarks —
SWE-bench and Terminal-Bench — through a common interface, so adding a new
benchmark, or a new environment shape within one, doesn't require touching the
agent-running code itself.

## Core abstractions (`benchmarks/base.py`)

| Type | Role |
|---|---|
| `BenchmarkTask` | `task_id`, `benchmark`, `instructions`, optional `workspace`, `metadata` |
| `BenchmarkResult` | `completed` (agent finished without crashing), `summary`, `patch`, `metadata`, `elapsed_s` |
| `PreparedEnvironment` | `prompt_hint` + private `context`, handed from an `ExecutionEnvironment` to `AgentBackend` |
| `BenchmarkProvider` (ABC) | `load_tasks(limit)`, `evaluate(task, result) -> dict` |
| `AgentBackend` (ABC) | `run_task(task) -> BenchmarkResult` |
| `ExecutionEnvironment` (ABC) | `prepare(cfg, task)`, `collect_artifacts(prepared, sandbox)`, `cleanup(prepared)` |

`completed` is deliberately separate from pass/fail: whether the agent finished
without crashing is a backend concern, while whether it actually solved the
task is `BenchmarkProvider.evaluate()`'s job, often deferring to a benchmark's
own official harness.

## Module layout

```
benchmarks/
├── base.py                    # core abstractions
├── cli.py                     # CLI implementation (run-benchmark is a thin shim)
├── runner.py                  # provider <-> backend orchestration, crash-safe result flushing
├── backends/
│   └── agency.py              # AgencyBackend -- the one agent backend
├── environments/
│   ├── host_workspace.py      # bind-mounted host workspace (SWE-bench)
│   └── container_image.py     # task-owned container image (Terminal-Bench)
└── providers/
    ├── swe_bench.py           # SWE-bench adapter
    └── terminal_bench.py      # Terminal-Bench (Harbor format) adapter
run-benchmark                  # CLI entry point
```

## One agent backend, pluggable execution environments

SWE-bench and Terminal-Bench both run the identical Agency agent — what
differs is how a task's files reach the sandbox and how artifacts are pulled
back out afterward, which isn't something "which agent backend" should own.
So there is a single `AgentBackend` (`AgencyBackend`), and the difference is
expressed as an `ExecutionEnvironment` strategy:

| Provider | Environment | Shape |
|---|---|---|
| SWE-bench | `HostWorkspaceEnvironment` | task workspace is a host git checkout, bind-mounted into a generic sandbox image |
| Terminal-Bench | `ContainerImageEnvironment` | task defines its own container image; the sandbox runs *inside* that image |

`_select_environment(provider_name)` in `benchmarks/cli.py` maps a provider
name to its environment class. Adding a benchmark with a new environment
shape means adding one `ExecutionEnvironment` implementation, not a new
`AgentBackend`.

## `AgencyBackend.run_task()` call sequence

```
run_task(task)
  |- cfg = agconfig.clone()
  |- prepared = environment.prepare(cfg, task)        # mount workspace / set base image
  |- skill = agskill(system_prompt=... + prepared.prompt_hint,
  |                   output_schema=agdata(status=str, summary=str))
  |- ag = agent(agconfig=cfg)
  |- result = ag.run(skill, agdata(instructions=task.instructions)); result.wait()
  `- finally:
       |- artifacts = environment.collect_artifacts(prepared, ag.sandbox)   # sandbox still live
       |- ag.sandbox.destroy()                                             # explicit, not __del__/atexit
       `- environment.cleanup(prepared)
```

`collect_artifacts()` always runs before `destroy()` — `ContainerImageEnvironment`
needs the live sandbox to retag its checkpoint image before it disappears.
`destroy()` is called explicitly rather than left to `__del__`/`atexit`, which
don't run on SIGKILL; podman/Docker containers share a session-keyring quota
(see `Design_resource_control.md`) that a crashed sweep would otherwise leave
zombie containers exhausting over hundreds of sequential tasks.

## `HostWorkspaceEnvironment` (SWE-bench)

- Copies `task.workspace` to a temp dir per run (never mutates the original)
  and bind-mounts it at `/workspace` via `agSandboxConfig.add_mount()` —
  this must happen before the sandbox is first provisioned, since mounts
  resolve once on first use and can't be added to a live sandbox.
- `collect_artifacts()` runs `git diff` on that temp copy — the container's
  own overlay is discarded rather than committed, so extraction is host-side.
- `cleanup()` removes the temp dir.
- The sandbox image has no per-repo dependencies installed: the agent edits
  source files only, it does not install dependencies or run tests.
  Test-suite evaluation is delegated to the official SWE-bench harness.

## `ContainerImageEnvironment` (Terminal-Bench)

- Points the sandbox at `task.metadata["image"]` via
  `agSandboxConfig.set_base_image()` instead of the default sandbox image.
  This works because Agency's container backend always runs
  `<runtime> run ... <image> tail -f /dev/null` regardless of the image's own
  `CMD`, so any task image stays alive for exec the same way the default
  image does. (Gap: an image with its own `ENTRYPOINT` would wrap
  `tail -f /dev/null` rather than replace it — not handled.)
- No `/workspace` mount — the task's own image *is* the workspace. The
  prompt hint tells the agent its real starting directory is
  `task.metadata["workdir"]` and to pass `workdir=` explicitly.
- `collect_artifacts()` retags the sandbox's live checkpoint to a stable,
  predictable name before the caller destroys it, via
  `type(sandbox._backend).tag_image(...)` — dispatched through the sandbox's
  own backend class, the same pattern `agSandbox.fork()` uses (not the bare
  `agSandbox.tag_image()` forwarder, which is hardcoded to the container
  backend and would be wrong for a chroot-backed sandbox). The tag comes back
  as a plain string in `metadata["checkpoint_image"]`, not the live
  `agSandbox` object, since the runner needs to JSON-serialize
  `BenchmarkResult`.

## Providers (`benchmarks/providers/`)

### `SWEBenchProvider`

- Loads tasks from a local JSONL export (standard SWE-bench/HuggingFace
  shape); clones each repo at `base_commit` into a shared bare mirror plus a
  per-instance checkout.
- Scope: task loading, Agency execution (via `HostWorkspaceEnvironment`),
  patch generation, and prediction export — not SWE-bench correctness
  evaluation. `evaluate()` never infers `passed` from patch size or
  similarity to the gold patch; it stays `None`, since only the official
  harness (applying the patch and running the instance's real test suite)
  can determine resolution.
- `write_predictions(records, path, model_name_or_path)` writes the official
  SWE-bench evaluation harness's predictions format
  (`instance_id`/`model_name_or_path`/`model_patch`) — the bridge from this
  integration's output to that harness.

### `TerminalBenchProvider`

- Loads tasks in the Harbor task format (`instruction.md` + `task.toml` +
  `environment/` + `tests/`) — walks the tree for every `task.toml`, builds
  or pulls the task's image, resolves its `WorkingDir` for
  `ContainerImageEnvironment`'s prompt hint.
- `evaluate()` spins up a container from `result.metadata["checkpoint_image"]`,
  copies in `tests/`, runs `test.sh`, reads the reward file, then removes
  both the eval container and the checkpoint image — grading owns final
  cleanup here since it needs that image to still exist after `run_task()`
  returns.
- All subprocess calls go through `agency.agsandbox.get_container_runtime()`
  rather than hardcoding `docker`, matching how Agency's own sandbox
  backends pick a runtime — a podman-only machine is a supported case, not
  an edge case.

## `Runner` and CLI

- `Runner` iterates tasks, calls `backend.run_task()` then
  `provider.evaluate()`, and flushes each result to `results.jsonl`
  immediately — crash safety for long sequential sweeps. A task that raises,
  in the backend or in `evaluate()`, is recorded as a failed task rather than
  aborting the rest of the sweep.
- Every `evaluate()` implementation is required to return a
  `"passed": bool | None` key; `pass_rate` is computed only over tasks with a
  known value, so a pure SWE-bench sweep correctly reports `n/a` rather than
  `0%`.
- `run-benchmark` is a thin shim over `benchmarks/cli.py`'s `main()` (kept
  importable, rather than in the hyphenated script itself, so it's
  unit-testable). There is no `--backend` flag — `--provider` determines the
  `ExecutionEnvironment` via `_select_environment()`.

## Known limitations

- Terminal-Bench: single-container tasks only (no `docker-compose`
  multi-service environments); only `[verifier].environment_mode = "shared"`;
  `task.toml` resource limits (`cpus`, `memory_mb`, `gpus`) and `network_mode`
  are read but not enforced.
- SWE-bench: no correctness evaluation of its own —
  `write_predictions()` bridges to the official harness for that.
- Sequential execution only. Parallelism, when added, should use
  `agent.fork()`/`agteam` (see `Design_parallelization.md`) rather than a
  bespoke thread pool.
- `AgentBackend` is the extension point for other coding agents (Claude
  Code, Aider, ...); none are implemented yet.
