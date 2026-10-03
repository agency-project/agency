# Profiling and viewing runs

Use this guide to launch the dashboard, reopen logs, or record a profile.
The dashboard shows live work, the profiler investigates recorded activity,
and Raw trace opens finalized traces in Perfetto.

First complete [getting started](../getting-started.md). Profiling requires
Linux with readable `/proc` and a cgroup v2 hierarchy. OpenTelemetry is a core
package dependency; no profiling extra is needed.

## Open a run

From the checkout, with `OPENAI_API_KEY` set:

```bash
AGENCY_PROFILE=1 AGENCY_PROFILE_SCOPE=workload \
  uv run python examples/quickstart.py --webui
```

The implemented `--webui` option calls `agwebui.run(main)`. It starts the
server, then runs the same native task as the README. The workload calls
`result.wait()` before returning, so pending agent work is included in the trace.

Open [http://localhost:7860/](http://localhost:7860/) for the run dashboard.
Use **Raw trace** for the embedded Perfetto viewer, or open
[http://localhost:7860/profiler](http://localhost:7860/profiler) for the existing
profiler UI. The server binds to **0.0.0.0**, not only loopback; use an
appropriate host/network boundary. On a remote Linux host, forward port 7860
to your browser machine. A port already in use prevents `agwebui.run()` from starting.

## Raw trace viewer

Missing or stale Perfetto assets are built automatically **before** the server
starts. This needs Git, internet access, and a writable checkout/cache; allow
several minutes. The builder fetches pinned Perfetto sources, runs
`tools/install-build-deps --ui`, and builds the UI and WebAssembly trace processor.
Build dependencies are prepared by that upstream script. Current cached assets
are reused without a download or rebuild; no separate Perfetto server runs.

If the build fails, the Web UI does not start. Read the terminal build error
and retry after fixing the reported dependency or network problem:

```bash
uv run python -m agency.observability.agwebui.build_perfetto
```

The builder installs assets under
`agency/observability/agwebui/static/perfetto/`; its sources and generated
assets are Git-ignored. It retains upstream's Apache license. `--force`
rebuilds a current viewer; `--skip-deps` assumes its build dependencies are
already installed. Concurrent starts share a build lock.

Agency patches flow arrows to connect source-span starts to destination-span
starts, including across tracks. Exported traces remain compatible with stock
Perfetto, whose public viewer uses its own rendering. Search, SQL queries,
zoom and span arguments are Perfetto's UI. Switching back to Run preserves
the viewer; reloading a trace creates a fresh viewer.

## When the trace is ready

`AGENCY_PROFILE=1` enables environment profiling; `0` disables it. The default
scope is `workload`, which ends when the function passed to `agwebui.run()`
returns. The completed files are written to
`agency_runs/<run-id>/profiler/`, beside `logs/`:
`agprof.trace.json`, `summary.json`, `summary.md`, and `profile_data.sqlite3`.
The example prints that run's directory. `AGENCY_RUNS_ROOT` changes the root;
`AGENCY_PROFILE_DIR` overrides only the profiler output directory, which the
server also reads.

After the terminal says the workload is done, click **Raw trace → Reload trace**.
This viewer loads a completed file; in-progress spans are not streamed into
Perfetto. The dashboard stays available until Ctrl+C. If you choose
`AGENCY_PROFILE_SCOPE=process`, the final trace is written on process exit,
so reopen the saved run afterward. Use workload scope for this first run.
If profiling reports that startup failed and continues unprofiled, there will
be no new completed trace; fix the reported Linux/cgroup or installation issue.

## Reopen a saved run

Replace `YOUR_RUN` with the printed run directory:

```bash
uv run python -m agency.observability.agwebui.server \
  --run-dir agency_runs/YOUR_RUN/logs --port 7860
```

Open the same URLs and reload the trace. The standalone server reads the
sibling `profiler/` directory by default. If the run used a custom output
directory, supply that same path:

```bash
AGENCY_PROFILE_DIR=/absolute/path/to/profile \
  uv run python -m agency.observability.agwebui.server \
  --run-dir /absolute/path/to/run/logs --port 7860
```

For the existing dashboard and profiler UI without the Perfetto build, add
`--no-perfetto` to the standalone command; the embedded raw viewer then needs
already built assets. **Open trace file** in the raw viewer can also open a
saved `agprof.trace.json` directly from your browser.

## Record a profile

For your own application, choose a fresh agent log directory. Using `cfg`
from the [basic example](../../README.md#basic-usage), keep its CPU-only sandbox
settings and change only agent logging:

```python
from agency import Agent
from agency.configs.agconfig import agconfig, agentconfig

run_cfg = agconfig(cfg.llm, cfg.sandbox, cfg.resources,
                   agentconfig(log_dir="runs/my-run/logs"))
worker = Agent(agconfig=run_cfg)
```

With `task` and `agdata` from that example, record an explicit session:

```python
from agency import agprof

with agprof.session("runs/my-run/profiler"):
    result = worker.run(task, agdata(request="Write a Python function and test it."))
    result.wait()
```

Wait inside the session so it includes the complete execution. Reopen
`runs/my-run/logs` with the standalone server above. Tutorials disable automatic
profiling; [example 09](../../examples/09_observability.py) records an explicit
session. [Profiler APIs](../api/runtime.md#profiler-callables) define lifecycle
and return values; [collection architecture](../architecture/observability.md)
explains attribution and measurement limits.

## Inspect a run

Continue with [profiler views and interpretation](profiling/views.md),
[live trajectory and replay](profiling/live-trajectory.md), or the
[screenshot gallery](profiling/screenshots.md). These are details of the same
viewer, with recorded, inferred, synthetic and unavailable data kept distinct.
