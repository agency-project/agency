# View runs

The browser dashboard shows agents, their context, and execution activity.
The profiler lets you inspect a trajectory, timings, resource use, and
comparisons between runs.

## Open a tutorial run

After running the [first example](../../examples/01_basic_agent.py), use the
artifact directory printed at the end. Replace the path below with its `logs`
directory:

```bash
uv run python -m agency.observability.agwebui.server \
  --run-dir runs/tutorials/YOUR_RUN/logs --port 7860 --no-perfetto
```

Open [the dashboard](http://localhost:7860/) or [the profiler](http://localhost:7860/profiler).
Keep the server running while viewing; stop it with Ctrl+C. You can start it
against the same log directory while an execution is still running.
`--no-perfetto` skips the optional raw trace viewer's download and build.

## Choose where your script writes logs

Use the model settings from the [README](../../README.md#basic-usage):

```python
from agency import Agent
from agency.configs.agconfig import agconfig, agentconfig

run_cfg = agconfig(cfg.llm, agentconfig(log_dir="runs/my-run/logs"))
worker = Agent(agconfig=run_cfg)
```

Run your skills on `worker`, then point the server's `--run-dir` at
`runs/my-run/logs`. Use a fresh directory for each run.

## Record a profile

With `worker`, `summarize`, and `agdata` from the README:

```python
from agency import agprof

with agprof.session("runs/my-run/profiler"):
    result = worker.run(summarize, agdata(topic="code review"))
    result.wait()
```

Waiting inside the session captures the complete execution. The session writes
`agprof.trace.json`, `summary.json`, and `summary.md` to that directory.
Tutorials disable automatic profiling; [example 09](../../examples/09_observability.py)
records a profile explicitly.

For more detail, see [profiler views and replay](../guides/profiling/views.md).
To use Perfetto, omit `--no-perfetto`; the first launch downloads and builds
the viewer. See [raw trace viewer setup](../guides/profiling.md#raw-trace-viewer).
