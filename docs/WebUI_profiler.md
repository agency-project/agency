# Profiler in the Agency web UI

The **Profiler** tab embeds actual Perfetto, served by the same FastAPI server
as the run dashboard. The local build changes the flow renderer to connect
source-span starts to destination-span starts, including across tracks.
Exported traces stay compatible with stock Perfetto. Its public website still
uses its own arrow rendering.

## Automatic build

Starting `agwebui.run(...)` or the standalone web UI server builds Perfetto
when assets are missing or stale. Current assets are reused without network
access or a build. The first build requires internet access and may take
several minutes; progress appears in the terminal. Startup stops with an
error if the build fails, and can be retried. Concurrent starts share a build lock.
The cache checks the pinned revision, builder/patch contents, and asset presence.

To build manually from the repository root:

```sh
python -m agency.observability.agwebui.build_perfetto
```

This downloads the pinned upstream Perfetto source and build dependencies,
builds the UI and WebAssembly trace processor, then installs the generated
assets in `agency/observability/agwebui/static/perfetto/`. Source and generated
assets are git-ignored. The upstream revision is pinned in the script, and
its Apache license is retained with the assets. No separate viewer server runs.
Use `--force` to rebuild a current viewer. `--skip-deps` skips dependency installation when the dependencies are already available.

## Open a trace

Start applications with `agwebui.run(...)`, then click **Profiler**. The web UI
uses the same profiler output directory as the application, including an
`AGENCY_PROFILE_DIR` override. Traces are available after the profiler writes
them; click **Reload trace** after completion. This is a completed-trace viewer,
not live streaming of in-progress spans.

For a saved run:

```sh
AGENCY_PROFILE_DIR=/path/to/profiler/output \
python -m agency.observability.agwebui.server \
  --run-dir /path/to/run/logs \
  --port 7860
```

Without `AGENCY_PROFILE_DIR`, the standalone server uses the `profiler` directory beside `run-dir` (normally `run-id/logs` and `run-id/profiler`).
You can also choose **Open trace file** to load another trace directly from
your browser. Files chosen this way are passed to the embedded viewer in
memory, not uploaded to a separate service.

Search, zoom, track navigation, SQL queries, and span arguments are Perfetto's
own UI. Select a span to display its connected flow arrows. Switching back to
**Run** preserves the loaded viewer. Reloading a trace creates a fresh viewer.
