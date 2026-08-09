"""
Profiler example (1 of 2): full-coverage profiling of the **native** engine.

This file and ``profiler_claude_code_example.py`` are deliberately the same
script -- same skills, same schemas, same phases, same report -- differing
only in ``ENGINE``. Run both against the same model and diff their
``summary.json`` to compare Agency's in-container ReAct loop against a
third-party CLI harness on identical work.

Native is the profiler's best case: it is the one engine that emits turn and
tool boundaries from *inside* the sandbox (Tier 3, exact), so ``turn{i}`` and
``tool:{name}`` spans here are measured, not inferred. See
``docs/Design_profiler_harness_integration.md`` §5.7 for the per-engine table.


What this exercises
-------------------
Every section ``summary.json`` can produce, and the phase that fills it:

  span_metrics       all phases -- one row per span label, with the
                     wall / cpu_ms / runqueue_ms / blocked_ms split
  run_metrics        phases 1-3 -- ``run{N}:{skill}:{agname}`` lane roots
  llm_metrics        all phases -- ``llm:attempt[0]`` spans minted by
                     agLLMTerminus, the single credentialed choke point every
                     engine's traffic passes through: exact TTFT, token
                     counts, attempts vs. retries
  tool_metrics       all phases -- ``tool:{name}``; **exact** for native
  sandbox_metrics    all phases -- per-container cgroup v2 sampling:
                     cpu, memory, io_read/io_write, **net_receive/net_transmit**
  process_metrics    phase 1-2 -- per-PID /proc sampling inside the workload
                     cgroup, plus ``process:{exe}`` spans that agProxyPtrace
                     records from execve/exit syscall events (Tier 2: needs no
                     harness cooperation, so it works identically on both
                     engines)
  workload_metrics   the aggregate cgroup rollup -- only populated when
                     AGENCY_PROFILE=1 re-execs into its own systemd slice
                     (see "Enabling" below)
  gpu_metrics /      NVML device sampling and ``sync:gpu_wait`` lease
  gpu_lease_metrics  intervals -- populated automatically on a GPU host
  resource_metrics   every raw sampler series as mean/min/max/last/total
  sampling           configured vs. effective Hz, GPU availability, raw
                     sample count -- read this before trusting anything else
  incomplete_spans   spans still open at session stop, reported with
                     ``outcome: "interrupted"`` rather than dropped


The network dimension
---------------------
Phase 1 exists specifically to move the network counters, and it is fussier
than it looks.

Network has no cgroup v2 controller, so agprof reads it from the container's
own netns via ``/proc/<pid>/net/dev`` -- per-container, and it excludes ``lo``.
Two consequences worth internalising:

  * The agent's **own LLM traffic is invisible here.** The harness talks to
    agProxyLLM over loopback, and agProxyLLM reaches agLLMTerminus over a
    bind-mounted unix socket. Neither crosses a counted interface. So a
    profiled agent run that only thinks and edits files reports ~0 net bytes,
    correctly. Use ``llm_metrics`` for model traffic, not net counters.
  * Only egress *the sandbox itself performs* shows up. That is why phase 1
    tells the agent to fetch over ``bash``. The host-side ``webfetch`` tool
    would satisfy the same prompt while bypassing the container's netns
    entirely -- a real and easy-to-miss way to measure nothing.

``NET_FETCH_URL`` overrides the default target. Pick something a megabyte or
so in size; a 2 KB response is inside the sampler's noise floor at 10 Hz.


Enabling
--------
The framework's spans are permanently present and compile down to a shared
``nullcontext`` when no session is active, so this script runs unprofiled with
no output and no measurable overhead. Two ways to turn it on:

    # Recommended -- one env var, no code change. Re-execs the process into a
    # dedicated systemd slice at import, so containers are parented under the
    # profiled cgroup and workload_metrics is populated. Needs passwordless
    # sudo for `systemd-run --scope`.
    AGENCY_PROFILE=1 AGENCY_PROFILE_DIR=./runs/profile \
        .venv/bin/python3 examples/profiler_native_example.py

    # Process-lifetime scope instead of the workload boundary below:
    AGENCY_PROFILE=1 AGENCY_PROFILE_SCOPE=process ... same command

``main()`` wraps its work in ``agprof.workload()``, which owns a session only
under the default ``workload`` scope and never stops one it did not start --
so both invocations above are correct and neither double-starts.

Requirements: **Linux only** (cgroups v2 + /proc), and the profiler extra::

    uv pip install -e ".[profiler]"     # opentelemetry-sdk, exporter, nvidia-ml-py

Artifacts land in ``AGENCY_PROFILE_DIR`` (default ``./agprof_trace``):

    summary.json        the complete metrics document (schema_version 4)
    summary.md          the same, rendered
    agprof.trace.json   Chrome-trace timeline -- drag into ui.perfetto.dev


LLM configuration
-----------------
Set ``LLM_BASE_URL`` / ``LLM_MODEL`` / ``LLM_API_KEY`` for any
OpenAI-compatible endpoint; with no ``LLM_BASE_URL`` this falls back to
Bedrock and picks up IAM / ``AWS_BEARER_TOKEN_BEDROCK`` from the environment.
The model needs tool-calling: a run with no tool calls leaves ``tool_metrics``
and the sandbox resource series empty, which looks like a profiler bug and
is not one.
"""

import os
from datetime import datetime
from pathlib import Path

from agency import AgError, agdata, agent, agprof, agskill, agsync
from agency.agconfig import agConfig
from agency.agllm_backends import agBedrockBackendConfig, agVLLMBackendConfig
from agency.agtype import agpath

# The only line that differs between this file and the Claude Code variant.
ENGINE = "native"

NET_FETCH_URL = os.environ.get("NET_FETCH_URL", "https://pypi.org/pypi/numpy/json")


if os.environ.get("LLM_BASE_URL"):
    cfg = agConfig(
        agVLLMBackendConfig(
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ.get("LLM_MODEL", ""),
            api_key=os.environ.get("LLM_API_KEY", ""),
            temperature=0.7,
            top_p=0.95,
        )
    )
else:
    cfg = agConfig(
        agBedrockBackendConfig(
            model=os.environ.get("LLM_MODEL", "us.anthropic.claude-sonnet-5"),
            region=os.environ.get("LLM_REGION", "us-east-1"),
        )
    )


def _make_run_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(__file__).parent.parent / "runs" / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Skills -- identical across both profiler examples
# ---------------------------------------------------------------------------

# Phase 1. Drives sandbox net_rx/net_tx, io_write, and (via execve) the
# ptrace-observed process spans. The "use bash, not webfetch" instruction is
# load-bearing: see the module docstring.
fetch_skill = agskill(
    name="net_fetch",
    system_prompt=(
        "You download files inside a Linux sandbox container.\n"
        "Use the bash tool and python3's urllib.request -- the container has "
        "python3 but no curl or wget.\n"
        "Do NOT use the webfetch tool: it executes on the host, outside the "
        "container's network namespace, and this task is measuring the "
        "container's own network usage.\n"
        "Download the URL to the given path, then report the exact byte count "
        "by running `stat -c %s <path>`."
    ),
    input_schema=agdata(url=str, dest_path=agpath),
    output_schema=agdata(status=str, path=agpath, bytes_downloaded=int),
)

# Phase 2. Drives container CPU and disk IO, and produces several turns and
# tool calls so the turn/tool latency distributions have more than one point.
analyze_skill = agskill(
    name="analyze",
    system_prompt=(
        "You analyse files inside a Linux sandbox container using the bash, "
        "read, and write tools.\n"
        "Prefer one bash command per step so each step is separately visible.\n"
        "Always verify a file you wrote by reading it back."
    ),
    input_schema=agdata(task=str, source_path=agpath, report_path=agpath),
    output_schema=agdata(status=str, report_path=agpath, summary=str),
)

# Phase 3. Runs on forked agents, each of which provisions its own container --
# so sandbox_metrics gains one row per fork and their spans interleave in the
# timeline instead of stacking.
note_skill = agskill(
    name="note",
    system_prompt=(
        "Write the given note to the given path with the write tool, then "
        "read it back to confirm. Keep it to one line."
    ),
    input_schema=agdata(note=str, file_path=agpath),
    output_schema=agdata(status=str, path=agpath),
)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value, unit: str = "") -> str:
    """None-safe formatter. An unmeasured metric is None, never 0 -- keep that
    distinction visible in the report, since a printed 0 reads as a finding."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.3f}{unit}"
    return f"{value:,}{unit}"


def _print_profile_report(profile_dir: Path) -> None:
    metrics = agprof.summary_metrics()
    if metrics is None:
        print(
            "\nNo profile was collected. Set AGENCY_PROFILE=1 (Linux only, with the\n"
            "`profiler` extra installed) to enable it -- see this file's docstring.\n"
        )
        return

    print("\n" + "=" * 78)
    print(f"agprof summary -- engine={ENGINE}  schema_version={metrics['schema_version']}")
    print("=" * 78)

    sampling = metrics["sampling"]
    print(
        f"\nsession {_fmt(metrics['duration_ms'], ' ms')}  |  sampler "
        f"{sampling['configured_hz']} Hz configured, {sampling['effective_hz']} Hz effective  "
        f"|  {_fmt(sampling['raw_samples'])} raw samples  "
        f"|  gpu requested={sampling['gpu_requested']} available={sampling['gpu_available']}"
    )

    runs = metrics["run_metrics"]
    print(
        f"\nRUNS      started={runs['started']} succeeded={runs['succeeded']} "
        f"failed={runs['failed']} interrupted={runs['interrupted']}  "
        f"p50={_fmt(runs['p50_ms'], ' ms')} max={_fmt(runs['max_ms'], ' ms')}"
    )

    llm = metrics["llm_metrics"]
    print(
        f"\nLLM       calls={llm['calls']} attempts={llm['attempts']} retries={llm['retries']}  "
        f"tokens in/out={llm['input_tokens']:,}/{llm['output_tokens']:,}  "
        f"out tok/s={_fmt(llm['output_tokens_per_second'])}"
    )
    print(
        f"          latency p50={_fmt(llm['latency']['p50_ms'], ' ms')} "
        f"p95={_fmt(llm['latency']['p95_ms'], ' ms')}   "
        f"TTFT p50={_fmt(llm['ttft']['p50_ms'], ' ms')} "
        f"p95={_fmt(llm['ttft']['p95_ms'], ' ms')}"
    )

    tools = metrics["tool_metrics"]
    print(
        f"\nTOOLS     started={tools['started']} succeeded={tools['succeeded']} "
        f"failed={tools['failed']}  p50={_fmt(tools['latency']['p50_ms'], ' ms')}"
    )
    for row in tools["by_tool"]:
        print(
            f"          {row['name']:<16} calls={row['completed']:<4} "
            f"p50={_fmt(row['p50_ms'], ' ms'):<14} max={_fmt(row['max_ms'], ' ms')}"
        )

    # The network capability. net_receive_mb / net_transmit_mb are integrated
    # from the container's own netns counters; loopback is excluded, so these
    # are strictly the sandbox's outbound work, not its LLM traffic.
    print("\nSANDBOXES (per-container cgroup v2 + netns sampling)")
    if not metrics["sandbox_metrics"]:
        print("          none -- no container was started, or cgroup registration failed")
    for row in metrics["sandbox_metrics"]:
        print(
            f"          {row['label']:<24} cpu avg/peak="
            f"{_fmt(row['cpu_average_percent'], '%')}/{_fmt(row['cpu_peak_percent'], '%')}  "
            f"mem peak={_fmt(row['memory_peak_mb'], ' MB')}"
        )
        print(
            f"          {'':<24} net rx/tx="
            f"{_fmt(row['net_receive_mb'], ' MB')}/{_fmt(row['net_transmit_mb'], ' MB')}  "
            f"disk r/w={_fmt(row['io_read_mb'], ' MB')}/{_fmt(row['io_write_mb'], ' MB')}"
        )

    workload = metrics["workload_metrics"]
    print("\nWORKLOAD  (aggregate cgroup -- empty unless AGENCY_PROFILE=1 owns the slice)")
    if workload:
        print(
            f"          cpu avg={_fmt(workload.get('cpu_average_percent'), '%')} "
            f"peak={_fmt(workload.get('cpu_peak_percent'), '%')}  "
            f"cpu time={_fmt(workload.get('cpu_time_seconds'), ' s')}  "
            f"mem peak={_fmt(workload.get('memory_peak_mb'), ' MB')}"
        )
    else:
        print("          not sampled")

    processes = metrics["process_metrics"]
    print(f"\nPROCESSES {len(processes)} observed (top 8 by cpu time)")
    for row in sorted(processes, key=lambda p: p["cpu_time_seconds"] or 0.0, reverse=True)[:8]:
        print(
            f"          {row['display_name']:<28} pid={_fmt(row['pid']):<8} "
            f"cpu={_fmt(row['cpu_time_seconds'], ' s'):<12} "
            f"rss peak={_fmt(row['rss_peak_mb'], ' MB'):<12} sandbox={row['sandbox']}"
        )

    if metrics["gpu_metrics"]:
        print("\nGPUS")
        for row in metrics["gpu_metrics"]:
            print(
                f"          gpu{row['gpu_id']}  util avg/peak="
                f"{_fmt(row['utilization_average_percent'], '%')}/"
                f"{_fmt(row['utilization_peak_percent'], '%')}  "
                f"mem peak={_fmt(row['memory_peak_mb'], ' MB')}  "
                f"energy={_fmt(row['energy_j'], ' J')}"
            )
        for row in metrics["gpu_lease_metrics"]:
            print(
                f"          lease gpu{row['gpu_id']} [{row['label']}] "
                f"n={row['leases']} total={_fmt(row['total_ms'], ' ms')}"
            )

    if metrics["incomplete_spans"]:
        print(f"\nINCOMPLETE ({len(metrics['incomplete_spans'])} spans open at session stop)")
        for span in metrics["incomplete_spans"][:8]:
            print(f"          {span['label']:<36} {_fmt(span['duration_ms'], ' ms')}")

    # Per-label wall/cpu/runqueue/blocked. `blocked_ms` (wall - cpu - runqueue)
    # is the interesting column for an agent workload: it is time spent waiting
    # on the model, the container, or IO rather than computing.
    print("\nSPANS (aggregated per label)")
    print(agprof.summary_table(sort_by="wall_ms", row_limit=25))

    print(f"\nArtifacts: {profile_dir}")
    print(f"  {profile_dir / 'summary.json'}")
    print(f"  {profile_dir / 'summary.md'}")
    print(f"  {profile_dir / 'agprof.trace.json'}   (drag into ui.perfetto.dev)")


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


def _workload() -> None:
    """The three phases. Everything below is ordinary Agency code -- the only
    profiler-aware lines are the ``agprof.span`` / ``agprof.annotate`` calls,
    which are no-ops when profiling is off."""
    ag = agent(agconfig=cfg, engine=ENGINE)

    # Phase 1 -- network.
    with agprof.span("example:phase1_network"):
        agprof.annotate(url=NET_FETCH_URL)
        print(f">> [1/3] net_fetch  {NET_FETCH_URL}")
        r1 = ag.run(
            fetch_skill,
            agdata(url=NET_FETCH_URL, dest_path="/workspace/payload.json"),
        )
        print(f"        status={r1.status!r}  bytes={r1.bytes_downloaded!r}")
        agprof.annotate(bytes_downloaded=r1.bytes_downloaded)

    # Phase 2 -- CPU and disk, on the file phase 1 fetched. Same agent, so this
    # is run1 on the same sandbox: sandbox:commit / sandbox:start spans between
    # the two runs show the checkpoint round-trip.
    with agprof.span("example:phase2_compute"):
        print(">> [2/3] analyze  (cpu + disk on the fetched payload)")
        r2 = ag.run(
            analyze_skill,
            agdata(
                task=(
                    "Report the payload's size in bytes, its sha256, and how many "
                    "lines it has. Compute the sha256 1000 times in a python3 loop "
                    "so the work is measurable, timing it. Write all of it to the "
                    "report path as plain text, then read the report back."
                ),
                source_path="/workspace/payload.json",
                report_path="/workspace/report.txt",
            ),
        )
        print(f"        status={r2.status!r}")
        print(f"        summary={r2.summary!r}")

    # Phase 3 -- concurrency. Each fork deep-copies history and gets its own
    # container, so their spans land on separate lanes and sandbox_metrics
    # grows a row per fork. agprof.spawn_traced keeps the forked run threads
    # attached to this trace instead of becoming disconnected roots.
    with agprof.span("example:phase3_fanout"):
        print(">> [3/3] fan-out  (2 forks, concurrent sandboxes)")
        notes = ["fork A checking in", "fork B checking in"]
        forks = [agent.fork(ag) for _ in notes]
        pending = [
            fork.run(note_skill, agdata(note=note, file_path=f"/workspace/note{i}.txt"))
            for i, (fork, note) in enumerate(zip(forks, notes))
        ]
        for i, result in enumerate(pending):
            # Touching a field on a pending agdata blocks -- that wait is the
            # `sync:result_wait` span.
            print(f"        fork {i}: status={result.status!r} path={result.path!r}")
        agsync(ag)

        # Pending results do not own a fork's sandbox lifecycle. Keep the
        # fork objects above and close them while profiling is still live so
        # their destroy spans cannot race the process-scope atexit flush.
        for fork in forks:
            if fork.sandbox is not None:
                fork.sandbox.destroy()
                fork.sandbox = None

    print(f"\nShared history : {len(ag.history.messages)} messages")

    # Keep process-scope profiles deterministic: leaving the parent sandbox
    # to object finalization can race agprof's own atexit handler, turning an
    # otherwise successful run into an interrupted ``sandbox:destroy`` span.
    # The fork sandboxes were closed above; explicitly close the parent while
    # the profiling session is still live.
    if ag.sandbox is not None:
        ag.sandbox.destroy()
        ag.sandbox = None


def main() -> None:
    run_dir = _make_run_dir(f"profiler_{ENGINE}_example")
    agent.log_dir = run_dir / "logs"
    agent.output_dir = run_dir / "agent_output"
    profile_dir = Path(os.environ.get("AGENCY_PROFILE_DIR", "agprof_trace"))

    print(f"Engine   : {ENGINE}")
    print(f"Run dir  : {run_dir}")
    print(f"Profiling: {'on' if os.environ.get('AGENCY_PROFILE') else 'off (see docstring)'}")
    print(f"Scope    : {agprof.profile_scope()}\n")

    # Names this thread for the per-process resource tracks. Linux truncates to
    # 15 chars; it is a naming call only and carries no span hierarchy.
    agprof.thread_name("agprof-example")

    # Owns a session only under AGENCY_PROFILE=1 with the default `workload`
    # scope; a no-op under `process` scope, where the session already started
    # at import and is closed by the atexit hook.
    with agprof.workload():
        try:
            _workload()
        except AgError as e:
            # Let the context exit normally: spans still open at stop() are
            # reported in `incomplete_spans` rather than silently dropped, so a
            # failed run still produces a readable profile.
            print(f"\nERROR: {e}")

    _print_profile_report(profile_dir)


if __name__ == "__main__":
    main()
