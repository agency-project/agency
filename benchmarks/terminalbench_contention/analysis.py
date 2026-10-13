"""Post-run reconstruction: agent trajectories, cross-resource metrics, summary.

Two kinds of evidence are kept apart and labelled in every output:

* measured application intervals: API admission wait, model response wait and
  model streaming (from this benchmark's API telemetry), and tool execution
  (from Agency's own profiler spans). Their timestamps were taken by the
  process that performed them.
* measured kernel quantities: CPU, memory and I/O pressure-stall time, CPU
  time, runqueue delay, reclaim, swap and OOM counters, sampled from cgroup and
  procfs files.

Time an agent spent runnable-but-not-running, or stalled on memory or I/O, is
reported as *amounts per sampling interval* from the kernel counters. It is
never placed on the timeline by guessing from application events.

The one inferred state is ``other``: time inside a task's execution window not
covered by any measured interval (harness and CLI bookkeeping, or idle).

Units: seconds unless a key says otherwise; token counts are tokens; CPU time
is CPU-seconds; rates are per second of the measured interval.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
from pathlib import Path

from .api_budget import read_ledger, waiting_statistics
from .util import atomic_json, read_json_or, read_jsonl

# Highest priority first when measured intervals overlap.
STATES = (
    "tool",
    "model_stream",
    "model_wait",
    "api_wait",
    "checkpoint",
    "harness_startup",
    "other",
)
STARTUP_SPANS = (
    "sandbox:ensure_daemon",
    "harness:launch",
    "harness:startup_ready",
    "harness:submit",
)
INFERENCE_STATES = ("model_wait", "model_stream")
STATE_SOURCE = {
    "api_wait": "measured: API telemetry (ready -> admitted)",
    "model_wait": "measured: API telemetry (admitted -> first streamed chunk)",
    "model_stream": "measured: API telemetry (first chunk -> stream end)",
    "tool": "measured: Agency profiler tool spans",
    "checkpoint": "measured: Agency's end-of-task container checkpoint (worker hook)",
    "harness_startup": "measured: Agency profiler spans for harness daemon and CLI start-up",
    "other": "inferred: uncovered time inside the task's execution window",
}
INFRASTRUCTURE_STATUSES = ("worker_died", "setup_failed", "setup_interrupted")
CPU_CONTENTION_THRESHOLD = 0.10  # fraction of an interval with CPU pressure ("some")


# ----------------------------------------------------------------- loaders


def load_requests(agent_dir: Path) -> list:
    return [r for r in read_jsonl(agent_dir / "api-requests.jsonl") if r.get("kind") == "request"]


def load_tool_spans(agent_dir: Path, epoch_offset_ns: "int | None") -> list:
    """Tool spans recorded by Agency, converted to the host monotonic clock."""
    if epoch_offset_ns is None:
        return []
    spans = []
    for database in sorted((agent_dir / "logs").glob("agent_*_data.sqlite3")):
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            continue
        try:
            rows = connection.execute(
                "select span_name, start_ts, end_ts, cpu_ms, runqueue_ms, blocked_ms "
                "from spans where span_name like 'tool:%'"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            connection.close()
        for name, start, end, cpu_ms, runqueue_ms, blocked_ms in rows:
            if start is None or end is None or end < start:
                continue
            spans.append(
                dict(
                    tool=name.split(":", 1)[1],
                    start_ns=int(start * 1e9) - epoch_offset_ns,
                    end_ns=int(end * 1e9) - epoch_offset_ns,
                    wall_seconds=end - start,
                    cpu_seconds=None if cpu_ms is None else cpu_ms / 1000,
                    runqueue_seconds=None if runqueue_ms is None else runqueue_ms / 1000,
                    blocked_seconds=None if blocked_ms is None else blocked_ms / 1000,
                )
            )
    return sorted(spans, key=lambda s: s["start_ns"])


def load_container_series(agent_dir: Path) -> list:
    """The worker's high-rate container counter rows (see worker.ContainerTrace)."""
    rows = []
    for row in read_jsonl(agent_dir / "container-series.jsonl"):
        if isinstance(row, list) and len(row) >= 8:
            rows.append(row)
    return rows


def series_delta(rows: list, start_ns: int, end_ns: int, column: int) -> "float | None":
    """Change of a cumulative column over [start, end], linearly interpolated.

    Returns None when the series does not reach the interval at all; an interval
    extending past the last row (container already stopped) is clipped to it.
    """
    if len(rows) < 2 or end_ns <= rows[0][0] or start_ns >= rows[-1][0]:
        return None

    def value(t):
        t = min(max(t, rows[0][0]), rows[-1][0])
        index = bisect.bisect_right(rows, t, key=lambda r: r[0])
        if index >= len(rows):
            return rows[-1][column]
        before, after = rows[index - 1], rows[index]
        span = after[0] - before[0]
        if span <= 0:
            return after[column]
        return before[column] + (after[column] - before[column]) * (t - before[0]) / span

    return value(end_ns) - value(start_ns)


def load_startup_spans(agent_dir: Path, epoch_offset_ns: "int | None") -> list:
    """Harness start-up intervals recorded by Agency's profiler session."""
    database = agent_dir / "profile" / "profile_data.sqlite3"
    if epoch_offset_ns is None or not database.exists():
        return []
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5)
        marks = ",".join("?" for _ in STARTUP_SPANS)
        rows = connection.execute(
            f"select start_ts, end_ts from spans where span_name in ({marks})", STARTUP_SPANS
        ).fetchall()
        connection.close()
    except sqlite3.Error:
        return []
    return [
        ("harness_startup", int(start * 1e9) - epoch_offset_ns, int(end * 1e9) - epoch_offset_ns)
        for start, end in rows
        if start is not None and end is not None and end > start
    ]


# -------------------------------------------------------------- trajectories


def request_intervals(requests: list) -> list:
    intervals = []
    for request in requests:
        ready, admitted = request.get("ready_ns"), request.get("admitted_ns")
        first, done = request.get("first_chunk_ns"), request.get("completed_ns")
        if ready and admitted and admitted > ready:
            intervals.append(("api_wait", ready, admitted))
        if admitted and (first or done) and (first or done) > admitted:
            intervals.append(("model_wait", admitted, first or done))
        if first and done and done > first:
            intervals.append(("model_stream", first, done))
    return intervals


def resolve_timeline(intervals: list, start_ns: int, end_ns: int) -> list:
    """Flatten overlapping (state, start, end) intervals into one sequence.

    Where intervals overlap the state earliest in ``STATES`` wins; uncovered
    time becomes ``other``.
    """
    if end_ns <= start_ns:
        return []
    rank = {state: index for index, state in enumerate(STATES)}
    points = {start_ns, end_ns}
    clipped = []
    for state, begin, finish in intervals:
        begin, finish = max(begin, start_ns), min(finish, end_ns)
        if finish > begin:
            clipped.append((state, begin, finish))
            points.update((begin, finish))
    ordered = sorted(points)
    segments = []
    for left, right in zip(ordered, ordered[1:]):
        active = [s for s, b, f in clipped if b <= left and f >= right]
        state = min(active, key=rank.__getitem__) if active else "other"
        if segments and segments[-1][0] == state and segments[-1][2] == left:
            segments[-1] = (state, segments[-1][1], right)
        else:
            segments.append((state, left, right))
    return segments


def state_seconds(segments: list) -> dict:
    totals = dict.fromkeys(STATES, 0.0)
    for state, begin, finish in segments:
        totals[state] += (finish - begin) / 1e9
    return totals


def overlap_seconds(segments: list, left: int, right: int) -> dict:
    totals = {}
    for state, begin, finish in segments:
        span = min(finish, right) - max(begin, left)
        if span > 0:
            totals[state] = totals.get(state, 0.0) + span / 1e9
    return totals


# ----------------------------------------------------------- resource series


def _delta(current, previous, *keys):
    for key in keys:
        current = current.get(key) if isinstance(current, dict) else None
        previous = previous.get(key) if isinstance(previous, dict) else None
    if current is None or previous is None:
        return None
    return current - previous


def host_intervals(samples: list) -> list:
    """Per-sampling-interval host and workload quantities (deltas of counters)."""
    rows = []
    for before, after in zip(samples, samples[1:]):
        seconds = (after["t"] - before["t"]) / 1e9
        if seconds <= 0:
            continue
        row = dict(
            start_ns=before["t"], end_ns=after["t"], seconds=seconds, active=after.get("active")
        )
        pool_before = (before.get("host", {}).get("cpu") or {}).get("pool")
        pool_after = (after.get("host", {}).get("cpu") or {}).get("pool")
        if pool_before and pool_after:
            total = sum(pool_after.values()) - sum(pool_before.values())
            idle = (pool_after["idle"] + pool_after["iowait"]) - (
                pool_before["idle"] + pool_before["iowait"]
            )
            if total > 0:
                row["pool_cpu_utilization"] = 1 - idle / total
                row["pool_iowait_fraction"] = (pool_after["iowait"] - pool_before["iowait"]) / total
        for resource in ("cpu", "memory", "io"):
            for scope in ("host", "workload"):
                for kind in ("some", "full"):
                    value = _delta(
                        after.get(scope), before.get(scope), "pressure", resource, f"{kind}_usec"
                    )
                    if value is not None:
                        row[f"{scope}_{resource}_{kind}_fraction"] = value / 1e6 / seconds
        usage = _delta(after.get("workload"), before.get("workload"), "cpu", "usage_usec")
        if usage is not None:
            row["workload_cpu_seconds"] = usage / 1e6
        delay = _delta(after.get("host"), before.get("host"), "schedstat", "run_delay_ns")
        if delay is not None:
            row["pool_run_delay_seconds"] = delay / 1e9
        workload_memory = (after.get("workload") or {}).get("memory") or {}
        if "current" in workload_memory:
            row["workload_memory_bytes"] = workload_memory["current"]
        if "swap_current" in workload_memory:
            row["workload_swap_bytes"] = workload_memory["swap_current"]
        for key in ("pswpin", "pswpout", "pgscan_direct", "pgscan_kswapd", "pgmajfault"):
            value = _delta(after.get("host"), before.get("host"), "vmstat", key)
            if value is not None:
                row[f"host_{key}"] = value
        for key in ("read_bytes", "write_bytes"):
            value = _delta(after.get("host"), before.get("host"), "disk", key)
            if value is not None:
                row[f"disk_{key}"] = value
        api = after.get("api") or {}
        for key in ("utilization", "waiters", "window_used", "window_requests"):
            if key in api:
                row[f"api_{key}"] = api[key]
        procs = (after.get("host", {}).get("cpu") or {}).get("procs_running")
        if procs is not None:
            row["host_procs_running"] = procs
        rows.append(row)
    return rows


def agent_series(samples: list) -> dict:
    """Per-agent kernel measurements for each agent-sampling interval."""
    series: dict = {}
    previous: dict = {}
    for sample in samples:
        for slot, roles in (sample.get("agents") or {}).items():
            last = previous.get(slot)
            if last is not None and sample["t"] > last["t"]:
                seconds = (sample["t"] - last["t"]) / 1e9
                row = dict(start_ns=last["t"], end_ns=sample["t"], seconds=seconds)
                for role in ("container", "host"):
                    now, then = roles.get(role), last["roles"].get(role)
                    if not now or not then:
                        continue
                    usage = _delta(now, then, "cpu", "usage_usec")
                    if usage is not None:
                        row[f"{role}_cpu_seconds"] = usage / 1e6
                    for resource in ("cpu", "memory", "io"):
                        stall = _delta(now, then, "pressure", resource, "some_usec")
                        if stall is not None:
                            row[f"{role}_{resource}_stall_seconds"] = stall / 1e6
                    delay = _delta(now, then, "threads", "run_delay_ns")
                    if delay is not None and delay >= 0:
                        row[f"{role}_run_delay_seconds_lower_bound"] = delay / 1e9
                    for key in ("voluntary_switches", "involuntary_switches"):
                        value = _delta(now, then, "threads", key)
                        if value is not None and value >= 0:
                            row[f"{role}_{key}_lower_bound"] = value
                    memory = now.get("memory") or {}
                    if "current" in memory:
                        row[f"{role}_memory_bytes"] = memory["current"]
                    threads = now.get("threads") or {}
                    row[f"{role}_runnable_threads"] = threads.get("runnable")
                    row[f"{role}_threads"] = threads.get("threads")
                    for key in ("read_bytes", "write_bytes", "rbytes", "wbytes"):
                        value = _delta(now, then, "io", key)
                        if value is not None and value >= 0:
                            name = {"rbytes": "read_bytes", "wbytes": "write_bytes"}.get(key, key)
                            row[f"{role}_io_{name}"] = value
                series.setdefault(slot, []).append(row)
            previous[slot] = dict(t=sample["t"], roles=roles)
    return series


def counter_totals(agent_dir: Path) -> dict:
    """Exact per-task totals from the counters the worker read at start and end."""
    start = read_json_or(agent_dir / "counters-start.json")
    end = read_json_or(agent_dir / "counters-end.json")
    if not start or not end:
        return {}
    totals = {}
    for role in ("container", "host"):
        now, then = end.get(role), start.get(role)
        if not now or not then:
            continue
        usage = _delta(now, then, "cpu", "usage_usec")
        if usage is not None:
            totals[f"{role}_cpu_seconds"] = usage / 1e6
            totals[f"{role}_cpu_user_seconds"] = (_delta(now, then, "cpu", "user_usec") or 0) / 1e6
            totals[f"{role}_cpu_system_seconds"] = (
                _delta(now, then, "cpu", "system_usec") or 0
            ) / 1e6
        for resource in ("cpu", "memory", "io"):
            for kind in ("some", "full"):
                stall = _delta(now, then, "pressure", resource, f"{kind}_usec")
                if stall is not None:
                    totals[f"{role}_{resource}_stall_{kind}_seconds"] = stall / 1e6
        memory = now.get("memory") or {}
        if "peak" in memory:
            totals[f"{role}_memory_peak_bytes"] = memory["peak"]
        if "swap_peak" in memory:
            totals[f"{role}_swap_peak_bytes"] = memory["swap_peak"]
        events_now = memory.get("events") or {}
        events_then = (then.get("memory") or {}).get("events") or {}
        for key in ("oom", "oom_kill", "max", "high"):
            if key in events_now:
                totals[f"{role}_memory_events_{key}"] = events_now[key] - events_then.get(key, 0)
        stat_now = memory.get("stat") or {}
        stat_then = (then.get("memory") or {}).get("stat") or {}
        for key in (
            "pgmajfault",
            "pgscan",
            "pgsteal",
            "workingset_refault_file",
            "workingset_refault_anon",
        ):
            if key in stat_now:
                totals[f"{role}_{key}"] = stat_now[key] - stat_then.get(key, 0)
        totals[f"{role}_cpu_max"] = now.get("cpu.max")
    totals["container_counters_source"] = end.get("container_source")
    final, then = end.get("container_final"), start.get("container")
    if final and then:
        usage = _delta(final, then, "cpu", "usage_usec")
        if usage is not None:
            totals["container_cpu_seconds_including_checkpoint"] = usage / 1e6
        memory = final.get("memory") or {}
        if "peak" in memory:
            totals["container_memory_peak_bytes"] = memory["peak"]
    cpu = [totals.get("container_cpu_seconds"), totals.get("host_cpu_seconds")]
    if any(v is not None for v in cpu):
        totals["cpu_seconds"] = sum(v for v in cpu if v is not None)
    return totals


# ---------------------------------------------------------------- statistics


def _stats(values: list) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(n=0)

    def quantile(q):
        return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]

    return dict(
        n=len(values),
        mean=sum(values) / len(values),
        p50=quantile(0.5),
        p90=quantile(0.9),
        p99=quantile(0.99),
        minimum=values[0],
        maximum=values[-1],
        total=sum(values),
    )


def _weighted_mean(rows: list, key: str, predicate=lambda row: True):
    total = weight = 0.0
    for row in rows:
        if key in row and row[key] is not None and predicate(row):
            total += row[key] * row["seconds"]
            weight += row["seconds"]
    return dict(mean=total / weight if weight else None, seconds=weight)


def api_summary(run_dir: Path, requests_by_agent: dict, makespan: "float | None") -> dict:
    requests = [r for rows in requests_by_agent.values() for r in rows]
    completed = [r for r in requests if r.get("outcome") == "completed"]
    try:
        ledger = read_ledger(run_dir / "api-metrics" / "ledger")
    except Exception as exc:
        ledger, ledger_error = [], f"{type(exc).__name__}: {exc}"
    else:
        ledger_error = None
    admits = [e for e in ledger if e.get("kind") == "admit"]
    estimated = sum(r.get("estimated_prompt_tokens") or 0 for r in completed)
    reported = sum(r.get("reported_prompt_tokens") or 0 for r in completed)
    ratios = [
        r["estimated_prompt_tokens"] / r["reported_prompt_tokens"]
        for r in completed
        if r.get("estimated_prompt_tokens") and r.get("reported_prompt_tokens")
    ]
    events = sorted(
        [(e["admitted_mono"], 1) for e in admits]
        + [
            (r["completed_ns"] / 1e9, -1)
            for r in requests
            if r.get("completed_ns") and r.get("admitted_ns")
        ]
    )
    concurrent = peak = 0
    for _, change in events:
        concurrent += change
        peak = max(peak, concurrent)
    summary = dict(
        requests=len(requests),
        completed_requests=len(completed),
        failed_requests=len(requests) - len(completed),
        http_attempts=sum(r.get("attempts") or 0 for r in requests),
        retries=sum(r.get("retries") or 0 for r in requests),
        http_429=sum(r.get("http_429") or 0 for r in requests),
        admissions=len(admits),
        admission_wait_seconds=waiting_statistics(ledger),
        wait_censored=sum(e.get("kind") == "wait_censored" for e in ledger),
        blocked_by=_count(reason for e in admits for reason in e.get("blocked_by", [])),
        response_latency_seconds=_stats([r.get("response_latency_seconds") for r in completed]),
        time_to_first_chunk_seconds=_stats(
            [r.get("time_to_first_chunk_seconds") for r in completed]
        ),
        stream_seconds=_stats([r.get("stream_seconds") for r in completed]),
        peak_concurrent_requests=peak,
        tokens=dict(
            estimated_prompt=estimated,
            reserved_completion=sum(r.get("reserved_completion_tokens") or 0 for r in completed),
            admission_charge=sum(e["charge"] for e in admits),
            reported_prompt=reported,
            reported_completion=sum(r.get("reported_completion_tokens") or 0 for r in completed),
            reported_cached=sum(r.get("reported_cached_tokens") or 0 for r in completed),
            reported_total=sum(r.get("reported_total_tokens") or 0 for r in completed),
            estimated_over_reported_prompt=estimated / reported if reported else None,
            per_request_estimate_ratio=_stats(ratios),
            requests_estimated_below_reported=sum(ratio < 1 for ratio in ratios),
            legacy_charge=sum(e.get("legacy_charge") or 0 for e in admits),
        ),
        ledger_error=ledger_error,
        note="estimated_* are admission-time estimates; reported_* are model-reported usage. "
        "Neither is asserted to equal what an upstream provider counted against its limits.",
    )
    if makespan:
        summary["throughput"] = dict(
            requests_per_second=len(completed) / makespan,
            estimated_tokens_per_minute=summary["tokens"]["admission_charge"] / makespan * 60,
            reported_tokens_per_minute=summary["tokens"]["reported_total"] / makespan * 60,
        )
    return summary


def _count(items) -> dict:
    counts: dict = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def cross_resource(timelines: dict, intervals: list, makespan: "float | None") -> dict:
    """Overlap of inference and tool execution, and utilization by agent state."""
    rows = []
    for row in intervals:
        counts = dict.fromkeys(STATES, 0.0)
        for segments in timelines.values():
            for state, seconds in overlap_seconds(segments, row["start_ns"], row["end_ns"]).items():
                counts[state] += seconds / row["seconds"]
        inference = counts["model_wait"] + counts["model_stream"]
        rows.append(
            dict(
                row,
                agents_tool=counts["tool"],
                agents_inference=inference,
                agents_api_wait=counts["api_wait"],
                agents_other=counts["other"],
            )
        )
    covered = sum(r["seconds"] for r in rows)

    def seconds_where(predicate):
        return sum(r["seconds"] for r in rows if predicate(r))

    cpu_pressure = (
        "workload_cpu_some_fraction"
        if any("workload_cpu_some_fraction" in r for r in rows)
        else "host_cpu_some_fraction"
    )
    contended = lambda r: (r.get(cpu_pressure) or 0) >= CPU_CONTENTION_THRESHOLD  # noqa: E731
    result = dict(
        covered_seconds=covered,
        makespan_seconds=makespan,
        cpu_pressure_source=cpu_pressure,
        cpu_contention_threshold=CPU_CONTENTION_THRESHOLD,
        mean_agents=dict(
            tool=_weighted_mean(rows, "agents_tool")["mean"],
            inference=_weighted_mean(rows, "agents_inference")["mean"],
            api_wait=_weighted_mean(rows, "agents_api_wait")["mean"],
            other=_weighted_mean(rows, "agents_other")["mean"],
        ),
        inference_tool_overlap=dict(
            seconds_both_present=seconds_where(
                lambda r: r["agents_tool"] >= 0.5 and r["agents_inference"] >= 0.5
            ),
            seconds_tool_only=seconds_where(
                lambda r: r["agents_tool"] >= 0.5 and r["agents_inference"] < 0.5
            ),
            seconds_inference_only=seconds_where(
                lambda r: r["agents_tool"] < 0.5 and r["agents_inference"] >= 0.5
            ),
            seconds_neither=seconds_where(
                lambda r: r["agents_tool"] < 0.5 and r["agents_inference"] < 0.5
            ),
            agent_pair_seconds=sum(
                r["agents_tool"] * r["agents_inference"] * r["seconds"] for r in rows
            ),
            definition="An interval counts for a state when the time-averaged number of "
            "agents in it is at least 0.5. agent_pair_seconds integrates "
            "(agents executing tools) x (agents in inference).",
        ),
        simultaneous_cpu_and_api_contention=dict(
            seconds=seconds_where(lambda r: contended(r) and r["agents_api_wait"] >= 0.5),
            seconds_cpu_contended=seconds_where(contended),
            seconds_api_waiting=seconds_where(lambda r: r["agents_api_wait"] >= 0.5),
        ),
        utilization_while_api_waiting=dict(
            pool_cpu=_weighted_mean(
                rows, "pool_cpu_utilization", lambda r: r["agents_api_wait"] >= 0.5
            ),
            pool_cpu_otherwise=_weighted_mean(
                rows, "pool_cpu_utilization", lambda r: r["agents_api_wait"] < 0.5
            ),
            api_budget=_weighted_mean(
                rows, "api_utilization", lambda r: r["agents_api_wait"] >= 0.5
            ),
        ),
        utilization_while_tools_run=dict(
            pool_cpu=_weighted_mean(
                rows, "pool_cpu_utilization", lambda r: r["agents_tool"] >= 0.5
            ),
            pool_cpu_otherwise=_weighted_mean(
                rows, "pool_cpu_utilization", lambda r: r["agents_tool"] < 0.5
            ),
            api_budget=_weighted_mean(rows, "api_utilization", lambda r: r["agents_tool"] >= 0.5),
            cpu_pressure=_weighted_mean(rows, cpu_pressure, lambda r: r["agents_tool"] >= 0.5),
        ),
        api_budget_while_cpu_contended=_weighted_mean(rows, "api_utilization", contended),
    )
    return result, rows


def resource_summary(intervals: list, samples: list) -> dict:
    if not intervals:
        return dict(samples=len(samples))
    total = sum(r["seconds"] for r in intervals)

    def integral(key):
        values = [r[key] * r["seconds"] for r in intervals if r.get(key) is not None]
        return sum(values) if values else None

    def total_of(key):
        values = [r[key] for r in intervals if r.get(key) is not None]
        return sum(values) if values else None

    def peak(key):
        values = [r[key] for r in intervals if r.get(key) is not None]
        return max(values) if values else None

    last = samples[-1] if samples else {}
    first = samples[0] if samples else {}
    events_last = ((last.get("workload") or {}).get("memory") or {}).get("events") or {}
    events_first = ((first.get("workload") or {}).get("memory") or {}).get("events") or {}
    summary = dict(
        samples=len(samples),
        sampled_seconds=total,
        pool_cpu_utilization=dict(
            mean=_weighted_mean(intervals, "pool_cpu_utilization")["mean"],
            peak=peak("pool_cpu_utilization"),
        ),
        workload_cpu_seconds=total_of("workload_cpu_seconds"),
        pool_run_delay_seconds=total_of("pool_run_delay_seconds"),
        stall_seconds={
            f"{scope}_{resource}_{kind}": integral(f"{scope}_{resource}_{kind}_fraction")
            for scope in ("host", "workload")
            for resource in ("cpu", "memory", "io")
            for kind in ("some", "full")
        },
        workload_memory_peak_bytes=peak("workload_memory_bytes"),
        workload_swap_peak_bytes=peak("workload_swap_bytes"),
        host_swap_in_pages=total_of("host_pswpin"),
        host_swap_out_pages=total_of("host_pswpout"),
        host_direct_reclaim_pages=total_of("host_pgscan_direct"),
        host_kswapd_reclaim_pages=total_of("host_pgscan_kswapd"),
        host_major_faults=total_of("host_pgmajfault"),
        disk_read_bytes=total_of("disk_read_bytes"),
        disk_write_bytes=total_of("disk_write_bytes"),
        workload_oom_kills=(
            events_last.get("oom_kill", 0) - events_first.get("oom_kill", 0)
            if events_last
            else None
        ),
        api_budget_utilization=dict(
            mean=_weighted_mean(intervals, "api_utilization")["mean"],
            peak=peak("api_utilization"),
        ),
        peak_api_waiters=peak("api_waiters"),
    )
    return summary


def perfetto_trace(timelines: dict, intervals: list, origin_ns: int, labels: dict) -> dict:
    """Chrome/Perfetto trace-event JSON: one track per agent plus counters."""
    events = [dict(name="process_name", ph="M", pid=1, tid=0, args=dict(name="agents"))]
    for slot, segments in sorted(timelines.items(), key=lambda item: int(item[0])):
        tid = int(slot)
        events.append(
            dict(
                name="thread_name",
                ph="M",
                pid=1,
                tid=tid,
                args=dict(name=labels.get(slot, f"agent {slot}")),
            )
        )
        for state, begin, finish in segments:
            events.append(
                dict(
                    name=state,
                    cat="agent_state",
                    ph="X",
                    pid=1,
                    tid=tid,
                    ts=(begin - origin_ns) / 1e3,
                    dur=(finish - begin) / 1e3,
                    args=dict(source=STATE_SOURCE[state]),
                )
            )
    counters = (
        ("pool_cpu_utilization", "pool CPU utilization"),
        ("workload_cpu_some_fraction", "workload CPU pressure (some)"),
        ("workload_memory_some_fraction", "workload memory pressure (some)"),
        ("workload_io_some_fraction", "workload I/O pressure (some)"),
        ("api_utilization", "API budget utilization"),
        ("api_waiters", "API waiters"),
        ("active", "executing agents"),
    )
    events.append(dict(name="process_name", ph="M", pid=2, tid=0, args=dict(name="host")))
    for row in intervals:
        for key, name in counters:
            if row.get(key) is not None:
                events.append(
                    dict(
                        name=name,
                        ph="C",
                        pid=2,
                        ts=(row["end_ns"] - origin_ns) / 1e3,
                        args={"value": row[key]},
                    )
                )
    return dict(
        traceEvents=events,
        displayTimeUnit="ms",
        metadata=dict(origin_monotonic_ns=origin_ns, clock="CLOCK_MONOTONIC"),
    )


# ------------------------------------------------------------------ finalize


def finalize(run_dir, run_record: dict, verdicts: dict, config: dict) -> dict:
    """Write trajectories, metrics and return the run summary."""
    run_dir = Path(run_dir)
    traces = run_dir / "traces"
    traces.mkdir(exist_ok=True)
    samples = read_jsonl(run_dir / "resource-metrics" / "samples.jsonl")
    start, end = run_record.get("measured_start_ns"), run_record.get("measured_end_ns")
    makespan = run_record.get("makespan_seconds")
    measured = [s for s in samples if start and end and start <= s["t"] <= end + int(2e9)]
    intervals = host_intervals(measured)
    series = agent_series(measured)
    timelines, requests_by_agent, labels, tool_rows, trajectory_rows = {}, {}, {}, [], []
    tasks = []
    writable_sources: dict = {}
    for record in run_record["tasks"]:
        agent_dir = run_dir / "agents" / record["directory"]
        state = read_json_or(agent_dir / "task.json") or {}
        slot = str(record["slot"])
        labels[slot] = f"{record['agent_id']} {record['task_id'].split('/')[-1]}"
        requests = load_requests(agent_dir) if agent_dir.is_dir() else []
        requests_by_agent[slot] = requests
        offset = (state.get("clock") or {}).get("epoch_offset_ns")
        tools = load_tool_spans(agent_dir, offset) if agent_dir.is_dir() else []
        row = dict(record)
        row.pop("collection", None)
        container = load_container_series(agent_dir) if agent_dir.is_dir() else []
        for tool in tools:
            # Attribution by interval: CPU the whole container used while this
            # call was open (overlapping calls and background work share it).
            usage = series_delta(container, tool["start_ns"], tool["end_ns"], 1)
            stall = series_delta(container, tool["start_ns"], tool["end_ns"], 4)
            tool["container_cpu_seconds"] = None if usage is None else usage / 1e6
            tool["container_cpu_stall_seconds"] = None if stall is None else stall / 1e6
        if record.get("start_ns") and record.get("end_ns"):
            raw = request_intervals(requests) + [
                ("tool", t["start_ns"], t["end_ns"]) for t in tools
            ]
            if state.get("commit_start_ns"):
                raw.append(("checkpoint", state["commit_start_ns"], record["end_ns"]))
            raw.extend(load_startup_spans(agent_dir, offset))
            segments = resolve_timeline(raw, record["start_ns"], record["end_ns"])
            timelines[slot] = segments
            seconds = state_seconds(segments)
            wall = (record["end_ns"] - record["start_ns"]) / 1e9
            row["state_seconds"] = seconds
            row["state_fraction"] = {k: (v / wall if wall else None) for k, v in seconds.items()}
            row["thinking_seconds"] = seconds["model_wait"] + seconds["model_stream"]
            row["doing_seconds"] = seconds["tool"]
            if container:
                by_state = {
                    name: dict(
                        cpu_seconds=0.0,
                        cpu_stall_seconds=0.0,
                        memory_stall_seconds=0.0,
                        io_stall_seconds=0.0,
                    )
                    for name in STATES
                }
                for state_name, begin, finish in segments:
                    for key, column in (
                        ("cpu_seconds", 1),
                        ("cpu_stall_seconds", 4),
                        ("memory_stall_seconds", 5),
                        ("io_stall_seconds", 6),
                    ):
                        value = series_delta(container, begin, finish, column)
                        if value is not None:
                            by_state[state_name][key] += value / 1e6
                row["container_by_state"] = by_state
                row["container_by_state_source"] = (
                    "container cgroup counters sampled by the worker, split by the agent's "
                    "state at the time (measured counters, measured state boundaries)"
                )
            for state_name, begin, finish in segments:
                trajectory_rows.append(
                    dict(
                        slot=record["slot"],
                        agent_id=record["agent_id"],
                        task_id=record["task_id"],
                        state=state_name,
                        start_ns=begin,
                        end_ns=finish,
                        seconds=(finish - begin) / 1e9,
                        source=STATE_SOURCE[state_name],
                    )
                )
        for tool in tools:
            tool_rows.append(
                dict(
                    slot=record["slot"],
                    agent_id=record["agent_id"],
                    task_id=record["task_id"],
                    **tool,
                )
            )
        row["tool_calls"] = len(tools)
        measured_cpu = [
            t["container_cpu_seconds"] for t in tools if t["container_cpu_seconds"] is not None
        ]
        row["tool_cpu_seconds"] = sum(measured_cpu) if measured_cpu else None
        row["tool_cpu_source"] = "container cgroup CPU over each tool call's wall interval"
        row["api"] = dict(
            requests=len(requests),
            wait_seconds=sum(r.get("api_wait_seconds") or 0 for r in requests),
            retries=sum(r.get("retries") or 0 for r in requests),
            http_429=sum(r.get("http_429") or 0 for r in requests),
            estimated_prompt_tokens=sum(r.get("estimated_prompt_tokens") or 0 for r in requests),
            reported_prompt_tokens=sum(r.get("reported_prompt_tokens") or 0 for r in requests),
            reported_completion_tokens=sum(
                r.get("reported_completion_tokens") or 0 for r in requests
            ),
        )
        row["resources"] = counter_totals(agent_dir) if agent_dir.is_dir() else {}
        rows = series.get(slot) or []
        row["waiting"] = dict(
            api_admission_seconds=row["api"]["wait_seconds"],
            cpu_stall_seconds=row["resources"].get("container_cpu_stall_some_seconds"),
            memory_stall_seconds=row["resources"].get("container_memory_stall_some_seconds"),
            io_stall_seconds=row["resources"].get("container_io_stall_some_seconds"),
            run_delay_seconds_lower_bound=(
                sum(r.get("container_run_delay_seconds_lower_bound") or 0 for r in rows)
                if rows
                else None
            ),
            source="cgroup pressure totals between task start and end (measured); run delay "
            "is summed over threads alive at each sample (lower bound)",
        )
        verdict = verdicts.get(record["directory"])
        if verdict is not None:
            row["verdict"] = dict(
                ok=bool(verdict.get("ok")),
                reward=verdict.get("reward"),
                rewards=verdict.get("rewards"),
                exception=verdict.get("exception"),
                error=verdict.get("error"),
            )
        isolation = read_json_or(agent_dir / "isolation.json") or {}
        for mount in isolation.get("writable_mounts") or []:
            writable_sources.setdefault(mount["source"], []).append(record["agent_id"])
        row["isolation_problems"] = isolation.get("problems")
        tasks.append(row)
    with (traces / "trajectories.jsonl").open("w") as stream:
        for item in trajectory_rows:
            stream.write(json.dumps(item) + "\n")
    with (traces / "tool-calls.jsonl").open("w") as stream:
        for item in tool_rows:
            stream.write(json.dumps(item) + "\n")
    with (traces / "agent-series.jsonl").open("w") as stream:
        for slot, rows in sorted(series.items(), key=lambda item: int(item[0])):
            for item in rows:
                stream.write(json.dumps(dict(slot=int(slot), **item)) + "\n")
    cross, cross_rows = cross_resource(timelines, intervals, makespan)
    with (traces / "timeline.jsonl").open("w") as stream:
        for item in cross_rows:
            stream.write(json.dumps(item) + "\n")
    atomic_json(traces / "cross-resource.json", cross)
    if start:
        atomic_json(
            traces / "run.trace.json",
            perfetto_trace(timelines, intervals, start, labels),
            indent=None,
        )
    atomic_json(
        traces / "index.json",
        dict(
            clock="CLOCK_MONOTONIC nanoseconds on one host; see metadata.json 'clock'",
            origin_monotonic_ns=start,
            run_trace="run.trace.json (Chrome/Perfetto trace-event JSON, ts in microseconds "
            "since measured start)",
            trajectories="trajectories.jsonl",
            state_sources=STATE_SOURCE,
            agent_traces={
                r["agent_id"]: f"../agents/{r['directory']}/profile/agprof.trace.json"
                for r in run_record["tasks"]
                if (run_dir / "agents" / r["directory"] / "profile" / "agprof.trace.json").exists()
            },
        ),
    )
    api = api_summary(run_dir, requests_by_agent, makespan)
    atomic_json(run_dir / "api-metrics" / "summary.json", api)
    resources = resource_summary(intervals, measured)
    atomic_json(run_dir / "resource-metrics" / "summary.json", resources)
    statuses = _count(task["status"] for task in tasks)
    graded = [t for t in tasks if t.get("verdict") and t["verdict"]["ok"]]
    grading_failures = [
        t["task_id"] for t in tasks if t.get("verdict") is not None and not t["verdict"]["ok"]
    ]
    rewards = [t["verdict"]["reward"] for t in graded if t["verdict"]["reward"] is not None]
    shared_writable = {s: a for s, a in writable_sources.items() if len(set(a)) > 1}
    completed = [t for t in tasks if t["status"] == "completed"]
    infrastructure = [t["task_id"] for t in tasks if t["status"] in INFRASTRUCTURE_STATUSES]
    # The harness or Agency itself returned an error (not a task outcome).
    agent_errors = [t["task_id"] for t in tasks if t["status"] == "agent_error"]
    walls = [t["wall_seconds"] for t in tasks if t.get("wall_seconds") is not None]
    summary = dict(
        schema_version=1,
        run_id=run_record["run_id"],
        status=run_record["status"],
        error=run_record.get("error"),
        run_label=run_record.get("run_label"),
        condition=run_record.get("condition"),
        repetition=run_record.get("repetition"),
        order_seed=run_record.get("order_seed"),
        agents=run_record["agents"],
        makespan_seconds=makespan,
        measured_start_ns=start,
        measured_end_ns=end,
        counts=dict(
            tasks=len(tasks),
            by_status=statuses,
            completed=len(completed),
            censored=sum(bool(t["censored"]) for t in tasks),
            not_started=statuses.get("not_started", 0),
            infrastructure_failures=len(infrastructure),
            agent_errors=len(agent_errors),
            collected=sum(bool(t["collection_ok"]) for t in tasks),
            graded=len(graded),
            grading_failures=len(grading_failures),
            passed=sum(r >= 1.0 for r in rewards),
        ),
        infrastructure_failed_tasks=infrastructure,
        agent_error_tasks=agent_errors,
        grading_failed_tasks=grading_failures,
        denominators=dict(
            mean_reward_graded="tasks with a verifier verdict",
            mean_reward_corpus="all tasks in the corpus; ungraded tasks count as 0",
            throughput="measured makespan in hours",
        ),
        rewards=dict(
            mean_reward_graded=sum(rewards) / len(rewards) if rewards else None,
            mean_reward_corpus=sum(rewards) / len(tasks) if tasks else None,
            passed_completed=sum(
                1
                for t in graded
                if t["status"] == "completed" and (t["verdict"]["reward"] or 0) >= 1
            ),
            passed_censored=sum(
                1 for t in graded if t["censored"] and (t["verdict"]["reward"] or 0) >= 1
            ),
        ),
        throughput=dict(
            completed_tasks_per_hour=len(completed) / makespan * 3600 if makespan else None,
            passed_tasks_per_hour=(
                sum(r >= 1.0 for r in rewards) / makespan * 3600 if makespan else None
            ),
        ),
        task_wall_seconds=_stats(walls),
        setup_seconds=_stats([t.get("setup_seconds") for t in tasks]),
        turnaround_seconds=_stats([t.get("turnaround_seconds") for t in tasks]),
        concurrency=dict(
            maximum_allowed=run_record["agents"],
            peak_in_flight=run_record.get("peak_in_flight"),
            peak_executing=run_record.get("peak_executing"),
        ),
        isolation=dict(
            all_verified=all(
                t.get("isolation_verified")
                for t in tasks
                if t["status"] not in ("not_started", "setup_failed")
            ),
            cross_agent_writable_mounts=shared_writable,
            leaked_containers=run_record.get("leaked_containers"),
            stale_units=run_record.get("stale_units"),
        ),
        api=api,
        resources=resources,
        cross_resource=cross,
        ledger_ok=(run_record.get("ledger") or {}).get("ok"),
        ledger_problems=(run_record.get("ledger") or {}).get("problems"),
        tasks=tasks,
    )
    return summary
