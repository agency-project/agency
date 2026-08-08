"""Perfetto-compatible Chrome trace output for :mod:`agency.profiler.agprof`.

The emitter deliberately has no dependency on torch or the OpenTelemetry SDK.
agprof's spans, resource samples, and GPU lease intervals all use
``time.perf_counter_ns()``, so they can be written onto one timeline without a
clock-sync event or timestamp matching pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


TRACE_FILENAME = "agprof.trace.json"


def _reiterable(values):
    """Materialize one-shot iterables without copying existing snapshots."""
    return values if isinstance(values, (list, tuple)) else list(values)


def _record_fields(record) -> tuple:
    """Decode both the legacy and explicit-parent agprof record shapes."""
    metadata = record[6] if len(record) > 6 else {}
    span_id = record[7] if len(record) > 7 else None
    parent_span_id = record[8] if len(record) > 8 else None
    return (*record[:6], metadata, span_id, parent_span_id)


def _trace_id(value) -> "str | None":
    """Keep 64-bit OTel IDs exact when the JSON is read by JavaScript."""
    if value is None:
        return None
    if isinstance(value, int):
        return f"{value:016x}"
    return str(value)


def _time_origin_ns(records, samples, leases, interrupted_spans, started_ns) -> int:
    if started_ns is not None:
        return started_ns
    candidates = [record[2] for record in records]
    candidates.extend(sample[0] for sample in samples)
    candidates.extend(lease[1] for lease in leases)
    candidates.extend(
        span["started_ns"] for span in interrupted_spans if span.get("started_ns") is not None
    )
    return min(candidates, default=0)


def _prepare_trace(
    records,
    samples,
    leases,
    process_info,
    interrupted_spans,
    started_ns,
    pid,
    observations,
) -> tuple:
    """Normalize trace inputs shared by in-memory and streaming output."""
    records = _reiterable(records)
    samples = _reiterable(samples)
    leases = _reiterable(leases)
    interrupted_spans = _reiterable(interrupted_spans)
    if process_info is None:
        from .agprof import _process_info

        process_info = _process_info
    process_info = dict(process_info)
    trace_pid = os.getpid() if pid is None else pid
    origin_ns = _time_origin_ns(records, samples, leases, interrupted_spans, started_ns)
    if observations is None:
        from .agprof import _resource_observations

        observations = _resource_observations(samples, process_info=process_info)
    else:
        observations = _reiterable(observations)
    return records, leases, process_info, interrupted_spans, observations, trace_pid, origin_ns


def _iter_trace_events(
    records,
    leases,
    process_info,
    interrupted_spans,
    observations,
    trace_pid,
    origin_ns,
):
    """Yield Chrome-trace events without retaining the full event array."""

    def to_us(timestamp_ns: int) -> float:
        return (timestamp_ns - origin_ns) / 1e3

    yield {
        "ph": "M",
        "pid": trace_pid,
        "tid": 0,
        "name": "process_name",
        "args": {"name": "agency profiler"},
    }
    thread_ids = set()
    for record in records:
        tid, name, t0, wall, cpu, runq, metadata, span_id, parent_span_id = _record_fields(record)
        thread_ids.add(tid)
        args = dict(metadata or {})
        if cpu is None:
            args.update(cpu_ms="n/a", runqueue_ms="n/a", blocked_ms="n/a", cpu_pct="n/a")
        else:
            blocked = max(0, wall - cpu - (runq or 0))
            args.update(
                cpu_ms=round(cpu / 1e6, 3),
                runqueue_ms=(round(runq / 1e6, 3) if runq is not None else "n/a"),
                blocked_ms=round(blocked / 1e6, 3),
                cpu_pct=(round(100 * cpu / wall, 1) if wall > 0 else 0.0),
            )
        if span_id is not None:
            args["span_id"] = _trace_id(span_id)
        if parent_span_id is not None:
            args["parent_span_id"] = _trace_id(parent_span_id)
        yield {
            "ph": "X",
            "pid": trace_pid,
            "tid": tid,
            "ts": to_us(t0),
            "dur": max(1.0, wall / 1e3),
            "name": name,
            "cat": "agprof",
            "args": args,
        }

    for interrupted in interrupted_spans:
        t0 = interrupted.get("started_ns")
        if t0 is None:
            continue
        tid = interrupted.get("thread_id", 0)
        thread_ids.add(tid)
        args = {
            key: value
            for key, value in interrupted.items()
            if key not in ("thread_id", "label", "started_ns", "duration_ms")
        }
        args["outcome"] = "interrupted"
        yield {
            "ph": "X",
            "pid": trace_pid,
            "tid": tid,
            "ts": to_us(t0),
            "dur": max(1.0, float(interrupted.get("duration_ms", 0.0)) * 1e3),
            "name": interrupted.get("label", "interrupted"),
            "cat": "agprof",
            "args": args,
        }

    for sort_index, tid in enumerate(sorted(thread_ids, key=str), start=1):
        yield {
            "ph": "M",
            "pid": trace_pid,
            "tid": tid,
            "name": "thread_name",
            "args": {"name": f"thread {tid}"},
        }
        yield {
            "ph": "M",
            "pid": trace_pid,
            "tid": tid,
            "name": "thread_sort_index",
            "args": {"sort_index": sort_index},
        }

    process_identities = {
        observation["process_identity"]
        for observation in observations
        if observation.get("process_identity") is not None
    }
    for sort_index, identity in enumerate(
        sorted(
            process_identities,
            key=lambda key: process_info.get(key, {}).get("first_seen_ns", 0),
        ),
        start=1,
    ):
        info = process_info.get(identity)
        if info is None:
            continue
        process_pid = info["trace_pid"]
        yield {
            "ph": "M",
            "pid": process_pid,
            "tid": 0,
            "name": "process_name",
            "args": {"name": info["display_name"]},
        }
        yield {
            "ph": "M",
            "pid": process_pid,
            "tid": 0,
            "name": "process_sort_index",
            "args": {"sort_index": sort_index},
        }
        yield {
            "ph": "M",
            "pid": process_pid,
            "tid": 0,
            "name": "process_labels",
            "args": {
                "labels": f"cgroup={info['cgroup']}; cmdline={info['cmdline'] or info['comm']}"
            },
        }

    for observation in observations:
        yield {
            "ph": "C",
            "pid": observation.get("trace_pid", trace_pid),
            "tid": 0,
            "ts": to_us(observation["timestamp_ns"]),
            "name": observation["trace_name"],
            "cat": "resource",
            "args": {"value": round(observation["value"], 2)},
        }

    lease_tids = set()
    for gpu_id, t0, t1, label in leases:
        tid = f"gpu{gpu_id}-lease"
        lease_tids.add((gpu_id, tid))
        yield {
            "ph": "X",
            "pid": trace_pid,
            "tid": tid,
            "ts": to_us(t0),
            "dur": max(1.0, (t1 - t0) / 1e3),
            "name": f"lease:{label}",
            "cat": "gpu_lease",
        }
    for gpu_id, tid in sorted(lease_tids):
        yield {
            "ph": "M",
            "pid": trace_pid,
            "tid": tid,
            "name": "thread_name",
            "args": {"name": f"GPU {gpu_id} lease"},
        }


def build_trace(
    records,
    samples,
    leases=(),
    *,
    process_info=None,
    interrupted_spans=(),
    started_ns: "int | None" = None,
    pid: "int | None" = None,
    observations=None,
) -> dict:
    """Build a Chrome-trace document consumable by ``ui.perfetto.dev``."""
    prepared = _prepare_trace(
        records,
        samples,
        leases,
        process_info,
        interrupted_spans,
        started_ns,
        pid,
        observations,
    )
    records, leases, process_info, interrupted_spans, observations, trace_pid, origin_ns = prepared
    return {
        "traceEvents": list(
            _iter_trace_events(
                records,
                leases,
                process_info,
                interrupted_spans,
                observations,
                trace_pid,
                origin_ns,
            )
        ),
        "displayTimeUnit": "ms",
        "otherData": {
            "agprof_clock": "perf_counter_ns",
            "agprof_time_origin_ns": origin_ns,
        },
    }


def write_trace(
    out_dir,
    records,
    samples,
    leases=(),
    *,
    process_info=None,
    interrupted_spans=(),
    started_ns: "int | None" = None,
    pid: "int | None" = None,
    observations=None,
) -> Path:
    """Atomically write ``agprof.trace.json`` under *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / TRACE_FILENAME
    temporary = out_dir / f".{TRACE_FILENAME}.{os.getpid()}.tmp"
    prepared = _prepare_trace(
        records,
        samples,
        leases,
        process_info,
        interrupted_spans,
        started_ns,
        pid,
        observations,
    )
    records, leases, process_info, interrupted_spans, observations, trace_pid, origin_ns = prepared
    events = _iter_trace_events(
        records,
        leases,
        process_info,
        interrupted_spans,
        observations,
        trace_pid,
        origin_ns,
    )
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write('{"traceEvents":[')
        for index, event in enumerate(events):
            if index:
                stream.write(",")
            json.dump(event, stream, separators=(",", ":"), default=str)
        stream.write('],"displayTimeUnit":"ms","otherData":')
        json.dump(
            {
                "agprof_clock": "perf_counter_ns",
                "agprof_time_origin_ns": origin_ns,
            },
            stream,
            separators=(",", ":"),
            default=str,
        )
        stream.write("}\n")
    temporary.replace(path)
    return path
