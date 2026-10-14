"""Read-only, bounded queries over the investigator's shared historical records.

No execution, replay transport, semantic matching, or second trace/graph cache.
Counts and distributions describe captured operations, not unobserved activity.
"""

from __future__ import annotations

import asyncio
import json
import math
from itertools import islice
from collections import deque
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .investigator import catalog, load_run, number as historical_number, reduce_samples

PAGE_SIZE = 100
TERMINAL = {"success", "failed", "failure", "error", "timeout", "cancelled", "completed"}
NON_TOOLS = {"model", "queue", "dependency", "setup", "unknown", "syscall", "control"}
# Span/call IDs are local trace identities, often generated or positional. Only
# an explicit task + invocation identity can establish cross-run correspondence.
MATCH_FIELDS = ("agency.task_id", "task_id", "agency.invocation_id", "invocation_id")


def number(value):
    return None if isinstance(value, bool) else historical_number(value)


def summary_file(entry):
    path = Path(entry["path"]).parent / "summary.json"
    try:
        value = json.loads(path.read_text()) if path.exists() else {}
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}  # Optional damaged summary is reported as unavailable.


def eligible(entry):
    summary = summary_file(entry)
    # Finalized profiler sessions may contain interrupted operations. Finalized
    # recording is not proof of successful task completion. Demo/live never enter.
    return entry.get("source") == "recorded" and (
        (bool(summary.get("generated_at")) and summary.get("data_source") != "mock")
        or ("resolved" in entry and entry.get("repetition") is not None)
    )


def historical_link(run_id, agent=None):
    params = {"run": run_id, "view": "trajectory"}
    if agent:
        params["agent"] = agent
    return "/profiler?" + urlencode(params)


def is_tool(action):
    return action.get("kind") not in NON_TOOLS


def duration(action):
    if (
        action.get("missing_start")
        or action.get("missing_end")
        or action.get("outcome") not in TERMINAL
    ):
        return None
    return number(action.get("duration"))


def offset(action):
    if action.get("missing_start") or "timestamp_missing" in str(action.get("source", "")):
        return None
    return number(action.get("start"))


def strict_sum(actions, key):
    total = 0
    seen = False
    for action in actions:
        seen = True
        value = number(action.get(key))
        if value is None or value < 0:
            return None
        total += value
    return total if seen else None


def identity(action):
    metadata = action.get("metadata", {})
    task = metadata.get("agency.task_id") or metadata.get("task_id")
    invocation = metadata.get("agency.invocation_id") or metadata.get("invocation_id")
    if isinstance(task, str) and task and isinstance(invocation, str) and invocation:
        return [task, invocation]
    return None


def operation(action):
    # Only bounded previews leave this endpoint. Full evidence is fetched on demand.
    row = {
        key: action.get(key)
        for key in (
            "id",
            "agent",
            "kind",
            "name",
            "model",
            "outcome",
            "tokens",
            "output_tokens",
            "timing",
        )
    }
    row.update(
        start=offset(action),
        duration=duration(action),
        command=str(action.get("command") or "")[:600],
        result_preview=str(action.get("result") or "")[:600],
        match=identity(action),
        missing_start=offset(action) is None,
        missing_end=duration(action) is None,
    )
    row["metadata"] = {
        key: action.get("metadata", {}).get(key)
        for key in MATCH_FIELDS
        if key in action.get("metadata", {})
    }
    row["error"] = action.get("error") or action.get("metadata", {}).get("error")
    if row["error"] is not None:
        row["error"] = str(row["error"])[:600]
    return row


def filtered(run, agent="", kind="", query="", start=None, end=None):
    query = query.casefold()
    for action in run.get("actions", ()):
        if agent and action.get("agent") != agent:
            continue
        if kind == "model" and action.get("kind") != "model":
            continue
        if kind == "tool" and not is_tool(action):
            continue
        at = offset(action)
        if start is not None and (at is None or at + (duration(action) or 0) < start):
            continue
        if end is not None and (at is None or at > end):
            continue
        if query and not any(
            query in str(action.get(key) or "").casefold()
            for key in ("id", "name", "model", "command", "arguments", "result", "intent")
        ):
            continue
        yield action


def page(run, *, skip=0, limit=PAGE_SIZE, tail=False, **filters):
    limit = min(PAGE_SIZE, max(1, limit))
    matches = filtered(run, **filters)
    if tail:
        rows = deque(maxlen=limit)
        count = 0
        for item in matches:
            count += 1
            rows.append(item)
        return {
            "operations": [operation(a) for a in rows],
            "next": None,
            "offset": max(0, count - limit),
            "limit": limit,
            "total": len(run.get("actions", ())),
            "extent": run.get("duration"),
            "time_basis": "First recorded span; execution start unavailable",
        }
    rows = list(islice(matches, skip, skip + limit + 1))
    return {
        "operations": [operation(a) for a in rows[:limit]],
        "next": skip + limit if len(rows) > limit else None,
        "offset": skip,
        "limit": limit,
        "total": len(run.get("actions", ())),
        "extent": run.get("duration"),
        "time_basis": "First recorded span; execution start unavailable",
    }


def stats(actions):
    # Aggregation uses references to the shared record, never copied payloads.
    count = errors = unresolved = timed = 0
    total_duration = 0.0
    bins = [0] * 7
    edges = [0.01, 0.1, 1, 10, 60, 300]
    for action in actions:
        count += 1
        state = action.get("outcome")
        errors += state in {"failed", "failure", "error", "timeout"}
        unresolved += state not in TERMINAL
        value = duration(action)
        if value is not None and value >= 0:
            timed += 1
            total_duration += value
            bins[next((i for i, bound in enumerate(edges) if value < bound), 6)] += 1
    return {
        "count": count,
        "errors": errors if not unresolved else None,
        "duration": total_duration if timed == count and count else None,
        "timed": timed,
        "unresolved": unresolved,
        "distribution": bins,
    }


def mean_field(actions, key):
    total = count = 0
    for action in actions:
        value = number(action.get("metadata", {}).get(key))
        if value is not None and value >= 0:
            total += value
            count += 1
    return {"value": total / count if count else None, "samples": count}


def activity(run):
    # One pass of scalar accumulators: no per-model/tool event collections.
    groups = {"models": {}, "tools": {}}
    for action in run.get("actions", ()):
        model = action.get("kind") == "model"
        if not model and not is_tool(action):
            continue
        name = (action.get("model") if model else action.get("name")) or "Not available"
        rows = groups["models" if model else "tools"]
        row = rows.setdefault(
            name,
            {
                "name": name,
                "count": 0,
                "errors": 0,
                "unresolved": 0,
                "duration": 0.0,
                "timed": 0,
                "distribution": [0] * 7,
                "input": 0,
                "output": 0,
                "retries": 0,
                "ttft_ms": {"value": 0, "samples": 0},
                "throughput": {"value": 0, "samples": 0},
            },
        )
        row["count"] += 1
        state = action.get("outcome")
        row["errors"] += state in {"failed", "failure", "error", "timeout"}
        row["unresolved"] += state not in TERMINAL
        elapsed = duration(action)
        if elapsed is not None and elapsed >= 0:
            row["timed"] += 1
            row["duration"] += elapsed
            row["distribution"][
                next(
                    (i for i, bound in enumerate([0.01, 0.1, 1, 10, 60, 300]) if elapsed < bound), 6
                )
            ] += 1
        for key, field in (("input", "tokens"), ("output", "output_tokens")):
            measured = number(action.get(field)) if model else None
            row[key] = (
                row[key] + measured
                if measured is not None and measured >= 0 and row[key] is not None
                else None
            )
        metadata = action.get("metadata", {})
        retry = metadata.get("retry_index")
        row["retries"] = (
            row["retries"] + int(retry > 0)
            if isinstance(retry, int) and retry >= 0 and row["retries"] is not None
            else None
        )
        for key, field in (("ttft_ms", "ttft_ms"), ("throughput", "output_tokens_per_second")):
            measured = number(metadata.get(field))
            if measured is not None and measured >= 0:
                row[key]["value"] += measured
                row[key]["samples"] += 1
    for rows in groups.values():
        for row in rows.values():
            if row["unresolved"]:
                row["errors"] = None
            if row["timed"] != row["count"]:
                row["duration"] = None
            for key in ("ttft_ms", "throughput"):
                stat = row[key]
                stat["value"] = stat["value"] / stat["samples"] if stat["samples"] else None
    totals = {}
    for key, rows in groups.items():
        values = list(rows.values())
        total = {
            field: sum(row[field] for row in values)
            if values and all(row[field] is not None for row in values)
            else None
            for field in ("count", "input", "output", "duration", "errors", "retries")
        }
        for field in ("ttft_ms", "throughput"):
            count = sum(row[field]["samples"] for row in values)
            total[field] = {
                "samples": count,
                "value": sum((row[field]["value"] or 0) * row[field]["samples"] for row in values)
                / count
                if count
                else None,
            }
        totals[key] = total
    return {**{key: list(rows.values()) for key, rows in groups.items()}, "totals": totals}


def run_metrics(run):
    actions = run.get("actions", ())
    models = lambda: (a for a in actions if a.get("kind") == "model")
    tools = lambda: (a for a in actions if is_tool(a))
    summary = run.get("summary", {})
    workload = summary.get("workload_metrics", {})
    recorded_rss = next(
        (
            r.get("max")
            for r in summary.get("resource_metrics", [])
            if r.get("name") == "workload_total:rss_mb"
        ),
        None,
    )
    return {
        # A profiler session duration includes setup/finalization. No makespan substitution.
        "makespan": number(run.get("execution_wall_time")),
        "recording_duration": number(summary.get("duration_ms")) / 1000
        if number(summary.get("duration_ms")) is not None
        else None,
        "status": run.get("status"),
        "agents": sum(a.get("id") != "workflow" for a in run.get("agents", ())),
        "requests": sum(1 for _ in models()),
        "input": strict_sum(models(), "tokens"),
        "output": strict_sum(models(), "output_tokens"),
        "tools": sum(1 for _ in tools()),
        "tool_errors": stats(tools())["errors"],
        "cpu": number(workload.get("cpu_time_seconds")),
        "rss": number(recorded_rss),
        "disk_read": number(workload.get("io_read_mb")),
        "disk_write": number(workload.get("io_write_mb")),
        "network_rx": number(workload.get("net_receive_mb")),
        "network_tx": number(workload.get("net_transmit_mb")),
    }


def descriptor(run):
    metadata = {
        key: run.get(key)
        for key in (
            "id",
            "task",
            "model",
            "harness",
            "condition",
            "repetition",
            "status",
            "time_origin",
            "execution_started_at",
            "configuration",
            "scheduler",
            "hardware",
            "commit",
            "environment",
            "parameters",
        )
        if run.get(key) is not None
    }
    # Profiler perf_counter offsets are never interpreted as Unix timestamps.
    return {
        "id": run["id"],
        "title": run.get("title"),
        "metrics": run_metrics(run),
        "metadata": metadata,
        "agents": [
            {key: agent.get(key) for key in ("id", "label", "model", "harness", "status")}
            for agent in islice(run.get("agents", ()), 100)
        ],
        "extent": run.get("duration"),
        "history_url": historical_link(run["id"]),
        "time_basis": run.get("clock", {}).get("origin", "First recorded span"),
        "coverage": run.get("summary", {}).get("coverage", run.get("coverage", {})),
        "note": "Finalized historical recording. Task completion and execution-start timestamp are not implied. "
        "Counts cover captured operations; missing usage makes token totals unavailable.",
    }


def agent_rows(run, skip=0, identifiers=None):
    roster = (a for a in run.get("agents", ()) if a.get("id") != "workflow")
    if identifiers is None:
        chosen = list(islice(roster, skip, skip + PAGE_SIZE))
    else:
        chosen = [a for a in roster if a["id"] in identifiers]
    result = {
        agent["id"]: {
            "id": agent["id"],
            "status": agent.get("status"),
            "duration": number(agent.get("duration")),
            "requests": 0,
            "tools": 0,
            "input": 0,
            "output": 0,
            "errors": 0,
            "cpu": None,
            "rss": None,
            "history_url": historical_link(run["id"], agent["id"]),
        }
        for agent in chosen
    }
    for action in run.get("actions", ()):
        row = result.get(action.get("agent"))
        if row is None:
            continue
        if action.get("kind") == "model":
            row["requests"] += 1
            for key, field in (("input", "tokens"), ("output", "output_tokens")):
                measured = number(action.get(field))
                row[key] = (
                    row[key] + measured
                    if measured is not None and measured >= 0 and row[key] is not None
                    else None
                )
        row["tools"] += is_tool(action)
        if action.get("outcome") not in TERMINAL:
            row["errors"] = None
        elif row["errors"] is not None:
            row["errors"] += action.get("outcome") in {"failed", "failure", "error", "timeout"}
    for row in result.values():
        if not row["requests"]:
            row["input"] = row["output"] = None
    total = sum(a.get("id") != "workflow" for a in run.get("agents", ()))
    return {
        "agents": list(result.values()),
        "next": skip + PAGE_SIZE if skip + PAGE_SIZE < total else None,
        "total": total,
    }


def series_info(name):
    # Exact, explicit metric scopes only. Resource instance IDs never imply
    # correspondence between sandboxes/processes belonging to different runs.
    parts = name.split(":")
    if parts[0] in {"host", "workload_total", "workload"} and len(parts) == 2:
        scope, metric = parts
        comparable = True
    elif parts[0] in {"sandbox", "process", "cg"} and len(parts) >= 3:
        scope, metric = ":".join(parts[:-1]), parts[-1]
        comparable = False
    else:
        scope, metric, comparable = "Scope not recorded", name, False
    unit = (
        "%"
        if metric.endswith("_pct")
        else "MB/s"
        if metric.endswith("_mb_s")
        else "MB"
        if metric.endswith("_mb")
        else "s"
        if metric.endswith("_seconds")
        else "Recorded units"
    )
    return {"name": name, "scope": scope, "metric": metric, "unit": unit, "comparable": comparable}


def systems(run, name="", start=0, end=None):
    counters = run.get("counters", {})
    series = [series_info(key) for key in counters if "_est" not in key and "power" not in key]
    points = counters.get(name, ())
    visible = [
        p
        for p in points
        if len(p) >= 2
        and number(p[0]) is not None
        and number(p[1]) is not None
        and p[0] >= start
        and (end is None or p[0] <= end)
    ]
    return {
        "series": series,
        "points": reduce_samples(visible, 400),
        "scope": series_info(name),
        "source": "Recorded samples, visualization downsampled; no interpolation",
        "time_basis": run.get("clock", {}).get("origin", "First recorded span"),
    }


def create_router(profile_dir):
    router = APIRouter()

    async def entries():
        import time

        return await asyncio.to_thread(catalog, str(profile_dir()), int(time.time() // 30))

    async def selected(run_id):
        entry = next((e for e in await entries() if e["id"] == run_id), None)
        if entry is None or not await asyncio.to_thread(eligible, entry):
            raise HTTPException(404, "Finalized recorded run not available.")
        path = Path(entry["path"])
        try:
            # Shared canonical historical cache. No deepcopy, replay, or graph projection.
            return await asyncio.to_thread(
                load_run, str(path), path.stat().st_mtime, json.dumps(entry, sort_keys=True)
            )
        except (ValueError, TypeError, OSError) as error:
            raise HTTPException(422, f"Historical data unavailable: {error}") from error

    @router.get("/compare")
    async def compare_page():
        return FileResponse(
            Path(__file__).parent / "static/compare.html", headers={"Cache-Control": "no-store"}
        )

    @router.get("/api/compare/runs")
    async def runs():
        def rows(values):
            return [
                {
                    key: e[key]
                    for key in ("id", "title", "subtitle", "task", "model", "harness")
                    if key in e
                }
                for e in values
                if eligible(e)
            ]

        return {"runs": await asyncio.to_thread(rows, await entries())}

    @router.get("/api/compare/runs/{run_id}")
    async def run(run_id: str):
        return await asyncio.to_thread(descriptor, await selected(run_id))

    @router.get("/api/compare/runs/{run_id}/operations")
    async def operations(
        run_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=100),
        agent: str = "",
        kind: str = "",
        tail: bool = False,
        query: str = Query("", max_length=200),
        start: float | None = Query(None, ge=0),
        end: float | None = Query(None, ge=0),
    ):
        if any(v is not None and not math.isfinite(v) for v in (start, end)):
            raise HTTPException(422, "Time bounds must be finite.")
        return await asyncio.to_thread(
            page,
            await selected(run_id),
            skip=skip,
            limit=limit,
            tail=tail,
            agent=agent,
            kind=kind,
            query=query,
            start=start,
            end=end,
        )

    @router.get("/api/compare/runs/{run_id}/operation")
    async def detail(run_id: str, event_id: str, offset: int = Query(0, ge=0)):
        def find(run):
            action = next((a for a in run.get("actions", ()) if a["id"] == event_id), None)
            if action is None:
                raise HTTPException(404, "Recorded operation not available.")
            text = json.dumps(action, ensure_ascii=False, indent=2)
            return {
                "operation": operation(action),
                "details": text[offset : offset + 8000],
                "offset": offset,
                "length": len(text),
                "next": offset + 8000 if offset + 8000 < len(text) else None,
            }

        return await asyncio.to_thread(find, await selected(run_id))

    @router.get("/api/compare/runs/{run_id}/match")
    async def match(run_id: str, task: str, invocation: str):
        def find(run):
            rows = list(
                islice((a for a in run.get("actions", ()) if identity(a) == [task, invocation]), 2)
            )
            return {
                "operation": operation(rows[0]) if len(rows) == 1 else None,
                "reason": "No unique recorded task/invocation match." if len(rows) != 1 else "",
            }

        return await asyncio.to_thread(find, await selected(run_id))

    @router.get("/api/compare/runs/{run_id}/agents")
    async def agents(
        run_id: str, skip: int = Query(0, ge=0), identifiers: str = Query("", max_length=20000)
    ):
        ids = None
        if identifiers:
            try:
                ids = json.loads(identifiers)
            except ValueError as error:
                raise HTTPException(422, "Agent identifiers must be a JSON array.") from error
            if (
                not isinstance(ids, list)
                or len(ids) > 200
                or not all(isinstance(i, str) for i in ids)
            ):
                raise HTTPException(422, "At most 200 explicit agent identifiers are supported.")
            ids = set(ids)
        return await asyncio.to_thread(agent_rows, await selected(run_id), skip, ids)

    @router.get("/api/compare/runs/{run_id}/activity")
    async def llm_tools(run_id: str):
        return await asyncio.to_thread(activity, await selected(run_id))

    @router.get("/api/compare/runs/{run_id}/systems")
    async def resources(
        run_id: str,
        name: str = "",
        start: float = Query(0, ge=0),
        end: float | None = Query(None, ge=0),
    ):
        if not math.isfinite(start) or end is not None and not math.isfinite(end):
            raise HTTPException(422, "Time bounds must be finite.")
        return await asyncio.to_thread(systems, await selected(run_id), name, start, end)

    @router.get("/api/compare/runs/{run_id}/critical")
    async def critical(run_id: str):
        await selected(run_id)
        # Existing build_graph explicitly declares system_critical_path unavailable
        # for saved data. Do not build/retain another full DAG for this page.
        return {
            "available": False,
            "reason": "Not available. The existing historical causal graph does not "
            "establish a system-wide critical path: cross-agent logical dependencies and complete lifecycle boundaries are not captured.",
            "history_url": "/profiler?" + urlencode({"run": run_id, "view": "critical"}),
        }

    return router
