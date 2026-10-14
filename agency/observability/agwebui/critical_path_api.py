"""Read-only historical critical path routes, with bounded response graphs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

from fastapi import APIRouter, HTTPException, Query

from . import investigator
from .critical_path_store import fingerprint, load_profile


def _bounded(profile, limit):
    operations = profile.get("operations", [])
    critical = set(profile.get("critical_path", {}).get("operation_ids", []))
    chosen = sorted(
        operations,
        key=lambda row: (row.get("id") not in critical, row.get("start") or 0, str(row.get("id"))),
    )[:limit]
    identifiers = {row["id"] for row in chosen}
    edges = profile.get("edges", [])
    agent = {row["id"]: row.get("agent") for row in chosen}
    eligible_edges = [
        edge
        for edge in edges
        if edge.get("source") in identifiers and edge.get("target") in identifiers
    ]
    eligible_edges.sort(
        key=lambda edge: (
            not edge.get("critical"),
            not edge.get("blocking"),
            agent.get(edge.get("source")) == agent.get(edge.get("target")),
            edge.get("confidence") not in {"recorded", "reconstructed"},
            str(edge.get("id")),
        )
    )
    visible_edges = eligible_edges[: limit * 2]
    processes, agents = profile.get("processes", []), profile.get("agents", [])
    return {
        **profile,
        "operations": sorted(chosen, key=lambda row: (row.get("start") or 0, str(row.get("id")))),
        "edges": visible_edges,
        "processes": processes[:1000],
        "agents": agents[:1000],
        "totals": {
            "operations": len(operations),
            "edges": len(edges),
            "processes": len(processes),
            "agents": len(agents),
        },
        "truncation": {
            "operations": len(operations) - len(chosen),
            "edges": len(edges) - len(visible_edges),
            "processes": max(0, len(processes) - 1000),
            "agents": max(0, len(agents) - 1000),
        },
    }


def router(profile_directory):
    routes = APIRouter(prefix="/api/critical-path")

    async def entries():
        values = await asyncio.to_thread(
            investigator.catalog, str(profile_directory()), int(time.time() // 30)
        )
        return [entry for entry in values if entry.get("source") == "recorded"]

    async def profile_for(run_id):
        entry = next((entry for entry in await entries() if entry["id"] == run_id), None)
        if entry is None:
            raise HTTPException(404, "Historical run not found. Refresh the catalog.")
        path = Path(entry["path"])
        try:
            revision = await asyncio.to_thread(fingerprint, path)
            return await asyncio.to_thread(
                load_profile, str(path), revision, json.dumps(entry, sort_keys=True)
            )
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            raise HTTPException(
                422, f"Historical critical path could not be analyzed: {exc}"
            ) from exc

    @routes.get("/runs")
    async def runs():
        return {
            "runs": [
                {key: value for key, value in entry.items() if key != "path"}
                for entry in await entries()
            ]
        }

    @routes.get("/{run_id}")
    async def profile(run_id: str, limit: int = Query(5000, ge=1, le=20000)):
        return _bounded(await profile_for(run_id), limit)

    @routes.get("/{run_id}/operations/{operation_id:path}")
    async def operation(run_id: str, operation_id: str):
        value = await profile_for(run_id)
        row = next((row for row in value.get("operations", []) if row["id"] == operation_id), None)
        if row is None:
            raise HTTPException(404, "Recorded operation not found.")
        return {
            "operation": row,
            "edges": [
                edge
                for edge in value.get("edges", [])
                if operation_id in {edge.get("source"), edge.get("target")}
            ],
            "processes": [
                process
                for process in value.get("processes", [])
                if operation_id in process.get("owner_ids", [])
            ],
        }

    return routes
