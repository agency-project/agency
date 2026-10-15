"""Saved CPU resources retain explicit ownership through the trace boundary."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from agency.observability.agwebui.investigator import normalize_trace
from agency.observability.profiler.agprof_trace import build_trace


def span(agent, sandbox=None, **extra):
    attrs = {"agency.agent_id": agent, **extra}
    if sandbox is not None:
        attrs["agency.sandbox_id"] = sandbox
    return {"ph": "X", "name": "tool:run", "ts": 0, "dur": 2_000_000, "args": attrs}


def counter(name, value=200, ts=2_000_000):
    return {"ph": "C", "name": name, "ts": ts, "args": {"value": value}}


def projected():
    records = [
        (
            i,
            "tool:run",
            0,
            2_000_000_000,
            None,
            None,
            {
                "agency.agent_id": agent,
                "agency.sandbox_id": sandbox,
                "agency.execution_side": "sandbox",
            },
            i,
            None,
        )
        for i, (agent, sandbox) in enumerate([("a", "shared"), ("b", "shared"), ("c", "own")], 1)
    ]
    samples = [
        (0, "cg:shared:cpu_us", 0),
        (2_000_000_000, "cg:shared:cpu_us", 4_000_000),
        (0, "cg:own:cpu_us", 0),
        (2_000_000_000, "cg:own:cpu_us", 1_000_000),
    ]
    return normalize_trace(build_trace(records, samples, process_info={}, started_ns=0), {})


def test_exported_trace_preserves_shared_and_individual_sandbox_ownership():
    run = projected()
    agents = {agent["id"]: agent for agent in run["agents"]}
    assert agents["a"]["sandbox"] == agents["b"]["sandbox"] == "shared"
    assert agents["c"]["sandbox"] == "own"
    assert run["sandbox_owners"] == {"shared": ["a", "b"], "own": ["c"]}
    assert run["counters"]["sandbox:shared:cpu_pct"] == [[2, 200]]
    assert run["counters"]["sandbox:own:cpu_pct"] == [[2, 50]]


def test_multiple_recorded_sandboxes_are_preserved_without_last_writer_ownership():
    run = normalize_trace([span("a", "old"), span("a", "new"), span("b", "old")], {})
    a = next(row for row in run["agents"] if row["id"] == "a")
    assert a["sandboxes"] == ["new", "old"]
    assert "sandbox" not in a
    assert run["sandbox_owners"] == {"old": ["a", "b"], "new": ["a"]}


def test_versioned_counter_paths_preserve_owners_without_retained_owner_spans():
    shared = {
        "version": 1,
        "path": [["shared", "Shared"], ["sandbox:box", "Display name"]],
        "shared_with": ["a", "b"],
        "label": "sandbox:box:cpu_pct",
    }
    solo = {
        "version": 1,
        "path": [["agent:c", "C"], ["sandbox:own", "Own"]],
        "label": "sandbox:own:cpu_pct",
    }
    run = normalize_trace(
        [
            span("a"),
            span("c"),
            counter("agency-counter-v1:" + json.dumps(shared)),
            counter("agency-counter-v1:" + json.dumps(solo), 50),
        ],
        {},
    )
    assert run["sandbox_owners"] == {"box": ["a", "b"], "own": ["c"]}
    assert next(a for a in run["agents"] if a["id"] == "a")["sandbox"] == "box"
    assert "b" not in [a["id"] for a in run["agents"]]  # no invented execution record


@pytest.mark.parametrize(
    "presentation", [None, [], "not json", {"path": [["shared", "Agent a sandbox box"]]}]
)
def test_missing_or_ambiguous_ownership_is_not_inferred_from_counter_labels(presentation):
    run = normalize_trace(
        [span("box", **{"agency_presentation": presentation}), counter("sandbox:box:cpu_pct")], {}
    )
    assert run["sandbox_owners"] == {}
    assert "sandbox" not in run["agents"][0]
    assert run["counters"]["sandbox:box:cpu_pct"] == [[2, 200]]


def test_legacy_unwrapped_counters_and_explicit_attributes_remain_usable():
    run = normalize_trace(
        [span("a", "box"), counter("cg:box:cpu_us", 0, ts=0), counter("cg:box:cpu_us", 4_000_000)],
        {},
    )
    assert run["agents"][0]["sandbox"] == "box"
    assert run["counters"]["cg:box:cpu_us"] == [[0, 0], [2, 4_000_000]]


def test_normalized_saved_trace_filters_through_actual_frontend():
    node = os.environ.get("AGENCY_TEST_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node required for cross-language projection; set AGENCY_TEST_NODE")
    module = Path(__file__).parents[2] / "agency/observability/agwebui/static/investigator-model.js"
    script = """
import fs from 'node:fs';
const {agentResources}=await import('data:text/javascript;base64,'+Buffer.from(fs.readFileSync(process.argv[1],'utf8')).toString('base64'));
const run=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify([['a'],['a','b'],['a','c'],['a','b','c']].map(ids=>agentResources(run,ids))));
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script, str(module)],
        input=json.dumps(projected()),
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(result.stdout)
    assert results[0]["cpu"] is None and results[0]["shared"] == 1
    assert results[1]["cpu"] == {"value": 200, "sandboxes": 1}
    assert results[2]["cpu"] == {"value": 50, "sandboxes": 1}
    assert results[3]["cpu"] == {"value": 250, "sandboxes": 2}
