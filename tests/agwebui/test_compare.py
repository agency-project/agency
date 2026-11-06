"""Comparison is a read-only projection of real historical schema fixtures."""

import json
import tracemalloc

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agency.observability.agwebui import compare
from agency.observability.agwebui.investigator import normalize_trace


def action(id="call", **kwargs):
    return {
        "id": id,
        "agent": "agent-a",
        "kind": "tool",
        "name": "exec",
        "start": 2.0,
        "duration": 3.0,
        "outcome": "success",
        "command": "echo x",
        "metadata": {},
        **kwargs,
    }


def run(actions=None, **kwargs):
    return {
        "id": "a",
        "title": "Recorded A",
        "source": "recorded",
        "duration": 10,
        "actions": actions if actions is not None else [action()],
        "agents": [{"id": "agent-a"}],
        "summary": {},
        **kwargs,
    }


def test_metrics_missing_partial_tokens_and_incomplete_calls():
    result = compare.run_metrics(
        run(
            [
                action(kind="model", tokens=0, output_tokens=10),
                action(id="two", kind="model", tokens=None, output_tokens=3),
                action(id="open", outcome="interrupted"),
            ]
        )
    )
    assert result["requests"] == 2
    assert result["input"] is None
    assert result["output"] == 13
    assert result["tool_errors"] is None
    assert result["makespan"] is result["cpu"] is result["rss"] is None
    assert result["status"] is None
    assert compare.operation(action(duration=500, outcome="interrupted"))["duration"] is None
    assert compare.operation(action(missing_start=True))["start"] is None


def test_missing_durations_survive_historical_normalizer():
    data = normalize_trace(
        {
            "traceEvents": [
                {"ph": "X", "name": "tool:exec", "ts": 10, "args": {"outcome": "success"}}
            ]
        },
        {"id": "a"},
    )
    assert data["actions"][0]["missing_end"]
    assert compare.operation(data["actions"][0])["duration"] is None


def test_pagination_different_lengths_search_and_relative_window():
    a = run([action(id=str(i), start=i * 2) for i in range(251)])
    b = run([action(id="only", start=4)])
    assert len(compare.page(a)["operations"]) == 100
    assert compare.page(a)["next"] == 100
    assert len(compare.page(a, skip=200)["operations"]) == 51
    assert compare.page(b)["next"] is None
    # The interval intersects the viewport, even when its start is earlier.
    assert [a["id"] for a in compare.page(a, start=7, end=8)["operations"]] == ["2", "3", "4"]
    assert compare.page(a, query="echo x", agent="unknown")["operations"] == []
    assert compare.page(a, query="ECHO")["operations"]
    assert compare.page(a, kind="model")["operations"] == []
    assert compare.page(a, skip=100000)["operations"] == []
    assert compare.page(a, query="echo", tail=True)["operations"][-1]["id"] == "250"
    assert compare.page(a, query="echo", tail=True)["offset"] == 151
    assert compare.number(True) is None


def test_large_projection_has_bounded_allocation_and_does_not_copy_payloads():
    # Many repeated references make the test cheap without executing any workload.
    original = action(result="x" * 100000)
    data = run([original] * 1_000_001)
    tracemalloc.start()
    page = compare.page(data, skip=999_900)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(page["operations"]) == 100
    assert peak < 1_000_000
    assert len(page["operations"][0]["result_preview"]) == 600
    assert len(original["result"]) == 100000
    assert data["actions"][0] is original


def test_matching_requires_explicit_compound_identity_not_order_or_span_id():
    assert compare.identity(action(id="action-0")) is None
    assert compare.identity(action(metadata={"span_id": "stable", "task_id": "task"})) is None
    assert compare.identity(action(metadata={"task_id": "task", "invocation_id": "invoke"})) == [
        "task",
        "invoke",
    ]


def test_agents_different_rosters_and_matching_outside_page():
    data = run(
        [action(agent="right", kind="model", tokens=3, output_tokens=4)],
        agents=[{"id": str(i)} for i in range(100)] + [{"id": "right"}],
    )
    assert compare.agent_rows(data)["next"] == 100
    assert compare.agent_rows(data, identifiers={"right"})["agents"][0]["input"] == 3
    assert compare.agent_rows(data, identifiers={"absent"})["agents"] == []
    assert compare.agent_rows(data, 100)["agents"][0]["duration"] is None


def test_activity_totals_missing_retries_distribution_and_overlapping_durations():
    data = run(
        [
            action(
                kind="model",
                model="m",
                tokens=2,
                output_tokens=3,
                metadata={"ttft_ms": 20, "retry_index": 0},
            ),
            action(
                id="second",
                kind="model",
                model="m",
                tokens=5,
                output_tokens=4,
                metadata={"ttft_ms": 40, "retry_index": 2},
            ),
            action(id="tool", duration=0.02),
            action(id="open", outcome="incomplete"),
        ]
    )
    result = compare.activity(data)
    assert result["models"][0]["retries"] == 1
    assert result["models"][0]["ttft_ms"] == {"value": 30, "samples": 2}
    assert result["totals"]["models"]["duration"] == 6  # Labelled interval sum, never makespan.
    assert result["totals"]["tools"]["duration"] is None
    assert sum(result["tools"][0]["distribution"]) == 1
    assert result["totals"]["models"]["throughput"]["value"] is None


def test_resource_scopes_and_downsampling_are_not_attribution_or_interpolation():
    points = [[i, i % 10] for i in range(5000)]
    data = run(counters={"host:cpu_util_pct": points, "sandbox:local:mem_mb": points})
    result = compare.systems(data, "host:cpu_util_pct", 10, 3000)
    assert len(result["points"]) <= 400
    assert result["scope"]["scope"] == "host"
    assert result["scope"]["unit"] == "%"
    assert compare.series_info("sandbox:local:mem_mb")["comparable"] is False
    assert compare.series_info("rss_mb")["scope"] == "Scope not recorded"
    assert len(points) == 5000


@pytest.fixture
def client(tmp_path, monkeypatch):
    entries = []
    saved = {}
    for id in ("a", "b", "not-finalized", "synthetic"):
        directory = tmp_path / id
        directory.mkdir()
        trace = directory / "agprof.trace.json"
        trace.write_text('{"traceEvents": []}')
        if id != "not-finalized":
            (directory / "summary.json").write_text(
                json.dumps({"generated_at": "2026-10-14T12:00:00Z", "data_source": "measured"})
            )
        entries.append(
            {
                "id": id,
                "title": id,
                "path": str(trace),
                "source": "synthetic" if id == "synthetic" else "recorded",
            }
        )
        saved[str(trace)] = run(id=id)
    monkeypatch.setattr(compare, "catalog", lambda *args: entries)
    monkeypatch.setattr(compare, "load_run", lambda path, *args: saved[path])
    app = FastAPI()
    app.include_router(compare.create_router(lambda: tmp_path))
    return TestClient(app), saved


def test_catalog_only_finalized_historical_recordings_and_no_execution_routes(client):
    browser, _ = client
    assert [r["id"] for r in browser.get("/api/compare/runs").json()["runs"]] == ["a", "b"]
    assert browser.get("/api/compare/runs/live").status_code == 404
    assert browser.get("/api/compare/runs/not-finalized").status_code == 404
    assert browser.get("/compare").status_code == 200
    assert browser.post("/api/compare/runs/a/execute").status_code == 404


def test_detail_pagination_match_ambiguity_and_historical_links(client):
    browser, saved = client
    data = next(value for value in saved.values() if value["id"] == "a")
    data["actions"][0]["result"] = "x" * 20000
    data["actions"][0]["metadata"] = {"task_id": "t", "invocation_id": "i"}
    response = browser.get("/api/compare/runs/a/operation", params={"event_id": "call"}).json()
    assert len(response["details"]) == 8000
    assert response["next"] == 8000
    assert (
        browser.get("/api/compare/runs/a/operation", params={"event_id": "absent"}).status_code
        == 404
    )
    assert (
        browser.get("/api/compare/runs/a/match", params={"task": "t", "invocation": "i"}).json()[
            "operation"
        ]["id"]
        == "call"
    )
    data["actions"].append(action(id="duplicate", metadata={"task_id": "t", "invocation_id": "i"}))
    assert (
        browser.get("/api/compare/runs/a/match", params={"task": "t", "invocation": "i"}).json()[
            "operation"
        ]
        is None
    )
    assert browser.get("/api/compare/runs/a/critical").json()["available"] is False
    assert (
        browser.get("/api/compare/runs/a").json()["history_url"]
        == "/profiler?run=a&view=trajectory"
    )
    assert (
        compare.historical_link("space / id", "agent & id")
        == "/profiler?run=space+%2F+id&view=trajectory&agent=agent+%26+id"
    )


def test_api_validates_bounded_queries(client):
    browser, _ = client
    for params in ({"limit": 101}, {"skip": -1}, {"start": "nan"}, {"end": "inf"}):
        assert browser.get("/api/compare/runs/a/operations", params=params).status_code == 422
    assert (
        browser.get(
            "/api/compare/runs/a/agents", params={"identifiers": json.dumps(["a"] * 201)}
        ).status_code
        == 422
    )
