"""Historical loading must not turn raw trace volume into graph volume."""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from agency.observability.agwebui import critical_path_store as store


def span(name, start=0, duration=1, **attrs):
    return {
        "ph": "X",
        "name": name,
        "ts": start * 1e6,
        "dur": duration * 1e6,
        "args": {"agency.agent_id": "worker", **attrs},
    }


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 64, 256 * 1024])
@pytest.mark.parametrize("object_root", [False, True])
def test_streaming_parser_handles_chunk_boundaries_and_metadata(tmp_path, chunk_size, object_root):
    events = [
        span("tool:Bash", 1.123456, 2, **{"tool.result": 'Unicode café 雨 and [\\" braces {}'}),
        None,
        {"ph": "C", "ts": 12345, "args": {"value": -1.23e4}},
    ]
    document = (
        {
            "displayTimeUnit": "ms",
            "traceEvents": events,
            "otherData": {"agprof_time_origin_ns": 123456789},
        }
        if object_root
        else events
    )
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(document))
    metadata = {}
    assert list(store.iter_trace_events(path, metadata, chunk_size=chunk_size)) == events
    assert metadata == ({"agprof_time_origin_ns": 123456789} if object_root else {})


@pytest.mark.parametrize("value", [1.23, -12.345, 1e20, 0.000001])
def test_primitive_numbers_can_split_at_decimal_and_exponent(tmp_path, value):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [], "scalar": value}))
    assert list(store.iter_trace_events(path, chunk_size=1)) == []


@pytest.mark.parametrize(
    "document",
    [
        '{"traceEvents":[',
        '{"traceEvents":[],}',
        "[{},]",
        '{"traceEvents":[],"traceEvents":[]}',
        '{"traceEvents":[]} trailing',
        '{"nothing":[]}',
        '{"traceEvents":12}',
        '{"traceEvents":[{"ph":"X","ts":NaN}]}',
    ],
)
def test_malformed_documents_are_rejected_explicitly(tmp_path, document):
    path = tmp_path / "trace.json"
    path.write_text(document)
    with pytest.raises(ValueError):
        list(store.iter_trace_events(path, chunk_size=3))


def test_raw_syscalls_and_python_stacks_are_not_retained(tmp_path):
    path = tmp_path / "agprof.trace.json"
    inherited = {"request_kind": "skill", "request_id": "run0"}
    events = [span("syscall:read", i / 100, 0.001, **inherited) for i in range(10000)]
    events.extend(
        [
            span("orchestrator.worker", 0, 100, **inherited),
            span("run0:skill:worker", 1, 90, span_id="envelope", **inherited),
            span("engine:execute", 2, 88, span_id="engine", **inherited),
            span("tool:Bash", 3, 4, span_id="tool", **inherited),
            span("process:gcc", 3.1, 3.5, span_id="process", **inherited),
        ]
    )
    path.write_text(json.dumps({"traceEvents": events}))
    run = store.load_compact_run(path, {"id": "saved", "title": "Saved", "source": "recorded"})
    assert len(run["actions"]) == 4
    assert {a["id"] for a in run["actions"]} == {"envelope", "engine", "tool", "process"}
    assert run["duration"] == 100
    assert run["coverage"]["events"] == 10005
    assert all(a["kind"] != "syscall" for a in run["actions"])


def test_missing_end_and_span_limit_are_disclosed(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MAX_OPERATIONS", 2)
    path = tmp_path / "agprof.trace.json"
    missing = span("tool:Bash", 1, span_id="missing")
    missing.pop("dur")
    path.write_text(
        json.dumps({"traceEvents": [missing, span("tool:read", 2, 1), span("tool:write", 8, 2)]})
    )
    run = store.load_compact_run(path, {"id": "saved"})
    assert run["actions"][0]["missing_end"]
    assert run["coverage"]["omitted_operations"] == 1
    assert run["duration"] == 9


def test_counter_identity_and_reduction_preserve_peak_without_pid_matching(tmp_path):
    path = tmp_path / "agprof.trace.json"
    name = "agency-counter-v1:" + json.dumps(
        {"label": "rss_mb", "path": [["process:123-99", "gcc"]]}
    )
    points = [
        {
            "ph": "C",
            "ts": i * 1000,
            "name": name,
            "pid": 123,
            "args": {"value": 999 if i == 711 else 1},
        }
        for i in range(10000)
    ]
    path.write_text(json.dumps({"traceEvents": [span("tool:Bash", 0, 10), *points]}))
    run = store.load_compact_run(path, {"id": "saved"})
    reduced = run["counters"]["process:123-99:rss_mb"]
    assert len(reduced) <= store.COUNTER_POINTS
    assert max(value for _, value in reduced) == 999
    assert run["coverage"]["counter_samples"] == 10000


def test_canonical_exchange_replaces_truncated_evidence_and_reads_dependencies(tmp_path):
    logs, profile = tmp_path / "logs", tmp_path / "profiler"
    logs.mkdir()
    profile.mkdir()
    path = profile / "agprof.trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    span(
                        "llm:attempt[0]",
                        0,
                        2,
                        span_id="model",
                        call_label="exchange",
                        request_id="run1",
                        **{"llm.messages": "cut", "llm.response": "cut"},
                    )
                ]
            }
        )
    )
    con = sqlite3.connect(logs / "global_data.sqlite3")
    con.executescript(
        "CREATE TABLE exchanges(call_label TEXT,name TEXT); CREATE TABLE exchange_chain(call_label TEXT,kind TEXT,seq INTEGER,hash TEXT); CREATE TABLE blocks(hash TEXT,payload TEXT); CREATE TABLE events(type TEXT,payload TEXT);"
    )
    con.execute("INSERT INTO exchanges VALUES('exchange','worker')")
    blocks = [
        ("prompt", {"type": "text", "role": "user", "text": "SECRET PROMPT"}),
        ("response", {"type": "tool_use", "role": "assistant", "id": "call", "name": "Bash"}),
        ("response", {"type": "metadata", "usage": {"prompt_tokens": 42}, "ttft_ms": 150}),
    ]
    for index, (kind, block) in enumerate(blocks):
        con.execute("INSERT INTO blocks VALUES(?,?)", (str(index), json.dumps(block)))
        con.execute(
            "INSERT INTO exchange_chain VALUES('exchange',?,?,?)", (kind, index, str(index))
        )
    dependency = {
        "request_id": "run1",
        "dependencies": [{"request_id": "run0", "kind": "input"}],
    }
    con.execute("INSERT INTO events VALUES('request_dependencies',?)", (json.dumps(dependency),))
    con.commit()
    con.close()
    run = store.load_compact_run(path, {"id": "saved"})
    model = run["actions"][0]
    assert model["causal_evidence"]["complete"]
    assert model["tool_call_ids"] == ["call"]
    assert model["tokens"] == 42 and model["metadata"]["ttft_ms"] == 150
    assert "SECRET PROMPT" not in json.dumps(run)
    assert run["request_dependencies"] == [dependency]
    with sqlite3.connect(logs / "global_data.sqlite3") as con:
        con.execute("DELETE FROM blocks WHERE hash='0'")
    incomplete = store.load_compact_run(path, {"id": "saved"})
    assert not incomplete["actions"][0]["causal_evidence"]["complete"]
    assert any(
        "missing block" in reason for reason in incomplete["coverage"]["causal_evidence"]["gaps"]
    )


def test_resource_series_and_global_point_budget_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MAX_COUNTER_SERIES", 2)
    monkeypatch.setattr(store, "MAX_COUNTER_POINTS", 8)
    path = tmp_path / "agprof.trace.json"
    points = [
        {"ph": "C", "ts": i * 1000, "name": f"series-{i % 3}", "args": {"value": i}}
        for i in range(1000)
    ]
    path.write_text(json.dumps({"traceEvents": [span("tool:run", 0, 1), *points]}))
    run = store.load_compact_run(path, {"id": "saved"})
    assert len(run["counters"]) <= 2
    assert sum(map(len, run["counters"].values())) <= 8
    assert run["coverage"]["counter_samples"] == 1000
    assert run["coverage"]["omitted_counter_samples"] > 0
    assert any("memory limits" in reason for reason in run["coverage"]["historical_warnings"])


def test_shared_sandbox_counter_names_never_merge_independent_scopes(tmp_path):
    path = tmp_path / "agprof.trace.json"
    points = []
    for sandbox, value in (("one", 10), ("two", 90)):
        name = "agency-counter-v1:" + json.dumps(
            {"label": "Memory pressure", "path": [[f"sandbox:{sandbox}", sandbox]]}
        )
        points.append({"ph": "C", "ts": 1000, "name": name, "args": {"value": value}})
    path.write_text(json.dumps({"traceEvents": [span("tool:Bash", 0, 1), *points]}))
    run = store.load_compact_run(path, {"id": "saved"})
    assert run["counters"]["sandbox:one:Memory pressure"] == [[0.001, 10]]
    assert run["counters"]["sandbox:two:Memory pressure"] == [[0.001, 90]]


def test_evidence_reference_budget_does_not_retain_partial_prompt_as_complete(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(store, "MAX_EVIDENCE_REFS", 1)
    path = tmp_path / "agprof.trace.json"
    messages = json.dumps([{"role": "user", "blocks": [{"type": "text", "text": "task"}]}])
    response = json.dumps(
        {"blocks": [{"role": "assistant", "type": "tool_use", "id": "tool", "name": "Bash"}]}
    )
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    span(
                        "llm:attempt[0]",
                        0,
                        1,
                        **{"llm.messages": messages, "llm.response": response},
                    )
                ]
            }
        )
    )
    run = store.load_compact_run(path, {"id": "saved"})
    assert run["actions"][0]["causal_evidence"] == {"prompt": [], "response": [], "complete": False}
    assert any(
        "evidence memory limit" in reason for reason in run["coverage"]["historical_warnings"]
    )


def test_cross_agent_clock_requires_recorded_export_declaration(tmp_path):
    path = tmp_path / "agprof.trace.json"
    events = [
        span("tool:one", 0, 1, **{"agency.agent_id": "one"}),
        span("tool:two", 0, 1, **{"agency.agent_id": "two"}),
    ]
    path.write_text(json.dumps({"traceEvents": events}))
    assert not store.load_compact_run(path, {"id": "saved"})["clock"]["shared"]
    path.write_text(
        json.dumps({"traceEvents": events, "otherData": {"agprof_clock": "perf_counter_ns"}})
    )
    assert store.load_compact_run(path, {"id": "saved"})["clock"]["shared"]


def test_oversized_optional_summary_and_canonical_block_remain_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "MAX_EVENT_BYTES", 4096)
    directory = tmp_path / "profiler"
    directory.mkdir()
    path = directory / "agprof.trace.json"
    path.write_text(
        json.dumps(
            {"traceEvents": [span("llm:attempt[0]", 0, 1, call_label="call", span_id="model")]}
        )
    )
    (directory / "summary.json").write_text(json.dumps({"large": "x" * 5000}))
    with sqlite3.connect(directory / "profile_data.sqlite3") as con:
        con.executescript(
            "CREATE TABLE exchanges(call_label TEXT,name TEXT); CREATE TABLE exchange_chain(call_label TEXT,kind TEXT,seq INTEGER,hash TEXT); CREATE TABLE blocks(hash TEXT,payload TEXT);"
        )
        con.execute("INSERT INTO exchanges VALUES('call','worker')")
        con.execute("INSERT INTO exchange_chain VALUES('call','prompt',0,'large')")
        con.execute(
            "INSERT INTO blocks VALUES('large',?)",
            (json.dumps({"type": "text", "role": "user", "text": "x" * 5000}),),
        )
    run = store.load_compact_run(path, {"id": "saved"})
    assert run["summary"] == {}
    assert not run["actions"][0]["causal_evidence"]["complete"]
    assert any("summary" in reason for reason in run["coverage"]["historical_warnings"])
    assert any("memory limit" in reason for reason in run["coverage"]["causal_evidence"]["gaps"])


def test_native_history_stream_retains_tool_identity_and_discloses_missing_prompts(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {
            "kind": "model_exchange",
            "event_id": "model",
            "timestamp_ns": 1_000_000_000,
            "duration_ns": 1_000_000_000,
            "agent_id": "native",
            "response": {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "call", "function": {"name": "Bash", "arguments": "{}"}}],
                },
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        },
        {
            "kind": "tool_annotation",
            "event_id": "tool",
            "model_tool_call_id": "call",
            "arguments": {"command": "echo evidence"},
            "annotation": {"status": "valid", "raw": {"purpose": "check"}},
        },
        {
            "kind": "tool_result",
            "event_id": "tool",
            "timestamp_ns": 2_000_000_000,
            "duration_ns": 1_000_000_000,
            "agent_id": "native",
            "tool_name": "Bash",
            "result": "evidence",
            "model_tool_call_id": "call",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\nmalformed observer record\n"
    )
    run = store.load_compact_native(path, {"id": "native", "model": "recorded"})
    assert run["duration"] == 2 and len(run["actions"]) == 2
    assert run["actions"][0]["tool_call_ids"] == ["call"]
    assert run["actions"][1]["model_tool_call_id"] == "call"
    assert run["coverage"]["partial_lines"] == 1
    assert not run["actions"][0]["causal_evidence"]["complete"]


def test_cache_revision_includes_canonical_wal_changes(tmp_path):
    path = tmp_path / "agprof.trace.json"
    path.write_text("[]")
    first = store.fingerprint(path)
    wal = tmp_path / "profile_data.sqlite3-wal"
    wal.write_text("new evidence")
    assert store.fingerprint(path) != first


def test_request_attribution_traverses_recorded_dropped_parent_spans_only(tmp_path):
    path = tmp_path / "agprof.trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    span(
                        "run0:skill:worker",
                        0,
                        9,
                        span_id="request",
                        request_id="run0",
                        request_kind="skill",
                    ),
                    span("runtime.harness", 1, 7, span_id="harness", parent_span_id="request"),
                    span("tool:Bash", 2, 2, span_id="tool", parent_span_id="harness"),
                    span("tool:unrelated", 3, 2, span_id="unrelated"),
                ]
            }
        )
    )
    run = store.load_compact_run(path, {"id": "saved"})
    actions = {action["id"]: action for action in run["actions"]}
    assert actions["tool"]["metadata"]["request_id"] == "run0"
    assert actions["tool"]["metadata"]["request_attribution"] == "recorded_parent_span_chain"
    assert "request_id" not in actions["unrelated"]["metadata"]
    assert "harness" not in actions


def test_historical_routes_are_allowlisted_compact_and_read_only(monkeypatch, tmp_path):
    from agency.observability.agwebui import investigator, server

    path = tmp_path / "agprof.trace.json"
    path.write_text(json.dumps({"traceEvents": [span("tool:Bash", 0, 3, span_id="tool")]}))
    entry = {"id": "saved", "title": "Saved", "source": "recorded", "path": str(path)}
    monkeypatch.setattr(investigator, "catalog", lambda *args: [entry])
    client = TestClient(server.app)
    assert client.get("/api/critical-path/runs").json()["runs"] == [
        {k: v for k, v in entry.items() if k != "path"}
    ]
    response = client.get("/api/critical-path/saved")
    assert response.status_code == 200
    assert response.json()["summary"]["wall_time"] == 3
    assert "traceEvents" not in response.text and '"path"' not in response.text
    assert client.get("/api/critical-path/unregistered").status_code == 404
    assert client.get("/api/critical-path/saved/operations/tool").status_code == 200
    assert client.post("/api/critical-path/saved").status_code == 405


def test_payload_limits_preserve_summary_and_report_omitted_nodes():
    from agency.observability.agwebui.critical_path_api import _bounded

    profile = {
        "summary": {"wall_time": 12},
        "operations": [{"id": "parallel", "start": 0}, {"id": "critical", "start": 10}],
        "critical_path": {"operation_ids": ["critical"]},
        "edges": [{"source": "parallel", "target": "critical"}],
        "agents": [],
        "processes": [],
    }
    bounded = _bounded(profile, 1)
    assert bounded["operations"][0]["id"] == "critical"
    assert bounded["summary"] == profile["summary"]
    assert bounded["truncation"] == {"operations": 1, "edges": 1, "agents": 0, "processes": 0}


def test_operation_detail_includes_owning_tool_processes(monkeypatch, tmp_path):
    from agency.observability.agwebui import investigator, server

    path = tmp_path / "agprof.trace.json"
    identity = {
        "agency.sandbox_id": "box",
        "agency.namespace_pid": 2,
        "agency.pid_namespace": "pid:[99]",
        "agency.process_start_ticks": 500,
    }
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    span("tool:Bash", 0, 3, span_id="tool", call_id="host-call"),
                    span(
                        "process:python",
                        0.1,
                        2,
                        span_id="process",
                        tool_call_id="host-call",
                        **identity,
                    ),
                ]
            }
        )
    )
    entry = {
        "id": "process-fixture",
        "title": "Process evidence",
        "source": "recorded",
        "path": str(path),
    }
    monkeypatch.setattr(investigator, "catalog", lambda *args: [entry])
    response = TestClient(server.app).get("/api/critical-path/process-fixture/operations/tool")
    assert response.status_code == 200
    assert len(response.json()["processes"]) == 1
    assert response.json()["processes"][0]["owner_ids"] == ["tool"]


def test_catalog_discovers_custom_runs_and_marks_mock_recordings(monkeypatch, tmp_path):
    from agency.observability.agwebui import investigator

    monkeypatch.setattr(investigator, "ROOT", tmp_path)
    real = tmp_path / "runs" / "custom" / "profiler"
    mock = tmp_path / "agency_runs" / "mock" / "profiler"
    for directory in (real, mock):
        directory.mkdir(parents=True)
        (directory / "agprof.trace.json").write_text('{"traceEvents":[]}')
    (real / "summary.json").write_text('["malformed optional summary"]')
    (mock / "summary.json").write_text(
        '{"data_source":"mock","duration_ms":"unknown","coverage":[]}'
    )
    entries = investigator.catalog(str(real), 987654321)
    assert {entry["source"] for entry in entries} == {"recorded", "synthetic"}
    assert all(entry["duration"] == 0 for entry in entries)
