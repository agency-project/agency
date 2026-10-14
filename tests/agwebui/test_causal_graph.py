import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from agency.observability.agwebui.causal_graph import build_graph, critical_path, exchange_evidence
from agency.observability.agwebui.trajectory import Trajectory, LiveSource, replay_events
from agency.observability.agwebui import server


def text(value, role="assistant"):
    return {"type": "text", "text": value, "role": role}


def use(key, value="x"):
    return {
        "type": "tool_use",
        "id": key,
        "name": "exec",
        "input": {"cmd": value},
        "role": "assistant",
    }


def result(key):
    return {"type": "tool_result", "tool_call_id": key, "content": "ok", "role": "user"}


def action(key, start, duration, kind="model", agent="a", prompt=(), response=(), **extra):
    return {
        "id": key,
        "agent": agent,
        "name": key,
        "kind": kind,
        "start": start,
        "duration": duration,
        "outcome": "success",
        "causal_evidence": exchange_evidence(prompt, response),
        **extra,
    }


def trace(*actions, duration=20):
    return {"actions": list(actions), "duration": duration}


def parallel():
    first = action("m1", 0, 2, response=[use("t1"), use("t2")])
    one = action("t1", 2, 3, "tool", model_id="m1", model_tool_call_id="t1")
    two = action("t2", 2, 7, "tool", model_id="m1", model_tool_call_id="t2")
    last = action(
        "m2",
        9,
        2,
        prompt=[use("t1"), use("t2"), result("t1"), result("t2")],
        response=[text("done")],
    )
    return trace(first, one, two, last, duration=11)


def test_parallel_join_and_critical_path_slack():
    graph = build_graph(parallel())
    assert graph["critical_path"]["duration"] == 11
    assert graph["summary"]["operation_time"] == {"LLM": 4, "Tool": 10, "System": 0}
    assert graph["joins"] == [{"node": "m2:start", "dependencies": ["t1:end", "t2:end"]}]
    assert "t2:end" in graph["critical_path"]["node_ids"]
    assert "t1:end" not in graph["critical_path"]["node_ids"]
    assert graph["critical_path"]["slack"]["t1:start"] == 4
    assert not graph["coverage"]["complete"]


def test_sequential_react():
    graph = build_graph(
        trace(
            action("m1", 0, 2, response=[use("t")]),
            action("t", 2, 4, "tool", model_id="m1", model_tool_call_id="t"),
            action("m2", 6, 3, prompt=[use("t"), result("t")]),
            duration=9,
        )
    )
    assert graph["critical_path"]["duration"] == 9
    assert {e["kind"] for e in graph["edges"]} >= {
        "duration",
        "issued_tool",
        "tool_result",
        "continuation",
    }


def test_history_is_provenance_and_never_an_extra_blocking_join():
    run = parallel()
    run["actions"].append(
        action("m3", 12, 1, prompt=[use("t1"), use("t2"), result("t1"), result("t2"), text("done")])
    )
    graph = build_graph(run)
    links = [e for e in graph["edges"] if e["target"] == "m3:start"]
    assert [e["source"] for e in links if e["blocking"]] == ["m2:end"]
    assert all(e["confidence"] == "provenance" for e in links if not e["blocking"])


@pytest.mark.parametrize("identical", [False, True])
def test_reused_tool_ids_are_scoped_to_emitting_exchange(identical):
    task = text("task", "user")
    old = use("same")
    new = old if identical else use("same", "other")
    p2 = [task, old, result("same")]
    p3 = [*p2, new, result("same")]
    run = trace(
        action("m1", 0, 1, prompt=[task], response=[old]),
        action("t1", 1, 2, "tool", model_id="m1", model_tool_call_id="same"),
        action("m2", 3, 1, prompt=p2, response=[new]),
        action("t2", 4, 2, "tool", model_id="m2", model_tool_call_id="same"),
        action("m3", 6, 1, prompt=p3),
    )
    graph = build_graph(run)
    links = [e for e in graph["edges"] if e["target"] == "m3:start" and e["kind"] == "tool_result"]
    assert [e["source"] for e in links] == ["t2:end"]
    assert graph["critical_path"]["duration"] == 7


def test_no_hooks_keeps_unmeasured_bounds_and_result_link():
    graph = build_graph(
        trace(
            action("m1", 0, 1, response=[use("t")]),
            action("m2", 5, 1, prompt=[use("t"), result("t")]),
        )
    )
    op = next(o for o in graph["operations"] if o["id"].startswith("issued:"))
    assert op["duration"] is None
    assert graph["summary"]["operation_time"]["Tool"] == 0
    assert any(e["kind"] == "tool_result" for e in graph["edges"])
    assert any(e["confidence"] == "unknown" and e["kind"] == "duration" for e in graph["edges"])
    assert graph["critical_path"]["duration"] == 6
    assert graph["critical_path"]["measured_work"] == 2
    assert graph["critical_path"]["unexplained_wait"] == 4  # no invented tool duration


def test_missing_result_and_compaction_do_not_invent_dependencies():
    graph = build_graph(
        trace(
            action("m1", 0, 2, response=[use("missing")]),
            action("m2", 10, 1, prompt=[text("compacted", "user")]),
            action("t", 2, 1, "tool"),
        )
    )
    assert not any(e["target"] == "m2:start" and e["blocking"] for e in graph["edges"])
    assert not any(e["kind"] == "issued_tool" and e["target"] == "t:start" for e in graph["edges"])
    assert any("not observed" in u["reason"] for u in graph["coverage"]["unknown"])


def test_retry_is_inferred_and_does_not_block_path():
    prompt = [text("task", "user")]
    graph = build_graph(
        trace(
            action("failed", 0, 3, prompt=prompt, outcome="failed"),
            action("retry", 5, 2, prompt=prompt),
        )
    )
    edge = next(e for e in graph["edges"] if e["kind"] == "retry")
    assert edge["confidence"] == "inferred" and not edge["blocking"]
    assert graph["critical_path"]["duration"] == 3


def test_concurrent_agents_same_tool_ids_never_cross_link():
    one = parallel()
    two = [
        {
            **a,
            "id": "b:" + a["id"],
            "agent": "b",
            **({"model_id": "b:" + a["model_id"]} if a.get("model_id") else {}),
        }
        for a in one["actions"]
    ]
    graph = build_graph(trace(*one["actions"], *two))
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert all(nodes[e["source"]]["agent"] == nodes[e["target"]]["agent"] for e in graph["edges"])
    assert graph["critical_path"]["duration"] == 11


def test_incomplete_and_missing_start_do_not_claim_duration():
    graph = build_graph(
        trace(
            action("open", 2, 99, outcome="running"),
            action("missing", 3, 10, "tool", missing_start=True),
        )
    )
    assert graph["critical_path"]["duration"] == 0
    assert {n["id"] for n in graph["nodes"]} == {"open:start", "open:open"}
    assert graph["operations"][0]["open_through"] == 20


def test_unexplained_gaps_are_exposed_and_excluded():
    graph = build_graph(
        trace(
            action("a", 0, 2, response=[text("response")]),
            action("b", 10, 3, prompt=[text("response")]),
        )
    )
    assert graph["critical_path"]["duration"] == 13
    assert graph["critical_path"]["measured_work"] == 5
    assert graph["coverage"]["gaps"][0]["duration"] == 8


def test_ancestry_never_blocks_and_invalid_time_is_not_verified():
    graph = build_graph(
        trace(
            action("m1", 0, 5, response=[use("t")]),
            action("t", 1, 2, "tool", model_id="m1", model_tool_call_id="t"),
            action("p", 1, 10, "process", metadata={"parent_id": "t"}),
        )
    )
    assert next(e for e in graph["edges"] if e["kind"] == "issued_tool")["confidence"] == "unknown"
    assert not any(e["kind"] == "process_owner" for e in graph["edges"])
    graph = build_graph(
        trace(
            action("t", 0, 3, "tool"),
            action("p", 1, 1, "process", metadata={"tool_action_id": "t"}),
        )
    )
    assert all(not e["blocking"] for e in graph["edges"] if e["kind"] == "process_owner")


def test_dag_cycle_is_reported():
    nodes = [{"id": "a"}, {"id": "b"}]
    edges = [
        {
            "id": f"{s}-{t}",
            "source": s,
            "target": t,
            "weight": 1,
            "confidence": "verified",
            "blocking": True,
        }
        for s, t in [("a", "b"), ("b", "a")]
    ]
    assert critical_path(nodes, edges)["duration"] is None
    assert "Cyclic" in critical_path(nodes, edges)["error"]


def test_live_updates_keep_ids_do_not_resend_unchanged_graph_and_replay_matches():
    model = Trajectory("x", "x", mode="replay")
    raw = {
        "id": "1",
        "ts": 0,
        "actor": "a",
        "type": "agent_state",
        "call_label": "m1",
        "payload": {
            "state": "waiting_llm",
            "causal_evidence": exchange_evidence([text("task", "user")], []),
        },
    }
    model.apply(raw)
    first = model.snapshot()["causal_graph"]
    model.patch()
    model.tick(2)
    assert "causal_graph" not in model.patch()["meta"]
    model.apply(
        {
            **raw,
            "id": "2",
            "ts": 3,
            "type": "model_result",
            "payload": {
                "causal_evidence": exchange_evidence([text("task", "user")], [use("t")]),
                "tool_call_ids": ["t"],
            },
        }
    )
    graph = model.patch()["meta"]["causal_graph"]
    assert first["nodes"][0]["id"] == graph["nodes"][0]["id"]
    replay = Trajectory("x", "x", mode="replay")
    for event in replay_events(model.snapshot()):
        replay.apply(event)
    assert (
        replay.snapshot()["causal_graph"]["critical_path"]["duration"]
        == graph["critical_path"]["duration"]
    )


def test_full_sqlite_evidence_survives_preview_limits(tmp_path):
    con = sqlite3.connect(tmp_path / "global_data.sqlite3")
    con.executescript("""CREATE TABLE events(id TEXT,type TEXT,timestamp REAL,name TEXT,call_label TEXT,payload TEXT);
    CREATE TABLE exchanges(id TEXT,type TEXT,timestamp REAL,name TEXT,call_label TEXT);
    CREATE TABLE blocks(hash TEXT,payload TEXT);
    CREATE TABLE exchange_chain(call_label TEXT,kind TEXT,seq INTEGER,hash TEXT);""")
    prompts = [text(str(i), "user") for i in range(90)] + [result("t")]
    for label, start, end, prompt, response in [
        ("m1", 100, 101, [], [use("t")]),
        ("m2", 105, 106, prompts, [text("done")]),
    ]:
        con.execute(
            "INSERT INTO events VALUES(?,?,?,?,?,?)",
            (label, "agent_state", start, "a", label, json.dumps({"state": "waiting_llm"})),
        )
        con.execute(
            "INSERT INTO exchanges VALUES(?,?,?,?,?)", (label, "llm_block", end, "a", label)
        )
        for kind, blocks in [("prompt", prompt), ("response", response)]:
            for i, b in enumerate(blocks):
                key = f"{label}-{kind}-{i}"
                con.execute("INSERT INTO blocks VALUES(?,?)", (key, json.dumps(b)))
                con.execute("INSERT INTO exchange_chain VALUES(?,?,?,?)", (label, kind, i, key))
    con.commit()
    con.close()
    source = LiveSource(tmp_path)
    source.refresh(force=True)
    model = source.model.actions["a:model:m2"]
    assert len(model["context"]["blocks"]) == 80
    assert len(model["causal_evidence"]["prompt"]) == 91
    assert source.model.snapshot()["causal_graph"]["state"] == "pending"
    source.model.apply(
        {
            "id": "done",
            "ts": 10,
            "actor": "workflow",
            "type": "done",
            "payload": {"status": "completed"},
        }
    )
    assert any(e["kind"] == "tool_result" for e in source.model.snapshot()["causal_graph"]["edges"])


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "unknown"])
def test_live_graph_waits_for_final_boundary_and_builds_once(monkeypatch, status):
    from agency.observability.agwebui import causal_graph

    original = causal_graph.build_graph
    calls = []

    def build(run):
        calls.append(len(run["actions"]))
        return original(run)

    monkeypatch.setattr(causal_graph, "build_graph", build)
    model = Trajectory("live", "live")
    for index in range(40):
        model.apply(
            {
                "id": str(index),
                "ts": index,
                "actor": "a",
                "type": "tool_call",
                "payload": {"call_id": str(index), "tool": "read"},
            }
        )
        model.tick(index + 1)
        patch = model.patch()
        assert not patch["meta"].get("causal_graph", {}).get("nodes")
        assert model.snapshot()["causal_graph"]["state"] == "pending"
    assert calls == []
    model.apply(
        {
            "id": "callback-return",
            "ts": 41,
            "actor": "workflow",
            "type": "workload_completed",
            "payload": {"status": status},
        }
    )
    assert model.snapshot()["causal_graph"]["state"] == "pending"
    assert calls == []
    model.apply(
        {
            "id": "done",
            "ts": 42,
            "actor": "workflow",
            "type": "done",
            "payload": {"status": status},
        }
    )
    patch = model.patch()
    assert patch["meta"]["causal_graph"]["nodes"]
    assert calls == [40]
    for tick in range(43, 50):
        model.tick(tick)
        assert "causal_graph" not in model.patch()["meta"]
        assert model.snapshot()["causal_graph"]["nodes"]
    assert calls == [40]


def test_completed_live_graph_waits_for_backlog_and_preserves_saved_replay(monkeypatch):
    from agency.observability.agwebui import causal_graph

    original = causal_graph.build_graph
    calls = []

    def build(run):
        calls.append(run["id"])
        return original(run)

    monkeypatch.setattr(causal_graph, "build_graph", build)
    live = Trajectory("live", "live")
    live.apply({"id": "done", "ts": 10, "type": "done", "actor": "a", "payload": {}})
    live.run["coverage"]["catching_up"] = True
    assert live.snapshot()["causal_graph"]["state"] == "pending"
    assert not calls
    live.run["coverage"]["catching_up"] = False
    assert "critical_path" in live.snapshot()["causal_graph"]
    replay = Trajectory("saved", "saved", mode="replay")
    assert "critical_path" in replay.snapshot()["causal_graph"]
    ongoing_replay = Trajectory("live", "live", mode="replay", defer_causal_graph=True)
    assert ongoing_replay.snapshot()["causal_graph"]["state"] == "pending"
    assert calls == ["live", "saved"]


def test_recorded_example_and_replay_api_have_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_run_dir", tmp_path)
    with TestClient(server.app) as client:
        run = client.get("/api/investigator/runs/demo-baseline").json()
        assert run["causal_graph"]["nodes"]
        assert run["causal_graph"]["coverage"]["confidence"] == "partial"
        with client.websocket_connect(
            "/ws/trajectory?mode=replay&run=demo-baseline&clock=20"
        ) as ws:
            frame = ws.receive_json()
            assert all(n["time"] <= 20 for n in frame["run"]["causal_graph"]["nodes"])


def test_large_trace():
    graph = build_graph(
        trace(
            *(action(f"t{i}", i, 1, "tool", agent=f"a{i % 32}") for i in range(10000)),
            duration=10000,
        )
    )
    assert len(graph["nodes"]) == 20000
    assert len(graph["edges"]) == 10000
    assert graph["critical_path"]["duration"] == 1


def test_late_short_tool_determines_join_and_unknown_wait_is_separate():
    run = parallel()
    run["actions"][1]["duration"] = 10
    run["actions"][2].update(start=15, duration=2)
    run["actions"][3].update(start=17, duration=2)
    graph = build_graph(run)
    assert graph["critical_path"]["duration"] == 19
    assert graph["critical_path"]["measured_work"] == 6
    assert graph["critical_path"]["unexplained_wait"] == 13
    assert "t2:end" in graph["critical_path"]["node_ids"]
    assert "t1:end" not in graph["critical_path"]["node_ids"]
    assert graph["critical_path"]["slack"]["t1:end"] == 5
    assert len(graph["coverage"]["gaps"]) == 1


def test_saved_truncated_spans_reuse_complete_canonical_exchange_evidence(tmp_path):
    from agency.observability.agwebui.investigator import load_run
    from agency.observability.agwebui.trajectory import historical_projection

    logs, profile = tmp_path / "logs", tmp_path / "profile"
    logs.mkdir()
    profile.mkdir()
    con = sqlite3.connect(logs / "agent_data.sqlite3")
    con.executescript("""CREATE TABLE exchanges(call_label TEXT,name TEXT,type TEXT);
    CREATE TABLE blocks(hash TEXT,payload TEXT);
    CREATE TABLE exchange_chain(call_label TEXT,kind TEXT,seq INTEGER,hash TEXT);""")
    for label, prompt, response in [
        ("m1", [], [use("t")]),
        ("m2", [use("t"), result("t")], [text("done")]),
    ]:
        con.execute("INSERT INTO exchanges VALUES(?,?,?)", (label, "a", "llm_block"))
        for kind, blocks in [("prompt", prompt), ("response", response)]:
            for index, block in enumerate(blocks):
                key = f"{label}-{kind}-{index}"
                con.execute("INSERT INTO blocks VALUES(?,?)", (key, json.dumps(block)))
                con.execute("INSERT INTO exchange_chain VALUES(?,?,?,?)", (label, kind, index, key))
    con.commit()
    con.close()
    path = profile / "agprof.trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "ts": start * 1e6,
                        "dur": 1e6,
                        "name": "llm:attempt[0]",
                        "args": {
                            "agency.agent_id": "a",
                            "call_label": label,
                            "llm.messages": "[truncated",
                            "span_id": label,
                        },
                    }
                    for label, start in [("m1", 0), ("m2", 5)]
                ]
            }
        )
    )
    run = historical_projection(
        load_run(
            str(path),
            path.stat().st_mtime,
            json.dumps({"id": "saved", "title": "Saved", "source": "recorded"}),
        )
    )
    assert run["coverage"]["causal_evidence"]["canonical_exchanges"] == 2
    assert any(e["kind"] == "tool_result" for e in run["causal_graph"]["edges"])
    assert run["causal_graph"]["critical_path"]["duration"] == 6
    assert run["actions"][0]["start"] == 0  # saved profiler clock unchanged


def test_replay_start_never_exposes_future_tool_issuance():
    from agency.observability.agwebui.trajectory import historical_projection

    run = {
        "id": "saved",
        "title": "saved",
        "source": "recorded",
        "status": "completed",
        "duration": 10,
        "actions": [
            {
                **action("m1", 0, 2, response=[use("future")]),
                "arguments": "",
                "command": "",
                "result": "",
                "files": [],
            }
        ],
        "edges": [],
    }
    # Only consume the reconstructed start, not its future response.
    model = Trajectory("saved", "saved", mode="replay")
    model.apply(replay_events(run)[0])
    assert not any(
        o.get("tool_id") == "future" for o in model.snapshot()["causal_graph"]["operations"]
    )
    assert historical_projection(run)["causal_graph"]["nodes"]


def test_reused_host_ids_without_model_id_resolve_with_prompt_scoped_windows():
    task = text("task", "user")
    p2 = [task, use("same"), result("same")]
    run = trace(
        action("m1", 0, 1, prompt=[task], response=[use("same")]),
        action("t1", 1, 2, "tool", model_tool_call_id="same"),
        action("m2", 3, 1, prompt=p2, response=[use("same")]),
        action("t2", 4, 2, "tool", model_tool_call_id="same"),
        action("m3", 6, 1, prompt=[*p2, use("same"), result("same")]),
    )
    graph = build_graph(run)
    assert graph["critical_path"]["duration"] == 7
    assert all(not o["id"].startswith("issued:") for o in graph["operations"])


def test_observed_completion_with_missing_start_remains_inspectable():
    run = trace(action("t", 0, 0, "tool", missing_start=True, end_ts=105))
    run["time_origin"] = 100
    graph = build_graph(run)
    assert [(n["id"], n["time"]) for n in graph["nodes"]] == [("t:end", 5)]
    assert graph["operations"][0]["duration"] is None
    assert graph["critical_path"]["duration"] == 0


def test_operation_path_excludes_syscalls_without_fabricating_bridges():
    run = parallel()
    run["actions"].append(action("syscall", 0, 900, "syscall"))
    graph = build_graph(run)
    assert graph["critical_path"]["duration"] == 900
    assert graph["operation_critical_path"]["duration"] == 11
    assert graph["operation_critical_path"]["measured_work"] == 11
    assert all(not n.startswith("syscall:") for n in graph["operation_critical_path"]["node_ids"])
    assert next(o for o in graph["operations"] if o["id"] == "syscall")["kind"] == "syscall"
    assert (
        graph["operation_critical_path"]["edge_ids"]
        == build_graph(parallel())["critical_path"]["edge_ids"]
    )


def test_successful_execve_is_process_evidence_not_syscall_duration_or_exit():
    a = action(
        "exec",
        3,
        0.002,
        "syscall",
        metadata={"pid": 42, "path": "/usr/bin/python", "return_value": 0},
    )
    a["name"] = "execve"
    a["arguments"] = json.dumps({"argv": ["python", "build.py"]})
    graph = build_graph(trace(a))
    process = next(o for o in graph["operations"] if o["kind"] == "process")
    assert process["label"] == "python"
    assert process["duration"] is None
    assert "end_node" not in process
    assert process["metrics"]["pid"] == 42
    assert process["metrics"]["argv"] == ["python", "build.py"]
    assert "return_value" not in process["metrics"]
    assert "exit_status" not in process["metrics"]
    assert graph["nodes"][-1]["timing"] == "execve_observation"
    assert not any(e["kind"] == "process_owner" for e in graph["edges"])
    assert any(o["kind"] == "syscall" for o in graph["operations"])
    a["metadata"]["return_value"] = -2
    assert not any(o["kind"] == "process" for o in build_graph(trace(a))["operations"])


def test_explicit_process_ancestry_and_ownership_are_nonblocking():
    tool = action("tool", 0, 5, "tool")
    parent = action("parent", 1, 4, "process", metadata={"pid": 10, "tool_action_id": "tool"})
    child = action(
        "child", 2, 1, "process", metadata={"pid": 11, "ppid": 10, "cpu_ms": 30, "rss_mb": 2}
    )
    graph = build_graph(trace(tool, parent, child))
    relationships = [e for e in graph["edges"] if e["kind"].startswith("process_")]
    assert {e["kind"] for e in relationships} == {"process_owner", "process_parent"}
    assert all(e["confidence"] == "provenance" and not e["blocking"] for e in relationships)
    assert all(not e["critical"] for e in relationships)
    assert graph["operation_critical_path"]["duration"] == 5
    duplicate_pid = action("reused-parent", 8, 1, "process", metadata={"pid": 10})
    graph = build_graph(trace(tool, parent, child, duplicate_pid))
    assert not any(e["kind"] == "process_parent" for e in graph["edges"])


def test_operation_path_ignores_unconnected_startup_bookkeeping():
    run = parallel()
    run["actions"].append(action("setup", 0, 200, "setup"))
    graph = build_graph(run)
    assert graph["critical_path"]["duration"] == 200
    assert graph["operation_critical_path"]["duration"] == 11


def _agent_chain(agent, offset, tool_seconds):
    first = action(f"{agent}-m1", offset, 1, agent=agent, response=[use(f"{agent}-t")])
    tool = action(
        f"{agent}-t",
        offset + 1,
        tool_seconds,
        "tool",
        agent=agent,
        model_id=f"{agent}-m1",
        model_tool_call_id=f"{agent}-t",
    )
    last = action(
        f"{agent}-m2",
        offset + 1 + tool_seconds,
        1,
        agent=agent,
        prompt=[use(f"{agent}-t"), result(f"{agent}-t")],
        response=[text("done")],
    )
    return [first, tool, last]


def test_each_agent_gets_its_own_known_chain_and_no_system_path_is_claimed():
    graph = build_graph(trace(*_agent_chain("a", 0, 10), *_agent_chain("b", 0.5, 2)))
    paths = graph["agent_critical_paths"]
    assert set(paths) == {"a", "b"}
    owner = {n["id"]: n["agent"] for n in graph["nodes"]}
    for agent, chain in paths.items():
        assert chain["node_ids"] and all(owner[node] == agent for node in chain["node_ids"])
        assert chain["measured_work"] == pytest.approx(
            chain["duration"] - chain["unexplained_wait"]
        )
    assert paths["a"]["duration"] == pytest.approx(12)
    assert paths["b"]["duration"] == pytest.approx(4)
    # The global longest chain alone would leave agent b without a path.
    assert set(graph["operation_critical_path"]["node_ids"]) <= {
        n for n in owner if owner[n] == "a"
    }
    system = graph["system_critical_path"]
    assert system["available"] is False and system["cross_agent_edges"] == 0
