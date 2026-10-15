"""Captured detail / correlation regression tests; no engines or model calls."""

import json
import sqlite3

from agency.observability.agwebui.trajectory import LiveSource, Trajectory, replay_events


def event(index, kind, *, actor="worker", label=None, ts=None, **payload):
    return {
        "id": f"detail/{index}",
        "type": kind,
        "actor": actor,
        "ts": index if ts is None else ts,
        "call_label": label,
        "payload": payload,
    }


def test_one_response_links_multiple_tools_with_explicit_ids_in_reverse_completion_order():
    projection = Trajectory("recorded", "Captured execution")
    projection.apply(event(0, "workload_started"))
    projection.apply(event(1, "agent_state", state="waiting_llm", label="model", request_id="r"))
    projection.apply(event(2, "tool_call", call_id="a", model_tool_call_id="x"))
    projection.apply(event(3, "tool_call", call_id="b", annotation={"model_tool_call_id": "y"}))
    projection.apply(event(4, "tool_result", call_id="b", result="b finished"))
    projection.apply(event(5, "tool_result", call_id="a", result="a finished"))
    projection.apply(
        event(
            6,
            "model_result",
            label="model",
            tool_call_ids=["x", "y", "x", {}, None],
            model="captured-model",
            request_id="r",
            result="Visible response",
        )
    )
    origin = projection.actions["worker:model:model"]
    assert origin["tool_call_ids"] == ["x", "y"]
    assert origin["model"] == "captured-model"
    assert origin["request_correlation"] == "explicit"
    assert projection.actions["worker:a"]["model_id"] == origin["id"]
    assert projection.actions["worker:b"]["model_id"] == origin["id"]
    assert projection.agents["worker"]["status"] == "waiting_llm"


def test_late_tool_identity_and_start_metadata_survive_result_projection():
    projection = Trajectory("recorded", "Captured execution")
    projection.apply(event(0, "workload_started"))
    projection.apply(event(1, "model_result", label="model", tool_call_ids=["x"], result=""))
    projection.apply(
        event(2, "tool_call", call_id="a", metadata={"cpu_ms": 0, "recorded_annotation": "keep"})
    )
    projection.apply(
        event(
            3,
            "tool_result",
            call_id="a",
            result={"exit_code": 1, "stdout": "FAILED specific test"},
            model_tool_call_id="x",
            metadata={"blocked_ms": 5},
        )
    )
    action = projection.actions["worker:a"]
    assert action["metadata"]["cpu_ms"] == 0
    assert action["metadata"]["blocked_ms"] == 5
    assert action["metadata"]["recorded_annotation"] == "keep"
    assert action["model_id"] == "worker:model:model"
    assert action["outcome"] == "failed"
    assert projection.agents["worker"]["status"] != "failed"


def test_exchange_preserves_safe_model_request_identity_without_native_reasoning():
    con = sqlite3.connect(":memory:")
    con.executescript(
        "CREATE TABLE blocks(hash TEXT, payload TEXT); CREATE TABLE exchange_chain(call_label TEXT, kind TEXT, seq INTEGER, hash TEXT);"
    )
    blocks = [
        ("prompt", "prompt", {"type": "text", "role": "user", "text": "Visible instruction"}),
        ("response", "reply", {"type": "text", "text": "Visible output"}),
        ("response", "tool-a", {"type": "tool_use", "id": "explicit-a"}),
        ("response", "tool-b", {"type": "tool_use", "id": "explicit-b"}),
        ("response", "hidden", {"type": "thinking", "text": "Hidden reasoning"}),
        (
            "response",
            "metadata",
            {
                "type": "metadata",
                "request_id": "run-owned",
                "ttft_ms": 15,
                "skill": "inspect",
                "usage": {"prompt_tokens": 4, "completion_tokens": 7},
                "data": {"model": "provider-model", "thinking": "Hidden native reasoning"},
            },
        ),
    ]
    for sequence, (kind, key, block) in enumerate(blocks):
        con.execute("INSERT INTO blocks VALUES(?,?)", (key, json.dumps(block)))
        con.execute("INSERT INTO exchange_chain VALUES(?,?,?,?)", ("m", kind, sequence, key))
    captured = LiveSource._exchange(con, "m", "llm_block", full=True)
    assert captured["model"] == "provider-model"
    assert captured["request_id"] == "run-owned"
    assert captured["metadata"]["ttft_ms"] == 15
    assert captured["tool_call_ids"] == ["explicit-a", "explicit-b"]
    assert [
        block["id"] for block in captured["model_response"]["blocks"] if block["type"] == "tool_use"
    ] == ["explicit-a", "explicit-b"]
    assert captured["tokens"] == 4 and captured["output_tokens"] == 7
    assert "Hidden" not in json.dumps(captured)
    assert "data" not in captured["metadata"]
    con.close()


def test_canonical_response_keeps_source_backed_tool_arguments_bounded_in_live_preview():
    con = sqlite3.connect(":memory:")
    con.executescript(
        "CREATE TABLE blocks(hash TEXT, payload TEXT); CREATE TABLE exchange_chain(call_label TEXT, kind TEXT, seq INTEGER, hash TEXT);"
    )
    arguments = {"command": "pytest target", "message": "safe captured text " * 10000}
    blocks = [
        {"type": "text", "text": "Inspect the failing test."},
        {"type": "tool_use", "id": "tool-a", "name": "Bash", "arguments": arguments},
        {"type": "thinking", "text": "private reasoning"},
    ]
    for index, block in enumerate(blocks):
        key = str(index)
        con.execute("INSERT INTO blocks VALUES(?,?)", (key, json.dumps(block)))
        con.execute("INSERT INTO exchange_chain VALUES(?,?,?,?)", ("m", "response", index, key))
    preview = LiveSource._exchange(con, "m", "llm_block")
    assert preview["output_truncated"]
    assert preview["model_response"]["blocks"][1]["name"] == "Bash"
    assert len(json.dumps(preview["model_response"])) < 2000
    full = LiveSource._exchange(con, "m", "llm_block", full=True)
    assert not full["output_truncated"]
    assert full["model_response"]["blocks"][1]["arguments"] == arguments
    assert "private reasoning" not in json.dumps(full)
    projection = Trajectory("captured", "Canonical response")
    projection.apply(event(0, "workload_started"))
    projection.apply(event(1, "model_result", label="m", **preview))
    action = projection.actions["worker:model:m"]
    assert action["output_truncated"]
    assert action["model_response"] == preview["model_response"]
    con.close()


def test_replay_keeps_recorded_links_without_emitting_future_tool_requests_at_model_start():
    run = {
        "id": "recorded",
        "title": "Old recording",
        "source": "recorded",
        "duration": 5,
        "actions": [
            {
                "id": "m",
                "agent": "worker",
                "kind": "model",
                "name": "model",
                "start": 0,
                "duration": 2,
                "outcome": "success",
                "command": "",
                "result": "Check the tool",
                "model": "recorded-model",
                "tool_call_ids": ["x"],
            },
            {
                "id": "t",
                "agent": "worker",
                "kind": "test",
                "name": "Bash",
                "start": 2,
                "duration": 3,
                "outcome": "failed",
                "command": "pytest target",
                "result": "FAILED target",
                "model_tool_call_id": "x",
                "model_id": "m",
                "request_id": "fallback",
                "request_correlation": "actor_context_fallback",
            },
        ],
    }
    events = replay_events(run)
    first = next(item for item in events if item["id"] == "replay/start/m")
    assert "tool_call_ids" not in first["payload"]
    projection = Trajectory("recorded", "Old recording", mode="replay")
    for item in events:
        projection.apply(item)
    assert projection.actions["m"]["tool_call_ids"] == ["x"]
    assert projection.actions["m"]["model"] == "recorded-model"
    assert projection.actions["t"]["model_id"] == "m"
    assert projection.actions["t"]["request_correlation"] == "actor_context_fallback"


def test_old_recording_without_correlation_does_not_invent_links():
    projection = Trajectory("recorded", "Old recording", mode="review")
    projection.apply(event(1, "model_result", label="model", result="Old model output"))
    projection.apply(
        event(2, "tool_call", call_id="a", tool="Bash", arguments={"command": "echo old"})
    )
    projection.apply(event(3, "tool_result", call_id="a", result="old output"))
    assert not projection.actions["worker:a"]["model_tool_call_id"]
    assert not projection.actions["worker:a"]["model_id"]
    assert not projection.actions["worker:model:model"].get("tool_call_ids")
    assert projection.actions["worker:a"]["request_correlation"] == "unavailable"


def test_truncated_model_input_marks_full_source_access_without_losing_preview():
    projection = Trajectory("recorded", "Captured execution")
    projection.apply(event(0, "workload_started"))
    projection.apply(
        event(
            1,
            "model_result",
            label="model",
            result="answer",
            context={
                "source": "recorded",
                "truncated": True,
                "blocks": [{"preview": "captured prompt preview"}],
            },
        )
    )
    action = projection.actions["worker:model:model"]
    assert action["input_truncated"]
    assert action["context"]["blocks"][0]["preview"] == "captured prompt preview"


def test_saved_native_observer_ids_keep_multiple_explicit_relationships_and_full_visible_response(
    tmp_path,
):
    from agency.observability.agwebui.investigator import normalize_native_events

    response = {
        "role": "assistant",
        "content": "Check both tools",
        "tool_calls": [
            {
                "id": "x",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"first.py"}'},
            },
            {
                "id": "y",
                "type": "function",
                "function": {"name": "Bash", "arguments": '{"command":"pytest target"}'},
            },
        ],
        "blocks": [{"type": "thinking", "text": "private reasoning"}],
    }
    rows = [
        {
            "kind": "model_exchange",
            "timestamp_ns": 1_000_000_000,
            "duration_ns": 100_000_000,
            "agent_id": "a",
            "response": {"message": response, "usage": {"prompt_tokens": 10}},
        },
        {
            "kind": "tool_annotation",
            "timestamp_ns": 2_000_000_000,
            "agent_id": "a",
            "event_id": "t1",
            "model_tool_call_id": "x",
            "arguments": {"path": "first.py"},
        },
        {
            "kind": "tool_result",
            "timestamp_ns": 3_000_000_000,
            "duration_ns": 100_000_000,
            "agent_id": "a",
            "event_id": "t1",
            "tool_name": "read_file",
            "result": "first",
            "category": "ok",
        },
        {
            "kind": "tool_result",
            "timestamp_ns": 4_000_000_000,
            "duration_ns": 100_000_000,
            "agent_id": "a",
            "event_id": "t2",
            "tool_name": "Bash",
            "model_tool_call_id": "y",
            "result": "FAILED target",
            "category": "error",
        },
    ]
    path = tmp_path / "native.events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    saved = normalize_native_events(
        path, {"id": "saved", "source": "recorded", "model": "captured-model"}
    )
    origin = next(action for action in saved["actions"] if action["kind"] == "model")
    assert origin["tool_call_ids"] == ["x", "y"]
    assert (
        origin["model_response"]["tool_calls"][1]["function"]["arguments"]
        == '{"command":"pytest target"}'
    )
    assert "private reasoning" not in json.dumps(saved)
    assert saved["actions"][1]["model_tool_call_id"] == "x"
    assert saved["actions"][2]["model_tool_call_id"] == "y"
    replay = Trajectory("saved", "Captured native observer", mode="replay")
    for observation in replay_events(saved):
        replay.apply(observation)
    assert replay.actions["t1"]["model_id"] == origin["id"]
    assert replay.actions["t2"]["model_id"] == origin["id"]
    assert replay.actions[origin["id"]]["model_response"] == origin["model_response"]


def test_saved_trace_without_explicit_tool_ids_does_not_use_adjacent_model_or_parent_span():
    from agency.observability.agwebui.investigator import normalize_trace

    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "name": "llm:attempt[0]",
                "ts": 0,
                "dur": 1e6,
                "args": {
                    "agency.agent_id": "a",
                    "span_id": "m",
                    "llm.response": '{"content":"Check the tool"}',
                },
            },
            {
                "ph": "X",
                "name": "tool:Bash",
                "ts": 1e6,
                "dur": 1e6,
                "args": {
                    "agency.agent_id": "a",
                    "span_id": "t",
                    "parent_span_id": "m",
                    "tool.arguments": '{"command":"echo old"}',
                },
            },
        ]
    }
    saved = normalize_trace(trace, {"id": "saved", "source": "recorded"})
    assert saved["actions"][0]["tool_call_ids"] == []
    assert saved["actions"][1]["model_tool_call_id"] is None
    assert saved["actions"][1]["model_id"] is None
