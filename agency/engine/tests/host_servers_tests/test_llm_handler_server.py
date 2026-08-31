# Tests for llm_handler_server.py -- host-side LLM routing, non-streaming
# dispatch, and the streaming relay (producer thread + _StreamHandle).

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agency.agconfig import agConfig
from agency.engine.host_servers import llm_handler_server as mod
from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
from agency.profiler import agprof

# ---------------------------------------------------------------------------
# Fakes -- duck-typed to match what serialize helpers read via getattr,
# same shape openai/anthropic SDK objects expose.
# ---------------------------------------------------------------------------


class _FakeToolCall:
    def __init__(self, index=0, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, **extra):
        self.content = content
        self.tool_calls = tool_calls
        self._extra = extra

    def model_dump(self):
        return {"content": self.content, "tool_calls": self.tool_calls, **self._extra}


class _FakeChoice:
    def __init__(self, message=None, delta=None, finish_reason=None):
        self.message = message
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResult:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _FakeChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _FakeClient:
    def __init__(self, create_fn):
        self._create_fn = create_fn
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return self._create_fn(**kwargs)

    def close(self):
        self.closed = True


class _FakeDataCollector:
    def __init__(self):
        self.events = []

    def record_event(self, type, payload, call_label=None, do_update=False, **_kw):
        self.events.append((type, payload, call_label, do_update))


def _cfg(**fields) -> agConfig:
    return agConfig({"agllm_backend": fields})


def _make_server(create_fn=None, **fields) -> "tuple[LlmHandlerServer, _FakeClient]":
    fields.setdefault("model", "gpt-test")
    server = LlmHandlerServer(_cfg(**fields), _FakeDataCollector())
    client = _FakeClient(create_fn)
    server._backend.make_client = lambda timeout: client
    return server, client


def _drain(handle) -> "list[dict]":
    item = handle.first()
    lines = [item]
    while item["type"] not in ("done", "error"):
        item = handle._queue.get()
        lines.append(item)
    return lines


# ---------------------------------------------------------------------------
# resolve_model / context_limit
# ---------------------------------------------------------------------------


def test_resolve_model_returns_backends_model():
    server, _ = _make_server(model="gpt-test")
    assert server.resolve_model() == "gpt-test"


def test_resolve_model_empty_string_when_unset():
    server, _ = _make_server()
    server._backend.model = None
    assert server.resolve_model() == ""


def test_context_limit_delegates_to_agllm_fetch_context_limit(monkeypatch):
    server, _ = _make_server()
    monkeypatch.setattr(mod.agllm, "fetch_context_limit", lambda backend: 12345)
    assert server.context_limit() == 12345


# ---------------------------------------------------------------------------
# dispatch (non-streaming)
# ---------------------------------------------------------------------------


def test_dispatch_returns_message_usage_stop_reason():
    def create(**kwargs):
        return _FakeResult(
            [_FakeChoice(message=_FakeMessage(content="hi there"), finish_reason="stop")],
            usage=_FakeUsage(5, 3),
        )

    server, client = _make_server(create_fn=create)
    result = server.dispatch({"messages": [{"role": "user", "content": "hi"}]})
    assert result == {
        "message": {
            "role": "assistant",
            "blocks": [{"type": "text", "index": 0, "text": "hi there"}],
        },
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "stop_reason": "stop",
    }
    assert client.closed is True


def test_dispatch_includes_tool_calls_when_present():
    def create(**kwargs):
        tc = _FakeToolCall(index=0, id="call_1", name="bash", arguments='{"cmd":"ls"}')
        return _FakeResult(
            [
                _FakeChoice(
                    message=_FakeMessage(content=None, tool_calls=[tc]), finish_reason="tool_calls"
                )
            ]
        )

    server, _ = _make_server(create_fn=create)
    result = server.dispatch({"messages": []})
    assert result["message"]["blocks"] == [
        {
            "type": "tool_use",
            "index": 0,
            "id": "call_1",
            "name": "bash",
            "arguments": '{"cmd":"ls"}',
        }
    ]


def test_dispatch_passes_tool_choice_through_when_given():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return _FakeResult([_FakeChoice(message=_FakeMessage(content="ok"), finish_reason="stop")])

    server, _ = _make_server(create_fn=create)
    server.dispatch({"messages": [], "tool_choice": "auto"})
    assert seen["tool_choice"] == "auto"


def test_dispatch_bad_request_raises_dispatch_error_and_closes_client(monkeypatch):
    monkeypatch.setattr(mod, "BAD_REQUEST_EXCS", (ValueError,))

    def create(**kwargs):
        raise ValueError("bad request")

    server, client = _make_server(create_fn=create)
    try:
        server.dispatch({"messages": []})
        assert False, "expected _DispatchError"
    except mod._DispatchError as e:
        assert e.status_code == 400
        assert e.transient is False
    assert client.closed is True


def test_dispatch_transient_error_raises_dispatch_error(monkeypatch):
    monkeypatch.setattr(mod, "TRANSIENT_DISPATCH_EXCS", (ConnectionError,))

    def create(**kwargs):
        raise ConnectionError("down")

    server, client = _make_server(create_fn=create)
    try:
        server.dispatch({"messages": []})
        assert False, "expected _DispatchError"
    except mod._DispatchError as e:
        assert e.status_code == 503
        assert e.transient is True
    assert client.closed is True


def test_dispatch_unclassified_exception_propagates_and_still_closes_client():
    def create(**kwargs):
        raise RuntimeError("boom")

    server, client = _make_server(create_fn=create)
    try:
        server.dispatch({"messages": []})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    assert client.closed is True


# ---------------------------------------------------------------------------
# start_stream / _StreamHandle -- happy path
# ---------------------------------------------------------------------------


def test_start_stream_relays_text_deltas_then_done():
    def create(**kwargs):
        return iter(
            [
                _FakeChunk([_FakeChoice(delta=_FakeDelta(content="Hel"))]),
                _FakeChunk([_FakeChoice(delta=_FakeDelta(content="lo"))]),
                _FakeChunk([_FakeChoice(delta=_FakeDelta(content=None))], usage=_FakeUsage(1, 2)),
            ]
        )

    server, client = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    items = _drain(handle)
    assert [i["type"] for i in items] == ["delta", "delta", "done"]
    assert items[0]["content"] == "Hel"
    assert items[1]["content"] == "lo"
    message = items[2]["message"]
    assert message["role"] == "assistant"
    assert len(message["blocks"]) == 1
    block = message["blocks"][0]
    assert block["type"] == "text" and block["text"] == "Hello"
    assert "ts_start" in block and "ts_end" in block
    assert items[2]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    handle._thread.join(timeout=2.0)
    assert client.closed is True


def test_start_stream_uses_spawn_traced(monkeypatch):
    calls = []
    real_spawn_traced = agprof.spawn_traced

    def record_spawn(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return real_spawn_traced(fn, *args, **kwargs)

    monkeypatch.setattr(agprof, "spawn_traced", record_spawn)
    server, _ = _make_server(create_fn=lambda **kwargs: iter([]))
    handle = server.start_stream({"messages": []})
    assert _drain(handle)[-1]["type"] == "done"
    handle._thread.join(timeout=2.0)

    assert len(calls) == 1
    assert calls[0][0] == server._run_stream_producer
    assert calls[0][2] == {"daemon": True}


def test_streaming_http_request_preserves_engine_run_parent_span(monkeypatch, tmp_path):
    """The LLM attempt remains a child of the agent run across HTTP + thread hops."""
    monkeypatch.setattr(agprof, "_require_linux", lambda: None)

    def create(**kwargs):
        return iter([_FakeChunk([_FakeChoice(delta=_FakeDelta(content="Hi"))])])

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("engine-run"):
            server = LlmHandlerServer(
                _cfg(model="gpt-test"),
                _FakeDataCollector(),
                parent_context=agprof.current_span_context(),
            )
            client = _FakeClient(create)
            server._backend.make_client = lambda timeout: client
            response = TestClient(server.build_app()).post(
                "/dispatch", json={"messages": [], "stream": True}
            )
            assert response.status_code == 200

    records = {record[1]: record for record in agprof._records}
    assert records["llm:attempt[0]"][8] == records["engine-run"][7]


def test_start_stream_accumulates_tool_call_argument_fragments():
    def create(**kwargs):
        return iter(
            [
                _FakeChunk(
                    [
                        _FakeChoice(
                            delta=_FakeDelta(tool_calls=[_FakeToolCall(0, "call_1", "bash", '{"c')])
                        )
                    ]
                ),
                _FakeChunk(
                    [
                        _FakeChoice(
                            delta=_FakeDelta(tool_calls=[_FakeToolCall(0, None, None, 'md":"ls"}')])
                        )
                    ]
                ),
            ]
        )

    server, _ = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    items = _drain(handle)
    done = items[-1]
    assert done["type"] == "done"
    blocks = done["message"]["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "tool_use"
    assert blocks[0]["index"] == 0
    assert blocks[0]["id"] == "call_1"
    assert blocks[0]["name"] == "bash"
    assert blocks[0]["arguments"] == '{"cmd":"ls"}'
    handle._thread.join(timeout=2.0)


def test_start_stream_preserves_unrecognized_delta_field_with_named_type():
    def create(**kwargs):
        return iter([_FakeChunk([_FakeChoice(delta=_FakeDelta(refusal="no"))])])

    server, _ = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    items = _drain(handle)
    done = items[-1]
    blocks = done["message"]["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "openai_chatcompletions_refusal"
    assert blocks[0]["data"] == ["no"]
    handle._thread.join(timeout=2.0)


def test_start_stream_no_chunks_at_all_yields_immediate_done():
    def create(**kwargs):
        return iter([])

    server, client = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    items = _drain(handle)
    assert items == [
        {
            "type": "done",
            "message": {"role": "assistant", "blocks": []},
            "usage": None,
            "stop_reason": None,
        }
    ]
    handle._thread.join(timeout=2.0)
    assert client.closed is True


def test_start_stream_first_chunk_bad_request_becomes_error_item(monkeypatch):
    monkeypatch.setattr(mod, "BAD_REQUEST_EXCS", (ValueError,))

    def create(**kwargs):
        raise ValueError("nope")

    server, client = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    item = handle.first()
    assert item == {"type": "error", "message": "nope", "transient": False, "status_code": 400}
    handle._thread.join(timeout=2.0)
    assert client.closed is True


def test_start_stream_first_chunk_transient_error_becomes_error_item(monkeypatch):
    monkeypatch.setattr(mod, "TRANSIENT_DISPATCH_EXCS", (ConnectionError,))

    def create(**kwargs):
        raise ConnectionError("down")

    server, _ = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    item = handle.first()
    assert item == {"type": "error", "message": "down", "transient": True, "status_code": 503}
    handle._thread.join(timeout=2.0)


def test_start_stream_mid_stream_exception_becomes_error_item_and_stops():
    def gen():
        yield _FakeChunk([_FakeChoice(delta=_FakeDelta(content="ok"))])
        raise RuntimeError("mid-stream failure")

    def create(**kwargs):
        return gen()

    server, client = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    items = _drain(handle)
    assert items[0]["type"] == "delta"
    assert items[-1] == {
        "type": "error",
        "message": "mid-stream failure",
        "transient": False,
        "status_code": 500,
    }
    handle._thread.join(timeout=2.0)
    assert client.closed is True


# ---------------------------------------------------------------------------
# cancellation / resource release
# ---------------------------------------------------------------------------


def test_handle_relay_stays_in_handles_on_completion():
    def create(**kwargs):
        return iter([_FakeChunk([_FakeChoice(delta=_FakeDelta(content="hi"))])])

    server, _ = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    first = handle.first()
    list(handle.relay(first))
    assert handle in server._handles


def test_cancel_sets_event_and_closes_stream_ref_once():
    close_calls = []

    class _Stream:
        def close(self):
            close_calls.append(1)

    server, _ = _make_server(create_fn=lambda **kw: iter([]))
    handle = mod._StreamHandle(mod.queue.Queue(), threading.Event())
    handle._set_stream_ref(_Stream())
    handle.cancel()
    handle.cancel()
    assert handle._cancel_event.is_set()
    assert close_calls == [1]


def test_producer_stops_early_when_cancelled_mid_stream():
    started = threading.Event()
    keep_going = threading.Event()

    def gen():
        for i in range(1000):
            if i == 1:
                started.set()
                keep_going.wait(timeout=5.0)
            yield _FakeChunk([_FakeChoice(delta=_FakeDelta(content=f"c{i}"))])

    server, client = _make_server(create_fn=lambda **kw: gen())
    handle = server.start_stream({"messages": []})
    handle.first()
    assert started.wait(timeout=2.0)
    handle.cancel()
    keep_going.set()
    handle._thread.join(timeout=2.0)
    assert not handle._thread.is_alive()
    assert client.closed is True


def test_stop_cancels_and_joins_all_handles():
    # Simulates a producer stuck reading the first chunk. Real cancellation
    # can only unblock this via closing the underlying connection (what
    # cancel() does) -- setting cancel_event alone wouldn't reach a thread
    # blocked inside a network read, so the fake's close() plays that role.
    keep_going = threading.Event()
    started = threading.Event()

    def gen():
        started.set()
        keep_going.wait(timeout=5.0)
        yield _FakeChunk([_FakeChoice(delta=_FakeDelta(content="x"))])

    class _ClosingClient(_FakeClient):
        def close(self):
            super().close()
            keep_going.set()

    server = LlmHandlerServer(_cfg(model="gpt-test"), _FakeDataCollector())
    client = _ClosingClient(lambda **kw: gen())
    server._backend.make_client = lambda timeout: client

    handle = server.start_stream({"messages": []})
    assert started.wait(timeout=2.0)
    t0 = time.monotonic()
    server.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0
    assert client.closed is True
    assert not handle._thread.is_alive()


def test_stop_with_no_handles_returns_immediately():
    server, _ = _make_server(create_fn=lambda **kw: iter([]))
    server.stop()  # should not raise or hang


# ---------------------------------------------------------------------------
# build_app / HTTP routes
# ---------------------------------------------------------------------------


def test_build_app_dispatch_route_non_streaming():
    def create(**kwargs):
        return _FakeResult([_FakeChoice(message=_FakeMessage(content="hi"), finish_reason="stop")])

    server, _ = _make_server(create_fn=create)
    client = TestClient(server.build_app())
    response = client.post("/dispatch", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json() == {
        "message": {
            "role": "assistant",
            "blocks": [{"type": "text", "index": 0, "text": "hi"}],
        },
        "usage": None,
        "stop_reason": "stop",
    }


def test_build_app_dispatch_route_bad_request_returns_400(monkeypatch):
    monkeypatch.setattr(mod, "BAD_REQUEST_EXCS", (ValueError,))

    def create(**kwargs):
        raise ValueError("bad")

    server, _ = _make_server(create_fn=create)
    client = TestClient(server.build_app())
    response = client.post("/dispatch", json={"messages": []})
    assert response.status_code == 400
    assert response.json() == {"error": {"message": "bad", "transient": False}}


def test_build_app_dispatch_route_streaming_relays_ndjson_lines():
    def create(**kwargs):
        return iter(
            [
                _FakeChunk([_FakeChoice(delta=_FakeDelta(content="Hi"))]),
                _FakeChunk([_FakeChoice(delta=_FakeDelta(content=None))]),
            ]
        )

    server, _ = _make_server(create_fn=create)
    client = TestClient(server.build_app())
    response = client.post("/dispatch", json={"messages": [], "stream": True})
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.strip().split("\n")]
    assert lines[0] == {"type": "delta", "content": "Hi"}
    assert lines[-1]["type"] == "done"
    assert lines[-1]["message"]["role"] == "assistant"
    blocks = lines[-1]["message"]["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Hi"


def test_build_app_dispatch_route_streaming_error_returns_error_status(monkeypatch):
    monkeypatch.setattr(mod, "BAD_REQUEST_EXCS", (ValueError,))

    def create(**kwargs):
        raise ValueError("nope")

    server, _ = _make_server(create_fn=create)
    client = TestClient(server.build_app())
    response = client.post("/dispatch", json={"messages": [], "stream": True})
    assert response.status_code == 400
    assert response.json() == {"error": {"message": "nope", "transient": False}}


def test_build_app_resolve_model_route():
    server, _ = _make_server(model="my-model")
    client = TestClient(server.build_app())
    response = client.get("/resolve_model")
    assert response.status_code == 200
    assert response.json() == {"model": "my-model"}


def test_build_app_context_limit_route(monkeypatch):
    server, _ = _make_server()
    monkeypatch.setattr(mod.agllm, "fetch_context_limit", lambda backend: 4096)
    client = TestClient(server.build_app())
    response = client.get("/context_limit")
    assert response.status_code == 200
    assert response.json() == {"context_limit": 4096}
