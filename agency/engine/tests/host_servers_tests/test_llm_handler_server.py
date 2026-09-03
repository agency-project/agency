# Tests for llm_handler_server.py -- host-side LLM routing, non-streaming
# dispatch, and the streaming relay (producer thread + _StreamHandle).

from __future__ import annotations

import asyncio
import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse

from agency._agent_control import AgentControl
from agency.agconfig import agConfig
from agency.engine.host_servers import llm_handler_server as mod
from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
from agency.llm.usage_tracker import LlmUsageTracker
from agency.observability.profiler import agprof

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

    def model_dump(self):
        return {"content": self.content, "tool_calls": self.tool_calls}


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

    def model_dump(self):
        return {
            "message": self.message.model_dump() if self.message is not None else None,
            "delta": self.delta.model_dump() if self.delta is not None else None,
            "finish_reason": self.finish_reason,
        }


class _FakeUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens

    def model_dump(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class _FakeResult:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage

    def model_dump(self):
        return {
            "choices": [c.model_dump() for c in self.choices],
            "usage": self.usage.model_dump() if self.usage is not None else None,
        }


class _FakeChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage

    def model_dump(self):
        return {
            "choices": [c.model_dump() for c in self.choices],
            "usage": self.usage.model_dump() if self.usage is not None else None,
        }


class _FakeClient:
    def __init__(self, create_fn):
        self._create_fn = create_fn
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return self._create_fn(**kwargs)

    def close(self):
        self.closed = True


class _FakeDataLogger:
    def __init__(self):
        self.events = []
        self.stream_deltas = []
        self.stream_delta_history = []
        self.finalized = []
        self.operations = []

    def record_event(self, type, payload, call_label=None, update_latest_snapshot=False, **_kw):
        self.events.append((type, payload, call_label, update_latest_snapshot))

    def record_stream_delta(self, type, payload, call_label=None, flush=False):
        entry = (type, payload, call_label)
        self.stream_deltas.append(entry)
        self.stream_delta_history.append(entry)
        self.operations.append(("delta", call_label, payload))

    def finalize_stream(self, call_label, type, payloads, term_message=None):
        self.stream_deltas = [d for d in self.stream_deltas if d[2] != call_label]
        self.finalized.append((call_label, type, payloads))
        self.operations.append(("finalize", call_label, type))


def _cfg(**fields) -> agConfig:
    return agConfig({"agllm_backend": fields})


def _make_server(create_fn=None, **fields) -> "tuple[LlmHandlerServer, _FakeClient]":
    fields.setdefault("model", "gpt-test")
    server = LlmHandlerServer(_cfg(**fields), _FakeDataLogger(), usage_tracker=LlmUsageTracker())
    client = _FakeClient(create_fn)
    server._backend.make_client = lambda timeout: client
    return server, client


def _drain(handle) -> "list[dict]":
    item = handle.first()
    lines = [item]
    while item["type"] not in ("done", "error"):
        item = handle.first()
        lines.append(item)
    return lines


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


async def _disconnect_after_first_response_body(response: StreamingResponse) -> "list[dict]":
    body_sent = asyncio.Event()
    sent = []

    async def receive():
        await body_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            body_sent.set()
            # Keep the response task at the first network write until the
            # disconnect watcher cancels it. This prevents the async iterator
            # from consuming queued backlog before cleanup begins.
            await asyncio.Event().wait()

    scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
    await response(scope, receive, send)
    return sent


async def _post_app_until_disconnect(app, payload: dict, disconnect: asyncio.Event) -> "list[dict]":
    encoded = json.dumps(payload).encode()
    request_sent = False
    sent = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": encoded, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/dispatch",
        "raw_path": b"/dispatch",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(encoded)).encode()),
        ],
        "client": ("test", 1234),
        "server": ("test", 80),
    }
    try:
        await app(scope, receive, send)
    except ClientDisconnect:
        pass
    return sent


def _completed_tool_history() -> "list[dict]":
    return [
        {
            "role": "user",
            "blocks": [{"type": "text", "index": 0, "text": "compare both"}],
        },
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "tool_use",
                    "index": 0,
                    "id": "call-a",
                    "name": "lookup",
                    "arguments": '{"item":"a"}',
                },
                {
                    "type": "tool_use",
                    "index": 1,
                    "id": "call-b",
                    "name": "lookup",
                    "arguments": '{"item":"b"}',
                },
            ],
        },
        {
            "role": "tool",
            "blocks": [
                {
                    "type": "tool_result",
                    "index": 0,
                    "tool_call_id": "call-a",
                    "text": "result-a",
                }
            ],
        },
        {
            "role": "tool",
            "blocks": [
                {
                    "type": "tool_result",
                    "index": 0,
                    "tool_call_id": "call-b",
                    "text": "result-b",
                }
            ],
        },
    ]


def _tool_message(call_id: str = "next") -> dict:
    return {
        "role": "assistant",
        "blocks": [
            {
                "type": "tool_use",
                "index": 0,
                "id": call_id,
                "name": "lookup",
                "arguments": "{}",
            }
        ],
    }


def _invocation_message_texts(messages: "list[dict]") -> "list[str]":
    return [
        block["text"]
        for message in messages
        if message.get("role") == "user"
        for block in message.get("blocks", [])
        if block.get("type") == "text"
        and block.get("text", "").startswith("[AGENCY INVOCATION MESSAGE]\n")
    ]


class _RecordingBackend:
    model = "recording"

    def __init__(self, *, final: bool = False) -> None:
        self.final = final
        self.requests: "list[tuple[str, dict]]" = []

    def _result(self) -> dict:
        message = (
            {
                "role": "assistant",
                "blocks": [{"type": "text", "index": 0, "text": "done"}],
            }
            if self.final
            else _tool_message()
        )
        return {
            "message": message,
            "usage": None,
            "stop_reason": "stop" if self.final else "tool_use",
        }

    def dispatch(self, request: dict) -> dict:
        self.requests.append(("nonstream", copy.deepcopy(request)))
        return self._result()

    def dispatch_stream(self, request: dict, *, on_client=None):
        del on_client
        self.requests.append(("stream", copy.deepcopy(request)))
        if self.final:
            yield {"type": "content", "index": 0, "block_type": "text", "text": "done"}
        else:
            yield {
                "type": "content",
                "index": 0,
                "block_type": "tool_use",
                "id": "next",
                "name": "lookup",
                "arguments": "{}",
            }
        yield {"type": "usage", "usage": None, "stop_reason": "stop"}


class _BlockingToolBackend(_RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def _block(self) -> None:
        self.entered.set()
        assert self.release.wait(timeout=2.0)

    def dispatch(self, request: dict) -> dict:
        self.requests.append(("nonstream", copy.deepcopy(request)))
        self._block()
        return self._result()

    def dispatch_stream(self, request: dict, *, on_client=None):
        del on_client
        self.requests.append(("stream", copy.deepcopy(request)))
        self._block()
        yield {
            "type": "content",
            "index": 0,
            "block_type": "tool_use",
            "id": "next",
            "name": "lookup",
            "arguments": "{}",
        }
        yield {"type": "usage", "usage": None, "stop_reason": "tool_use"}


class _BlockingFinalOnceBackend(_RecordingBackend):
    def __init__(self) -> None:
        super().__init__(final=True)
        self.entered = threading.Event()
        self.release = threading.Event()

    def _block_first(self) -> None:
        if len(self.requests) == 1:
            self.entered.set()
            assert self.release.wait(timeout=2.0)

    def dispatch(self, request: dict) -> dict:
        self.requests.append(("nonstream", copy.deepcopy(request)))
        self._block_first()
        return self._result()

    def dispatch_stream(self, request: dict, *, on_client=None):
        del on_client
        self.requests.append(("stream", copy.deepcopy(request)))
        self._block_first()
        yield {"type": "content", "index": 0, "block_type": "text", "text": "draft"}
        yield {"type": "usage", "usage": None, "stop_reason": "stop"}


class _TextThenBlockingToolBackend(_RecordingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.after_text = threading.Event()
        self.release = threading.Event()

    def dispatch_stream(self, request: dict, *, on_client=None):
        del on_client
        self.requests.append(("stream", copy.deepcopy(request)))
        yield {"type": "content", "index": 0, "block_type": "text", "text": "working"}
        self.after_text.set()
        assert self.release.wait(timeout=2.0)
        yield {
            "type": "content",
            "index": 1,
            "block_type": "tool_use",
            "id": "next",
            "name": "lookup",
            "arguments": "{}",
        }
        yield {"type": "usage", "usage": None, "stop_reason": "tool_use"}


def _controlled_server(backend, invocation) -> LlmHandlerServer:
    server = LlmHandlerServer(
        _cfg(model="gpt-test"),
        _FakeDataLogger(),
        invocation=invocation,
        usage_tracker=LlmUsageTracker(),
    )
    server._backend = backend
    return server


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
    content_blocks = [b for b in result["message"]["blocks"] if b["type"] != "metadata"]
    assert result["message"]["role"] == "assistant"
    assert content_blocks == [{"type": "text", "index": 0, "text": "hi there"}]
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    assert result["stop_reason"] == "stop"
    assert client.closed is True


def test_dispatch_tags_metadata_block_with_new_prompt_tokens_across_exchanges():
    """Two successive dispatch() calls on the same server share one
    LlmUsageTracker -- the second exchange's metadata block must report
    only the NEW prompt tokens, not the full cumulative prompt_tokens."""
    responses = iter(
        [
            _FakeResult(
                [_FakeChoice(message=_FakeMessage(content="first"), finish_reason="stop")],
                usage=_FakeUsage(100, 20),
            ),
            _FakeResult(
                [_FakeChoice(message=_FakeMessage(content="second"), finish_reason="stop")],
                usage=_FakeUsage(135, 15),
            ),
        ]
    )

    def create(**kwargs):
        return next(responses)

    server, _client = _make_server(create_fn=create)

    first_request = {"messages": [{"role": "user", "content": "hi"}]}
    first_result = server.dispatch(first_request)
    first_metadata = next(b for b in first_result["message"]["blocks"] if b["type"] == "metadata")
    assert first_metadata["new_prompt_tokens"] == 100

    second_request = {
        "messages": first_request["messages"]
        + [first_result["message"], {"role": "user", "content": "and then?"}]
    }
    second_result = server.dispatch(second_request)
    second_metadata = next(b for b in second_result["message"]["blocks"] if b["type"] == "metadata")
    # 135 - (100 + 20) == 15 tokens of genuinely new prompt content.
    assert second_metadata["new_prompt_tokens"] == 15


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
    content_blocks = [b for b in result["message"]["blocks"] if b["type"] != "metadata"]
    assert content_blocks == [
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
    assert server._data_logger.finalized == [
        (
            server._data_logger.events[0][2],
            "llm_stream_error",
            [{"error": "ValueError: bad request"}],
        )
    ]


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
    assert server._data_logger.finalized == [
        (
            server._data_logger.events[0][2],
            "llm_stream_error",
            [{"error": "ConnectionError: down"}],
        )
    ]


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
    assert server._data_logger.finalized == [
        (
            server._data_logger.events[0][2],
            "llm_stream_error",
            [{"error": "RuntimeError: boom"}],
        )
    ]


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
    content_blocks = [b for b in message["blocks"] if b["type"] != "metadata"]
    assert len(content_blocks) == 1
    block = content_blocks[0]
    assert block["type"] == "text" and block["text"] == "Hello"
    assert "ts_start" in block and "ts_end" in block
    assert items[2]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    metadata_block = next(b for b in message["blocks"] if b["type"] == "metadata")
    assert metadata_block["data"][-1]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
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
                _FakeDataLogger(),
                parent_context=agprof.current_span_context(),
                usage_tracker=LlmUsageTracker(),
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
    assert server._data_logger.finalized == [
        (handle.call_label, "llm_stream_error", [{"error": "ValueError: nope"}])
    ]


def test_start_stream_first_chunk_transient_error_becomes_error_item(monkeypatch):
    monkeypatch.setattr(mod, "TRANSIENT_DISPATCH_EXCS", (ConnectionError,))

    def create(**kwargs):
        raise ConnectionError("down")

    server, _ = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    item = handle.first()
    assert item == {"type": "error", "message": "down", "transient": True, "status_code": 503}
    handle._thread.join(timeout=2.0)
    assert server._data_logger.finalized == [
        (handle.call_label, "llm_stream_error", [{"error": "ConnectionError: down"}])
    ]


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
    assert server._data_logger.finalized == [
        (
            handle.call_label,
            "llm_stream_error",
            [{"error": "RuntimeError: mid-stream failure"}],
        )
    ]


# ---------------------------------------------------------------------------
# invocation controls / message safe boundaries
# ---------------------------------------------------------------------------


def test_invocation_messages_are_fifo_protocol_valid_and_stable_across_stream_replay():
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _RecordingBackend()
    server = _controlled_server(backend, invocation)
    history = _completed_tool_history()

    invocation.send_message("first")
    invocation.send_message("second")
    server.dispatch({"messages": history})

    first_request = backend.requests[-1][1]
    assert first_request["messages"][: len(history)] == history
    assert _invocation_message_texts(first_request["messages"]) == [
        "[AGENCY INVOCATION MESSAGE]\nfirst",
        "[AGENCY INVOCATION MESSAGE]\nsecond",
    ]
    assert [message["role"] for message in first_request["messages"][-4:]] == [
        "tool",
        "tool",
        "user",
        "user",
    ]

    # Streaming is transport-only and therefore reuses the same semantic
    # boundary assignment rather than consuming newly queued invocation messages.
    invocation.send_message("third")
    stream = server.start_stream({"messages": history, "stream": True})
    assert _drain(stream)[-1]["type"] == "done"
    stream._thread.join(timeout=2.0)
    replay_request = backend.requests[-1][1]
    assert _invocation_message_texts(replay_request["messages"]) == [
        "[AGENCY INVOCATION MESSAGE]\nfirst",
        "[AGENCY INVOCATION MESSAGE]\nsecond",
    ]

    expanded = history + [
        _tool_message(),
        {
            "role": "tool",
            "blocks": [
                {
                    "type": "tool_result",
                    "index": 0,
                    "tool_call_id": "next",
                    "text": "next-result",
                }
            ],
        },
    ]
    server.dispatch({"messages": expanded})
    assert _invocation_message_texts(backend.requests[-1][1]["messages"]) == [
        "[AGENCY INVOCATION MESSAGE]\nfirst",
        "[AGENCY INVOCATION MESSAGE]\nsecond",
        "[AGENCY INVOCATION MESSAGE]\nthird",
    ]
    assert invocation.phase == "boundary"


@pytest.mark.parametrize("streaming", [False, True])
def test_final_response_establishes_closing_fence(streaming: bool):
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _RecordingBackend(final=True)
    server = _controlled_server(backend, invocation)
    invocation.send_message("before final")
    request = {"messages": _completed_tool_history(), "stream": streaming}

    if streaming:
        stream = server.start_stream(request)
        assert _drain(stream)[-1]["type"] == "done"
        stream._thread.join(timeout=2.0)
    else:
        assert server.dispatch(request)["message"]["blocks"][0]["text"] == "done"

    assert invocation.phase == "closing"
    with pytest.raises(RuntimeError, match="phase is closing"):
        invocation.send_message("too late")


def test_failed_attempt_reuses_pre_boundary_message_on_identical_retry():
    class _FailOnceBackend(_RecordingBackend):
        def __init__(self) -> None:
            super().__init__(final=True)
            self.failed = False

        def dispatch(self, request: dict) -> dict:
            self.requests.append(("nonstream", copy.deepcopy(request)))
            if not self.failed:
                self.failed = True
                raise RuntimeError("provider disconnected")
            return self._result()

    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _FailOnceBackend()
    server = _controlled_server(backend, invocation)
    invocation.send_message("assigned before attempt")
    request = {"messages": _completed_tool_history()}

    with pytest.raises(RuntimeError, match="provider disconnected"):
        server.dispatch(request)

    assert invocation.phase == "model"
    invocation.send_message("deliver after retry")

    server.dispatch(request)
    assert [_invocation_message_texts(item[1]["messages"]) for item in backend.requests] == [
        ["[AGENCY INVOCATION MESSAGE]\nassigned before attempt"],
        ["[AGENCY INVOCATION MESSAGE]\nassigned before attempt"],
        [
            "[AGENCY INVOCATION MESSAGE]\nassigned before attempt",
            "[AGENCY INVOCATION MESSAGE]\ndeliver after retry",
        ],
    ]


@pytest.mark.parametrize("streaming", [False, True])
def test_message_during_model_replaces_final_draft_at_next_safe_boundary(streaming: bool):
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _BlockingFinalOnceBackend()
    server = _controlled_server(backend, invocation)
    request = {"messages": _completed_tool_history(), "stream": streaming}

    if streaming:
        stream = server.start_stream(request)
    else:
        outcome = {}

        def dispatch() -> None:
            outcome["result"] = server.dispatch(request)

        worker = threading.Thread(target=dispatch, daemon=True)
        worker.start()

    assert backend.entered.wait(timeout=2.0)
    assert invocation.phase == "model"
    invocation.send_message("incorporate this before answering")
    backend.release.set()

    if streaming:
        terminal = _drain(stream)[-1]
        stream._thread.join(timeout=2.0)
        assert terminal["type"] == "done"
        assert terminal["message"]["blocks"][0]["text"] == "done"
    else:
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert outcome["result"]["message"]["blocks"][0]["text"] == "done"

    assert len(backend.requests) == 2
    assert _invocation_message_texts(backend.requests[0][1]["messages"]) == []
    assert _invocation_message_texts(backend.requests[1][1]["messages"]) == [
        "[AGENCY INVOCATION MESSAGE]\nincorporate this before answering"
    ]
    assert invocation.phase == "closing"


def test_incomplete_tool_batch_does_not_drain_or_inject_messages():
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _RecordingBackend()
    server = _controlled_server(backend, invocation)
    complete = _completed_tool_history()
    invocation.send_message("wait for every result")

    server.dispatch({"messages": complete[:-1]})
    assert _invocation_message_texts(backend.requests[-1][1]["messages"]) == []

    server.dispatch({"messages": complete})
    assert _invocation_message_texts(backend.requests[-1][1]["messages"]) == [
        "[AGENCY INVOCATION MESSAGE]\nwait for every result"
    ]


def test_internal_compaction_bypasses_controls_and_strips_marker():
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _RecordingBackend()
    server = _controlled_server(backend, invocation)
    invocation.send_message("for the next user-visible generation")
    history = _completed_tool_history()

    server.dispatch({"messages": history, "agency_internal_kind": "compaction"})
    compaction_request = backend.requests[-1][1]
    assert "agency_internal_kind" not in compaction_request
    assert _invocation_message_texts(compaction_request["messages"]) == []
    assert invocation.phase == "starting"

    server.dispatch({"messages": history})
    assert _invocation_message_texts(backend.requests[-1][1]["messages"]) == [
        "[AGENCY INVOCATION MESSAGE]\nfor the next user-visible generation"
    ]


@pytest.mark.parametrize(
    ("destroyed", "streaming", "expected"),
    [
        (False, False, "agent invocation cancelled"),
        (False, True, "agent invocation cancelled"),
        (True, False, "agent destroyed"),
        (True, True, "agent destroyed"),
    ],
)
def test_pre_model_stop_returns_conflict_without_calling_provider(
    destroyed: bool, streaming: bool, expected: str
):
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _RecordingBackend()
    server = _controlled_server(backend, invocation)
    if destroyed:
        control.destroy()
    else:
        invocation.cancel()

    response = TestClient(server.build_app()).post(
        "/dispatch", json={"messages": _completed_tool_history(), "stream": streaming}
    )

    assert response.status_code == 409
    assert response.json() == {"error": {"message": expected, "transient": False}}
    assert backend.requests == []
    assert server._data_logger.events == []
    assert server._data_logger.finalized == []


@pytest.mark.parametrize("streaming", [False, True])
def test_pause_during_model_parks_post_model_boundary_before_tool_delivery(streaming: bool):
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _BlockingToolBackend()
    server = _controlled_server(backend, invocation)
    request = {"messages": _completed_tool_history(), "stream": streaming}
    finished = threading.Event()
    outcome = {}

    if streaming:
        stream = server.start_stream(request)
    else:

        def dispatch() -> None:
            try:
                outcome["result"] = server.dispatch(request)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                finished.set()

        worker = threading.Thread(target=dispatch, daemon=True)
        worker.start()

    assert backend.entered.wait(timeout=2.0)
    assert invocation.phase == "model"
    invocation.pause()
    assert control.is_paused_actual() is False
    backend.release.set()

    assert _wait_until(control.is_paused_actual)
    if streaming:
        assert stream._queue.empty()
        assert stream._thread.is_alive()
    else:
        assert finished.is_set() is False

    invocation.resume()
    if streaming:
        assert _drain(stream)[-1]["type"] == "done"
        stream._thread.join(timeout=2.0)
        assert not stream._thread.is_alive()
    else:
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert "error" not in outcome
        assert outcome["result"]["message"]["blocks"][0]["type"] == "tool_use"
    assert invocation.phase == "boundary"


@pytest.mark.parametrize("streaming", [False, True])
def test_cancel_during_model_suppresses_post_model_tool_delivery(streaming: bool):
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _BlockingToolBackend()
    server = _controlled_server(backend, invocation)
    request = {"messages": _completed_tool_history(), "stream": streaming}

    if streaming:
        stream = server.start_stream(request)
    else:
        outcome = {}

        def run_dispatch() -> None:
            try:
                outcome["result"] = server.dispatch(request)
            except BaseException as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=run_dispatch, daemon=True)
        worker.start()

    assert backend.entered.wait(timeout=2.0)
    assert invocation.phase == "model"
    invocation.cancel()
    backend.release.set()

    if streaming:
        terminal = _drain(stream)[-1]
        stream._thread.join(timeout=2.0)
        assert terminal == {
            "type": "error",
            "message": "agent invocation cancelled",
            "transient": False,
            "status_code": 409,
        }
    else:
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert "result" not in outcome
        assert str(outcome["error"]) == "agent invocation cancelled"
        assert outcome["error"].status_code == 409

    logger = server._data_logger
    assert len(logger.events) == 1
    assert logger.finalized == [
        (
            logger.events[0][2],
            "llm_stream_error",
            [{"error": "_DispatchError: agent invocation cancelled"}],
        )
    ]
    if streaming:
        assert logger.events[0][2] == stream.call_label


def test_unclassified_first_stream_read_error_always_wakes_consumer():
    def create(**kwargs):
        del kwargs
        raise RuntimeError("setup exploded")

    server, client = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})

    assert handle.first() == {
        "type": "error",
        "message": "setup exploded",
        "transient": False,
        "status_code": 500,
    }
    handle._thread.join(timeout=2.0)
    assert not handle._thread.is_alive()
    assert client.closed is True
    assert server._data_logger.finalized == [
        (
            handle.call_label,
            "llm_stream_error",
            [{"error": "RuntimeError: setup exploded"}],
        )
    ]
    assert server.get_main_transcript() == []


def test_controlled_paths_preserve_logger_call_labels_and_finalization():
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _RecordingBackend()
    server = _controlled_server(backend, invocation)
    handle = server.start_stream({"messages": _completed_tool_history(), "stream": True})

    assert _drain(handle)[-1]["type"] == "done"
    handle._thread.join(timeout=2.0)
    logger = server._data_logger
    assert logger.events == [("agent_state", {"state": "waiting_llm"}, handle.call_label, True)]
    assert logger.finalized[-1][0] == handle.call_label
    assert logger.finalized[-1][1] == "llm_block"


def test_malformed_nonstream_result_finalizes_the_logger_call_label():
    class MalformedBackend(_RecordingBackend):
        def dispatch(self, request: dict) -> dict:
            self.requests.append(("nonstream", copy.deepcopy(request)))
            return {"usage": None, "message": {"role": "assistant", "blocks": []}}

    control = AgentControl()
    invocation = control.begin_invocation("external")
    server = _controlled_server(MalformedBackend(), invocation)

    with pytest.raises(KeyError, match="stop_reason"):
        server.dispatch({"messages": _completed_tool_history()})

    logger = server._data_logger
    assert logger.finalized == [
        (
            logger.events[0][2],
            "llm_stream_error",
            [{"error": "KeyError: 'stop_reason'"}],
        )
    ]


@pytest.mark.parametrize("failure_point", ["span", "annotate"])
def test_nonstream_outer_infrastructure_failure_finalizes_once(monkeypatch, failure_point):
    server, _ = _make_server(create_fn=lambda **_kwargs: None)
    if failure_point == "span":
        monkeypatch.setattr(
            agprof,
            "span",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("span failed")),
        )
    else:
        monkeypatch.setattr(
            mod,
            "_annotate",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("annotate failed")),
        )

    with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
        server.dispatch({"messages": []})

    logger = server._data_logger
    assert logger.finalized == [
        (
            logger.events[0][2],
            "llm_stream_error",
            [{"error": f"RuntimeError: {failure_point} failed"}],
        )
    ]
    assert [operation[0] for operation in logger.operations] == ["finalize"]


def test_stream_spawn_failure_finalizes_the_logger_call_label(monkeypatch):
    server, _ = _make_server(create_fn=lambda **_kwargs: iter([]))

    def fail_spawn(*_args, **_kwargs):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(agprof, "spawn_traced", fail_spawn)

    with pytest.raises(RuntimeError, match="spawn failed"):
        server.start_stream({"messages": []})

    logger = server._data_logger
    assert logger.finalized == [
        (
            logger.events[0][2],
            "llm_stream_error",
            [{"error": "RuntimeError: spawn failed"}],
        )
    ]


def test_stream_spawn_failure_observes_control_cancel(monkeypatch):
    control = AgentControl()
    invocation = control.begin_invocation("external")
    server = _controlled_server(_RecordingBackend(), invocation)

    def cancel_then_fail(*_args, **_kwargs):
        invocation.cancel()
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(agprof, "spawn_traced", cancel_then_fail)

    with pytest.raises(mod._DispatchError, match="agent invocation cancelled"):
        server.start_stream({"messages": _completed_tool_history()})

    logger = server._data_logger
    assert logger.finalized == [
        (
            logger.events[0][2],
            "llm_stream_error",
            [{"error": "_DispatchError: agent invocation cancelled"}],
        )
    ]


def test_stream_spawn_failure_after_disconnect_finalizes_as_cancelled(monkeypatch):
    server, _ = _make_server(create_fn=lambda **_kwargs: iter([]))
    abort_event = threading.Event()

    def abort_then_fail(*_args, **_kwargs):
        abort_event.set()
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(agprof, "spawn_traced", abort_then_fail)

    with pytest.raises(mod._RequestAborted):
        server.start_stream({"messages": []}, abort_event=abort_event)

    logger = server._data_logger
    assert logger.finalized == [
        (logger.events[0][2], "llm_stream_cancelled", [{"cancelled": True}])
    ]


def test_stream_thread_start_failure_finalizes_and_removes_the_handle(monkeypatch):
    class StartFailure:
        def start(self):
            raise RuntimeError("thread start failed")

    server, _ = _make_server(create_fn=lambda **_kwargs: iter([]))
    monkeypatch.setattr(agprof, "spawn_traced", lambda *_args, **_kwargs: StartFailure())

    with pytest.raises(RuntimeError, match="thread start failed"):
        server.start_stream({"messages": []})

    assert server._handles == []
    logger = server._data_logger
    assert logger.finalized == [
        (
            logger.events[0][2],
            "llm_stream_error",
            [{"error": "RuntimeError: thread start failed"}],
        )
    ]


def test_outer_stream_producer_failure_wakes_consumer_and_finalizes(monkeypatch):
    server, _ = _make_server(create_fn=lambda **_kwargs: iter([]))
    monkeypatch.setattr(
        mod,
        "_annotate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("span annotation failed")),
    )

    handle = server.start_stream({"messages": []})

    assert handle.first() == {
        "type": "error",
        "message": "span annotation failed",
        "transient": False,
        "status_code": 500,
    }
    handle._thread.join(timeout=2.0)
    assert not handle._thread.is_alive()
    assert server._data_logger.finalized == [
        (
            handle.call_label,
            "llm_stream_error",
            [{"error": "RuntimeError: span annotation failed"}],
        )
    ]


def test_rejected_stream_error_enqueue_finalizes_as_cancelled():
    class _FailingBackend:
        model = "failing"

        def dispatch_stream(self, request, *, on_client=None):
            del request, on_client
            raise RuntimeError("provider failed")
            yield  # pragma: no cover

    class _RejectErrorHandle(mod._StreamHandle):
        def register_stream_exchange(self, item=None, **entry_fields):
            if item is not None and item.get("type") == "error":
                self.cancel()
                return False
            return super().register_stream_exchange(item, **entry_fields)

    server = LlmHandlerServer(
        _cfg(model="gpt-test"), _FakeDataLogger(), usage_tracker=LlmUsageTracker()
    )
    server._backend = _FailingBackend()
    handle = _RejectErrorHandle(mod.queue.Queue(), threading.Event(), "rejected-error")

    server._run_stream_producer({"messages": []}, None, False, handle)

    assert server._data_logger.finalized == [
        ("rejected-error", "llm_stream_cancelled", [{"cancelled": True}])
    ]


def test_stream_logger_records_every_provider_item_in_order_before_finalize():
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _RecordingBackend()
    server = _controlled_server(backend, invocation)
    handle = server.start_stream({"messages": _completed_tool_history(), "stream": True})

    assert _drain(handle)[-1]["type"] == "done"
    handle._thread.join(timeout=2.0)
    expected_items = [
        {
            "type": "content",
            "index": 0,
            "block_type": "tool_use",
            "id": "next",
            "name": "lookup",
            "arguments": "{}",
        },
        {"type": "usage", "usage": None, "stop_reason": "stop"},
    ]
    logger = server._data_logger
    assert [entry[1] for entry in logger.stream_delta_history] == expected_items
    assert logger.operations == [
        ("delta", handle.call_label, expected_items[0]),
        ("delta", handle.call_label, expected_items[1]),
        ("finalize", handle.call_label, "llm_block"),
    ]


# ---------------------------------------------------------------------------
# cancellation / resource release
# ---------------------------------------------------------------------------


def test_handle_relay_stays_in_handles_on_completion():
    def create(**kwargs):
        return iter([_FakeChunk([_FakeChoice(delta=_FakeDelta(content="hi"))])])

    server, _ = _make_server(create_fn=create)
    handle = server.start_stream({"messages": []})
    first = handle.first()

    async def consume() -> None:
        async for _chunk in handle.relay(first):
            pass

    asyncio.run(consume())
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


def test_stream_ref_published_after_cancel_is_closed_immediately_once():
    close_calls = []

    class _Stream:
        def close(self):
            close_calls.append(1)

    handle = mod._StreamHandle(mod.queue.Queue(), threading.Event())
    handle.cancel()
    handle._set_stream_ref(_Stream())
    handle.cancel()

    assert close_calls == [1]


def test_failed_stream_close_remains_retryable():
    close_calls = []

    class _Stream:
        def close(self):
            close_calls.append(1)
            if len(close_calls) == 1:
                raise RuntimeError("close failed")

    handle = mod._StreamHandle(mod.queue.Queue(), threading.Event())
    handle._set_stream_ref(_Stream())

    with pytest.raises(RuntimeError, match="close failed"):
        handle.cancel()
    handle.cancel()
    assert close_calls == [1, 1]


def test_disconnect_cleanup_joins_producer_even_when_initial_close_raises():
    server, _ = _make_server(create_fn=lambda **kw: iter([]))
    producer_release = threading.Event()
    producer = threading.Thread(target=producer_release.wait, daemon=True)
    handle = mod._StreamHandle(mod.queue.Queue(), threading.Event())
    handle._thread = producer
    producer.start()
    close_calls = []

    class _Stream:
        def close(self):
            close_calls.append(1)
            if len(close_calls) == 1:
                raise RuntimeError("close failed")
            producer_release.set()

    handle._set_stream_ref(_Stream())

    async def receive():
        return {"type": "http.disconnect"}

    request = SimpleNamespace(receive=receive)
    first_worker, first_result = server._spawn_http_worker(handle.first)

    with pytest.raises(RuntimeError, match="close failed"):
        asyncio.run(
            server._wait_for_http_worker(
                request,
                first_worker,
                first_result,
                handle.cancel,
                handle.cancel_and_join,
            )
        )

    assert close_calls == [1, 1]
    assert not first_worker.is_alive()
    assert not producer.is_alive()


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


def test_streaming_response_disconnect_unblocks_full_queue_and_joins_producer():
    q = mod.queue.Queue(maxsize=1)
    handle = mod._StreamHandle(q, threading.Event())
    assert handle.register_stream_exchange({"type": "delta", "content": "queued"})

    producer_entered = threading.Event()
    producer_finished = threading.Event()
    enqueue_result = []

    def produce() -> None:
        producer_entered.set()
        enqueue_result.append(
            handle.register_stream_exchange(
                {"type": "done"},
                response={"role": "assistant", "blocks": [{"type": "tool_use"}]},
            )
        )
        producer_finished.set()

    producer = threading.Thread(target=produce, daemon=True)
    handle._thread = producer
    producer.start()
    assert producer_entered.wait(timeout=2.0)
    assert producer_finished.wait(timeout=0.05) is False

    response = StreamingResponse(
        handle.relay({"type": "delta", "content": "first"}),
        media_type="application/x-ndjson",
    )
    asyncio.run(_disconnect_after_first_response_body(response))

    assert producer_finished.is_set()
    assert enqueue_result == [False]
    assert not producer.is_alive()
    assert handle._cancel_event.is_set()
    assert handle.get_transcript()[0]["response"] is None


def test_streaming_response_disconnect_interrupts_paused_post_model_checkpoint():
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _TextThenBlockingToolBackend()
    server = _controlled_server(backend, invocation)
    handle = server.start_stream({"messages": _completed_tool_history(), "stream": True})
    first = handle.first()

    assert first == {"type": "delta", "content": "working"}
    assert backend.after_text.wait(timeout=2.0)
    invocation.pause()
    backend.release.set()
    assert _wait_until(control.is_paused_actual)
    assert handle._thread.is_alive()

    response = StreamingResponse(handle.relay(first), media_type="application/x-ndjson")
    asyncio.run(_disconnect_after_first_response_body(response))

    assert not handle._thread.is_alive()
    assert control.is_pause_requested()
    assert server._data_logger.finalized == [
        (handle.call_label, "llm_stream_cancelled", [{"cancelled": True}])
    ]


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

    server = LlmHandlerServer(
        _cfg(model="gpt-test"), _FakeDataLogger(), usage_tracker=LlmUsageTracker()
    )
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
    server.stop()


def test_stop_raises_while_producer_is_alive_and_can_be_retried(monkeypatch):
    server, _ = _make_server(create_fn=lambda **kw: iter([]))
    release = threading.Event()
    producer = threading.Thread(target=release.wait, daemon=True)
    handle = mod._StreamHandle(mod.queue.Queue(), threading.Event())
    handle._thread = producer
    server._handles.append(handle)
    producer.start()
    monkeypatch.setattr(mod, "_STREAM_JOIN_TIMEOUT_S", 0.0)

    with pytest.raises(RuntimeError, match="1 LLM stream producer.*did not stop"):
        server.stop()
    assert producer.is_alive()

    release.set()
    producer.join(timeout=2.0)
    assert not producer.is_alive()
    server.stop()


def test_stop_linearizes_against_concurrent_stream_start(monkeypatch):
    server, _ = _make_server(create_fn=lambda **kw: iter([]))
    start_entered = threading.Event()
    release_start = threading.Event()
    start_outcome = {}
    stop_outcome = {}

    class _SlowStartThread:
        def __init__(self, handle):
            self.handle = handle
            self.started = False
            self.alive = False

        def start(self):
            start_entered.set()
            assert release_start.wait(timeout=2.0)
            self.started = True
            self.alive = True

        def join(self, timeout=None):
            del timeout
            if not self.started:
                raise RuntimeError("cannot join thread before it is started")
            if self.handle._cancel_event.is_set():
                self.alive = False

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(
        agprof,
        "spawn_traced",
        lambda *_args, **_kwargs: _SlowStartThread(_args[-1]),
    )

    def start() -> None:
        try:
            start_outcome["handle"] = server.start_stream({"messages": []})
        except BaseException as error:
            start_outcome["error"] = error

    def stop() -> None:
        try:
            server.stop()
        except BaseException as error:
            stop_outcome["error"] = error
        finally:
            stop_outcome["finished"] = True

    starter = threading.Thread(target=start, daemon=True)
    stopper = threading.Thread(target=stop, daemon=True)
    starter.start()
    assert start_entered.wait(timeout=2.0)
    stopper.start()
    assert stopper.join(timeout=0.05) is None
    assert "finished" not in stop_outcome

    release_start.set()
    starter.join(timeout=2.0)
    stopper.join(timeout=2.0)

    assert "error" not in start_outcome
    assert "error" not in stop_outcome
    assert stop_outcome["finished"] is True
    handle = start_outcome["handle"]
    assert handle._cancel_event.is_set()
    assert not handle._thread.is_alive()

    with pytest.raises(RuntimeError, match="LLM handler server is stopping"):
        server.start_stream({"messages": []})


def test_stop_with_no_handles_returns_immediately():
    server, _ = _make_server(create_fn=lambda **kw: iter([]))
    server.stop()  # should not raise or hang


# ---------------------------------------------------------------------------
# build_app / HTTP routes
# ---------------------------------------------------------------------------


def test_http_worker_prefers_disconnect_when_completion_is_simultaneously_ready(monkeypatch):
    server, _ = _make_server(create_fn=lambda **kw: iter([]))
    result = mod.Future()
    result.set_result("complete")
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join(timeout=2.0)
    aborted = threading.Event()

    async def receive():
        return {"type": "http.disconnect"}

    request = SimpleNamespace(receive=receive)
    real_wait = asyncio.wait

    async def wait_for_both(awaitables, *, return_when):
        del return_when
        return await real_wait(awaitables, return_when=asyncio.ALL_COMPLETED)

    monkeypatch.setattr(mod.asyncio, "wait", wait_for_both)
    disconnected = asyncio.run(server._wait_for_http_worker(request, worker, result, aborted.set))

    assert disconnected is True
    assert aborted.is_set()


def test_build_app_dispatch_route_non_streaming():
    def create(**kwargs):
        return _FakeResult([_FakeChoice(message=_FakeMessage(content="hi"), finish_reason="stop")])

    server, _ = _make_server(create_fn=create)
    client = TestClient(server.build_app())
    response = client.post("/dispatch", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    body = response.json()
    content_blocks = [b for b in body["message"]["blocks"] if b["type"] != "metadata"]
    assert body["message"]["role"] == "assistant"
    assert content_blocks == [{"type": "text", "index": 0, "text": "hi"}]
    assert body["usage"] is None
    assert body["stop_reason"] == "stop"


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


def test_stream_http_disconnect_while_paused_before_model_leaves_no_logger_call():
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _RecordingBackend()
    server = _controlled_server(backend, invocation)
    invocation.pause()

    async def scenario() -> None:
        disconnect = asyncio.Event()
        request = asyncio.create_task(
            _post_app_until_disconnect(
                server.build_app(),
                {"messages": _completed_tool_history(), "stream": True},
                disconnect,
            )
        )
        assert await asyncio.to_thread(_wait_until, control.is_paused_actual)
        disconnect.set()
        await asyncio.wait_for(request, timeout=2.0)

    asyncio.run(scenario())

    assert backend.requests == []
    assert server._data_logger.events == []
    assert server._data_logger.finalized == []
    assert control.is_pause_requested()


def test_stream_http_disconnect_before_first_item_joins_producer():
    backend = _BlockingToolBackend()
    server = _controlled_server(backend, invocation=None)

    async def scenario() -> None:
        disconnect = asyncio.Event()
        request = asyncio.create_task(
            _post_app_until_disconnect(
                server.build_app(),
                {"messages": _completed_tool_history(), "stream": True},
                disconnect,
            )
        )
        assert await asyncio.to_thread(backend.entered.wait, 2.0)
        # Closing a real provider stream wakes its first read. This fake models
        # that wake independently so the event loop may block until cleanup
        # has fully joined the producer.
        wake_provider = threading.Timer(0.05, backend.release.set)
        wake_provider.start()
        disconnect.set()
        await asyncio.wait_for(request, timeout=2.0)
        wake_provider.join(timeout=2.0)

    asyncio.run(scenario())

    logger = server._data_logger
    assert logger.finalized == [
        (logger.events[0][2], "llm_stream_cancelled", [{"cancelled": True}])
    ]
    assert all(
        handle._thread is None or not handle._thread.is_alive() for handle in server._handles
    )


def test_nonstream_http_disconnect_interrupts_paused_post_model_checkpoint():
    control = AgentControl()
    invocation = control.begin_invocation("external")
    backend = _BlockingToolBackend()
    server = _controlled_server(backend, invocation)

    async def scenario() -> None:
        disconnect = asyncio.Event()
        request = asyncio.create_task(
            _post_app_until_disconnect(
                server.build_app(),
                {"messages": _completed_tool_history()},
                disconnect,
            )
        )
        assert await asyncio.to_thread(backend.entered.wait, 2.0)
        invocation.pause()
        backend.release.set()
        assert await asyncio.to_thread(_wait_until, control.is_paused_actual)
        disconnect.set()
        await asyncio.wait_for(request, timeout=2.0)

    asyncio.run(scenario())

    logger = server._data_logger
    assert logger.finalized == [
        (logger.events[0][2], "llm_stream_cancelled", [{"cancelled": True}])
    ]
    assert control.is_pause_requested()


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
