# Tests for llm_handler_server.py -- host-side LLM routing, non-streaming
# dispatch, and the streaming relay (producer thread + _StreamHandle).

from __future__ import annotations

import asyncio
import copy
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from starlette.responses import StreamingResponse

from agency.configs.agconfig import agconfig, llmconfig
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

    def record_final_transcript(
        self, call_label, type, payloads, term_message=None, print_to_terminal=True
    ):
        self.stream_deltas = [d for d in self.stream_deltas if d[2] != call_label]
        self.finalized.append((call_label, type, payloads))
        self.operations.append(("finalize", call_label, type))


def _cfg(**fields) -> agconfig:
    fields.setdefault("base_url", "http://x/v1")
    return agconfig(llmconfig(**fields))


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


def _controlled_server(backend) -> LlmHandlerServer:
    server = LlmHandlerServer(
        _cfg(model="gpt-test"),
        _FakeDataLogger(),
        request_id="test-request",
        skill_name="test-skill",
        usage_tracker=LlmUsageTracker(),
    )
    server._backend = backend
    return server


# ---------------------------------------------------------------------------
# _payload_hash -- transcript dedup identity
# ---------------------------------------------------------------------------


def test_payload_hash_treats_native_and_replayed_tool_use_as_the_same_call():
    """A tool_use block gets logged once in its rich, backend-native shape
    (with extra bookkeeping fields like text/signature/data/citations/
    ts_start/ts_end) and again in the reduced shape native_harness's
    OpenAI-chatcompletions wire protocol reconstructs when that same call is
    replayed back into a later request (id/name/arguments only, and often a
    different `index`) -- these must hash identically so the second one
    doesn't get double-logged."""
    rich = {
        "role": "assistant",
        "type": "tool_use",
        "index": 1,
        "text": "",
        "signature": "",
        "id": "call_1",
        "name": "write",
        "arguments": '{"a": 1}',
        "data": None,
        "citations": None,
        "ts_start": 123.0,
        "ts_end": 123.0,
    }
    replayed = {
        "role": "assistant",
        "type": "tool_use",
        "index": 0,
        "id": "call_1",
        "name": "write",
        "arguments": '{"a": 1}',
    }
    assert mod._payload_hash(rich) == mod._payload_hash(replayed)


def test_payload_hash_distinguishes_different_tool_use_calls():
    a = {"role": "assistant", "type": "tool_use", "index": 0, "id": "call_1", "name": "write"}
    b = {"role": "assistant", "type": "tool_use", "index": 0, "id": "call_2", "name": "write"}
    assert mod._payload_hash(a) != mod._payload_hash(b)


def test_payload_hash_distinguishes_different_tool_results():
    a = {"role": "tool", "type": "tool_result", "index": 0, "tool_call_id": "call_1", "text": "ok"}
    b = {"role": "tool", "type": "tool_result", "index": 0, "tool_call_id": "call_2", "text": "ok"}
    assert mod._payload_hash(a) != mod._payload_hash(b)


def test_payload_hash_treats_native_and_resent_text_as_the_same_message():
    """A text block gets logged once in its rich response shape (ts_start/
    ts_end/id/etc.) and again in the reduced shape it takes when the same
    message is resent inside a later request's history (role/type/text/
    index only, often a different index) -- these must hash identically so
    the resent copy doesn't get double-logged, corrupting replay."""
    rich = {
        "role": "assistant",
        "type": "text",
        "index": -1,
        "text": "Let me look.",
        "ts_start": 1.0,
        "ts_end": 1.5,
    }
    resent = {"role": "assistant", "type": "text", "index": 0, "text": "Let me look."}
    assert mod._payload_hash(rich) == mod._payload_hash(resent)


def test_payload_hash_falls_back_to_whole_payload_for_metadata_blocks():
    """metadata has no id-like stable identity field and is never resent
    inside a later request, so it is still hashed on full content."""
    a = {"role": "assistant", "type": "metadata", "index": 2**31 - 1, "stop_reason": "stop"}
    b = {"role": "assistant", "type": "metadata", "index": 2**31 - 1, "stop_reason": "tool_use"}
    assert mod._payload_hash(a) != mod._payload_hash(b)
    assert mod._payload_hash(a) == mod._payload_hash(dict(a))


# ---------------------------------------------------------------------------
# _new_transcript_payloads across turns -- the resent-text bug end to end
# ---------------------------------------------------------------------------


def test_resent_response_text_does_not_reappear_as_new_content():
    """Two consecutive turns, shaped the way native_harness actually resends
    history: turn 2's request includes turn 1's own answer back as an
    assistant message. Before the fix, that resent text was not recognized
    as a duplicate and was logged again as "new" -- landing in turn 2's own
    recorded group at the same index as turn 2's real tool_use call. A
    replayer merges same-index blocks together, so the tool_use turned into
    a text block carrying its name and no arguments, and the call never
    fired."""
    server, _ = _make_server()
    turn1_request = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "index": 0, "text": "Find the bug"}]}
        ]
    }
    turn1_response = {
        "role": "assistant",
        "blocks": [
            {"type": "text", "index": -1, "text": "Let me look.", "ts_start": 1.0, "ts_end": 1.5},
            {"type": "tool_use", "index": 0, "id": "call_1", "name": "glob", "arguments": "{}"},
            {"type": "metadata", "index": 2**31 - 1, "stop_reason": "tool_use"},
        ],
    }
    first = server._new_transcript_payloads(turn1_request, turn1_response)
    assert len(first) == 4  # the user prompt, plus all three response blocks

    turn2_request = {
        "messages": [
            {"role": "user", "blocks": [{"type": "text", "index": 0, "text": "Find the bug"}]},
            {"role": "assistant", "blocks": [{"type": "text", "index": 0, "text": "Let me look."}]},
            {
                "role": "tool",
                "blocks": [
                    {"type": "tool_result", "index": 0, "tool_call_id": "call_1", "text": "a.py"}
                ],
            },
        ]
    }
    turn2_response = {
        "role": "assistant",
        "blocks": [
            {"type": "text", "index": -1, "text": "Found it.", "ts_start": 2.0, "ts_end": 2.5},
            {
                "type": "tool_use",
                "index": 0,
                "id": "call_2",
                "name": "read",
                "arguments": '{"path": "a.py"}',
            },
            # Distinct from turn 1's metadata (usage differs) -- otherwise it dedups too,
            # for the unrelated, correct reason that it would be byte-identical.
            {
                "type": "metadata",
                "index": 2**31 - 1,
                "stop_reason": "tool_use",
                "usage": {"total_tokens": 2},
            },
        ],
    }
    second = server._new_transcript_payloads(turn2_request, turn2_response)

    # Only what's genuinely new this turn: the tool result and this turn's own three
    # response blocks -- not turn 1's text, resent as history.
    assert [p["type"] for p in second] == ["tool_result", "text", "tool_use", "metadata"]
    assert second[1]["text"] == "Found it."
    # Exactly one *assistant* block at index 0: the real tool_use call. The resent text
    # was also role=assistant, index 0 (reduced shape) -- that's the actual collision a
    # replayer hits, once role=tool content is filtered out downstream.
    assistant_index_zero = [p for p in second if p["role"] == "assistant" and p.get("index") == 0]
    assert len(assistant_index_zero) == 1
    assert assistant_index_zero[0]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# resolve_model / context_limit
# ---------------------------------------------------------------------------


def test_resolve_model_returns_backends_model():
    server, _ = _make_server(model="gpt-test")
    assert server.resolve_model() == "gpt-test"


def test_resolve_model_empty_string_when_unset():
    server, _ = _make_server()
    server._backend.agconfig.llm.model = None
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
    # Non-streaming calls report ttft_ms too -- the same moment the whole
    # response becomes available, since there's no earlier partial content.
    assert isinstance(first_metadata["ttft_ms"], float)

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
    server.dispatch(
        {
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "tool_choice": "auto",
        }
    )
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


def test_start_stream_accumulates_deltas_and_delivers_one_done_item():
    """Deltas are never individually delivered to the harness -- only the
    one final "done" event carries the fully-assembled message. This is
    what lets the producer fully drain the upstream provider regardless of
    the harness's own state (paused, slow, or gone): the queue-full
    backpressure wait only ever triggers for a real delivered item, and
    there is only one of those now."""

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
    assert [i["type"] for i in items] == ["done"]
    message = items[0]["message"]
    assert message["role"] == "assistant"
    content_blocks = [b for b in message["blocks"] if b["type"] != "metadata"]
    assert len(content_blocks) == 1
    block = content_blocks[0]
    assert block["type"] == "text" and block["text"] == "Hello"
    assert "ts_start" in block and "ts_end" in block
    assert items[0]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    metadata_block = next(b for b in message["blocks"] if b["type"] == "metadata")
    assert metadata_block["data"][-1]["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    # Streaming calls persist TTFT into the durable record, not just an
    # agprof span annotation -- a replay backend needs it without depending
    # on profiling having been active during the original run.
    assert isinstance(metadata_block["ttft_ms"], float) and metadata_block["ttft_ms"] >= 0
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


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.skipif(sys.platform != "linux", reason="agprof requires Linux /proc and cgroups")
def test_llm_trace_includes_transcript_and_token_counts(tmp_path, streaming):
    def create(**kwargs):
        usage = _FakeUsage(prompt_tokens=12, completion_tokens=3, total_tokens=15)
        if streaming:
            return iter(
                [
                    _FakeChunk([_FakeChoice(delta=_FakeDelta(content="Hello"))]),
                    _FakeChunk([_FakeChoice(delta=_FakeDelta(), finish_reason="stop")]),
                    _FakeChunk([], usage=usage),
                ]
            )
        return _FakeResult(
            [_FakeChoice(message=_FakeMessage(content="Hello"), finish_reason="stop")], usage
        )

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False, auto_functions=False):
        server, _ = _make_server(create_fn=create)
        request = {"messages": [{"role": "user", "content": "hi"}]}
        if streaming:
            handle = server.start_stream(request)
            assert _drain(handle)[-1]["type"] == "done"
            handle._thread.join(timeout=2)
        else:
            server.dispatch(request)
    trace = json.loads((tmp_path / "agprof.trace.json").read_text())
    args = next(e["args"] for e in trace["traceEvents"] if e.get("name") == "llm:attempt[0]")
    assert args["input_tokens"] == 12
    assert args["output_tokens"] == 3
    assert args["total_tokens"] == 15
    assert args["stop_reason"] == "stop"
    assert json.loads(args["llm.messages"])[0]["content"] == "hi"
    assert json.loads(args["llm.response"])["blocks"][0]["text"] == "Hello"
    assert args["llm.response_truncated"] is False


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

    records = {record[1]: record for record in agprof.profile_records()}
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
    # The "ok" delta is never individually delivered -- only the error item is.
    assert items == [
        {
            "type": "error",
            "message": "mid-stream failure",
            "transient": False,
            "status_code": 500,
        }
    ]
    handle._thread.join(timeout=2.0)
    assert client.closed is True
    assert server._data_logger.finalized == [
        (
            handle.call_label,
            "llm_stream_error",
            [{"error": "RuntimeError: mid-stream failure"}],
        )
    ]


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
    backend = _RecordingBackend()
    server = _controlled_server(backend)
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

    server = _controlled_server(MalformedBackend())

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

    server._run_stream_producer({"messages": []}, handle)

    assert server._data_logger.finalized == [
        ("rejected-error", "llm_stream_cancelled", [{"cancelled": True}])
    ]


def test_stream_logger_records_every_provider_item_in_order_before_finalize():
    backend = _RecordingBackend()
    server = _controlled_server(backend)
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
    # Deltas are never individually relayed -- only the one final "done" line.
    assert len(lines) == 1
    assert lines[0]["type"] == "done"
    assert lines[0]["message"]["role"] == "assistant"
    blocks = lines[0]["message"]["blocks"]
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


def test_stream_http_disconnect_before_first_item_joins_producer():
    backend = _BlockingToolBackend()
    server = _controlled_server(backend)

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


def test_cached_routes_do_not_retain_server_after_app_is_released():
    import gc
    import weakref

    server = _make_server()[0]
    reference = weakref.ref(server)
    app = server.build_app()
    # FastAPI caches endpoint classification independently of the app lifetime.
    from fastapi.routing import APIRoute

    cached_endpoints = [route.endpoint for route in app.routes if isinstance(route, APIRoute)]
    del server
    gc.collect()
    assert reference() is not None, "the live app must own its server"
    del app
    gc.collect()
    assert reference() is None, "cached endpoints must not retain completed requests"
    assert cached_endpoints
