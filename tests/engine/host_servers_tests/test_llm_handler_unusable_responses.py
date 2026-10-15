"""LlmHandlerServer and provider responses that are not a finished turn.

Regressions for what recorded Terminal-Bench runs showed going wrong through
Codex -> Agency -> LiteLLM -> Bedrock:

* 41 exchanges came back HTTP 200 with one chunk -- finish_reason "stop", no
  content, no tool call, no usage -- and were delivered as completed turns;
* 30 responses stopped at the output-token limit, 29 of them inside a tool
  call's JSON arguments, and were delivered as completed tool calls;
* a stream that broke after it started was reported as a permanent failure.

The scripted backend speaks Agency's own stream items, so these tests hold
for every provider; the SDK-level shapes are covered at the end.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agency.configs.agconfig import agconfig, llmconfig
from agency.engine.host_servers import llm_handler_server as mod
from agency.engine.host_servers.llm_handler_server import LlmHandlerServer, _DispatchError
from agency.llm import StreamInterruptedError, UnusableResponseError
from agency.llm.agllm import ProviderResponseError, current_dispatch_attempt
from agency.llm.openai import _OpenAICompatibleBackend
from agency.llm.usage_tracker import LlmUsageTracker

METADATA_INDEX = 2**31 - 1
CUT = '{"cmd":"cat <<\'EOF\' > notes.txt\\n1. Linux scheduling determines which'
REQUEST = {"messages": [{"role": "user", "blocks": [{"type": "text", "index": 0, "text": "go"}]}]}


# --------------------------------------------------------------- stream items


def text(value, index=0):
    return {"type": "block_delta", "index": index, "block_type": "text", "text": value}


def reasoning(signature="openai_responses:rsn", index=0):
    return {"type": "block_delta", "index": index, "block_type": "thinking", "signature": signature}


def tool(arguments, call_id="call_1", name="exec_command", index=0):
    return {
        "type": "block_delta",
        "index": index,
        "block_type": "tool_use",
        "id": call_id,
        "name": name,
        "arguments": arguments,
    }


def end(stop_reason="stop", usage=None):
    return [
        {
            "type": "block_delta",
            "index": METADATA_INDEX,
            "block_type": "metadata",
            "data": {"usage": usage, "stop_reason": stop_reason},
        },
        {"type": "usage", "usage": usage, "stop_reason": stop_reason},
    ]


EMPTY_200 = end("stop")  # the recorded Luna shape: a finish reason and nothing else
GOOD_TOOL = [
    tool('{"cmd":"nproc"}'),
    *end("tool_calls", {"prompt_tokens": 9, "completion_tokens": 3}),
]
GOOD_TEXT = [text("done"), *end("stop", {"prompt_tokens": 9, "completion_tokens": 1})]
TRUNCATED_TOOL = [tool(CUT), *end("length", {"prompt_tokens": 9, "completion_tokens": 4096})]


class Raises:
    """A script step that fails the stream at this point."""

    def __init__(self, error):
        self.error = error


class ScriptedBackend:
    """Plays one scripted stream per ask and records how it was asked."""

    model = "scripted"

    def __init__(self, scripts, cap=4096, log=None):
        self.scripts = scripts
        self.cap = cap
        self.log = log if log is not None else []

    def output_token_cap(self):
        return self.cap

    def with_output_token_cap(self, tokens):
        return ScriptedBackend(self.scripts, tokens, self.log)

    def _next(self):
        self.log.append(dict(cap=self.cap, attempt=current_dispatch_attempt()))
        return self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]

    def dispatch_stream(self, request, on_client=None):
        for step in self._next():
            if isinstance(step, Raises):
                raise step.error
            yield step

    def dispatch(self, request):
        blocks, usage, stop_reason = {}, None, None
        for step in self._next():
            if isinstance(step, Raises):
                raise step.error
            if step["type"] == "usage":
                usage, stop_reason = step["usage"], step["stop_reason"]
                continue
            block = blocks.setdefault(
                step["index"], {"type": step["block_type"], "index": step["index"]}
            )
            block.update(
                {k: v for k, v in step.items() if k not in ("type", "index", "block_type")}
            )
        message = {"role": "assistant", "blocks": [blocks[i] for i in sorted(blocks)]}
        return {"message": message, "usage": usage, "stop_reason": stop_reason}


class Logger:
    _default_name = "agent"

    def __init__(self):
        self.events, self.finalized = [], []

    def record_event(self, type, payload, call_label=None, **_kw):
        self.events.append((type, payload, call_label))

    def record_stream_delta(self, *args, **kwargs):
        pass

    def record_llm_exchange(
        self, call_label, *, exchange_type, message_store, response_message_hashes, **_
    ):
        response = [
            payload
            for message_hash in response_message_hashes
            for _hash, payload in message_store.payloads(message_hash)
        ]
        self.finalized.append((exchange_type, response))

    def retries(self):
        return [payload for kind, payload, _ in self.events if kind == "llm_retry"]


def server_with(scripts, **llm):
    llm.setdefault("unusable_response_backoff_s", 0)
    config = agconfig(llmconfig(base_url="http://x/v1", model="gpt-test", **llm))
    server = LlmHandlerServer(config, Logger(), usage_tracker=LlmUsageTracker())
    server._backend = ScriptedBackend(scripts)
    return server, server._backend


def run_stream(server):
    handle = server.start_stream(dict(REQUEST))
    item = handle.first()
    handle._thread.join(timeout=5.0)
    assert not handle._thread.is_alive()
    return item


def content_blocks(item):
    return [b for b in item["message"]["blocks"] if b["type"] != "metadata"]


def metadata(item):
    return next(b for b in item["message"]["blocks"] if b["type"] == "metadata")


# ------------------------------------------------------------ empty responses


def test_empty_200_is_asked_again_and_the_next_answer_is_delivered():
    server, backend = server_with([EMPTY_200, GOOD_TOOL])
    item = run_stream(server)
    assert item["type"] == "done" and item["stop_reason"] == "tool_calls"
    assert content_blocks(item)[0]["arguments"] == '{"cmd":"nproc"}'
    assert len(backend.log) == 2
    # The discarded ask is on the durable record, twice: as an event and on
    # the delivered exchange's own metadata.
    assert [r["reason"] for r in server._data_logger.retries()] == ["empty_response"]
    assert metadata(item)["retries"][0]["reason"] == "empty_response"
    assert server._data_logger.finalized[-1][0] == "llm_block"


def test_empty_200_that_keeps_coming_is_a_reported_failure_not_a_finished_turn():
    server, backend = server_with([EMPTY_200])
    item = run_stream(server)
    assert item["type"] == "error"
    assert item["status_code"] == 502 and item["transient"] is True
    assert "no text and no tool call" in item["message"]
    assert "reason=empty_response, attempts=3" in item["message"]
    assert len(backend.log) == 3  # the first ask plus unusable_response_retries=2
    kind, chain = server._data_logger.finalized[-1]
    assert kind == "llm_stream_error" and "empty_response" in chain[0]["error"]


def test_reasoning_with_no_answer_is_empty():
    server, backend = server_with([[reasoning(), *end("stop")], GOOD_TEXT])
    assert run_stream(server)["type"] == "done"
    assert server._data_logger.retries()[0]["reason"] == "empty_response"


def test_empty_text_beside_a_valid_tool_call_is_an_ordinary_turn():
    server, backend = server_with(
        [[text("", index=0), tool('{"cmd":"ls"}', index=1), *end("tool_calls")]]
    )
    item = run_stream(server)
    assert item["type"] == "done" and len(backend.log) == 1
    assert server._data_logger.retries() == []


def test_retry_budget_is_configurable_and_zero_means_report_at_once():
    server, backend = server_with([EMPTY_200], unusable_response_retries=0)
    item = run_stream(server)
    assert item["type"] == "error" and "attempts=1" in item["message"]
    assert len(backend.log) == 1


def test_empty_response_passes_through_only_when_configured_to():
    server, backend = server_with([EMPTY_200], fail_on_empty_response=False)
    item = run_stream(server)
    assert item["type"] == "done" and content_blocks(item) == []
    assert len(backend.log) == 1


def test_retries_know_which_attempt_they_are_and_why():
    server, backend = server_with([EMPTY_200, EMPTY_200, GOOD_TEXT])
    assert run_stream(server)["type"] == "done"
    attempts = [entry["attempt"] for entry in backend.log]
    assert [a["attempt"] for a in attempts] == [0, 1, 2]
    assert [a["retry_reason"] for a in attempts] == [None, "empty_response", "empty_response"]
    assert len({a["call_label"] for a in attempts}) == 1


# ------------------------------------------------------------------ truncation


def test_truncated_tool_call_is_never_delivered_as_a_runnable_call():
    server, backend = server_with([TRUNCATED_TOOL])  # truncation_retries defaults to 0
    item = run_stream(server)
    assert item["type"] == "done" and item["stop_reason"] == "length"
    block = content_blocks(item)[0]
    assert block["incomplete"] is True
    assert block["id"] == "call_1" and block["arguments"] == CUT
    assert len(backend.log) == 1


def test_truncation_is_asked_again_with_a_larger_output_limit():
    server, backend = server_with(
        [TRUNCATED_TOOL, GOOD_TOOL], truncation_retries=1, truncation_max_output_tokens=32768
    )
    item = run_stream(server)
    assert item["type"] == "done" and item["stop_reason"] == "tool_calls"
    assert "incomplete" not in content_blocks(item)[0]
    assert [entry["cap"] for entry in backend.log] == [4096, 8192]
    retry = server._data_logger.retries()[0]
    assert retry["reason"] == "truncated_output"
    assert (retry["output_token_cap"], retry["next_output_token_cap"]) == (4096, 8192)
    assert retry["usage"]["completion_tokens"] == 4096


def test_truncation_escalation_is_bounded_by_count_and_by_the_ceiling():
    server, backend = server_with(
        [TRUNCATED_TOOL], truncation_retries=5, truncation_max_output_tokens=10000
    )
    item = run_stream(server)
    # 4096 -> 8192 -> 10000 (ceiling) -> no room left: delivered, flagged.
    assert [entry["cap"] for entry in backend.log] == [4096, 8192, 10000]
    assert item["type"] == "done" and item["stop_reason"] == "length"
    assert content_blocks(item)[0]["incomplete"] is True


def test_truncation_and_empty_retries_have_separate_budgets():
    server, backend = server_with(
        [TRUNCATED_TOOL, EMPTY_200, EMPTY_200, GOOD_TOOL],
        truncation_retries=1,
        unusable_response_retries=2,
    )
    assert run_stream(server)["type"] == "done"
    assert [r["reason"] for r in server._data_logger.retries()] == [
        "truncated_output",
        "empty_response",
        "empty_response",
    ]
    # The raised limit is kept for the asks that follow.
    assert [entry["cap"] for entry in backend.log] == [4096, 8192, 8192, 8192]


def test_malformed_tool_arguments_without_truncation_are_asked_again():
    malformed = [tool(CUT), *end("tool_calls")]
    server, backend = server_with([malformed, GOOD_TOOL])
    assert run_stream(server)["type"] == "done"
    assert server._data_logger.retries()[0]["reason"] == "malformed_tool_call"

    server, backend = server_with([malformed])
    item = run_stream(server)
    assert item["type"] == "error" and "reason=malformed_tool_call" in item["message"]


# ------------------------------------------------------- interrupted streams


def test_stream_that_breaks_after_it_started_is_asked_again():
    broken = [text("partial "), Raises(ConnectionError("peer closed connection"))]
    server, backend = server_with([broken, GOOD_TEXT])
    item = run_stream(server)
    assert item["type"] == "done" and content_blocks(item)[0]["text"] == "done"
    retry = server._data_logger.retries()[0]
    assert retry["reason"] == "stream_interrupted"
    assert "ConnectionError: peer closed connection" in retry["detail"]


def test_stream_with_no_terminal_event_is_interrupted_not_complete():
    cut = [tool(CUT), Raises(StreamInterruptedError("stream ended without a terminal event"))]
    server, backend = server_with([cut])
    item = run_stream(server)
    assert item["type"] == "error" and item["transient"] is True and item["status_code"] == 502
    assert "reason=stream_interrupted, attempts=3" in item["message"]


def test_malformed_streaming_event_is_asked_again_and_keeps_its_cause():
    garbled = [text("x"), Raises(ValueError("Expecting value: line 1 column 1 (char 0)"))]
    server, backend = server_with([garbled])
    item = run_stream(server)
    assert item["type"] == "error" and "ValueError: Expecting value" in item["message"]
    assert len(backend.log) == 3


def test_provider_error_inside_a_200_is_retried_only_when_asking_again_can_help():
    server, backend = server_with(
        [[Raises(ProviderResponseError("upstream overloaded", code="server_error"))], GOOD_TEXT]
    )
    assert run_stream(server)["type"] == "done" and len(backend.log) == 2

    server, backend = server_with(
        [[Raises(ProviderResponseError("invalid encrypted reasoning", code="validation_error"))]]
    )
    item = run_stream(server)
    assert (item["type"], item["status_code"], item["transient"]) == ("error", 400, False)
    assert len(backend.log) == 1


def test_rate_limit_and_bad_request_mid_stream_are_never_retried_in_place(monkeypatch):
    class RateLimited(Exception):
        pass

    class BadRequest(Exception):
        pass

    monkeypatch.setattr(mod, "RATE_LIMIT_EXCS", (RateLimited,))
    monkeypatch.setattr(mod, "BAD_REQUEST_EXCS", (BadRequest,))
    monkeypatch.setattr(mod, "TRANSIENT_DISPATCH_EXCS", (RateLimited,))

    server, backend = server_with([[text("x"), Raises(RateLimited("429"))]])
    item = run_stream(server)
    assert (item["type"], item["status_code"], item["transient"]) == ("error", 503, True)
    assert len(backend.log) == 1

    server, backend = server_with([[text("x"), Raises(BadRequest("nope"))]])
    item = run_stream(server)
    assert (item["type"], item["status_code"], item["transient"]) == ("error", 400, False)
    assert len(backend.log) == 1

    quota = ProviderResponseError("slow down", code="rate_limit_exceeded")
    server, backend = server_with([[Raises(quota)]])
    item = run_stream(server)
    assert (item["status_code"], item["transient"]) == (429, True) and len(backend.log) == 1


def test_failure_before_any_response_is_left_to_the_sdk_and_the_harness(monkeypatch):
    monkeypatch.setattr(mod, "TRANSIENT_DISPATCH_EXCS", (ConnectionError,))
    server, backend = server_with([[Raises(ConnectionError("connect failed"))]])
    item = run_stream(server)
    assert item == {
        "type": "error",
        "message": "connect failed",
        "transient": True,
        "status_code": 503,
    }
    assert len(backend.log) == 1


def test_upstream_call_ids_are_kept_in_the_reported_failure():
    ids = {
        "type": "block_delta",
        "index": METADATA_INDEX,
        "block_type": "metadata",
        "data": {"response_ids": {"x-litellm-call-id": "call-7", "x-request-id": "req-9"}},
    }
    server, backend = server_with([[ids, *EMPTY_200]], unusable_response_retries=0)
    item = run_stream(server)
    assert "upstream=x-litellm-call-id=call-7,x-request-id=req-9" in item["message"]


# ---------------------------------------------------------------- cancellation


def test_cancellation_during_retry_backoff_stops_asking():
    server, backend = server_with([EMPTY_200], unusable_response_backoff_s=30)
    handle = server.start_stream(dict(REQUEST))
    deadline = threading.Event()
    for _ in range(200):
        if server._data_logger.retries():
            break
        deadline.wait(0.01)
    assert server._data_logger.retries(), "the first empty response was never seen"
    handle.cancel()
    handle._thread.join(timeout=5.0)
    assert not handle._thread.is_alive()
    assert len(backend.log) == 1
    assert server._data_logger.finalized[-1][0] == "llm_stream_cancelled"


def test_cancelled_request_is_not_retried():
    cancelled = threading.Event()
    log = []

    class Backend(ScriptedBackend):
        def dispatch_stream(self, request, on_client=None):
            self.log.append(1)
            cancelled.set()
            handle.cancel()
            raise ConnectionError("stream closed by cancel")
            yield

    server, _ = server_with([EMPTY_200])
    server._backend = Backend([EMPTY_200], log=log)
    handle = server._new_stream_handle()
    server.start_stream(dict(REQUEST), _handle=handle)
    handle._thread.join(timeout=5.0)
    assert cancelled.is_set() and len(log) == 1
    assert server._data_logger.finalized[-1][0] == "llm_stream_cancelled"


# ---------------------------------------------------------------- non-streaming


def test_non_streaming_empty_response_is_retried_then_raised():
    server, backend = server_with([EMPTY_200, GOOD_TEXT])
    result = server.dispatch(dict(REQUEST))
    assert result["stop_reason"] == "stop" and len(backend.log) == 2

    server, backend = server_with([EMPTY_200])
    with pytest.raises(_DispatchError) as raised:
        server.dispatch(dict(REQUEST))
    assert raised.value.status_code == 502 and raised.value.transient is True
    assert "reason=empty_response, attempts=3" in str(raised.value)
    assert server._data_logger.finalized[-1][0] == "llm_stream_error"


def test_non_streaming_truncation_escalates_then_flags():
    server, backend = server_with([TRUNCATED_TOOL, GOOD_TOOL], truncation_retries=1)
    assert server.dispatch(dict(REQUEST))["stop_reason"] == "tool_calls"
    assert [entry["cap"] for entry in backend.log] == [4096, 8192]

    server, backend = server_with([TRUNCATED_TOOL])
    result = server.dispatch(dict(REQUEST))
    assert result["stop_reason"] == "length"
    assert result["message"]["blocks"][0]["incomplete"] is True


# ------------------------------------------- the recorded SDK-level shapes


def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None, choices=True):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None)
    delta.model_dump = lambda: {"content": content}
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    chunk = SimpleNamespace(choices=[choice] if choices else [], usage=usage)
    chunk.model_dump = lambda: {"finish_reason": finish_reason}
    return chunk


def _call(index, call_id, name, arguments):
    return SimpleNamespace(
        index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def chat_server(streams, monkeypatch, **llm):
    """A real Chat Completions backend over scripted SDK chunks."""
    asked = []

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            asked.append(kwargs)
            return iter(streams.pop(0) if len(streams) > 1 else streams[0])

        def close(self):
            pass

    monkeypatch.setattr(_OpenAICompatibleBackend, "make_client", lambda self, timeout: Client())
    llm.setdefault("unusable_response_backoff_s", 0)
    config = agconfig(llmconfig(base_url="http://x/v1", model="gpt-6-luna", **llm))
    return LlmHandlerServer(config, Logger(), usage_tracker=LlmUsageTracker()), asked


def test_recorded_luna_empty_chunk_no_longer_completes_a_turn(monkeypatch):
    """The exact chunk sequence of the 41 recorded empty exchanges."""
    empty = [_chunk(finish_reason="stop")]
    good = [_chunk(content="done"), _chunk(finish_reason="stop")]
    server, asked = chat_server([empty, good], monkeypatch)
    item = run_stream(server)
    assert item["type"] == "done" and len(asked) == 2

    server, asked = chat_server([empty], monkeypatch)
    item = run_stream(server)
    assert item["type"] == "error" and "reason=empty_response" in item["message"]


def test_recorded_truncated_tool_call_escalates_the_real_request_limit(monkeypatch):
    """finish_reason "length" at exactly the 4,096-token cap, inside the
    arguments of an exec_command call."""
    cut = [
        _chunk(tool_calls=[_call(0, "call_9", "exec_command", "")]),
        _chunk(tool_calls=[_call(0, None, None, CUT)]),
        _chunk(finish_reason="length"),
    ]
    whole = [
        _chunk(tool_calls=[_call(0, "call_10", "exec_command", '{"cmd":"ls"}')]),
        _chunk(finish_reason="tool_calls"),
    ]
    server, asked = chat_server(
        [cut, whole], monkeypatch, max_completion_tokens=4096, truncation_retries=1
    )
    item = run_stream(server)
    assert item["type"] == "done" and item["stop_reason"] == "tool_calls"
    assert [kwargs["max_completion_tokens"] for kwargs in asked] == [4096, 8192]
    assert content_blocks(item)[0]["id"] == "call_10"

    server, asked = chat_server([cut], monkeypatch, max_completion_tokens=4096)
    item = run_stream(server)
    assert item["stop_reason"] == "length" and content_blocks(item)[0]["incomplete"] is True
    assert content_blocks(item)[0]["id"] == "call_9"


def test_chat_stream_that_closes_without_a_finish_reason_is_interrupted(monkeypatch):
    cut_off = [
        _chunk(tool_calls=[_call(0, "call_9", "exec_command", "")]),
        _chunk(tool_calls=[_call(0, None, None, '{"cmd":"ls')]),
    ]
    server, asked = chat_server([cut_off], monkeypatch, unusable_response_retries=1)
    item = run_stream(server)
    assert item["type"] == "error" and item["transient"] is True
    assert "without a finish_reason" in item["message"]
    assert "reason=stream_interrupted, attempts=2" in item["message"]
    assert len(asked) == 2


def test_unusable_response_error_is_an_ordinary_runtime_error():
    error = UnusableResponseError("empty_response", "nothing came back", details={"a": 1})
    assert isinstance(error, RuntimeError)
    assert (error.retryable, error.transient, error.status_code) == (True, True, 502)
    assert error.details == {"a": 1}
