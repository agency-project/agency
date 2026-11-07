"""Native wire capture with real session HTTP servers and a local HTTP double."""

import asyncio
import copy
import gzip
import json
import os
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pyte
import pytest

from agency.llm.message_store import split_tool_schema
from agency.observability.agdatalogger import read_llm_exchanges, reconstruct_llm_messages
from agency.sessions import capture as module
from agency.sessions.capture import CaptureConfig
from agency.sessions.codex import CodexSession
from agency.sessions import session as session_module
from agency.sessions.claude_code import ClaudeCodeSession

from .helpers import FIXTURES, actions, eventually, otlp, trajectory


def sse(*events):
    return b"".join(
        b"data: " + json.dumps(e, ensure_ascii=False).encode() + b"\r\n\r\n" for e in events
    )


class Stream(httpx.AsyncByteStream):
    def __init__(self, data, *, hang=False, fail=False):
        self.data, self.hang, self.fail = data, hang, fail
        self.closed = threading.Event()

    async def __aiter__(self):
        # Split through UTF-8 codepoints and SSE delimiters.
        for offset in range(0, len(self.data), 7):
            yield self.data[offset : offset + 7]
        if self.fail:
            raise httpx.ReadError("fixture transport interrupted")
        if self.hang:
            await asyncio.sleep(3600)

    async def aclose(self):
        self.closed.set()


@pytest.fixture
def captured(make_session, monkeypatch):
    monkeypatch.setenv("LITE_LLM_KEY", "capture-fixture-upstream-key")
    original = httpx.AsyncClient
    state = SimpleNamespace(requests=[], handler=None)

    async def handle(request):
        state.requests.append(request)
        result = state.handler(request)
        result = await result if asyncio.iscoroutine(result) else result
        if result.is_stream_consumed:
            result = httpx.Response(
                result.status_code, headers=result.headers, stream=Stream(result.content)
            )
        return result

    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )

    def make(protocol="messages", handler=None):
        state.handler = handler
        config = CaptureConfig(
            "gpt-6-luna" if protocol == "responses" else "claude-haiku-5-5",
            "https://capture-upstream.invalid",
        )
        kwargs = {"capture": config}
        if protocol == "responses":
            kwargs.update(session_class=CodexSession, binary=FIXTURES / "fake_codex_session.py")
        session = make_session(**kwargs)
        state.session = session
        return state

    return make


def exchange(session):
    def check():
        rows = read_llm_exchanges(session._agent_logger.db_path)
        assert rows
        return rows[-1]

    return eventually(check)


def archive(session):
    def check():
        records = [
            json.loads(p.read_text()) for p in (session.run_dir / "capture").glob("*/manifest.json")
        ]
        assert records and "ended_at" in records[-1]
        return records[-1]

    return eventually(check)


@pytest.mark.parametrize("path", ["responses", "responses/compact"])
def test_provider_rejecting_server_history_or_compaction_is_not_retried_or_rewritten(
    captured, path
):
    state = captured(
        "responses",
        lambda req: httpx.Response(
            400,
            json={
                "error": {
                    "message": "fixture: opaque compaction or previous_response_id not supported",
                    "code": "unsupported_state",
                }
            },
        ),
    )
    body = {
        "model": "gpt-6-luna",
        "previous_response_id": "missing-parent",
        "input": [{"type": "compaction", "encrypted_content": "opaque-fixture"}],
    }
    response = request(state.session, path, body)
    assert response.status_code == 400
    assert len(state.requests) == 1
    assert json.loads(state.requests[0].content) == body
    row = exchange(state.session)
    assert row["type"] == "llm_stream_error"
    metadata = next(b for b in row["response"] if b["type"] == "metadata")
    assert not metadata["capture"]["complete"]
    assert metadata["capture"]["compaction"]
    assert metadata["data"]["response"] == response.json()


def test_invalid_json_is_archived_without_contacting_upstream(captured):
    state = captured("responses", lambda req: pytest.fail("invalid JSON reached upstream"))
    response = httpx.post(
        f"http://127.0.0.1:{state.session.port}/capture/v1/responses",
        content=b'{"model":',
        headers={"authorization": "Bearer " + state.session.host_token},
    )
    assert response.status_code == 400
    manifest = archive(state.session)
    assert manifest["error"] == "invalid_native_request"
    assert (
        state.session.run_dir / "capture" / manifest["call_label"] / "request.bin"
    ).read_bytes() == b'{"model":'
    assert exchange(state.session)["type"] == "llm_stream_error"


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_provider_rejection_of_malformed_fields_remains_reconstructible(captured, protocol):
    result = {"error": {"message": "invalid role and tools", "future": {"retained": True}}}
    state = captured(protocol, lambda req: httpx.Response(400, json=result))
    native_message = {"role": None, "content": "malformed native request"}
    body = {
        "model": state.session.capture.model,
        "tools": 42,
        "messages" if protocol == "messages" else "input": [native_message],
    }
    response = request(state.session, protocol, body)
    assert response.status_code == 400
    assert json.loads(state.requests[0].content) == body
    row = exchange(state.session)
    assert row["type"] == "llm_stream_error"
    metadata = next(b for b in row["response"] if b["type"] == "metadata")
    assert metadata["data"]["request"] == body
    assert metadata["data"]["response"] == result
    with sqlite3.connect(state.session._agent_logger.db_path) as connection:
        messages = reconstruct_llm_messages(connection, row["call_label"])
    chat, tools, _, extra = split_tool_schema(messages)
    assert tools is None and extra["tools"] == 42
    assert chat[0]["role"] == "unknown" and chat[0]["capture_native_role"] is None


def request(session, protocol, body, **kwargs):
    return httpx.post(
        f"http://127.0.0.1:{session.port}/capture/v1/{protocol}",
        json=body,
        headers={"Authorization": "Bearer " + session.host_token},
        **kwargs,
    )


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_native_bodies_streams_headers_unknowns_and_sqlite_are_preserved(captured, protocol):
    opaque = "opaque-provider-state"
    if protocol == "responses":
        output = [
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "héllo 🌍"}],
            },
            {"type": "reasoning", "encrypted_content": opaque, "summary": [], "future": 17},
            {"type": "function_call", "call_id": "call-1", "name": "run", "arguments": '{"x":1}'},
        ]
        result = {
            "id": "resp-1",
            "model": "gpt-6-luna",
            "output": output,
            "usage": {"input_tokens": 23, "output_tokens": 5},
            "unknown": {"retained": True},
        }
        wire = sse(
            {"type": "response.created", "response": {"id": "resp-1"}},
            {"type": "future.event", "opaque": opaque},
            {"type": "response.completed", "response": result},
        )
        body = {
            "model": "gpt-6-luna",
            "stream": True,
            "instructions": "system instructions",
            "input": [
                {"role": "user", "content": "hello", "native_unknown": {"retained": True}},
                {"type": "reasoning", "encrypted_content": opaque, "summary": []},
                {
                    "type": "additional_tools",
                    "role": "developer",
                    "tools": [{"type": "namespace", "name": "native", "tools": []}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "run",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
        }
    else:
        blocks = [
            {"type": "text", "text": "héllo 🌍"},
            {"type": "thinking", "thinking": "", "signature": opaque},
            {"type": "tool_use", "id": "call-1", "name": "run", "input": {"x": 1}},
        ]
        result = {
            "id": "msg-1",
            "model": "claude-haiku-5-5",
            "content": blocks,
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 13},
            "unknown": {"retained": True},
        }
        wire = sse(
            {"type": "message_start", "message": {**result, "content": []}},
            *[
                {"type": "content_block_start", "index": i, "content_block": block}
                for i, block in enumerate(blocks)
            ],
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        )
        body = {
            "model": "claude-haiku-5-5",
            "stream": True,
            "system": [
                {
                    "type": "text",
                    "text": "system instructions",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": "hello", "native_unknown": {"retained": True}},
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {"name": "run", "input_schema": {"type": "object"}, "future_schema_field": 99}
            ],
        }
    body.update(unknown={"nested": [1, opaque]}, tool_choice={"type": "auto"})
    original_body = copy.deepcopy(body)
    stream = Stream(wire)
    state = captured(
        protocol,
        lambda req: httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "upstream-1",
                "x-future-header": "preserved",
            },
            stream=stream,
        ),
    )
    session = state.session
    raw = json.dumps(body, indent=2, ensure_ascii=False).encode()
    response = httpx.post(
        f"http://127.0.0.1:{session.port}/capture/v1/{protocol}?beta=true",
        content=raw,
        headers={
            "authorization": "Bearer " + session.host_token,
            "content-type": "application/json",
            "anthropic-beta": "fixture-beta",
            "x-future-request": "preserved",
        },
    )
    assert response.status_code == 200
    assert response.content == wire
    assert response.headers["x-future-header"] == "preserved"
    [forwarded] = state.requests
    assert forwarded.content == raw
    assert forwarded.url.path == f"/v1/{protocol}"
    assert forwarded.url.query == b"beta=true"
    assert forwarded.headers["authorization"] == "Bearer capture-fixture-upstream-key"
    assert forwarded.headers["anthropic-beta"] == "fixture-beta"
    assert forwarded.headers["x-future-request"] == "preserved"
    row = exchange(session)
    assert row["type"] == "llm_block"
    metadata = next(b for b in row["response"] if b["type"] == "metadata")
    assert metadata["data"]["request"] == original_body
    assert metadata["usage"]["prompt_tokens"] == 23
    assert metadata["usage"]["completion_tokens"] == 5
    with sqlite3.connect(session._agent_logger.db_path) as connection:
        prompt_messages = reconstruct_llm_messages(connection, row["call_label"])
        response_messages = reconstruct_llm_messages(connection, row["call_label"], "response")
        tables = {
            r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"message_block_lookup", "message_structure_lookup", "per_exchange_messages"} <= tables
    assert prompt_messages == module._prompt(original_body, protocol)
    chat, tools, tool_choice, extra = split_tool_schema(prompt_messages)
    assert tools == body["tools"]
    assert tool_choice == body["tool_choice"]
    assert extra["unknown"] == body["unknown"]
    user_messages = [m for m in chat if m["role"] == "user"]
    assert user_messages[0]["native_unknown"] == {"retained": True}
    if protocol == "messages":
        assert len(user_messages) == 2
    reconstructed_metadata = next(
        b for m in response_messages for b in m["blocks"] if b["type"] == "metadata"
    )
    assert reconstructed_metadata["data"]["request"] == original_body
    expected_result = {**result, **({"stop_reason": "tool_use"} if protocol == "messages" else {})}
    assert reconstructed_metadata["data"]["response"] == expected_result
    if protocol == "responses":
        assert response_messages[0]["phase"] == "commentary"
    if protocol == "responses":
        assert any(
            b["role"] == "developer" and b["type"] == "openai_responses_additional_tools"
            for b in row["prompt"]
        )
    assert any(b.get("type") == "tool_use" and b["id"] == "call-1" for b in row["response"])
    assert opaque in json.dumps(row)
    manifest = archive(session)
    directory = session.run_dir / "capture" / manifest["call_label"]
    assert (directory / "request.bin").read_bytes() == raw
    assert (directory / "response.bin").read_bytes() == wire
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(session.run_dir.stat().st_mode) == 0o700
    for p in directory.iterdir():
        assert stat.S_IMODE(p.stat().st_mode) == 0o600
        assert b"capture-fixture-upstream-key" not in p.read_bytes()
        assert session.host_token.encode() not in p.read_bytes()
    assert stream.closed.is_set()
    model = trajectory(session)
    assert model.agents[session.agent_name]["execution_state"] == "agent_idle"
    [call] = actions(model, "model")
    assert call["context"]["available"] and call["context"]["complete"]
    assert "hello" in json.dumps(call["context"])
    # OTel tool/lifecycle telemetry still runs; model telemetry cannot duplicate.
    if protocol == "messages":
        otlp(session, "api_request", manifest["ended_at"], model=body["model"], duration_ms=100)
    else:
        otlp(
            session,
            "codex.api_request",
            manifest["ended_at"],
            **{"conversation.id": "c1", "http.response.status_code": 200},
        )
        otlp(
            session,
            "codex.sse_event",
            manifest["ended_at"],
            **{"conversation.id": "c1", "event.kind": "response.completed"},
        )
    assert len(read_llm_exchanges(session._agent_logger.db_path)) == 1


@pytest.mark.parametrize("status", [400, 429, 500])
def test_provider_error_status_headers_body_and_no_retry(captured, status):
    wire = b'{"error":{"message":"provider rejected","unknown":true}}'
    state = captured(
        "responses",
        lambda req: httpx.Response(
            status,
            headers={"content-type": "application/json", "retry-after": "5"},
            stream=Stream(wire),
        ),
    )
    response = request(state.session, "responses", {"model": "gpt-6-luna", "input": "x"})
    assert response.status_code == status and response.content == wire
    assert response.headers["retry-after"] == "5"
    assert exchange(state.session)["type"] == "llm_stream_error"
    assert len(state.requests) == 1


def test_previous_response_id_and_compaction_are_forwarded_without_inventing_context(captured):
    state = captured(
        "responses",
        lambda req: httpx.Response(
            200,
            json={
                "id": "compaction-1",
                "object": "response.compaction",
                "output": [
                    {
                        "type": "compaction",
                        "encrypted_content": "opaque-compaction",
                        "unknown": True,
                    }
                ],
            },
        ),
    )
    body = {
        "model": "gpt-6-luna",
        "previous_response_id": "not-locally-recorded",
        "input": [{"role": "user", "content": "delta only"}],
    }
    assert request(state.session, "responses", body).status_code == 200
    row = exchange(state.session)
    meta = next(b for b in row["response"] if b["type"] == "metadata")
    assert meta["data"]["request"] == body
    assert meta["capture"]["server_state"]
    assert not trajectory(state.session).actions[next(iter(trajectory(state.session).actions))][
        "context"
    ]["complete"]
    compact = {
        "model": "gpt-6-luna",
        "input": [{"type": "compaction", "encrypted_content": "opaque-compaction"}],
    }
    response = request(state.session, "responses/compact", compact)
    assert response.json()["output"][0]["encrypted_content"] == "opaque-compaction"
    assert state.requests[-1].content == json.dumps(compact, separators=(",", ":")).encode()

    def compacted():
        rows = read_llm_exchanges(state.session._agent_logger.db_path)
        assert len(rows) == 2
        return next(b for b in rows[-1]["response"] if b["type"] == "metadata")

    assert eventually(compacted)["capture"]["compaction"]
    assert not eventually(compacted)["capture"]["context_editing"]


def test_silent_upstream_is_cancelled_and_partial_capture_is_reconstructible(captured):
    wire = sse(
        {
            "type": "response.output_text.delta",
            "delta": "partial reply",
            "output_index": 0,
            "content_index": 0,
        }
    )
    stream = Stream(wire, hang=True)
    state = captured(
        "responses",
        lambda req: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        ),
    )
    session = state.session
    with httpx.stream(
        "POST",
        f"http://127.0.0.1:{session.port}/capture/v1/responses",
        json={"model": "gpt-6-luna", "input": "cancel me", "stream": True},
        headers={"authorization": "Bearer " + session.host_token},
    ) as response:
        assert next(response.iter_raw())
    assert stream.closed.wait(3), "upstream must close without waiting for its next token"
    row = exchange(session)
    assert row["type"] == "llm_stream_cancelled"
    assert archive(session)["cancelled"]
    assert any(b.get("text") == "partial reply" for b in row["response"])


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_cli_disconnect_after_terminal_event_is_a_completed_model_call(captured, protocol):
    if protocol == "responses":
        wire = sse(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp-fixture",
                    "status": "completed",
                    "output": [],
                },
            }
        )
    else:
        wire = sse(
            {"type": "message_start", "message": {"id": "msg-fixture", "content": []}},
            {"type": "message_stop"},
        )
    stream = Stream(wire, hang=True)
    state = captured(
        protocol,
        lambda req: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        ),
    )
    session = state.session
    with httpx.stream(
        "POST",
        f"http://127.0.0.1:{session.port}/capture/v1/{protocol}",
        json={"model": session.capture.model, "stream": True},
        headers={"authorization": "Bearer " + session.host_token},
    ) as response:
        received = b""
        for chunk in response.iter_raw():
            received += chunk
            if len(received) == len(wire):
                break
    assert stream.closed.wait(3)
    assert exchange(session)["type"] == "llm_block"
    manifest = archive(session)
    assert manifest["client_disconnected"]
    assert not manifest["cancelled"]


def test_broken_stream_records_error_and_retains_received_bytes(captured):
    wire = sse({"type": "response.output_text.delta", "delta": "partial"})
    state = captured(
        "responses",
        lambda req: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Stream(wire, fail=True)
        ),
    )
    with pytest.raises(httpx.RemoteProtocolError):
        request(state.session, "responses", {"model": "gpt-6-luna", "stream": True, "input": "x"})
    assert exchange(state.session)["type"] == "llm_stream_error"
    assert archive(state.session)["error"] == "upstream_stream_error"


def test_authentication_model_mismatch_and_endpoint_boundaries(captured):
    state = captured("responses", lambda req: pytest.fail("must not reach upstream"))
    session = state.session
    url = f"http://127.0.0.1:{session.port}/capture/v1/responses"
    assert httpx.post(url, json={"model": "gpt-6-luna"}).status_code == 401
    assert (
        request(session, "responses", {"model": "forbidden-model", "input": "x"}).status_code == 400
    )
    assert request(session, "responses", {"input": "missing model"}).status_code == 400
    assert request(session, "unknown", {}).status_code == 404
    assert not state.requests


def test_token_count_is_archived_without_a_duplicate_model_action(captured):
    state = captured("messages", lambda req: httpx.Response(200, json={"input_tokens": 9}))
    response = request(
        state.session,
        "messages/count_tokens",
        {"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "count"}]},
    )
    assert response.json() == {"input_tokens": 9}
    assert archive(state.session)["path"] == "v1/messages/count_tokens"
    assert read_llm_exchanges(state.session._agent_logger.db_path) == []


def test_compressed_requests_and_responses_preserve_original_bytes(captured):
    wire = gzip.compress(
        json.dumps(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "compressed"}],
                    }
                ]
            }
        ).encode()
    )
    state = captured(
        "responses",
        lambda req: httpx.Response(
            200,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
            stream=Stream(wire),
        ),
    )
    raw = gzip.compress(b'{"model":"gpt-6-luna","input":"compressed input"}')
    session = state.session
    response = httpx.post(
        f"http://127.0.0.1:{session.port}/capture/v1/responses",
        content=raw,
        headers={"authorization": "Bearer " + session.host_token, "content-encoding": "gzip"},
    )
    assert "compressed" in response.text
    assert state.requests[0].content == raw
    assert exchange(session)["type"] == "llm_block"
    directory = session.run_dir / "capture" / archive(session)["call_label"]
    assert (directory / "response.bin").read_bytes() == wire


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_child_configuration_routes_locally_without_upstream_key(captured, protocol):
    state = captured(protocol, lambda req: pytest.fail("no model call expected"))
    session = state.session
    env, argv = session._environment(), session._cli_argv()
    assert "LITE_LLM_KEY" not in env
    assert "capture-fixture-upstream-key" not in " ".join(argv)
    if protocol == "messages":
        assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{session.port}/capture"
        assert env["ANTHROPIC_AUTH_TOKEN"] == session.host_token
        assert all(
            env[k] == session.capture.model
            for k in (
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
            )
        )
    else:
        assert any(
            "agency_capture.base_url" in arg and f":{session.port}/capture/v1" in arg
            for arg in argv
        )
        assert 'model_provider="agency_capture"' in argv
        assert "model_providers.agency_capture.supports_websockets=false" in argv


def test_capture_requires_explicit_key_and_rejects_credential_urls(monkeypatch):
    monkeypatch.delenv("LITE_LLM_KEY", raising=False)
    with pytest.raises(ValueError, match="requires LITE_LLM_KEY"):
        module.CaptureGateway(None, CaptureConfig("gpt-6-luna"))
    with pytest.raises(ValueError, match="without credentials"):
        CaptureConfig("gpt-6-luna", "https://user:secret@example.com")


@pytest.mark.parametrize("enabled", [False, True])
def test_codex_conversation_start_registers_its_actor(make_session, monkeypatch, enabled):
    monkeypatch.setenv("LITE_LLM_KEY", "registration-fixture-key")
    session = make_session(
        session_class=CodexSession, capture=CaptureConfig("gpt-6-luna") if enabled else None
    )
    status, _ = otlp(session, "codex.conversation_starts", time.time(), model="gpt-6-luna")
    assert status == 200


def test_disconnect_before_response_headers_cancels_the_upstream(captured):
    closed = threading.Event()

    async def wait_forever(req):
        try:
            await asyncio.sleep(3600)
        finally:
            closed.set()

    state = captured("responses", wait_forever)
    with pytest.raises(httpx.ReadTimeout):
        request(
            state.session,
            "responses",
            {"model": "gpt-6-luna", "input": "cancel before headers"},
            timeout=0.2,
        )
    assert closed.wait(3)
    assert exchange(state.session)["type"] == "llm_stream_cancelled"


def test_connection_errors_are_recorded_without_retry_or_credentials(captured):
    def fail(req):
        raise httpx.ConnectError("fixture refused connection")

    state = captured("responses", fail)
    response = request(state.session, "responses", {"model": "gpt-6-luna", "input": "x"})
    assert response.status_code == 502
    assert exchange(state.session)["type"] == "llm_stream_error"
    assert len(state.requests) == 1
    assert "capture-fixture-upstream-key" not in response.text


@pytest.mark.parametrize(
    "session_class,command", [(ClaudeCodeSession, "claude"), (CodexSession, "codex")]
)
def test_scope_reexec_keeps_capture_options_and_credential_out_of_argv(
    tmp_path, monkeypatch, session_class, command
):
    monkeypatch.setenv("LITE_LLM_KEY", "scope-fixture-secret")
    monkeypatch.delenv(session_module._SCOPED_VARIABLE, raising=False)
    monkeypatch.setattr(session_module, "_user_scope_available", lambda: True)

    class Reexec(Exception):
        pass

    invocation = []

    def execvp(binary, args):
        invocation.extend(args)
        assert os.environ["LITE_LLM_KEY"] == "scope-fixture-secret"
        raise Reexec

    monkeypatch.setattr(session_module.os, "execvp", execvp)
    argv = [
        "--terminal",
        "browser",
        "--cwd",
        str(tmp_path),
        f"--{command}",
        sys.executable,
        "--capture",
        "--capture-model",
        "gpt-6-luna",
        "--",
        "--no-alt-screen",
    ]
    with pytest.raises(Reexec):
        session_module.main(session_class, command, argv)
    assert invocation[0] == "systemd-run"
    assert invocation[-len(argv) :] == argv
    assert "scope-fixture-secret" not in " ".join(invocation)


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not session_module._user_scope_available(),
    reason="needs systemd user scopes",
)
@pytest.mark.parametrize(
    "command,native", [("claude", False), ("codex", False), ("codex", True), ("claude", True)]
)
def test_real_cli_entry_reexec_preserves_capture_mode(tmp_path, command, native):
    from websockets.sync.client import connect
    from agency.sessions.terminal import _Screen

    # No upstream model request: real execvp, Agency scope and terminal scope.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ)
    env.update(
        LITE_LLM_KEY="scope-fixture-secret",
        AGENCY_RUNS_ROOT=str(tmp_path / "runs"),
        AGENCY_PROFILE="0",
    )
    env.pop(session_module._SCOPED_VARIABLE, None)
    if native:
        state = tmp_path / "codex-home"
        state.mkdir()
        env["CODEX_HOME"] = str(state)
        env.pop("OPENAI_BASE_URL", None)
        env.pop("LITE_LLM_KEY", None)
        if command == "claude":
            env["CLAUDE_CONFIG_DIR"] = str(state)
            for variable in (
                "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX",
                "CLAUDE_CODE_USE_FOUNDRY",
            ):
                env.pop(variable, None)
    log = tmp_path / "scope.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agency.cli",
                command,
                "--terminal",
                "browser",
                "--cwd",
                str(tmp_path),
                f"--{command}",
                str(FIXTURES / f"fake_{command}_session.py"),
                "--port",
                str(port),
                "--capture-native" if native else "--capture",
                "--no-linger",
            ],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 20
        while True:
            paths = list((tmp_path / "runs").glob("*/logs/terminal_session.json"))
            if paths:
                private = json.loads(paths[-1].read_text())
                break
            assert process.poll() is None, "scoped entry exited before session startup"
            assert time.monotonic() < deadline, "scoped entry failed to start"
            time.sleep(0.1)
        assert private["capture"]["model"] == (None if native else "gpt-6-luna")
        assert private["capture"]["authentication"] == ("native" if native else "litellm")
        with connect(
            f"ws://127.0.0.1:{private['host_port']}/terminal",
            additional_headers={"X-Agency-Token": private["host_token"]},
        ) as ws:
            ws.send(json.dumps({"cols": 100, "rows": 30}))
            screen = _Screen(100, 30)
            parser = pyte.ByteStream(screen)
            while f"fake {command} ready" not in "\n".join(screen.display):
                frame = ws.recv(timeout=10)
                if isinstance(frame, bytes):
                    parser.feed(frame)
            ws.send(b"q")
        assert process.wait(timeout=20) == 0
        assert b"scope-fixture-secret" not in log.read_bytes()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
