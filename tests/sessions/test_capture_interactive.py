"""Opt-in acceptance test using unmodified interactive CLI binaries and real PTYs.

AGENCY_CAPTURE_TEST_CLAUDE=/path/to/claude AGENCY_CAPTURE_TEST_CODEX=/path/to/codex
uv run pytest tests/sessions/test_capture_interactive.py -v

By default a local scripted upstream supplies native protocol fixtures. Add
AGENCY_CAPTURE_LIVE=1 and LITE_LLM_KEY for the same test against LiteLLM. No
alternative provider, credential, or model is used by the live mode.
"""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.codex import CodexAdapter
from agency.observability.agdatalogger import read_llm_exchanges
from agency.sessions.capture import CaptureConfig, NativeCaptureConfig
from agency.sessions.claude_code import ClaudeCodeSession
from agency.sessions.codex import CodexSession

from .helpers import TerminalClient, actions, eventually, trajectory


class BrowserTerminal(TerminalClient):
    """The terminal WebSocket used by xterm, including cursor-query replies."""

    def type(self, data):
        super().type(data.encode() if isinstance(data, str) else data)

    def pump(self, timeout=0.1):
        from websockets.exceptions import ConnectionClosed

        try:
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline:
                message = self.socket.recv(timeout=timeout)
                if isinstance(message, bytes):
                    self.stream.feed(message)
                    if b"\x1b[6n" in message:
                        self.type(f"\x1b[{self.screen.cursor.y + 1};{self.screen.cursor.x + 1}R")
                else:
                    self.status.append(json.loads(message))
        except (TimeoutError, ConnectionClosed):
            return


class NativeUpstream:
    """Scripted native Messages/Responses server. No model inference."""

    def __init__(self, protocol, model):
        self.requests = []
        self.cancelled = threading.Event()
        self.closing = threading.Event()
        adapter = CodexAdapter(agconfig())
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                # A native WebSocket-capable client can fall back to SSE after
                # a valid HTTP rejection; this fixture only implements SSE.
                self.protocol_version = "HTTP/1.1"
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(body)
                if self.path.endswith("count_tokens"):
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"input_tokens":1000}')
                    return
                if self.path.endswith("/compact"):
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "id": "compact-fixture",
                                "object": "response.compaction",
                                "output": [
                                    {
                                        "type": "compaction",
                                        "encrypted_content": "opaque-compaction-fixture",
                                    }
                                ],
                            }
                        ).encode()
                    )
                    return
                history = body.get("messages", body.get("input", []))
                latest = next(
                    (
                        m
                        for m in reversed(history)
                        if isinstance(m, dict) and m.get("role") == "user"
                    ),
                    {},
                )
                contents = latest.get("content", [])
                contents = [{"text": contents}] if isinstance(contents, str) else contents
                if not owner.cancelled.is_set() and any(
                    isinstance(b, dict)
                    and str(b.get("text", "")).startswith("CAPTURE_CANCEL_PROMPT")
                    for b in contents
                ):
                    partial = (
                        [
                            {
                                "type": "response.created",
                                "response": {"id": "cancel-fixture", "status": "in_progress"},
                            },
                            {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": {
                                    "id": "msg_cancel",
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                            },
                            {
                                "type": "response.output_text.delta",
                                "item_id": "msg_cancel",
                                "output_index": 0,
                                "content_index": 0,
                                "delta": "PARTIAL_CAPTURE",
                            },
                        ]
                        if protocol == "responses"
                        else [
                            {
                                "type": "message_start",
                                "message": {
                                    "id": "cancel-fixture",
                                    "role": "assistant",
                                    "model": model,
                                    "content": [],
                                    "type": "message",
                                    "usage": {"input_tokens": 1000, "output_tokens": 0},
                                },
                            },
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            },
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": "PARTIAL_CAPTURE"},
                            },
                        ]
                    )
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(
                        "".join(
                            f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in partial
                        ).encode()
                    )
                    self.wfile.flush()
                    self.connection.settimeout(0.2)
                    while not owner.closing.is_set():
                        try:
                            if not self.connection.recv(1):
                                owner.cancelled.set()
                                break
                        except socket.timeout:
                            pass
                        except ConnectionResetError:
                            owner.cancelled.set()
                            break
                    return
                text = json.dumps(history)
                title_request = "Generate a concise, single-line task title" in text
                turn = 2 if "CAPTURE_SECOND_PROMPT" in text else 1
                marker = f"CAPTURE_TOOL_{turn}"
                tools = body.get("tools", [])
                names = [t.get("name") for t in tools]
                code_tools = [
                    t
                    for i in history
                    if isinstance(i, dict) and i.get("type") == "additional_tools"
                    for ns in i.get("tools", [])
                    for t in ns.get("tools", [])
                ]
                code_mode = any(t.get("name") == "exec" for t in code_tools) and (
                    "Generate a concise, single-line task title" not in text
                )
                tool = next(
                    (n for n in ("Bash", "exec_command", "shell_command", "shell") if n in names),
                    None,
                )
                results = [
                    b
                    for m in history
                    if isinstance(m, dict)
                    for b in (m.get("content", []) if isinstance(m.get("content"), list) else [])
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                results += [
                    i
                    for i in history
                    if isinstance(i, dict)
                    and i.get("type") in {"function_call_output", "custom_tool_call_output"}
                ]
                ran = any(
                    marker in json.dumps(r.get("output", r.get("content", "")))
                    and not r.get("is_error")
                    for r in results
                )
                call_id = f"capture_call_{turn}"
                if code_mode:
                    tool = "exec"
                if tool and not ran:
                    arguments = {"cmd" if tool == "exec_command" else "command": f"printf {marker}"}
                    if tool == "shell":
                        arguments = {"command": ["sh", "-c", f"printf {marker}"]}
                    if tool == "exec":
                        arguments = {
                            "input": f'text(await tools.exec_command({{cmd:"printf {marker}"}}));'
                        }
                    block = {
                        "type": "tool_use",
                        "index": 0,
                        "id": call_id,
                        "name": tool,
                        "input": arguments,
                        "arguments": json.dumps(arguments),
                    }
                else:
                    block = {"type": "text", "index": 0, "text": f"CAPTURE_TURN_{turn}_DONE"}
                if title_request:
                    block = {"type": "text", "index": 0, "text": "Capture fixture title"}
                if len(owner.requests) > 20:
                    block = {"type": "text", "index": 0, "text": "SCRIPTED_UPSTREAM_TOOL_LOOP"}
                if protocol == "responses":
                    frames = "".join(
                        adapter._format_agency_stream_to_harness(
                            iter(
                                [
                                    {
                                        "type": "done",
                                        "message": {"role": "assistant", "blocks": [block]},
                                        "usage": {"prompt_tokens": 1000, "completion_tokens": 20},
                                        "stop_reason": "tool_use"
                                        if block["type"] == "tool_use"
                                        else "stop",
                                    }
                                ]
                            ),
                            model,
                            tool_routes={
                                "exec": {
                                    "tool": {"type": "custom", "name": "exec"},
                                    "namespace": "functions",
                                }
                            }
                            if code_mode
                            else None,
                        )
                    )
                else:
                    native = {k: v for k, v in block.items() if k not in {"index", "arguments"}}
                    delta = (
                        {"type": "input_json_delta", "partial_json": json.dumps(native["input"])}
                        if native["type"] == "tool_use"
                        else {"type": "text_delta", "text": native["text"]}
                    )
                    native = (
                        {**native, "input": {}}
                        if native["type"] == "tool_use"
                        else {"type": "text", "text": ""}
                    )
                    events = [
                        {
                            "type": "message_start",
                            "message": {
                                "id": f"msg_{len(owner.requests)}",
                                "type": "message",
                                "role": "assistant",
                                "model": model,
                                "content": [],
                                "usage": {"input_tokens": 1000, "output_tokens": 0},
                            },
                        },
                        {"type": "content_block_start", "index": 0, "content_block": native},
                        {"type": "content_block_delta", "index": 0, "delta": delta},
                        {"type": "content_block_stop", "index": 0},
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": "tool_use" if tool and not ran else "end_turn",
                                "stop_sequence": None,
                            },
                            "usage": {"output_tokens": 20},
                        },
                        {"type": "message_stop"},
                    ]
                    frames = "".join(
                        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
                    )
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(frames.encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.closing.set()
        if hasattr(self, "_websocket_server"):
            self._websocket_server.should_exit = True
            self._websocket_thread.join(5)
        self.server.shutdown()
        self.server.server_close()

    def enable_websocket(self):
        """Native WS fixture over the same scripted provider, with stored state.

        The CLI sends incremental input after previous_response_id. Hydration
        happens inside this fake provider, never inside Agency's capture path.
        """
        import anyio
        import httpx
        import uvicorn
        from fastapi import FastAPI, Request, WebSocket
        from fastapi.responses import Response
        from starlette.websockets import WebSocketDisconnect

        app, original_url = FastAPI(), self.url
        self.native_auth_seen = []

        @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
        async def http(request: Request, path: str):
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.request(
                    request.method, original_url + "/" + path, content=await request.body()
                )
                return Response(
                    response.content,
                    status_code=response.status_code,
                    headers={
                        "content-type": response.headers.get("content-type", "application/json")
                    },
                )

        @app.websocket("/{path:path}")
        async def websocket(ws: WebSocket, path: str):
            self.native_auth_seen.append(
                (
                    ws.headers.get("authorization") == "Bearer fixture-native-bearer",
                    ws.headers.get("chatgpt-account-id") == "fixture-account",
                    ws.headers.get("x-agency-token") is None,
                )
            )
            await ws.accept()
            history = []
            active = {"scope": None, "response_id": None}
            sender, receiver = anyio.create_memory_object_stream(1)
            async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
                async with anyio.create_task_group() as group:

                    async def receive():
                        try:
                            while True:
                                body = await ws.receive_json()
                                if body.get("type") == "response.interrupt":
                                    if active["scope"] is not None:
                                        active["scope"].cancel()
                                    await ws.send_json(
                                        {
                                            "type": "response.cancelled",
                                            "response_id": active["response_id"]
                                            or body.get("response_id"),
                                        }
                                    )
                                else:
                                    await sender.send(body)
                        except WebSocketDisconnect:
                            pass
                        finally:
                            group.cancel_scope.cancel()

                    async def generate():
                        nonlocal history
                        async with receiver:
                            async for body in receiver:
                                if body.get("generate") is False:
                                    await ws.send_json(
                                        {
                                            "type": "response.completed",
                                            "response": {
                                                "id": "fixture-prewarm",
                                                "status": "completed",
                                                "output": [],
                                            },
                                        }
                                    )
                                    continue
                                context = (
                                    [*history, *body.get("input", [])]
                                    if body.get("previous_response_id")
                                    else body.get("input", [])
                                )
                                request_body = {**body, "input": context}
                                with anyio.CancelScope() as scope:
                                    active["scope"] = scope
                                    async with client.stream(
                                        "POST", original_url + "/responses", json=request_body
                                    ) as response:
                                        async for line in response.aiter_lines():
                                            if line.startswith("data: ") and line != "data: [DONE]":
                                                event = json.loads(line[6:])
                                                if event.get("type") == "response.created":
                                                    active["response_id"] = event["response"]["id"]
                                                await ws.send_text(line[6:])
                                                if event.get("type") == "response.completed":
                                                    history = [
                                                        *context,
                                                        *event["response"].get("output", []),
                                                    ]
                                active["scope"] = None

                    group.start_soon(receive)
                    group.start_soon(generate)

        self._websocket_server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=2
            )
        )
        self._websocket_thread = threading.Thread(target=self._websocket_server.run, daemon=True)
        self._websocket_thread.start()
        deadline = time.monotonic() + 10
        while not self._websocket_server.started:
            assert time.monotonic() < deadline, "native WS fixture did not start"
            time.sleep(0.01)
        self.url = (
            f"http://127.0.0.1:{self._websocket_server.servers[0].sockets[0].getsockname()[1]}"
        )


@pytest.mark.parametrize("protocol", ["messages", "responses", "responses-native"])
def test_real_interactive_capture(make_session, monkeypatch, tmp_path, protocol):
    claude = protocol == "messages"
    native = protocol == "responses-native"
    protocol = "responses" if native else protocol
    binary = os.environ.get("AGENCY_CAPTURE_TEST_CLAUDE" if claude else "AGENCY_CAPTURE_TEST_CODEX")
    if not binary:
        pytest.skip("set AGENCY_CAPTURE_TEST_CLAUDE / AGENCY_CAPTURE_TEST_CODEX to opt in")
    live = os.environ.get("AGENCY_CAPTURE_LIVE") == "1"
    model = "claude-haiku-5-5" if claude and not live else "gpt-6-luna"
    if live and claude:
        model = os.environ.get("AGENCY_CAPTURE_TEST_CLAUDE_MODEL", model)
        assert model in {"gpt-6-luna", "claude-haiku-5-5"}
    upstream = None
    if live:
        assert os.environ.get("LITE_LLM_KEY"), "live test requires LITE_LLM_KEY"
        config = (
            NativeCaptureConfig(api_base_url="https://router.js-park.info/v1")
            if native
            else CaptureConfig(model)
        )
    else:
        monkeypatch.setenv("LITE_LLM_KEY", "scripted-upstream-fixture-key")
        upstream = NativeUpstream(protocol, model)
        if native:
            upstream.enable_websocket()
        config = (
            NativeCaptureConfig(api_base_url=upstream.url, chatgpt_base_url=upstream.url)
            if native
            else CaptureConfig(model, upstream.url)
        )
    if claude:
        state = tmp_path / "claude-state"
        state.mkdir()
        (state / ".claude.json").write_text(
            json.dumps(
                {
                    "hasCompletedOnboarding": True,
                    "theme": "dark",
                    "projects": {str(tmp_path): {"hasTrustDialogAccepted": True}},
                }
            )
        )
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(state))
        cli_args = ["--model", model, "--allowedTools", "Bash(printf *)"]
    else:
        cli_args = [
            "--no-alt-screen",
            "--sandbox",
            # A user network namespace hides host-root ownership of bwrap.
            # The offline native fixture runs only scripted printf commands.
            "danger-full-access" if native and not live else "workspace-write",
            "-a",
            "never",
            "--dangerously-bypass-hook-trust",
        ]
        if native:
            # Exercise Codex-owned API authentication without real subscription
            # credentials or account-discovery requests. Live mode is LiteLLM
            # only; the native subscription bearer path is covered offline.
            state = tmp_path / "codex-state"
            state.mkdir(mode=0o700)
            with os.fdopen(
                os.open(state / "auth.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
            ) as credential_file:
                json.dump({"OPENAI_API_KEY": os.environ["LITE_LLM_KEY"]}, credential_file)
            monkeypatch.setenv("CODEX_HOME", str(state))
            monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
            monkeypatch.setenv("OPENAI_API_KEY", os.environ["LITE_LLM_KEY"])
            cli_args += ["--model", model, "-c", 'cli_auth_credentials_store="file"']
    session = make_session(
        session_class=ClaudeCodeSession if claude else CodexSession,
        binary=binary,
        cli_args=cli_args,
        use_scope=True,
        capture=config,
    )
    viewer = BrowserTerminal(session, columns=140, rows=50)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            viewer.pump()
            screen = viewer.text()
            if "trust this folder" in screen.lower() or "trust the contents" in screen.lower():
                viewer.type("\r")
            if "❯" in screen if claude else "gpt-6-luna" in screen.lower():
                break
        else:
            pytest.fail("CLI did not become ready:\n" + viewer.text())
        for turn in (1, 2):
            label = "CAPTURE_FIRST_PROMPT" if turn == 1 else "CAPTURE_SECOND_PROMPT"
            viewer.type(
                f"{label}: Run the shell command printf CAPTURE_TOOL_{turn}, then reply exactly CAPTURE_TURN_{turn}_DONE."
            )
            viewer.pump()
            viewer.type("\r")
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                viewer.pump()
                session._agent_logger.flush()
                exchanges = read_llm_exchanges(session._agent_logger.db_path)
                if (
                    any(
                        b.get("type") == "text" and f"CAPTURE_TURN_{turn}_DONE" in b.get("text", "")
                        for e in exchanges
                        for b in e["response"]
                    )
                    and session._main.state == "agent_idle"
                ):
                    break
            else:
                pytest.fail(
                    f"turn {turn} failed; archives={len(list((session.run_dir / 'capture').glob('*/manifest.json')))}\n"
                    + viewer.text()
                )
            # The terminal must receive the actual response as well as Agency.
            viewer.wait_for(f"CAPTURE_TURN_{turn}_DONE", timeout=10)

        def completed():
            view = trajectory(session)
            models = actions(view, "model")
            tools = actions(view, "tool")
            assert len(models) >= 4
            assert len(tools) >= 2, "native Pre/PostToolUse hooks were not preserved"
            assert all(t["outcome"] == "success" for t in tools)
            assert all(m["context"] and m["context"]["blocks"] for m in models)
            return models, tools

        models, tools = eventually(completed)
        archives = [
            json.loads(p.read_text()) for p in (session.run_dir / "capture").glob("*/manifest.json")
        ]
        assert all(not a["cancelled"] for a in archives)
        assert all(a["status"] in ({200, 101, 404} if native else {200}) for a in archives)
        assert session.scope_dir is not None
        if not live:
            viewer.type("CAPTURE_CANCEL_PROMPT: stream until I interrupt")
            viewer.pump()
            viewer.type("\r")
            if claude or native:
                # Claude can defer rendering a text block until block_stop;
                # interrupt while the native stream is still inside it.
                def partial_received():
                    viewer.pump()
                    assert any(
                        b"PARTIAL_CAPTURE" in p.read_bytes()
                        for p in (session.run_dir / "capture").glob("*/response.bin")
                    )

                eventually(partial_received)
            else:
                viewer.wait_for("PARTIAL_CAPTURE", timeout=15)
            viewer.type("\x1b")
            assert upstream.cancelled.wait(5), "Escape did not close the native model stream"

            def cancelled_exchange():
                session._agent_logger.flush()
                rows = read_llm_exchanges(session._agent_logger.db_path)
                assert any(
                    r["type"] == "llm_stream_cancelled"
                    and any(b.get("text") == "PARTIAL_CAPTURE" for b in r["response"])
                    for r in rows
                )

            eventually(cancelled_exchange)
            print(
                f"{protocol}: real PTY Escape cancelled upstream; partial SQLite response retained"
            )
        if os.environ.get("AGENCY_CAPTURE_COMPACT") == "1":
            before = len(list((session.run_dir / "capture").glob("*/manifest.json")))
            viewer.type("/compact")
            viewer.pump()
            viewer.type("\r")

            def compacted():
                viewer.pump()
                records = [
                    json.loads(p.read_text())
                    for p in (session.run_dir / "capture").glob("*/manifest.json")
                ]
                assert len(records) > before, "CLI compact command sent no model request"
                assert all("ended_at" in r for r in records)
                assert "compacted" in viewer.text().lower(), (
                    "CLI did not report completed compaction"
                )
                return records

            records = eventually(compacted, timeout=45)
            screen_path = tmp_path / "compaction-terminal.txt"
            screen_path.write_text(viewer.text())
            screen_path.chmod(0o600)
            print(f"{protocol}: /compact captured; paths={sorted({r['path'] for r in records})}")
        print(
            f"{protocol}: {len(models)} model exchanges, {len(tools)} tools, {len(archives)} wire archives; scope={session.scope_unit}; live={live}; native_auth={native}"
        )
    finally:
        viewer.close()
        if upstream:
            upstream.close()
