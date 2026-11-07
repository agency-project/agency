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
from agency.sessions.capture import CaptureConfig
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
                self.send_response(404)
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
                    return
                text = json.dumps(history)
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
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_real_interactive_capture(make_session, monkeypatch, tmp_path, protocol):
    claude = protocol == "messages"
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
        config = CaptureConfig(model)
    else:
        monkeypatch.setenv("LITE_LLM_KEY", "scripted-upstream-fixture-key")
        upstream = NativeUpstream(protocol, model)
        config = CaptureConfig(model, upstream.url)
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
            "workspace-write",
            "-a",
            "never",
            "--dangerously-bypass-hook-trust",
        ]
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
                if any(
                    b.get("type") == "text" and f"CAPTURE_TURN_{turn}_DONE" in b.get("text", "")
                    for e in exchanges
                    for b in e["response"]
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
        assert all(a["status"] == 200 for a in archives)
        assert session.scope_dir is not None
        if not live:
            viewer.type("CAPTURE_CANCEL_PROMPT: stream until I interrupt")
            viewer.pump()
            viewer.type("\r")
            if claude:
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
            f"{protocol}: {len(models)} model exchanges, {len(tools)} tools, {len(archives)} wire archives; scope={session.scope_unit}; live={live}"
        )
    finally:
        viewer.close()
        if upstream:
            upstream.close()
