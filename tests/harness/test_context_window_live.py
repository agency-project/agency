"""The installed harness CLIs, not a restatement of their rules, compact where
Agency configures them to.

CI installs the latest Claude Code and Grok and the pinned Codex
(.github/workflows/tests.yml), so a release that changes how one sizes
auto-compaction fails here before it reaches a run. Each test talks to a local mock API that reports the usage
Agency's adapter would forward; nothing leaves the machine. Skipped when the
CLI is not on PATH.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agency.configs.agconfig import agconfig
from agency.harness.adapters.claude_code import claude_code_window_settings
from agency.harness.adapters.codex import CodexAdapter
from agency.harness.adapters.openai_chat_completions import _forward_usage
from agency.harness.adapters.streaming import upstream_error_body
from agency.harness.clients.host_services_client import HostDispatchError

pytestmark = pytest.mark.timeout(300)


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class _MockAPI:
    """Anthropic Messages and OpenAI Chat Completions, one tool call then a
    final answer, every reply reporting `usage`."""

    def __init__(
        self,
        anthropic_usage: "dict | None" = None,
        chat_usage: "dict | None" = None,
        reject: "tuple[dict, int] | None" = None,
        tool_rounds: int = 1,
        command: str = "true",
    ):
        """`reject`: (JSON body, HTTP status) to answer the last request that
        carries tools before the final answer with, once; `tool_rounds`: tool
        calls before that answer; `command`: what each one runs."""
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                api.requests.append(body)
                with_tools = [r for r in api.requests if r.get("tools")]
                if reject and body.get("tools") and len(with_tools) == tool_rounds + 1:
                    payload = json.dumps(reject[0]).encode()
                    self.send_response(reject[1])
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    api.rejected = len(api.requests)
                    return
                results = sum(
                    sum(1 for b in m["content"] if b.get("type") == "tool_result")
                    if isinstance(m.get("content"), list)
                    else int(m.get("role") == "tool")
                    for m in body.get("messages", [])
                )
                answered = results >= tool_rounds
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                if self.path.startswith("/v1/messages"):
                    self.wfile.write(api._anthropic(body, answered))
                else:
                    self.wfile.write(api._chat(body, answered))

        self.command = command
        self.anthropic_usage = anthropic_usage or {"input_tokens": 1_000}
        self.chat_usage = chat_usage or {"prompt_tokens": 1_000, "completion_tokens": 10}
        self.requests: "list[dict]" = []
        self.rejected: "int | None" = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()

    def _anthropic(self, body, answered) -> bytes:
        usage = {"output_tokens": 1, **self.anthropic_usage}
        out = _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": body.get("model"),
                    "stop_reason": None,
                    "usage": usage,
                },
            },
        )
        if answered:
            block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": "done"}
            stop = "end_turn"
        else:
            block = {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps({"command": self.command, "description": "noop"}),
            }
            stop = "tool_use"
        out += _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": block},
        )
        out += _sse(
            "content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}
        )
        out += _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        out += _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop},
                "usage": {"output_tokens": 5},
            },
        )
        return out + _sse("message_stop", {"type": "message_stop"})

    def _chat(self, body, answered) -> bytes:
        tools = [t.get("function", {}).get("name", "") for t in body.get("tools") or []]
        shell = next(
            (t for t in tools if any(k in t.lower() for k in ("bash", "shell", "run"))), None
        )
        usage = self.chat_usage
        if answered or shell is None:
            # The final answer, or a request without tools (a compaction
            # summary, a title): small, so it never triggers anything itself.
            delta = {"role": "assistant", "content": "done"}
            finish = "stop"
            if shell is None:
                usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        else:
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": shell,
                            "arguments": json.dumps({"command": "true", "description": "noop"}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
        out = b""
        for choice in (
            {"index": 0, "delta": delta, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": finish},
        ):
            out += f"data: {json.dumps({**chunk, 'choices': [choice]})}\n\n".encode()
        out += f"data: {json.dumps({**chunk, 'choices': [], 'usage': usage})}\n\n".encode()
        return out + b"data: [DONE]\n\n"


def _run_claude(
    tmp_path, api, model, argv=(), env=None, prompt="run true once"
) -> subprocess.CompletedProcess:
    (tmp_path / ".claude.json").write_text('{"hasCompletedOnboarding": true}')
    try:
        return subprocess.run(
            [
                shutil.which("claude"),
                "-p",
                prompt,
                "--model",
                model,
                "--setting-sources",
                "",
                "--permission-mode",
                "bypassPermissions",
                "--max-turns",
                "8",
                "--tools",
                "Bash",
                "--debug",
                *argv,
            ],
            cwd=tmp_path,
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(tmp_path),
                "CLAUDE_CONFIG_DIR": str(tmp_path),
                "ANTHROPIC_BASE_URL": api.url,
                "ANTHROPIC_AUTH_TOKEN": "x",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                # As in the adapter: bypass mode is refused to root otherwise.
                "IS_SANDBOX": "1",
                **(env or {}),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=240,
        )
    finally:
        api.close()


@pytest.mark.skipif(shutil.which("claude") is None, reason="Claude Code is not installed")
@pytest.mark.parametrize(
    "model,limit",
    [
        ("gpt-6-luna", 922_000),  # not a model Claude Code knows: capped at 200K before
        ("open-weights-model", 131_072),
        ("small-open-weights-model", 64_000),  # below --autocompact's 100K floor
        ("claude-haiku-5-5", 1_000_000),
        ("claude-future-9-small", 200_000),
    ],
)
def test_claude_code_compacts_at_90_percent(model, limit, tmp_path):
    model, argv, env = claude_code_window_settings(model, limit)
    api = _MockAPI()
    _run_claude(tmp_path, api, model, argv, env)
    assert api.requests, "Claude Code made no request"
    assert all("[" not in request["model"] for request in api.requests)
    debug = "".join(p.read_text(errors="replace") for p in tmp_path.rglob("debug/*") if p.is_file())
    windows = [int(w) for w in re.findall(r"autocompact: .*?effectiveWindow=(\d+)", debug)]
    assert windows, "Claude Code logged no autocompact check; its debug format changed"
    # Claude Code compacts 13K below its effective window.
    threshold = windows[-1] - 13_000
    assert abs(threshold - 0.9 * limit) <= 0.01 * limit, (model, limit, threshold)


@pytest.mark.skipif(shutil.which("claude") is None, reason="Claude Code is not installed")
def test_claude_code_takes_any_backends_overflow_for_one(tmp_path):
    """An OpenAI-style "maximum context length" error, as Agency's adapter
    passes it on, must start Claude Code's reactive compaction (shrink the
    conversation and retry). Unworded, Claude Code just fails the turn."""
    message = (
        "litellm 400: This model's maximum context length is 128000 tokens. However, your "
        "messages resulted in 250000 tokens."
    )
    rejection = upstream_error_body(HostDispatchError({"message": message, "status_code": 400}))
    api = _MockAPI(reject=rejection, tool_rounds=3)
    _run_claude(tmp_path, api, "gpt-6-luna", prompt="run true three times, one call each")
    assert api.rejected is not None
    debug = "".join(p.read_text(errors="replace") for p in tmp_path.rglob("debug/*") if p.is_file())
    assert "Reactive compact" in debug, "Claude Code did not take the error for an overflow"


@pytest.mark.skipif(shutil.which("grok") is None, reason="Grok is not installed")
@pytest.mark.parametrize("share,compacts", [(0.80, False), (0.90, True)])
def test_grok_compacts_on_a_cached_anthropic_prompt(share, compacts, tmp_path):
    """Usage as the Anthropic backend reports it, almost all cache reads,
    forwarded the way the adapter forwards it: Grok must see the whole prompt
    (it compacts at 85% of context_window)."""
    window = 100_000
    whole = int(window * share)
    anthropic_usage = {
        "prompt_tokens": 500,
        "completion_tokens": 10,
        "total_tokens": 510,
        "cache_read_tokens": whole - 500,
        "cache_write_tokens": 0,
    }
    api = _MockAPI(chat_usage=_forward_usage(anthropic_usage))
    home = tmp_path / "home"
    (home / ".grok").mkdir(parents=True)
    (home / ".grok" / "config.toml").write_text(
        '[models]\ndefault = "agency-proxy"\n\n[model.agency-proxy]\nmodel = "m"\n'
        f'base_url = "{api.url}/v1"\napi_key = "x"\napi_backend = "chat_completions"\n'
        f"context_window = {window}\n\n[ui]\nprompt_suggestions = false\n"
    )
    try:
        result = subprocess.run(
            [
                shutil.which("grok"),
                "-p",
                "run true once with your shell tool",
                "--always-approve",
                "--no-memory",
                "--no-plan",
                "--no-subagents",
                "--disable-web-search",
                "--output-format",
                "streaming-json",
            ],
            cwd=tmp_path,
            env={"PATH": os.environ["PATH"], "HOME": str(home), "GROK_HOME": str(home / ".grok")},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=240,
        )
    finally:
        api.close()
    assert len(api.requests) >= 2, result.stdout[-2000:] + result.stderr[-2000:]
    assert ("auto_compact_started" in result.stdout) is compacts


# -- Codex -------------------------------------------------------------------

_CODEX_MODEL = "agency-test-model"


class _MockResponses:
    """The Responses API, answered with the frames Agency's own Codex adapter
    produces: one shell call, then "done". The first reply reports
    `first_prompt` tokens, later ones 1,000."""

    def __init__(self, first_prompt: int):
        mock = self
        adapter = CodexAdapter(agconfig())
        self.requests: "list[dict]" = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                mock.requests.append(body)
                names = [t.get("name") for t in body.get("tools") or []]
                shell = next(
                    (t for t in ("exec_command", "shell_command", "shell") if t in names), None
                )
                ran = any(i.get("type") == "function_call_output" for i in body.get("input", []))
                if shell and not ran:
                    arguments = {
                        "exec_command": {"cmd": "true"},
                        "shell_command": {"command": "true"},
                    }
                    block = {
                        "type": "tool_use",
                        "index": 0,
                        "id": f"call{len(mock.requests)}",
                        "name": shell,
                        "arguments": json.dumps(arguments.get(shell, {"command": ["true"]})),
                    }
                    stop = "tool_use"
                else:
                    block, stop = {"type": "text", "index": 0, "text": "done"}, "stop"
                prompt = first_prompt if len(mock.requests) == 1 else 1_000
                item = {
                    "type": "done",
                    "message": {"role": "assistant", "blocks": [block]},
                    "usage": {"prompt_tokens": prompt, "completion_tokens": 10},
                    "stop_reason": stop,
                }
                frames = "".join(
                    adapter._format_agency_stream_to_harness(iter([item]), _CODEX_MODEL)
                )
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.write(frames.encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _run_codex(tmp_path, name, context_limit, first_prompt) -> "tuple[list[dict], str]":
    api = _MockResponses(first_prompt)
    home = tmp_path / name
    home.mkdir()
    adapter = CodexAdapter(agconfig())
    adapter._write_codex_config(home, api.url, _CODEX_MODEL, context_limit=context_limit)
    # Agency's own MCP server isn't part of this test.
    config = (home / "config.toml").read_text()
    (home / "config.toml").write_text(
        re.sub(r"\n\[mcp_servers\.agency[^\n]*\](?:\n(?!\[)[^\n]*)*", "", config)
    )
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    try:
        subprocess.run(
            [
                shutil.which("codex"),
                "exec",
                "--skip-git-repo-check",
                "-C",
                str(work),
                "run the shell command `true` once, then say done",
            ],
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(home),
                "CODEX_HOME": str(home),
                adapter._ENV_KEY_NAME: "x",
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
        )
    finally:
        api.server.shutdown()
    assert api.requests, "Codex made no request"
    return api.requests, str(home)


def _compacted(requests) -> bool:
    """Codex's compaction request is the one without tools."""
    return any(not request.get("tools") for request in requests)


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex is not installed")
def test_codex_catalog_changes_nothing_but_the_window(tmp_path):
    """Above 272K the adapter hands Codex a catalog entry (codex_model_catalog);
    Codex must send the same request it sends on its own fallback metadata."""
    fallback, fallback_home = _run_codex(tmp_path, "fallback", 200_000, 1_000)
    catalog, catalog_home = _run_codex(tmp_path, "catalog", 400_000, 1_000)
    assert (tmp_path / "catalog" / "model_catalog.json").exists()
    assert not (tmp_path / "fallback" / "model_catalog.json").exists()
    a, b = fallback[0], catalog[0]
    for key in sorted(
        (set(a) | set(b)) - {"client_metadata", "prompt_cache_key", "input", "tools"}
    ):
        assert a.get(key) == b.get(key), key

    def tools(request):
        # spawn_agent (inside the multi_agent namespace) lists the catalog's
        # other models, which the catalog leaves out: Agency serves every
        # request with its own model anyway.
        def blank(tool):
            if tool.get("name") == "spawn_agent":
                tool = {**tool, "description": ""}
            if "tools" in tool:
                tool = {**tool, "tools": [blank(t) for t in tool["tools"]]}
            return tool

        return [blank(t) for t in request.get("tools") or []]

    assert tools(a) == tools(b)

    def contents(request, home):
        return [
            json.dumps({k: v for k, v in item.items() if k != "id"}).replace(home, "HOME")
            for item in request["input"]
        ]

    assert contents(a, fallback_home) == contents(b, catalog_home)


@pytest.mark.skipif(shutil.which("codex") is None, reason="Codex is not installed")
@pytest.mark.parametrize("prompt,compacts", [(300_000, False), (370_000, True)])
def test_codex_compacts_at_90_percent_of_a_large_window(prompt, compacts, tmp_path):
    """400K limit: 90% is 360K. Codex's own cap for a model it doesn't know
    would compact at 245K."""
    requests, _ = _run_codex(tmp_path, "home", 400_000, prompt)
    assert _compacted(requests) is compacts
