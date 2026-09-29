"""Live compatibility smoke test for the current OpenAI and Anthropic models.

Not a tutorial lesson and not part of pytest: every run makes real (tiny)
API calls. For each selected model it checks three things, all through
Agency's own code rather than the raw provider SDK:

1. visible -- the model appears in Agency's model listing for the provider,
   and what context limit Agency resolves for it (and from where);
2. basic   -- one tiny streamed request through Agency's LLM backend
   (``agllm.for_config(...).dispatch_stream``), expecting AGENCY_MODEL_OK;
3. e2e     -- a two-turn tool loop through the same path a sandboxed harness
   uses: harness wire request -> harness adapter proxy -> HostServicesClient
   -> LlmHandlerServer (over a real Unix socket) -> backend -> provider. The
   native harness's own LLMClient drives the Chat Completions wire; the Codex
   (Responses) and Claude Code (Messages) wires are sent as those harnesses
   send them. Only the sandbox and harness binaries are left out.

A model entry can declare a known incompatibility (`expected_e2e_error`):
its e2e runs must then fail with exactly that provider message, reported as
XFAIL. The script exits 0 only if every stage passes or fails as expected.

Credentials come from the environment, or from a local .env file (default:
the repository root's) for variables not already set. Key values are never
printed; error text is redacted before it is shown.

Examples:

    uv run python examples/smoke_test_models.py
    uv run python examples/smoke_test_models.py --provider anthropic
    uv run python examples/smoke_test_models.py --model gpt-6-luna --harness codex
    uv run python examples/smoke_test_models.py --skip-e2e
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from agency.configs.agconfig import agconfig, llmconfig

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TEXT = "AGENCY_MODEL_OK"
BASIC_PROMPT = f"Reply with exactly: {EXPECTED_TEXT}"
SYSTEM_PROMPT = "You are a terse test assistant. Follow instructions exactly."
SECRET_WORD = "PLUM-7"
TOOL_PROMPT = (
    "Call the get_secret_word tool, then reply with exactly the secret word it "
    "returns and nothing else."
)
TOOL_NAME = "get_secret_word"
TOOL_DESCRIPTION = "Return the secret word for this test."
TOOL_PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}
PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "openai_responses": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class ModelSpec:
    model: str
    provider: str
    # None leaves the provider default in place. GPT-6 Sol/Luna accept
    # function tools on Chat Completions only with "none"; Astra rejects
    # "none" outright, so it keeps its default -- and can only run tools
    # through provider="openai_responses".
    reasoning_effort: "str | None" = None
    # The harness whose wire protocol matches the provider, exercised in
    # addition to the native harness.
    home_harness: str = "native"
    # A known incompatibility: the provider message every e2e run of this
    # spec must fail with. A matching failure is XFAIL and doesn't fail the
    # run; any other failure, or an unexpected pass, does.
    expected_e2e_error: "str | None" = None


MODELS: "tuple[ModelSpec, ...]" = (
    # Chat Completions rejects function tools for Astra at every effort it
    # accepts; it runs agents only through provider="openai_responses".
    ModelSpec(
        "gpt-6-astra",
        "openai",
        home_harness="codex",
        expected_e2e_error=(
            "Function tools with reasoning_effort are not supported for "
            "gpt-6-astra in /v1/chat/completions"
        ),
    ),
    ModelSpec("gpt-6-sol", "openai", reasoning_effort="none", home_harness="codex"),
    ModelSpec("gpt-6.1-sol", "openai", reasoning_effort="none", home_harness="codex"),
    ModelSpec("gpt-6-luna", "openai", reasoning_effort="none", home_harness="codex"),
    ModelSpec("gpt-6-astra", "openai_responses", home_harness="codex"),
    ModelSpec("claude-fable-5-1", "anthropic", home_harness="claude_code"),
    ModelSpec("claude-opus-5-5", "anthropic", home_harness="claude_code"),
    ModelSpec("claude-sonnet-5-5", "anthropic", home_harness="claude_code"),
)
HARNESSES = ("native", "codex", "claude_code")


# ---------------------------------------------------------------------------
# Credentials and redaction
# ---------------------------------------------------------------------------


def load_env_file(path: Path) -> "list[str]":
    """Set KEY=VALUE pairs from `path` that aren't already in the
    environment. Returns only the variable names that were set."""
    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


_KEY_LIKE_RE = re.compile(r"\b(?:sk|key)-[A-Za-z0-9_\-]{8,}")


def redact(text: object) -> str:
    """Strip every configured provider key (and anything key-shaped) from
    text that is about to be printed."""
    out = str(text)
    for env_key in PROVIDER_ENV_KEYS.values():
        secret = os.environ.get(env_key)
        if secret:
            out = out.replace(secret, "<redacted>")
    return _KEY_LIKE_RE.sub("<redacted>", out)


def _short(text: object, limit: int = 300) -> str:
    return redact(" ".join(str(text).split()))[:limit]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def build_config(spec: ModelSpec, *, reasoning_effort: "str | None" = None) -> agconfig:
    fields: dict = dict(
        provider=spec.provider,
        model=spec.model,
        api_key=os.environ.get(PROVIDER_ENV_KEYS[spec.provider]),
        max_completion_tokens=2048,
        stream_timeout=180.0,
        idle_timeout=180.0,
    )
    effort = reasoning_effort if reasoning_effort is not None else spec.reasoning_effort
    if spec.provider in ("openai", "openai_responses"):
        fields["base_url"] = OPENAI_BASE_URL
        if effort is not None:
            fields["reasoning_effort"] = effort
    return agconfig(llmconfig(**fields))


def select_models(provider: "str | None", models: "list[str] | None") -> "list[ModelSpec]":
    selected = [s for s in MODELS if provider is None or s.provider == provider]
    if models:
        known = {s.model for s in MODELS}
        unknown = sorted(set(models) - known)
        if unknown:
            raise SystemExit(f"unknown model(s) {unknown}; choose from {sorted(known)}")
        selected = [s for s in selected if s.model in models]
    return selected


def harnesses_for(spec: ModelSpec, choice: str) -> "list[str]":
    if choice == "default":
        return ["native"] if spec.home_harness == "native" else ["native", spec.home_harness]
    return [choice]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def classify_status(status: "int | None") -> str:
    if status in (401, 403):
        return "account access"
    if status == 404:
        return "model unavailable"
    if status == 429:
        return "rate limit/quota"
    if status is not None and status >= 500:
        return "provider error"
    if status == 400:
        return "request rejected"
    return "error"


def classify_exception(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return classify_status(status)
    name = type(exc).__name__
    if "Connection" in name or "Timeout" in name:
        return "network"
    return "error"


# ---------------------------------------------------------------------------
# Stage 1 + 2: listing, context limit, basic streamed call
# ---------------------------------------------------------------------------


def check_visibility(backend, model: str) -> "tuple[bool | None, str]":
    try:
        listed = backend.list_models()
    except Exception as exc:
        return None, f"listing failed ({classify_exception(exc)}): {_short(exc)}"
    entry = next((m for m in listed if getattr(m, "id", None) == model), None)
    if entry is None:
        return False, f"not in the provider's model list ({len(listed)} models listed)"
    # fetch_context_limit prints its own fallback warnings; keep them out of
    # the report and describe where the value came from instead.
    with contextlib.redirect_stdout(io.StringIO()):
        limit = backend.fetch_context_limit()
    if getattr(entry, "max_input_tokens", None):
        source = "provider /models"
    elif backend.known_context_limit(model) is not None:
        source = "Agency static table"
    else:
        source = "Agency default_context_limit (provider reports none)"
    return True, f"context limit {limit:,} from {source}"


def basic_call(backend) -> "tuple[bool, str]":
    request = {
        "messages": [
            {"role": "system", "blocks": [{"type": "text", "index": 0, "text": SYSTEM_PROMPT}]},
            {"role": "user", "blocks": [{"type": "text", "index": 0, "text": BASIC_PROMPT}]},
        ]
    }
    started = time.monotonic()
    try:
        text, stop_reason, usage = "", None, None
        for item in backend.dispatch_stream(request):
            if item.get("type") == "block_delta" and item.get("block_type") == "text":
                text += item.get("text") or ""
            elif item.get("type") == "usage":
                stop_reason = item.get("stop_reason") or stop_reason
                usage = item.get("usage") or usage
    except Exception as exc:
        return False, f"{classify_exception(exc)}: {_short(exc)}"
    elapsed = time.monotonic() - started
    tokens = f"{usage.get('prompt_tokens')}/{usage.get('completion_tokens')}" if usage else "?"
    detail = f"stream {elapsed:.1f}s, stop={stop_reason}, tokens in/out={tokens}"
    if EXPECTED_TEXT not in text:
        return False, f"unexpected reply {text[:60]!r} ({detail})"
    return True, detail


# ---------------------------------------------------------------------------
# Stage 3: Agency harness-proxy path
# ---------------------------------------------------------------------------


class UpstreamErrorLog:
    """LlmHandlerServer's data-logger seam. Keeps nothing but the error text
    of failed exchanges: on the native wire, a permanent upstream error
    reaches the harness only as a truncated stream, so the host's record is
    the one place the provider's actual message is still visible."""

    def __init__(self) -> None:
        self.errors: "list[str]" = []

    def record_event(self, *args, **kwargs) -> None:
        pass

    def record_stream_delta(self, *args, **kwargs) -> None:
        pass

    def record_llm_exchange(self, call_label, *, exchange_type, response_chain, **kwargs) -> None:
        if not exchange_type.endswith("_error"):
            return
        for _payload_hash, marker in response_chain:
            if "error" in marker:
                self.errors.append(marker["error"])


class _Server:
    """Run one ASGI app under uvicorn in a daemon thread."""

    def __init__(self, app, *, uds: "str | None" = None) -> None:
        import uvicorn

        self._sock = None
        if uds is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.bind(("127.0.0.1", 0))
            self.base_url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        self._server = uvicorn.Server(uvicorn.Config(app, uds=uds, log_level="warning"))
        sockets = [self._sock] if self._sock is not None else None
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": sockets}, daemon=True
        )

    def __enter__(self) -> "_Server":
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started and self._thread.is_alive():
            if time.monotonic() > deadline:
                raise RuntimeError("smoke-test server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
        if self._sock is not None:
            self._sock.close()


@contextlib.contextmanager
def agency_proxy(cfg: agconfig):
    """Yield (harness_base_url, token, upstream_error_log): the harness-facing
    adapter routes of every harness, wired to a real LlmHandlerServer through
    a real HostServicesClient over a Unix socket -- the host/sandbox split of
    a live run without the sandbox."""
    from fastapi import FastAPI

    from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
    from agency.harness.adapters.claude_code import ClaudeCodeAdapter
    from agency.harness.adapters.codex import CodexAdapter
    from agency.harness.adapters.native import NativeAdapter
    from agency.harness.clients.host_services_client import HostServicesClient
    from agency.llm.usage_tracker import LlmUsageTracker

    upstream_error_log = UpstreamErrorLog()
    handler = LlmHandlerServer(cfg, upstream_error_log, LlmUsageTracker())
    host_app = FastAPI()
    host_app.mount("/llm", handler.build_app())
    token = "agency-smoke-test"
    with tempfile.TemporaryDirectory(prefix="agsmoke") as tmp:
        uds = os.path.join(tmp, "host.sock")
        with _Server(host_app, uds=uds):
            host = HostServicesClient(uds, timeout_s=300)
            host.register_attempt_token(token)
            harness_app = FastAPI()
            for adapter_cls in (NativeAdapter, CodexAdapter, ClaudeCodeAdapter):
                adapter_cls(cfg).register(harness_app, host)
            try:
                with _Server(harness_app) as harness:
                    yield harness.base_url, token, upstream_error_log
            finally:
                host.clear_attempt_token(token)
                host.client.close()
                handler.stop()


class _E2EFailure(Exception):
    def __init__(self, message: str, status: "int | None" = None) -> None:
        super().__init__(message)
        self.status_code = status


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code != 200:
        resp.read()
        raise _E2EFailure(f"HTTP {resp.status_code}: {resp.text}", resp.status_code)


def _iter_sse(resp: httpx.Response):
    for line in resp.iter_lines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            yield json.loads(line[6:])


def _native_tool_loop(base_url: str, token: str, model: str) -> str:
    """The native harness's own LLMClient over the Chat Completions wire."""
    from agency.native_harness.llm_client import LLMClient

    client = LLMClient(base_url, token, timeout_s=300)
    tools = [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "parameters": TOOL_PARAMETERS,
            },
        }
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TOOL_PROMPT},
    ]
    first = client.dispatch(model, messages, tools)
    if "error" in first:
        status = re.search(r"dispatch failed: (\d{3})", first["error"])
        raise _E2EFailure(first["error"], int(status.group(1)) if status else None)
    calls = first["message"].get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != TOOL_NAME:
        raise _E2EFailure(f"no {TOOL_NAME} call in first turn: {first['message']}")
    messages += [
        first["message"],
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": SECRET_WORD},
    ]
    second = client.dispatch(model, messages, tools)
    if "error" in second:
        raise _E2EFailure(second["error"])
    return second["message"].get("content") or ""


def _codex_turn(http: httpx.Client, body: dict) -> "list[dict]":
    items = []
    with http.stream("POST", "/v1/responses", json=body) as resp:
        _raise_for_status(resp)
        for event in _iter_sse(resp):
            if event.get("type") == "response.output_item.done":
                items.append(event["item"])
    return items


def _codex_tool_loop(base_url: str, token: str, model: str) -> str:
    """Codex's Responses wire: output items are replayed as next-turn input."""
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=300) as http:
        body = {
            "model": model,
            "instructions": SYSTEM_PROMPT,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": TOOL_PROMPT}],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "parameters": TOOL_PARAMETERS,
                    "strict": False,
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "stream": True,
        }
        output = _codex_turn(http, body)
        call = next((i for i in output if i.get("type") == "function_call"), None)
        if call is None or call.get("name") != TOOL_NAME:
            raise _E2EFailure(f"no {TOOL_NAME} call in first turn: {output}")
        body["input"] = [
            *body["input"],
            *output,
            {"type": "function_call_output", "call_id": call["call_id"], "output": SECRET_WORD},
        ]
        output = _codex_turn(http, body)
    return "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content") or []
    )


def _claude_code_turn(http: httpx.Client, body: dict) -> "list[dict]":
    blocks: "dict[int, dict]" = {}
    json_parts: "dict[int, list[str]]" = {}
    with http.stream("POST", "/v1/messages", json=body) as resp:
        _raise_for_status(resp)
        for event in _iter_sse(resp):
            etype = event.get("type")
            if etype == "content_block_start":
                blocks[event["index"]] = dict(event["content_block"])
            elif etype == "content_block_delta":
                block, delta = blocks[event["index"]], event["delta"]
                if delta["type"] == "text_delta":
                    block["text"] = block.get("text", "") + delta["text"]
                elif delta["type"] == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta["thinking"]
                elif delta["type"] == "signature_delta":
                    block["signature"] = delta["signature"]
                elif delta["type"] == "input_json_delta":
                    json_parts.setdefault(event["index"], []).append(delta["partial_json"])
    for index, parts in json_parts.items():
        blocks[index]["input"] = json.loads("".join(parts) or "{}")
    return [blocks[i] for i in sorted(blocks)]


def _claude_code_tool_loop(base_url: str, token: str, model: str) -> str:
    """Claude Code's Messages wire: assistant content (including any signed
    thinking blocks) is replayed verbatim on the next turn."""
    headers = {"Authorization": f"Bearer {token}", "anthropic-version": "2023-06-01"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=300) as http:
        body = {
            "model": model,
            "max_tokens": 2048,
            "system": [{"type": "text", "text": SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": TOOL_PROMPT}],
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "input_schema": TOOL_PARAMETERS,
                }
            ],
            "stream": True,
        }
        content = _claude_code_turn(http, body)
        call = next((b for b in content if b.get("type") == "tool_use"), None)
        if call is None or call.get("name") != TOOL_NAME:
            raise _E2EFailure(f"no {TOOL_NAME} call in first turn: {content}")
        body["messages"] = [
            *body["messages"],
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call["id"], "content": SECRET_WORD}
                ],
            },
        ]
        content = _claude_code_turn(http, body)
    return "".join(b.get("text", "") for b in content if b.get("type") == "text")


_TOOL_LOOPS = {
    "native": _native_tool_loop,
    "codex": _codex_tool_loop,
    "claude_code": _claude_code_tool_loop,
}


XFAIL = "xfail"


def judge_e2e_failure(
    exc: BaseException, upstream_errors: "list[str]", expected_error: "str | None"
) -> "tuple[bool | str, str]":
    """A failure is expected (XFAIL) only when the spec names one and that
    exact provider message shows up -- in the harness-visible error or the
    host's upstream record. Any other failure is a real FAIL."""
    evidence = [str(exc), *upstream_errors]
    if expected_error and any(expected_error in text for text in evidence):
        return XFAIL, f"expected failure: {expected_error}"
    detail = f"{classify_exception(exc)}: {_short(exc)}"
    if upstream_errors:
        detail += f" (upstream: {_short(upstream_errors[-1])})"
    if expected_error:
        detail += f" -- expected {expected_error!r} instead"
    return False, detail


def judge_e2e_success(reply: str, elapsed: float, expected_error: "str | None"):
    if expected_error:
        # Strict: an expected failure that stops happening means the known
        # incompatibility changed, so the spec needs revisiting either way.
        return False, f"passed, but was expected to fail with {expected_error!r}"
    if SECRET_WORD not in reply:
        return False, f"tool loop finished but reply was {reply[:60]!r}"
    return True, f"tool call + tool result round-trip OK in {elapsed:.1f}s"


def e2e_call(
    cfg: agconfig, harness: str, expected_error: "str | None" = None
) -> "tuple[bool | str, str]":
    started = time.monotonic()
    upstream_error_log = UpstreamErrorLog()
    try:
        with agency_proxy(cfg) as (base_url, token, upstream_error_log):
            reply = _TOOL_LOOPS[harness](base_url, token, cfg.llm.model)
    except Exception as exc:
        return judge_e2e_failure(exc, upstream_error_log.errors, expected_error)
    return judge_e2e_success(reply, time.monotonic() - started, expected_error)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _mark(ok: "bool | str | None") -> str:
    return {True: "PASS", False: "FAIL", None: "SKIP", XFAIL: "XFAIL"}[ok]


def run(args: argparse.Namespace) -> int:
    from agency.llm.agllm import agllm

    specs = select_models(args.provider, args.model)
    rows = []
    for spec in specs:
        env_key = PROVIDER_ENV_KEYS[spec.provider]
        print(f"\n== {spec.model} ({spec.provider})")
        if not os.environ.get(env_key):
            print(f"  SKIP  {env_key} is not set")
            rows.append((spec, None, None, {}))
            continue
        cfg = build_config(spec, reasoning_effort=args.reasoning_effort)
        if cfg.llm.reasoning_effort is not None:
            print(f"  reasoning_effort={cfg.llm.reasoning_effort}")
        backend = agllm.for_config(cfg)
        print(f"  backend: {type(backend).__name__}")

        visible, detail = check_visibility(backend, spec.model)
        print(f"  {_mark(visible)}  visible  {detail}")
        basic_ok, detail = basic_call(backend)
        print(f"  {_mark(basic_ok)}  basic    {detail}")

        e2e: "dict[str, bool | str | None]" = {}
        if not args.skip_e2e:
            for harness in harnesses_for(spec, args.harness):
                ok, detail = e2e_call(cfg, harness, spec.expected_e2e_error)
                e2e[harness] = ok
                print(f"  {_mark(ok)}  e2e:{harness:<11s} {detail}")
        rows.append((spec, visible, basic_ok, e2e))

    print("\n| Model | Provider | API visible | Basic call | Agency E2E |")
    print("|---|---|---|---|---|")
    for spec, visible, basic_ok, e2e in rows:
        e2e_text = ", ".join(f"{h} {_mark(ok)}" for h, ok in e2e.items()) or "SKIP"
        print(
            f"| {spec.model} | {spec.provider} | {_mark(visible)} | {_mark(basic_ok)} "
            f"| {e2e_text} |"
        )
    failed = any(
        visible is False or basic_ok is False or False in e2e.values()
        for _, visible, basic_ok, e2e in rows
    )
    return 1 if failed else 0


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=sorted(PROVIDER_ENV_KEYS))
    parser.add_argument(
        "--model", action="append", help="model ID to test (repeatable); default: all"
    )
    parser.add_argument(
        "--harness",
        choices=("default", *HARNESSES),
        default="default",
        help="harness wire for the e2e stage; default: native plus the provider's own",
    )
    parser.add_argument("--skip-e2e", action="store_true", help="only list + basic call")
    parser.add_argument(
        "--reasoning-effort", help="override the per-model reasoning_effort (OpenAI only)"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="dotenv file for unset credentials (default: repo-root .env)",
    )
    return parser.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)
    loaded = load_env_file(args.env_file)
    if loaded:
        print(f"loaded {', '.join(sorted(loaded))} from {args.env_file}")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
