"""A local, scripted OpenAI-compatible endpoint for smoke tests.

It stands in for LiteLLM so the complete Codex -> Agency -> HTTP path can be
exercised without a paid model call. Stateless: each reply is derived from the
conversation in the request, so any number of agents can share one server.

Script per conversation:
  1. no tool result yet        -> call the shell tool with ``--command``
  2. shell result, no submit   -> call Agency's submit tool
  3. otherwise                 -> a short final message

Failure injection (for the validation suite):
  --fail-429 N      answer the first N requests with HTTP 429 (+ Retry-After)
  --fail-500 N      answer the first N requests with HTTP 500
  --hang-first N    accept the first N requests and never answer
  --latency S       seconds before the first byte
  --stream-seconds S  spread the streamed chunks over S seconds

    python -m benchmarks.terminalbench_contention.mock_model --port 0 --ready-file ready.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SHELL_TOOLS = ("exec_command", "shell", "local_shell", "bash", "container.exec")


def _find_tool(tools, predicate):
    for tool in tools or []:
        function = tool.get("function") or {}
        if predicate(function.get("name", "")):
            return function
    return None


def _fill_schema(schema, text):
    """Smallest value satisfying a JSON schema, using ``text`` for strings."""
    if not isinstance(schema, dict):
        return text
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind == "object" or "properties" in schema:
        properties = schema.get("properties") or {}
        required = schema.get("required") or list(properties)
        return {name: _fill_schema(properties.get(name, {}), text) for name in required}
    if kind == "array":
        return []
    if kind in ("integer", "number"):
        return 0
    if kind == "boolean":
        return True
    return text


def plan_reply(payload: dict, command: str, final_text: str) -> dict:
    """Decide the scripted assistant turn for one request body."""
    tools = payload.get("tools") or []
    messages = payload.get("messages") or []
    called = [
        call.get("function", {}).get("name", "")
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]
    shell = _find_tool(tools, lambda name: name in SHELL_TOOLS)
    submit = _find_tool(tools, lambda name: "submit" in name.lower())
    if shell is not None and shell["name"] not in called:
        name = shell["name"]
        properties = (shell.get("parameters") or {}).get("properties") or {}
        if "cmd" in properties:
            arguments = {"cmd": command}
        elif properties.get("command", {}).get("type") == "array":
            arguments = {"command": ["bash", "-lc", command]}
        else:
            arguments = {"command": command}
        return dict(tool=name, arguments=arguments)
    if submit is not None and submit["name"] not in called:
        return dict(
            tool=submit["name"], arguments=_fill_schema(submit.get("parameters"), final_text)
        )
    return dict(text=final_text)


class MockState:
    def __init__(self, args) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.count = 0
        self.active = 0
        self.peak_active = 0
        self.log = Path(args.log) if args.log else None

    def record(self, **fields) -> None:
        if self.log is None:
            return
        with self.lock, self.log.open("a") as stream:
            stream.write(json.dumps(dict(mono_ns=time.monotonic_ns(), **fields)) + "\n")


def make_handler(state: MockState):
    args = state.args

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # keep test output quiet
            return

        def _json(self, status, body, headers=None):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._json(200, {"object": "list", "data": [{"id": args.model, "object": "model"}]})
            elif self.path.rstrip("/").endswith("/stats"):
                with state.lock:
                    self._json(200, dict(requests=state.count, peak_active=state.peak_active))
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._json(404, {"error": {"message": "not found"}})
                return
            with state.lock:
                state.count += 1
                number = state.count
                state.active += 1
                state.peak_active = max(state.peak_active, state.active)
                active = state.active
            started = time.monotonic_ns()
            try:
                self._respond(number, body, active)
            finally:
                with state.lock:
                    state.active -= 1
                state.record(
                    event="request",
                    number=number,
                    started_ns=started,
                    bytes=len(body),
                    concurrent=active,
                )

        def _respond(self, number, body, active):
            if number <= args.hang_first:
                time.sleep(args.hang_seconds)
                return
            if number <= args.fail_429:
                self._json(
                    429,
                    {"error": {"message": "mock rate limit", "type": "rate_limit_error"}},
                    {"Retry-After": str(args.retry_after), "x-ratelimit-reset-tokens": "1s"},
                )
                return
            if number <= args.fail_429 + args.fail_500:
                self._json(500, {"error": {"message": "mock upstream failure"}})
                return
            try:
                payload = json.loads(body)
            except ValueError:
                self._json(400, {"error": {"message": "invalid JSON"}})
                return
            if args.latency:
                time.sleep(args.latency)
            reply = plan_reply(payload, args.command, args.final_text)
            prompt_tokens = args.prompt_tokens or max(1, len(body) // 4)
            if "tool" in reply:
                arguments = json.dumps(reply["arguments"])
                completion_tokens = max(1, len(arguments) // 4) + 8
            else:
                completion_tokens = max(1, len(reply["text"]) // 4)
            usage = dict(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_details=dict(cached_tokens=args.cached_tokens),
            )
            completion_id = "chatcmpl-mock-" + uuid.uuid4().hex[:16]
            base = dict(
                id=completion_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=payload.get("model") or args.model,
            )
            if not payload.get("stream"):
                message = {"role": "assistant", "content": reply.get("text")}
                if "tool" in reply:
                    message["tool_calls"] = [
                        dict(
                            id="call_" + uuid.uuid4().hex,
                            type="function",
                            function=dict(
                                name=reply["tool"], arguments=json.dumps(reply["arguments"])
                            ),
                        )
                    ]
                self._json(
                    200,
                    dict(
                        id=completion_id,
                        object="chat.completion",
                        created=base["created"],
                        model=base["model"],
                        usage=usage,
                        choices=[
                            dict(
                                index=0,
                                message=message,
                                finish_reason="tool_calls" if "tool" in reply else "stop",
                            )
                        ],
                    ),
                )
                return
            chunks = [
                dict(
                    base,
                    choices=[
                        dict(index=0, delta=dict(role="assistant", content=""), finish_reason=None)
                    ],
                )
            ]
            if "tool" in reply:
                arguments = json.dumps(reply["arguments"])
                call_id = "call_" + uuid.uuid4().hex
                chunks.append(
                    dict(
                        base,
                        choices=[
                            dict(
                                index=0,
                                finish_reason=None,
                                delta=dict(
                                    tool_calls=[
                                        dict(
                                            index=0,
                                            id=call_id,
                                            type="function",
                                            function=dict(name=reply["tool"], arguments=""),
                                        )
                                    ]
                                ),
                            )
                        ],
                    )
                )
                half = max(1, len(arguments) // 2)
                for piece in (arguments[:half], arguments[half:]):
                    if piece:
                        chunks.append(
                            dict(
                                base,
                                choices=[
                                    dict(
                                        index=0,
                                        finish_reason=None,
                                        delta=dict(
                                            tool_calls=[
                                                dict(index=0, function=dict(arguments=piece))
                                            ]
                                        ),
                                    )
                                ],
                            )
                        )
                finish = "tool_calls"
            else:
                for word in reply["text"].split(" "):
                    chunks.append(
                        dict(
                            base,
                            choices=[
                                dict(index=0, finish_reason=None, delta=dict(content=word + " "))
                            ],
                        )
                    )
                finish = "stop"
            chunks.append(dict(base, choices=[dict(index=0, delta={}, finish_reason=finish)]))
            if (payload.get("stream_options") or {}).get("include_usage"):
                chunks.append(dict(base, choices=[], usage=usage))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("x-request-id", completion_id)
            self.end_headers()
            pause = args.stream_seconds / max(1, len(chunks))
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                if pause:
                    time.sleep(pause)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="mock-luna")
    parser.add_argument("--command", default="echo agency-mock-tool-ran")
    parser.add_argument("--final-text", default="Mock agent finished.")
    parser.add_argument("--latency", type=float, default=0.0)
    parser.add_argument("--stream-seconds", type=float, default=0.0)
    parser.add_argument("--prompt-tokens", type=int, default=0)
    parser.add_argument("--cached-tokens", type=int, default=0)
    parser.add_argument("--fail-429", type=int, default=0)
    parser.add_argument("--fail-500", type=int, default=0)
    parser.add_argument("--retry-after", type=float, default=1)
    parser.add_argument("--hang-first", type=int, default=0)
    parser.add_argument("--hang-seconds", type=float, default=3600)
    parser.add_argument("--log")
    parser.add_argument("--ready-file")
    return parser


def start(argv=None):
    """Start the server in a daemon thread; returns (server, base_url, state)."""
    args = build_parser().parse_args(argv or [])
    state = MockState(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{args.host}:{server.server_address[1]}/v1", state


def main(argv=None):
    args = build_parser().parse_args(argv)
    state = MockState(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    server.daemon_threads = True
    base_url = f"http://{args.host}:{server.server_address[1]}/v1"
    if args.ready_file:
        Path(args.ready_file).write_text(json.dumps(dict(base_url=base_url, model=args.model)))
    print(base_url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
