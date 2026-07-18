# LLM gateway for harness-driven agents (`agharness_internal/agproxy_llm.py`)

Agency's LLM client (`agllm.py`, see [agllm.md](../agllm.md)) is server-less — it only ever makes
outbound calls. Wiring a harness's `ANTHROPIC_BASE_URL`/`model_providers.base_url`/custom
`provider` block at the agent's own `agConfig` backend requires becoming a server for the harness
to connect *to*. `agProxyLLM` is that server: one process-wide FastAPI app, started on demand,
routing each harness's requests to the specific agent that launched it (via a per-run bearer
token), forwarded to that agent's own backend client.

**Chat-completions passthrough only.** This already matches `agllm`'s internal wire format 1:1
(see [Design_harness_integration.md](../Design_harness_integration.md)'s Component 1), so the route
just forwards the request body to `ag.llm.backend.make_client(...)` and streams the response back
unmodified — no reshaping. This is what opencode's `@ai-sdk/openai-compatible` provider speaks
natively. An Anthropic Messages API adapter (for Claude Code) and an OpenAI Responses API adapter
(for Codex, which no longer supports chat-completions at all) are **not implemented** — both
`agharness_internal/agharness_backends/claude_code.py` and `.../codex.py` currently leave the harness's own LLM
endpoint untouched rather than routing it through this gateway, and say so in their own docstrings.

## API

```python
from agency.agharness_internal.agproxy_llm import agProxyLLM, get_shared_gateway

gateway = get_shared_gateway(agconfig)   # one per process, lazily started
token = "..."                             # minted per harness launch
gateway.register(token, ag)
base_url = gateway.base_url               # e.g. "http://127.0.0.1:54321"
# ... launch the harness pointed at base_url with this token as its bearer credential ...
gateway.unregister(token)
```

- `agProxyLLM(agconfig=None)` — one instance is reusable; `.start()`/`.stop()` manage the
  background uvicorn server (idempotent both ways).
- `.register(token, ag)` / `.unregister(token)` — the routing table; a request with an unknown or
  missing bearer token gets `401`.
- `get_shared_gateway(agconfig=None)` — process-wide singleton, since routing is per-token, not
  per-port; every harness-driven agent in the process shares one running server.

## Config

`agProxyLLMConfig` (`_OWNER = "agproxy_llm"`):

| Field | Tier | Default |
|---|---|---|
| `bind_host` | global | `"127.0.0.1"` |
| `port` | dynamic | `0` (OS-assigned; read the real bound port from `start()`'s return value, never assume this field's value is what actually got bound) |
| `request_timeout_s` | global | `300` |

## Implementation notes

- **Route handler annotations must resolve from module-level globals.** With
  `from __future__ import annotations` active, FastAPI/Starlette resolve a route's parameter
  annotations (e.g. `request: Request`) via `typing.get_type_hints()` against the function's
  `__globals__` at runtime — a function-local `from fastapi import Request` inside the app-building
  method breaks this silently (every call 422s with "field required: request" instead of
  dispatching, since FastAPI can't resolve the string annotation and guesses it's a query
  parameter). `FastAPI`/`Request`/response classes must be imported at module level. Hit and fixed
  during development.
- Both streaming (SSE, one `data: {...}\n\n` chunk per upstream chunk, ending `data: [DONE]\n\n`)
  and non-streaming requests are supported, dispatched on the request body's own `"stream"` field —
  same as the real OpenAI-compatible API this passes through to.
