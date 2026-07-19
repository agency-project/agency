# LLM gateway for harness-driven agents (`agharness_internal/agproxy_llm.py`)

Agency's LLM client (`agllm.py`, see [agllm.md](../agllm.md)) is server-less — it only ever makes
outbound calls. Wiring a harness's `ANTHROPIC_BASE_URL`/`model_providers.base_url`/custom
`provider` block at the agent's own `agConfig` backend requires becoming a server for the harness
to connect *to*. `agProxyLLM` is that server: one process-wide FastAPI app, started on demand,
routing each harness's requests to the specific agent that launched it (via a per-run bearer
token), forwarded to that agent's own backend client.

**Three routes, two modes.** `/v1/chat/completions` is a straight passthrough — it already matches
`agllm`'s internal wire format 1:1 (see [Design_harness_integration.md](../Design_harness_integration.md)'s
Component 1), so the route just forwards the request body to `ag.llm.backend.make_client(...)` and
streams the response back unmodified. This is what opencode's `@ai-sdk/openai-compatible` provider
and Grok Build's `api_backend = "chat_completions"` speak natively (`gateway_mode="passthrough"`).

`/v1/messages` (+ `/v1/messages/count_tokens`) and `/v1/responses` are **translated**
(`gateway_mode="translate"`): the incoming Anthropic Messages API request (Claude Code) or OpenAI
Responses API request (Codex) is reshaped into `client.chat.completions.create(**kwargs)` — the
exact same call every backend uniformly exposes — and the (possibly streaming) response reshaped
back into the harness's native format. The conversion functions themselves live in
`agproxy_llm_adapters.py` (a separate module, pure functions, no FastAPI/network dependency), kept
apart from this file's routing/registry concerns. See that module's docstring for the fidelity cost
this implies: extended-thinking blocks, prompt-cache breakpoints (`cache_control`), and image
content blocks all have no chat-completions equivalent and are silently dropped rather than erroring
— this is translation, not a transparent proxy to the harness's real provider.

`count_tokens` has no real tokenizer wired up — it returns a `chars / 4` heuristic, good enough for
Claude Code's own context-usage estimates (the only thing that endpoint feeds).

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
  and non-streaming requests are supported on `/v1/chat/completions`, dispatched on the request
  body's own `"stream"` field. `/v1/messages` and `/v1/responses` stream via their own native SSE
  framing instead (`event: <type>\ndata: {...}\n\n`, matching the real Anthropic/OpenAI wire
  formats — Anthropic's SDK dispatches on the `event:` line, not just the JSON payload's own
  `"type"` field, so both must be present and correct).
- `/v1/messages`'s translated-response path was verified against the real `claude` CLI (not just
  mocked) — see `tests/agharness_internal/agharness_backends/test_claude_code.py`'s `real_claude`-marked tests,
  which now point `ANTHROPIC_BASE_URL` at this gateway rather than leaving Claude Code on its own
  host credentials. `/v1/responses` is unverified against a live `codex` binary (none installable in
  this environment) — implemented from documented Responses API streaming event shapes only.
