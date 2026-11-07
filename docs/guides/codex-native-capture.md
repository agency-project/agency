# Codex capture with native authentication

Launch the real Codex terminal while capturing model traffic using its existing
OpenAI sign-in:

```sh
uv run agency codex --capture-native
# Normal Codex arguments still follow --:
uv run agency codex --capture-native -- --model YOUR_NORMAL_CODEX_MODEL
```

Sign in through Codex first if necessary. Agency does not log in for you, read
`auth.json`, copy credentials, or refresh tokens. `LITE_LLM_KEY` is unnecessary.
The existing `--capture --capture-model gpt-6-luna` remains a separate explicit
LiteLLM mode; the two capture modes cannot be combined.

## Model and authentication path

```text
User's original terminal
          |
Unmodified Codex: native settings, login, permissions and tools
          |
Agency loopback capture: HTTP/SSE and WebSocket messages
          |                                  |
Original OpenAI endpoint                     Private archives
                                             SQLite exchanges/context
                                             Dashboard / inspector
```

Agency adds a session-local Responses provider with
`requires_openai_auth=true`. Codex supplies its native bearer and account
headers; its own login, account discovery and refresh still use its normal
services. The capture provider forwards model traffic carrying
`ChatGPT-Account-Id` to `https://chatgpt.com/backend-api/codex`; native API-key
traffic uses `https://api.openai.com/v1`. There is no LiteLLM fallback, model
substitution, gateway retry, or redirect following. This uses Codex's
[documented OpenAI-authenticated proxy support](https://learn.chatgpt.com/docs/enterprise/connect-to-a-gateway?gateway-auth=api-key).

A separate environment-backed Agency header authorizes the local route. That
header never reaches the upstream. Provider authentication passes upstream in
memory and is excluded from archives. A new bearer supplied by Codex after a
refresh is forwarded unchanged; HTTP and WebSocket handshake 401 responses pass
back to Codex without Agency attempting its own refresh.

Model selection, history, tools, permissions and compaction remain with Codex.
WebSocket transport is supported rather than forcibly disabled. HTTP errors,
partial streams and cancelled responses use the existing capture projection.
Hooks, tool policies, optional ptrace, resource profiling and workload scopes
continue independently of model capture. The default terminal is inherited;
Agency adds no PTY. `--terminal browser` remains an explicit alternative.

## Configuration and boundaries

This mode currently supports the built-in `openai` provider. It resolves routing
from the user configuration, selected profile and command-line overrides, and
rejects other selected providers, OSS mode and custom ChatGPT service routing.
It does not silently replace another provider's authentication.

An explicit `openai_base_url` or `OPENAI_BASE_URL` is retained as the original
upstream for either authentication type. Configure that destination only if it
is intended to receive your native credential. Upstreams require HTTPS, with
loopback HTTP permitted for local fixtures. Incoming requests cannot choose a
different upstream. Proxy environment variables and Codex's
`CODEX_CA_CERTIFICATE` / `SSL_CERT_FILE` CA bundle are used for upstream TLS;
certificate verification stays enabled. No certificate installation or TLS
inspection is needed for this Codex mode.

The session-local provider changes the provider identifier. Managed settings
can forbid that override. This mode does not claim compatibility with every
workspace's routing, residency rules, model catalog or authentication policy;
those require acceptance testing with that actual account and Codex version.
It does not capture model traffic deliberately routed through other providers
by independently configured agents or external services.

## Archives and context inspection

The ordinary private capture directories, SQLite MessageStore records and
context inspector are reused. Request bodies, unknown fields, tool schemas,
opaque state and streaming response content remain authoritative in the raw
archives. Model OTel records are suppressed during capture to avoid duplicates;
tool and lifecycle telemetry remain active.

For WebSockets, `request.bin` contains the original create-message payload.
`response.bin` concatenates the original response-message payload bytes.
`response.frames.jsonl` records each message's offset, byte length and text or
binary type as it arrives; the final manifest also contains that index.
WebSocket protocol control/heartbeat frames, TLS packets, compression framing
and TCP chunk boundaries are not archived. Unknown application messages and
handshake failures are retained as auxiliary captures. Non-generating prewarm
requests are auxiliary traffic rather than additional model exchanges.

Capture observes the context Codex actually transmits. It does not reveal
hidden server-side reasoning, decode encrypted compaction data, recover an
uncaptured earlier conversation, or implement context editing. Provider-side
references such as `previous_response_id` and compaction remain marked as
incomplete context. A raw archive can survive interruption or a process crash
even when the final SQLite projection has not been written.

## Validation scope

Automated tests cover native bearer/account forwarding, API and ChatGPT route
selection, arbitrary native model selection, 401 forwarding and a changed
bearer, local-route authorization, credential exclusion, HTTP/WS body fidelity,
multiple WS turns, errors, partial cancellation, SQLite/context reconstruction,
compaction limits and scope re-exec.

The opt-in real Codex test also supports `responses-native`, using only a
scripted local API-login fixture unless live LiteLLM testing is explicitly
enabled. Synthetic ChatGPT credentials establish request-construction mechanics;
they cannot prove real subscription billing, account entitlements, token refresh
or online account discovery. No real subscription account was used to validate
this change.

Validation on Linux with Codex 0.162.1:

* Full repository Python suite: **4,173 passed, 97 skipped**. Additional final
  session regressions: **187 passed, 5 skipped**. The full run preceded the last
  targeted interrupt/fixture additions; those additions passed separately.
* All pre-commit checks passed; all **20 JavaScript test files** passed.
* Real interactive Codex through native WebSockets: two user turns, two real
  `printf` tools, six model exchanges before the interruption/compaction checks,
  Escape cancellation with partial SQLite data retained, successful `/compact`,
  context inspection and a real systemd workload scope.
* A separate real Codex run loaded a synthetic ChatGPT login from its own private
  fixture home and completed a tool turn. The native bearer and account headers
  reached the upstream; Agency's local token did not. Online account integrations
  were disabled in this fixture and its processes were retired after the turn;
  this is not an online account-service shutdown or refresh test.

Both real-binary checks ran in a private network namespace with only loopback
enabled and synthetic credentials. Fixture tools ran without a nested Codex
sandbox because user namespaces hide host-root ownership of the trusted sandbox
helper. The production launcher does not change the user's sandbox or approval
settings. No live model calls or real subscription credentials were used.
