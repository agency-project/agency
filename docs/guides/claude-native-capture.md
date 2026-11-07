# Claude Code capture with native authentication

Use your existing Claude Code sign-in and model selection while capturing its
Messages requests and responses:

```sh
uv run agency claude --capture-native
# Normal Claude Code arguments follow --:
uv run agency claude --capture-native -- --resume
```

The default terminal is inherited: Agency adds no PTY. Native settings, tool
permissions, conversation management and compaction remain with the real CLI.
Hooks, Agency tool policy, optional ptrace, profiling and systemd controls use
the same runtime as other interactive sessions.

## Traffic and authentication

```text
User's original terminal
          |
Unmodified Claude Code + native login
          | verified TLS for api.anthropic.com
Agency loopback HTTPS proxy
          |                          |
          | verified upstream TLS    +--> Raw bodies / SQLite / context UI
          v
Original Anthropic API

Other HTTPS destinations --> opaque CONNECT tunnel --> original destination
```

Claude Code uses its original API hostname. Agency does not replace its OAuth
token, API key, model or credential store. Authentication headers pass upstream
without entering the capture archive. Login/refresh connections to other hosts
remain encrypted end to end, and non-Messages API traffic is forwarded without
body archiving because it can contain credentials.

This uses Claude Code's documented [HTTPS proxy and additional CA
support](https://code.claude.com/docs/en/network-config). A separate per-session
proxy credential restricts the loopback listener; that credential never reaches
the provider. Native HTTP errors, including 401s, pass back without Agency
refreshing credentials, retrying, changing models or following redirects.
There is no LiteLLM fallback and `LITE_LLM_KEY` is unnecessary. The existing
`--capture --capture-model gpt-6-luna` remains a separate explicit LiteLLM mode.

## Certificate and network lifecycle

Agency generates a private session CA and a leaf certificate for
`api.anthropic.com`. Only the launched CLI environment receives the combined
`NODE_EXTRA_CA_CERTS` bundle. Nothing is installed in the system trust store,
and TLS verification remains enabled on both legs.

Certificate files live in a mode-0700 temporary directory outside the run logs,
with mode-0600 files. The CA signing key is never saved. Clean shutdown removes
the directory; a hard crash can leave a private temporary directory behind.
Certificates expire after seven days; restart capture for a longer session.
They are not part of the normal run export.

Existing shell and known Claude settings-file network configuration is retained:
additional CAs, bundled/system CA selection, client certificates and the
original HTTP(S) proxy and bypass rules. Relative certificate paths resolve
against the session directory. Claude's launch environment points to the local
capture proxy and bypasses local hooks/telemetry; upstream connections still
use the original proxy/bypass settings. The capture CA is never added to
Agency's upstream trust.

## Capture and boundaries

Only `api.anthropic.com:443` is decrypted. Messages and token-count bodies use
the existing private raw archives; the MessageStore projects model exchanges
into the existing SQLite context inspector. This retains tool schemas, history,
unknown fields, thinking/signatures, opaque content, compressed bodies, provider
errors and partial responses. OTel model records are suppressed during capture
to avoid duplicates; tool/lifecycle telemetry continues.

Capture supports the native HTTP/1.1 Messages/SSE transport. It does not archive
TLS packets, TCP/HTTP chunk boundaries, unrelated encrypted tunnels or login
bodies. API WebSocket upgrades are not supported by this mode; unrelated
WebSockets remain opaque tunnels.

The mode currently requires the native Anthropic endpoint. Alternate cloud
providers, custom API base URLs and disabled TLS verification are rejected
rather than silently changing their authentication. Managed settings can impose
additional restrictions. Existing background supervisors, remote/cloud-owned
work and independent processes that ignore this launch environment are outside
the owned session's capture/control guarantees.

Raw capture does not implement context editing, reveal server-hidden reasoning,
decode opaque provider state, or recover a conversation from before capture
started. Client-side compaction is visible as summarization and subsequent
summarized requests; earlier expanded context is available only if captured.
See the shared [capture and reconstruction limits](interactive-capture.md).

## Validation

Tests use local scripted responses and synthetic credentials. Coverage includes
OAuth/API-key forwarding, changed credentials, exact body/SSE capture, opaque
thinking state, gzip and chunked input, HTTP errors, cancellation, original CA
and proxy routing, private certificate cleanup, active-stream shutdown,
SQLite/context reconstruction and systemd re-exec.

The real Claude Code 2.1.299 acceptance test runs offline in a private network
namespace. It exercises multiple interactive turns, real `printf` tools, Escape
cancellation with partial data retained, `/compact` and a systemd workload scope.
The native Codex acceptance test is also rerun to check the shared capture path.
No real subscription credential or live model inference is used. Actual plan
billing, online OAuth refresh and account-specific policies remain unverified.

Validation record on this Linux machine:

* Both real native-auth CLI acceptance tests passed offline: Claude Code
  2.1.299 and Codex 0.162.1, with multiple turns, tools, cancellation,
  compaction and systemd scopes.
* The final session/thread-audit suite passed 213 tests, with six opt-in
  cases skipped. All 22 focused Claude TLS/proxy tests and all 20 JavaScript
  test files passed. Pre-commit checks passed.
* The full Python run had 4,197 passing tests and 99 skips. Its single failure
  was the new listener's missing service-thread audit entry. That entry was
  classified alongside the existing hook server, and all three thread-audit
  tests passed on rerun. Later certificate/settings refinements were checked
  with the targeted and real-CLI suites rather than another full run.
