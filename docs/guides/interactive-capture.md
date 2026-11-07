# Interactive model context capture

For capture that retains your existing sign-in and model selection, use
[`agency codex --capture-native`](codex-native-capture.md) or
[`agency claude --capture-native`](claude-native-capture.md). The `--capture` examples below
describe the separate LiteLLM route.

`agency claude` and `agency codex` launch the real interactive CLI in your
existing terminal. Capture is optional and independent of the terminal mode:

```sh
# Export LITE_LLM_KEY using your existing secret configuration first.
uv run agency claude --capture --capture-model gpt-6-luna
uv run agency codex --capture --capture-model gpt-6-luna
```

Agency forwards native Messages or Responses HTTP traffic to
`https://router.js-park.info`. The CLI only accepts these two capture models;
Luna is the default for both CLIs, validated live through their native HTTP
interfaces. Haiku remains an explicit choice; this router currently rejects
even a minimal native Haiku Messages request with `unsupported_responses_model`.
That error is preserved rather than retried with another model. Missing
`LITE_LLM_KEY` fails before launch. There is no credential, provider, model, or
protocol fallback. Put CLI arguments after `--`, as with ordinary interactive
sessions. The flags survive Agency's systemd scope re-exec.

An existing local setup may name the router key `LITELLM_API_KEY`; explicitly
export it as `LITE_LLM_KEY` before starting capture. The gateway does not search
or load dotenv files or silently select another credential.

The CLI authenticates to the loopback capture route with a per-session token.
Agency replaces that credential with `LITE_LLM_KEY` at the upstream boundary
and removes the upstream key from the child environment. Claude's alternate
cloud-provider modes are disabled in capture mode. Codex uses a session-local
Responses provider, HTTP streaming, and `--no-daemon` so it cannot reuse a
daemon configured for another session. Normal CLI settings, hooks, terminal
interaction, and tool permission decisions continue to apply. A desktop
parent's chat identity and tool pipe are removed from the child environment.
Codex code mode needs the matching `codex-code-mode-host` installed alongside
the CLI; capture does not supply or replace that runtime.

## Recording and reconstruction

Each authenticated supported request has a private directory under
`<run>/logs/capture/<call-label>/`:

* `request.bin`: original request body bytes, before forwarding.
* `response.bin`: original upstream body bytes, appended and flushed as they
  arrive, including SSE frames, unknown events, errors, and partial responses.
* `manifest.json`: method, path, query, safe headers, timestamps, HTTP status,
  body hashes, response length, cancellation, parsing errors, and coverage.

The capture directories and run's logs directory are mode 0700; capture files
are mode 0600. The logs directory also protects the SQLite projection. Bodies include the user's
complete supplied context and may contain sensitive content. Authentication
and cookie headers are excluded. These files record HTTP entity bodies, not
TLS packets or transport chunk boundaries. Hop-by-hop headers are removed;
other native protocol headers and bytes pass through. No SDK translation,
retry, redirect following, history fitting, or request editing occurs.

At completion or cancellation, Agency projects visible text, tools, tool
results, history, and schemas through the shared MessageStore into SQLite
`message_block_lookup`, `message_structure_lookup`, `per_exchange_messages`,
and `exchanges` records. Message roles, boundaries, extra fields and positions are
retained. Response metadata also stores the complete parsed native request
and reconstructed response, plus an archive reference. The byte archive is
authoritative for unknown frames and opaque provider state. Projection
failures do not alter the forwarded response. A process crash can leave an
unfinished archive without a final SQLite exchange; completed records do
not require the CLI's own transcript to reconstruct their captured input.

The context inspector uses the existing SQLite records and paginates long
input. Native provider metadata and hidden reasoning remain outside the
ordinary visible context UI. Raw archives preserve provider state. Model
OTel events are ignored only in capture mode to avoid duplicate model
records. Tool and lifecycle hooks/OTel remain active. Gateway calls are
attributed to the session's parent agent; a native HTTP request does not
reliably identify which subagent sent it.

Token-count and retrieval/cancellation requests are archived as auxiliary
traffic rather than counted as extra model generations. HTTP errors and
incomplete streams produce error exchanges; disconnects close the upstream
connection and retain the received bytes as cancelled exchanges. Closing a
connection cannot guarantee a remote provider has stopped computation.

## Context limits

Capture records the input actually sent by the CLI. It **does not implement
context editing**, decode provider state, or hydrate server-side history.

* `previous_response_id` and `conversation` refer to provider-side context
  absent from the current HTTP body. They pass through unchanged, and the
  context UI marks coverage incomplete. A missing/expired parent error passes
  through without a retry that invents a replacement conversation.
* `/v1/responses/compact` and native compaction items pass through unchanged.
  Encrypted compaction data is opaque. Such records are marked incomplete;
  the earlier expanded context cannot be recovered by decoding that data.
* Client-side compaction appears as the CLI's summarization request and
  subsequent summarized input. Earlier requests are available only if this
  Agency session captured them. Resuming a previously uncaptured session
  cannot recover earlier calls.
* LiteLLM or its selected model may reject compaction, stored responses,
  unknown native fields, or a CLI beta protocol. Capture preserves that
  rejection; it does not claim those upstream capabilities or emulate them.
* Codex WebSocket model transport is disabled for this capture provider.
  Explicit `--terminal browser` mode uses its usual terminal WebSocket relay.

The [Responses conversation-state guide](https://developers.openai.com/api/docs/guides/conversation-state)
and [compact endpoint reference](https://developers.openai.com/api/reference/resources/responses/methods/compact)
describe the provider-side mechanisms; their availability through a particular
LiteLLM route must be tested separately.

## Reproducible checks

```sh
uv run pytest tests/sessions tests/agwebui -q
node --test tests/agwebui/*.test.mjs tests/agwebui/*.cjs
```

The browser-mode opt-in acceptance test launches unmodified interactive binaries,
attaches through the terminal WebSocket, submits two prompts, executes shell commands,
and checks SQLite context and systemd scopes. By default its upstream is a
local scripted native protocol server, with no model inference:

```sh
AGENCY_CAPTURE_TEST_CLAUDE=/path/to/claude \
AGENCY_CAPTURE_TEST_CODEX=/path/to/codex \
uv run pytest tests/sessions/test_capture_interactive.py -v -s
```

The scripted mode also interrupts an open native stream with Escape and
checks the partial SQLite response. Add `AGENCY_CAPTURE_COMPACT=1` to exercise
the CLI's `/compact` command and require its completed-compaction notice.

To run the same acceptance test against the specified LiteLLM router, export
`LITE_LLM_KEY` and add `AGENCY_CAPTURE_LIVE=1`. It selects Luna for both CLIs.
`AGENCY_CAPTURE_TEST_CLAUDE_MODEL=claude-haiku-5-5` explicitly tests Haiku's native
route, including its current upstream rejection. Only Haiku and Luna are allowed.
The script requires the model to actually call tools, so a successful prose
reply alone does not pass. Test archives contain actual CLI prompts; keep
them private when sharing evidence.

The [runtime guide](interactive-runtime.md) describes the default inherited
terminal and shared tracing/policy. The [native validation report](inherited-terminal-validation.md)
records direct-terminal, real-shell job-control and three-way live checks.
