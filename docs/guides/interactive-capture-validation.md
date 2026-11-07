# Linux interactive capture validation — 2026-11-06

This records the earlier browser-terminal capture stage. The current
inherited-terminal implementation and updated branch baseline are covered in
[the native runtime validation](inherited-terminal-validation.md).

Implementation is an uncommitted diff in
`/home/eric/Documents/Codex/agency-context-capture`, branch
`codex/interactive-context-capture`. Its base and fetched
`origin/eric/interactive-sessions` both equal
`e6a42a9dae65a1319f79d32d025c7eef2011689c`. The original checkout remains
on `eric/interactive-sessions`; existing work there was preserved. No new
commit, push, or merge was made. HANDOFF.md was not executed.

The working router credential was located in
`/home/eric/Documents/Codex/agency/.env` as `LITELLM_API_KEY`, paired with
`LITELLM_BASE_URL=https://router.js-park.info`. Validation explicitly loaded
that value as `LITE_LLM_KEY` in the test process. Secret values were not
printed, copied into source, or added to a new dotenv file.

## Results

| Check | Result |
| --- | --- |
| Sessions, context, and agwebui Python regression suite | 337 passed, 3 skipped |
| Final focused gateway suite | 27 passed |
| agwebui JavaScript suite | 205 passed |
| Live Claude Code 2.1.299 through native Messages | Passed two turns, two Bash tools, SQLite context checks, and `/compact`, explicitly selecting `gpt-6-luna` |
| Live Codex 0.162.1 through native Responses | Passed two turns, two shell tools through native code mode, SQLite context checks, and `/compact`, explicitly selecting `gpt-6-luna` |
| Full CLI entry / systemd re-exec | Both real CLIs separately completed a live Luna tool turn after actual Agency scope re-exec; each native terminal also had its own scope |
| Live database reconstruction | Six model exchanges and six wire archives per CLI, all HTTP 200, no cancelled model calls; original body hashes and native request reconstruction verified |
| Scripted cancellation / compaction | Both real PTYs passed Escape during a partial response and then `/compact`; nine exchanges/archives per CLI, including one cancelled partial response |
| Live stored-response lineage | Parent HTTP 200; `previous_response_id` continuation HTTP 200 and returned the remembered code; missing parent HTTP 403 preserved |
| Live native `/responses/compact` | HTTP 200, returned message and opaque compaction items, archived without translation; context marked incomplete |
| Browser context UI | Inspected both live runs in the actual browser: captured input, replies, tool results, read-only capture notice, idle lifecycle state, enabled scope controls, and sampled resources |
| Haiku native Messages on this router | **Upstream limitation:** HTTP 400 `unsupported_responses_model`, reproduced with a minimal request and the real Claude Code CLI; exact errors archived |
| Ruff check, format, git diff check | Passed |

Luna is the capture default for both CLIs because both native routes passed
live testing. Haiku remains an explicit option. The gateway never changes
an incoming Haiku request into a Luna request or retries it with another
model. The router's Haiku model group currently reports that stateful
Responses requests require a Responses-served model, even for a minimal
Messages request. This requires an upstream route/configuration change;
none was made.

The real CLI binaries were kept under `/tmp`. Claude came from its official
distribution with the manifest SHA256 verified. Codex's code-mode helper came
from the matching official 0.162.1 GitHub release, with its published digest
verified. The host's existing CLI installation was not replaced.

Claude and Codex's `/compact` commands used ordinary Messages and Responses
summarization calls respectively for these session configurations. Each CLI
reported completed compaction. The separate live native compact endpoint
probe also succeeded. Neither result makes encrypted compaction decodable.
The lineage probe demonstrates server history exists outside a subsequent
HTTP request: Agency captures the reference and marks the supplied context
incomplete, rather than claiming the history was present in that body.
Capture does not implement context editing or hydrate unseen server state.

Gateway tests cover arbitrary request fields, schemas (including Codex's
`additional_tools` envelopes), roles/history, unknown SSE events,
signatures/encrypted state, fragmented UTF-8, compressed bodies, HTTP errors,
connection failures, cancellation before headers and during a silent stream,
truncated streams, auxiliary endpoints, authentication, model enforcement,
and OTel deduplication. Tests also caught and fixed the existing Codex
conversation-start registration error and the stale waiting-for-model state
left by background title generation.

Live Codex testing identified normal client disconnects after
`response.completed`. These now remain successful model exchanges; the
manifest separately retains `client_disconnected`. Two regression tests cover
this behavior for Messages and Responses. Interrupted streams without a
terminal event remain cancelled, with their partial bytes and SQLite output
preserved.

## Evidence on this machine

Private generated artifacts are in the ignored directory
`artifacts/interactive-capture-validation/` in this worktree:

* `live-summary.json`: live exchange counts and hash/reconstruction checks.
* `live-luna/messages/`, `live-luna/responses/`: live SQLite databases, native
  request/response archives, manifests, and terminal text after compaction.
* `live-haiku-rejection/messages/`: the four native HTTP 400 exchanges from
  Claude Code, each preserved separately, without gateway retries/fallback.
* `live-state/results.json`, `live-state/capture/`, and its databases: stored
  parent, continuation, missing-parent error, and native compact results.
* `live-entry/summary.json`: actual re-executed Agency and terminal scopes,
  live turn/tool results, and run directories containing profiler evidence.
* `messages/`, `responses/`, `summary.json`: scripted real-PTY evidence,
  including Escape cancellation and subsequent compaction.
* `final-regression.log`, `gateway-final-live.log`, `live-luna.log`,
  `live-state-probe.log`, `pty-final.log`, `js-tests-final.log`: test results.

Session authentication/configuration files are omitted from copied acceptance
runs. Original archive references still name their original test paths.
The upstream credential was checked to be absent from the generated evidence.

To repeat the live native PTY checks, explicitly export the existing router
key as `LITE_LLM_KEY`, then run:

```sh
AGENCY_CAPTURE_LIVE=1 \
AGENCY_CAPTURE_TEST_CLAUDE=/tmp/agency-context-capture-tools/claude \
AGENCY_CAPTURE_TEST_CODEX=/tmp/agency-merge-validation/clis/codex/codex \
AGENCY_CAPTURE_COMPACT=1 \
uv run pytest tests/sessions/test_capture_interactive.py \
  -o tmp_path_retention_policy=all --timeout=300 -v -s
```

Only `https://router.js-park.info` and the allowed Luna/Haiku models were used
for live model calls. No credentials or models from another provider were
used as fallbacks.
