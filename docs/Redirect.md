# Redirecting a skill execution

```python
result = ag.run(skill, skill_input)
ag.redirect(result, "Use the corrected requirements")
result.wait()
```

`run()` returns the existing `agdata`, with private execution identity attached
by the orchestrator. `redirect()` requires that result and its owning agent.
Identity survives `wait()`, `await`, and field access and is excluded from output
serialization. There is no public invocation object.

A redirect either reaches that execution's active harness or calls
`queue_message()` once. Early, late, unsupported, paused, and failed deliveries
use the queue. Enqueueing does not wait for the target and does not change
already submitted work. The next submission after the enqueue receives the
message through the existing ordered context chain.

The orchestrator resolves only the target's own engine. The engine's existing
service lock serializes redirect RPCs with teardown, avoiding requests to a
hibernated daemon. The daemon checks the execution ID and performs delivery
under its existing control lock. Attempt retirement clears the callback under
that lock; its existing attempt lock prevents another attempt starting first.
An RPC delayed past the target's lifetime is rejected even if another run is
active. Output-repair attempts carry the same execution ID.

Claude's adapter owns a fresh traced PTY process per attempt and resumes native
conversation files using the existing session-blob protocol. It waits for native
startup input, submits bracketed paste plus Enter, and requires an exact
`UserPromptSubmit` hook acknowledgment. Redirect sends one Ctrl-C, waits for a
new native interruption record (or clears Claude's restored input draft with
double Escape), and submits the new prompt. Control characters that would turn
message text into keyboard commands fall back to queued context.

Native Stop detection and redirect acceptance share an adapter lock. Completion
requires the current submitted prompt and final assistant response to be flushed
to the transcript. A failed or unacknowledged partial redirect terminates its
process before returning failure. Claude HTTP cancellation propagates upstream
so an interrupted model request closes instead of keeping a blocked worker.
Pause time is excluded from terminal readiness/completion deadlines.

No other adapter registers the optional delivery callback yet. They receive
queued context; implementing their native delivery callback requires no changes
to the public API, scheduler, or RPC protocol.

Tests:

```sh
pytest tests/test_redirect.py tests/harness/test_claude_pty.py tests/harness/test_traced_pty.py
# Linux, installed Claude and Docker; uses synthetic model responses, no API key:
AGENCY_TEST_CLAUDE_PTY=1 pytest tests/harness/test_claude_pty_live.py
```
