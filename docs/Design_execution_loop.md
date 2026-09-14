# Execution engine and harness lifecycle

The execution path is:

```text
agent.run() / agskill.run()
  -> GlobalAgentOrchestrator -> AgentEngine
  -> HostServerManager + sandbox HarnessManager
  -> adapter -> ptrace-supervised harness process
```

## Host and sandbox services

Each dispatched skill gets a fresh engine and a working copy of its predecessor
context. The engine lazily obtains the agent's sandbox and holds its lock for
input preparation, harness execution, service teardown, and commit or discard.
Host services provide model routing, interaction/policy callbacks, and host MCP
tools. The engine calls the sandbox daemon through a Unix socket; the daemon
calls host services through a separate Unix socket. Harness-facing HTTP and
sandbox MCP services live inside the daemon.

Each attempt receives a fresh token. The host accepts traffic only for its
bound token and holds a lease through the complete ASGI response. Retirement
closes admission and drains those leases before another token can be bound.
Output-schema repair may start another attempt within the same execution ID.

## Harnesses and controls

Claude Code, Codex, Grok Build, and OpenCode use interactive PTYs. Their adapters
own prompt acknowledgment, transcript/session parsing, interruption, and process
retirement. Native uses a standalone Python ReAct process with pipe output.
All harness processes use the ptrace supervisor; see [PTY.md](PTY.md).

`redirect(result, message)` resolves the result's exact engine and sends its
execution ID to the daemon. The engine serializes delivery with service
teardown. The daemon validates the ID and serializes delivery with process
controls and attempt retirement. An early, late, paused, unsupported, or failed
delivery becomes ordered future context. Native redirects use that queue path.
There is no host-side model-generation redirect inbox.

Pause and resume are persistent agent-wide process controls. They do not gate
scheduler admission. The engine synchronizes both paused and resumed states
when it obtains a daemon, using the same per-agent control lock as public
pause/resume calls. A paused active harness retains its execution capacity.

Cancellation first marks the owned request under the orchestrator condition.
A running request also receives an execution-scoped daemon kill RPC. A stale
RPC cannot kill another request. Cancellation after daemon admission but before
process registration is applied when that process registers. Queued cancellation
is observed when dependencies and capacity allow dispatch; it does not settle
unresolved input dependencies early.

## Transaction and context publication

Before commit, the engine atomically claims completion against cancellation.
If cancellation wins, the engine discards the working container and context.
If completion wins, later cancellation cannot invalidate that transaction.
A commit failure still fails the request and discards its working container.

An ordinary executed skill failure appends a rollback notice to a copy of the
committed predecessor. Cancellation passes the predecessor through without the
notice. On success, session blobs and retained-message cursors are published
only after commit. Stateful harnesses receive retained messages beyond their
cursor; stateless harnesses receive all retained messages on each call.

Host services close before commit. The sandbox hibernates after commit when
process tracking reports no pending background work. Native and PTY adapters
reap their process trees before removing attempt-local configuration.

The worker posts its completion to the scheduler. The scheduler settles context,
releases request/capacity bookkeeping, and then publishes the public result.
Fork and save hold a consistent context/checkpoint snapshot, retrying when a
submission advances the context while they wait for the sandbox lock.

## Observability

`agency/observability/agdatalogger.py` owns event and stream persistence. Provider
fragments enter temporary `stream_deltas` rows; stream finalization writes durable
events and clears temporary rows. Profiler spans are opt-in and belong to their
profiling session. See [Design_orchestrator.md](Design_orchestrator.md) for
scheduling, settlement, and shutdown.
