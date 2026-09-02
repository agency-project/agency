# Execution loop design

This document follows one engine-backed request after the global scheduler admits it. Submission ordering and dispatch policy are described in [Design_orchestrator.md](Design_orchestrator.md); the caller-facing controls are described in [Invocation_API.md](Invocation_API.md).

## Ownership and transaction boundary

The orchestrator creates a fresh `AgentEngine` for each eligible request and dispatches that request onto a reusable execution-worker pool. A worker receives the exact public `Invocation`, copies the committed predecessor context into a working context, materializes resolved input dependencies, and only then lazily creates the agent sandbox and harness infrastructure.

The engine owns one sandbox transaction. It holds the sandbox lock while it starts host services, runs the harness, validates output, closes services, and either commits or removes the working container. The orchestrator remains the authority for request state, capacity, result publication, and engine lifetime; `agent.engine` is only an active/latest compatibility reference.

Each request therefore has two isolated views:

- the committed predecessor context and last successful sandbox checkpoint;
- a working context and container transaction that are publishable only after successful completion.

An ordinary skill failure discards the working transaction and publishes a clean predecessor copy with the canonical rollback notice. Cancellation and destruction discard work without that notice.

## Safe-boundary controls

The exact invocation is propagated through `AgentEngine`, `HostServerManager`, the host interaction service, and native or external harness control paths. Pause, suspension, cancellation, destruction, and invocation messages are observed only at explicit boundaries:

| Boundary | What has completed | Control behavior |
|---|---|---|
| Before infrastructure | Fresh engine and worker job exist; contexts are copied | Cancel or destroy avoids sandbox and service creation |
| Before harness execution | Sandbox lock acquired | Pause/suspension may park; cancel/destroy discards |
| Before model generation | Prior history/tool result is complete | FIFO invocation messages may be attached when the protocol permits a user turn |
| After model output / before tools | Provider call is complete | Cancel/destroy is observed; a final answer atomically hands off any in-flight messages or establishes the closing fence |
| After each tool result is published | Tool call is complete | Pause/cancel/destroy is observed before another model turn |
| Before sandbox commit | Harness attempt returned and host-side services are closed | Completion must be claimed atomically; the sandbox daemon may remain until hibernation or removal |
| Before public success | Commit and session update succeeded | Orchestrator rechecks the same monotonic completion claim |

There is no asynchronous interruption halfway through a provider request or tool function. A pause can remain parked indefinitely at a boundary; execution-owned RPCs do not expire the attempt merely because a valid pause lasts longer than a readiness probe.

At a final model result, Agency atomically checks the exact invocation inbox. If a message was accepted while the provider call was in flight, the LLM service performs a follow-up generation with that message before it may close. Otherwise the invocation changes to `closing`. New messages and pause requests are rejected after this fence. Cancellation and destruction race with a monotonic completion claim: if either control wins, the container and working context are discarded; if completion wins, a later control cannot retroactively invalidate the already claimed transaction.

## Native and external harnesses

The native ReAct loop checkpoints before generation, after internal compaction, after model output, and after each published tool result. Internal compaction requests explicitly bypass ordinary invocation messages, but a post-compaction checkpoint still observes pause, cancellation, and destruction before task generation.

External harnesses send model traffic through the invocation-bound LLM service. The service fingerprints the semantic request to derive a stable boundary ID, admits invocation messages only where adding a user turn preserves tool-call pairing, and retains the assigned overlay by sequence and a separately derived history anchor. An identical provider or CLI retry reuses that boundary assignment instead of consuming a message twice. If external compaction removes an old anchor, an already admitted instruction moves to the current generation instead of disappearing. A message that arrives during a final provider call is assigned at an atomic post-model boundary and produces a traced follow-up generation.

Both paths use the same `Invocation` control state. A harness cannot select another invocation by supplying an ID.

## Attempt isolation and disconnects

Every harness attempt receives a fresh unguessable token in `HarnessAttemptRequest`. The host manager binds exactly one token, and middleware requires it for LLM, interaction, and MCP traffic. Authorization is leased for the complete ASGI request or streaming response lifetime. Clearing a token first retires admission, waits for its in-flight leases to drain, and only then permits a successor, so missing, unknown, stale, or late traffic cannot reach a later attempt.

External harness launchers have a separate, narrowly scoped one-shot authorization for their exact initial executable path, PID, and arguments. Descendant execution remains subject to normal policy.

Checkpoint HTTP handlers watch for caller disconnects. A disconnected client aborts its interruptible boundary wait and wakes the underlying condition without consuming or assigning an invocation message. LLM stream shutdown closes provider streams, wakes producers and consumers on both generic errors and cancellation, joins their workers, and leaves failed close operations retryable.

## Streaming telemetry

The lowercase `agdatacollector.py` collector remains authoritative. Each provider fragment is appended to the temporary `stream_deltas` table in arrival order. Exactly one success, error, or cancellation finalizer clears those temporary rows and writes durable events under the existing call label. Lifecycle delivery does not bypass or replace collector streaming.

## Retained context and sessions

An ordered `queue_message()` entry lives in `agcontext.retained_messages` independently of the latest transcript. Before a harness attempt, the engine prefixes entries beyond that harness's retained-message cursor.

When a successful stateful harness returns a restorable session, the engine stages the session blob and the highest incorporated retained sequence. It publishes both only after sandbox commit succeeds. Cancellation or destruction that wins before the completion claim, along with harness failure, schema failure, rollback, or commit failure, clears the staged update and cannot advance the cursor. A later control that loses to an existing completion claim is a no-op, so the already-claimed commit and session publication proceed. Stateless harnesses have no advancing session cursor, so retained messages remain available on later calls.

Save, load, fork, and context copying preserve transcripts, session blobs, retained messages, and per-harness cursors.

## Settlement and teardown

After execution returns, the worker posts a completion event; it does not publish futures itself. The scheduler event thread then settles the output-context future, finalizes invocation and request bookkeeping, releases the agent and global capacity slot, and only afterward settles the public result future. Callbacks therefore observe a settled context and may safely inspect history or submit follow-up work.

Host services and clients close idempotently even when startup, provider streaming, sandbox commit, or an earlier close step fails. A failed transaction removes its working container. Explicit orchestrator shutdown closes admission, settles work that cannot run, drains active executions and asynchronous context continuations, joins the scheduler, and retires the reusable worker pool.
