# Harness History & Continuity Design

> **Status:** implemented for Claude Code and Codex. Each invocation is still a fresh OS process,
> but compatible native session state is extracted to the agent and restored before the next call.
> Native state is an optimization: portable `agcontext.messages` remains the engine-neutral source
> of truth and the fallback when native resume is missing, stale, incompatible, or rejected. This
> document explains that design and the rejected request-splicing alternative.

## Constraint this design must satisfy

**History must remain a property of the *agent*, never of a specific sandbox instance.** This is
the same guarantee native agents already get: `agcontext` (`ag.ctx`) is a plain, portable object —
`agent.save()`/`load()` (`agent.py:691-841`) checkpoints it alongside (not inside) the sandbox's own
filesystem image, so an agent's history survives being reattached to a different sandbox, a
restored checkpoint, or a different backend entirely. Any design that makes a harness-driven agent's
continuity depend on state living *inside* a specific container's committed filesystem — e.g.
persisting the harness's own config/session directory as part of what `commit()`/`export_image()`
capture — violates this guarantee outright: it ties history to the sandbox, backwards from how the
rest of this framework treats the relationship. Whatever mechanism this document adopts has to
extract history out to the agent's own state, not leave it embedded in the container.

## Rejected: splicing `agcontext.messages` into the harness's outbound LLM request

The first design considered: since Agency never constructs the LLM call in the harness-driven path
(the harness's own internal code does), and the only agency-controlled seam into that traffic is
`agproxy_llm` (which already reshapes every request between the harness's wire format and the
uniform `chat.completions.create()` shape every backend expects), have the gateway splice
`ag.ctx.messages` into the *first* request of each new harness launch — tracked per bearer token,
prepended after any leading system message(s) and before the harness's own first turn, then passed
through unmodified for the rest of that invocation (the harness's own internal loop naturally
resends its growing conversation from there).

This is mechanically cheap to wire in — `/v1/messages`'s body is parsed at `agproxy_llm.py:176`,
translated via `anthropic_messages_to_openai` at `:190`, dispatched at `:202`; the splice point is
the gap between those two lines, and `register()`/`unregister()` (`:119-125`) already track
`token -> agent` per launch, so adding a "has this token's history already been spliced" flag is a
small addition. **It was built as a standalone probe and empirically tested against the real
translation code, and it breaks:**

- The function that builds every real outbound Anthropic Messages API request (both the direct
  Anthropic API and Bedrock) is `_openai_messages_to_anthropic`
  (`agllm_backends/anthropic.py:100-144`). It coalesces consecutive `tool`-role messages into a
  single `user` turn (`:132-142`, the only case any existing caller has ever produced, tested at
  `test_anthropic.py:338-370`) — it has **no logic at all** for merging consecutive plain
  `user`/`user` or `assistant`/`assistant` entries (`:111-131`), because nothing in the current
  codebase has ever needed to feed it non-alternating input.
- A synthetic `ag.ctx.messages` list was built with the shape realistic accumulated history
  plausibly has — a tool-call/result pair plus a couple of consecutive same-role turns — spliced in
  at the identified point, and run through that real function. Result:

  ```
  BROKEN: 2 strict-alternation violation(s) found:
    - index 4->5: consecutive 'user' messages
    - index 6->7: consecutive 'assistant' messages
  ```

  This is exactly the shape the real Anthropic Messages API rejects with a 400
  (`roles must alternate between "user" and "assistant"...`). No existing test in
  `test_agproxy_llm.py`/`test_agproxy_llm_adapters.py`/`test_anthropic.py` (109 tests, all passing)
  exercises this, because the current code never produces the input shape that breaks it.
- Separately, "first request of this launch" is not safely detectable by arrival order alone:
  Claude Code also calls `/v1/messages/count_tokens` (`agproxy_llm.py:205-226`) on the same token
  for context-size estimation, which never reaches the model — if splice-tracking isn't scoped
  strictly to the two real inference routes, that call can consume the "first" slot and the actual
  first inference request goes unspliced. `body = await request.json()` is also a genuine async
  yield point before any flag check would occur, so concurrent requests/retries can race past an
  unlocked boolean.

**Conclusion: rejected.** Making this work would require adding a same-role-run coalescing pass with
its own correctness surface (how do you merge two consecutive assistant turns that made different
tool calls without silently dropping one?) plus a real lock around splice-tracking — new,
un-precedented engineering with no existing pattern to lean on, for a mechanism that's structurally
fighting the wire format instead of using it. See the adopted approach below instead.

## Adopted: treat the harness's own native session storage as a portable blob

Where a harness exposes a verified continuation mechanism, extract its native on-disk state to the
*agent's* portable checkpoint after each call and re-inject it before the next one — regardless of
which sandbox instance handles that call. The blob format and resume command remain backend-local,
unlike the engine-neutral task/result/process abstractions.

### What was verified against the real `claude` CLI (v2.1.220 installed, not assumed)

- **Location and format.** Claude Code stores each conversation as
  `~/.claude/projects/<slug>/<session-id>.jsonl`, where `<slug>` is the launch's working directory
  with every non-alphanumeric character replaced by `-`. Confirmed against real files already on
  disk from earlier development testing: cwd `/data/tmp/agharness-alex_0000-3dl4c7mp` produced
  directory `-data-tmp-agharness-alex-0000-3dl4c7mp`. Each line is a structured record (`role`,
  `content`, `sessionId`, `cwd`, `version`, ...) — a real, already-existing JSON history of the run.
- **`CLAUDE_CONFIG_DIR` relocates the entire storage root**, including session files (confirmed via
  strings in the installed binary: `"CLAUDE_CONFIG_DIR=/tmp for ephemeral local writes"`), not just
  settings. This means sessions never need to touch the real host `~/.claude` at all.
- **Copying the file into a fresh directory and resuming actually works.** A real session file from
  an earlier test run was copied into a brand-new directory (standing in for "a different sandbox"),
  its matching slug computed, and `claude -p --resume <session-id> --model sonnet --output-format
  json --setting-sources "" "What exact instruction did I give you in the very first message of
  this session? Reply with only that instruction verbatim, nothing else."` was run from there. It
  answered `"Say hi in exactly two words."` — the exact original instruction from the copied
  transcript — with genuine prompt-cache hits (`cache_read_input_tokens: 24154`), not a coincidence.
- **Negative control confirms causation.** The identical command run from a different fresh
  directory with *no* copied session file failed immediately and cleanly: `No conversation found
  with session ID: b82d7947-6b6c-484c-b85a-f3772ee9722e` — not a silent fallback to something wrong.

### Concrete mechanism (Claude Code)

1. `claude_code.py` sets `"CLAUDE_CONFIG_DIR": str(config_home)` — `config_home` is
   already the isolated, disposable per-launch directory `materialize_config_home` creates
   (`agharness.py:29-35`) and already cleaned up unconditionally after every launch
   (`cleanup_config_home`, `agharness.py:38-39`), so this adds no new lifecycle to manage.
2. Before launch, if the agent's own saved state carries a session blob + session id from a prior
   call in this lineage, write the blob to `config_home/projects/<slug(cwd)>/<session_id>.jsonl` and
   add `--resume <session_id>` to `argv`.
3. After the run, read that path back out (the run may have appended to it) and store the bytes plus
   session metadata in `ag._harness_sessions`. `agent.save()` serializes that map alongside history,
   not inside `container.tar`, and `fork()` deep-copies it.
4. `cleanup_config_home` runs exactly as it does today — full removal, zero host trace, so the
   "leaves no trace in the user's own `~/.claude`" isolation goal (`agharness.py:6-9`) is unaffected.

### Codex rollout mechanism (verified with CLI 0.147.0)

Codex writes timestamped rollout JSONL below isolated `CODEX_HOME/sessions/`. Agency captures the
matching rollout after a completed turn and stores its bytes, relative path, thread id, context
revision, and version evidence in `ag._harness_sessions["codex"]`; fork and checkpoint operations
carry that record independently of the sandbox image.

Restore requires a canonical UUID, safe `sessions/.../*.jsonl` path, valid base64 and `session_meta`,
matching workspace/context revision, and the exact detected Codex/rollout version. Passing those
guards selects `codex exec resume <thread-id>` and avoids duplicating portable history. A failed
guard starts a fresh native thread with portable history. If Codex itself reports a recognized
resume-state failure, Agency retries that way once; unrelated failures are not retried. The live
container E2E verified capture, resume in a fresh container, and stale-version fallback.

### Interaction with running the harness inside the container

The project decided to run the harness process inside the sandbox container rather than keeping it
on the host behind a FUSE view (see [Design_harness_filesystem.md](Design_harness_filesystem.md),
now superseded, and [Design_harness_integration.md](Design_harness_integration.md)'s "Prerequisites"
for the adopted in-container supervisor bridge). A host `mkdtemp()` is invisible there, so
`materialize_config_home_in_container()` creates the config home inside docker/podman and each
backend points `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, or its equivalent at that path.

This is a net simplification for the history mechanism specifically: reading the session blob back
out after a run and writing it back in before the next one now goes through `agsandbox`'s existing
`read_file_bytes`/`write_file_bytes` (`agsandbox.py:277-286`) — the same primitives already used
elsewhere for sandbox file I/O — rather than plain host filesystem calls. The slug-computation and
`--resume` mechanics from the empirical validation above are otherwise unchanged: the slug still
keys off whatever `cwd` this launch uses, and `--resume` still doesn't care whether that path is
stable across launches, only that the file is present at the path matching *this* launch's `cwd`
before the CLI starts. The chroot backend needs none of this change — its harness launch already
runs against a real host directory (the jail), so `materialize_config_home`'s existing host
`tempfile.mkdtemp()` behavior is unaffected there.

## Fallback: `agcontext.messages` recap for cross-engine continuity

The native session file only ever covers that one engine's own turns. It's the wrong mechanism —
and shouldn't be attempted — when:

- the agent's history crosses between the native engine (its own in-container process, not the
  retired host-process `execute_react()`) and a harness, or between two different harness engines
  (the file is meaningless to the other side);
- the agent is deliberately moving to a materially different environment (fresh base image, clean
  workspace) where the old session's tool-call results reference files/state that no longer match —
  resuming natively there risks feeding the model stale, misleading context rather than helping it.

For these cases, fall back to what already exists: `build_harness_messages()` carries the relevant
prior `agcontext.messages`, and the selected backend renders them into the fresh task. This
is lossier (no step-by-step tool-call fidelity, no native prompt-cache reuse) but always available
and never depends on sandbox or environment compatibility — the correct degraded mode for a
boundary the native mechanism structurally can't cross.

## Verification scope

Claude Code and Codex have both been verified against real CLIs. OpenCode and Grok still need the
same live verification; their session-id support is not proof that complete on-disk state is
portable. Treat native resume as a per-engine capability, with portable history covering engines
that lack it.

## Open items — not yet verified or resolved

- **Format is undocumented and already shows version drift** — the sample record read during
  verification was stamped `"version":"2.1.217"`; the CLI installed in the same environment is
  `2.1.220`. Resume failing loudly on a future incompatible format change (parse error, or the same
  `No conversation found`) is an acceptable failure mode, but there's currently no way to distinguish
  "the format changed under us" from "the blob is simply missing" — worth guarding explicitly.
- **Native blobs grow with the conversation.** Codex rollouts are base64-encoded in agent state;
  long sessions still need a measured size/pruning policy.
- **Mixed native/harness history within one agent's lifetime has no unified representation.** The
  native session file covers only that engine's span; whether/how `agcontext` should note "a harness
  session covers this range, unavailable in raw message form" for logging/webui purposes is
  undesigned.
- **Schema retries require captured state.** Codex issues a bounded MCP-output correction only when
  the just-completed rollout was captured; otherwise it fails explicitly rather than retrying
  without context.
