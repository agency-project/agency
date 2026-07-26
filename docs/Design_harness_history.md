# Harness History & Continuity Design

> **Status:** proposed, not yet implemented. Extends
> [Design_harness_integration.md](Design_harness_integration.md) — read that first. This document
> covers a gap the shipped design doesn't address at all: a harness-driven agent has no continuity
> across separate invocations today. Every call to a harness backend (`claude_code.py`, `codex.py`,
> ...) launches the harness binary as a fresh, one-shot process; whatever happened inside is
> collapsed to exactly `prev_ctx.messages = [user_msg, assistant_msg]` (`claude_code.py:155`) before
> being handed back. There is no session id, no `--resume`, no cross-call memory of any kind in the
> shipped code — this document designs that in, and rejects one plausible-looking approach along
> the way for a concrete, empirically demonstrated reason.

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

Every harness already has its own internal multi-turn continuation mechanism (`--resume` for Claude
Code, and presumably an equivalent for Codex/opencode/Grok — not yet verified, see "Scope" below).
Rather than reconstructing continuity ourselves, extract that mechanism's on-disk state out to the
*agent's* own portable checkpoint after each call, and re-inject it before the next one — regardless
of which sandbox instance handles that next call. This reuses machinery the harness has already
built and tested, at the cost of it being harness-specific rather than uniform across engines (the
opposite tradeoff from Components 1–3, which are deliberately harness-agnostic).

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

### Concrete mechanism

1. `claude_code.py`'s `envp` gains `"CLAUDE_CONFIG_DIR": str(config_home)` — `config_home` is
   already the isolated, disposable per-launch directory `materialize_config_home` creates
   (`agharness.py:29-35`) and already cleaned up unconditionally after every launch
   (`cleanup_config_home`, `agharness.py:38-39`), so this adds no new lifecycle to manage.
2. Before launch, if the agent's own saved state carries a session blob + session id from a prior
   call in this lineage, write the blob to `config_home/projects/<slug(cwd)>/<session_id>.jsonl` and
   add `--resume <session_id>` to `argv`.
3. After the run, in the existing `finally` block, read that same path back out (the run may have
   appended to it) and store the bytes plus the session id as part of the agent's own portable
   state — a new field in `agent.save()`/`load()`'s `state.json` (`agent.py:691-841`) alongside
   `history`/`sandbox_image_kind`, not inside `container.tar`. This is what makes it portable: the
   blob travels with the agent's checkpoint, not with any particular sandbox's committed image.
4. `cleanup_config_home` runs exactly as it does today — full removal, zero host trace, so the
   "leaves no trace in the user's own `~/.claude`" isolation goal (`agharness.py:6-9`) is unaffected.

### Interaction with running the harness inside the container

The project decided to run the harness process inside the sandbox container rather than keeping it
on the host behind a FUSE view (see [Design_harness_filesystem.md](Design_harness_filesystem.md),
now superseded, and [Design_harness_integration.md](Design_harness_integration.md)'s "Prerequisites"
for the adopted in-container supervisor bridge). That changes where `config_home` itself has to
live: `materialize_config_home` (`agharness.py:29-35`) creates a host-side `tempfile.mkdtemp()`
today, which the harness can no longer see once it's launched inside the container's own filesystem
namespace via the `docker exec`-based entrypoint. For a docker/podman-backed launch, config-home
materialization needs to create its directory *inside* the container instead — via the sandbox's
existing `exec()`/file-write primitives (`agsandbox.py:274`, `write_file`/`write_file_bytes`,
`agsandbox.py:283-286`), not a host `mkdtemp()`. `CLAUDE_CONFIG_DIR` then points at that in-container
path, and `cwd` for the launched process is whatever real in-container directory the entrypoint
uses (no FUSE mount, no virtual root — it's simply a real path in the container's own filesystem,
the same way native tool execution already sees it via `_container_exec`).

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

- the agent's history crosses between native `execute_react` and a harness, or between two
  different harness engines (the file is meaningless to the other side);
- the agent is deliberately moving to a materially different environment (fresh base image, clean
  workspace) where the old session's tool-call results reference files/state that no longer match —
  resuming natively there risks feeding the model stale, misleading context rather than helping it.

For these cases, fall back to what already exists: build the new call's prompt by recapping the
relevant prior `agcontext.messages` as text, through the same `build_user_turn_prompt`/
`_build_user_content` path that already constructs every call's prompt (`agharness.py:42-47`). This
is lossier (no step-by-step tool-call fidelity, no native prompt-cache reuse) but always available
and never depends on sandbox or environment compatibility — the correct degraded mode for a
boundary the native mechanism structurally can't cross.

## Scope: verified for Claude Code only

Everything empirical in this document was checked against the real `claude` CLI. Codex, opencode,
and Grok Build each need the same kind of direct verification against their own binaries — session
storage location, on-disk format, and `--resume`-equivalent semantics may all differ, and shouldn't
be assumed to generalize from this one case. Treat the native-session-blob mechanism as a per-engine
capability, added one backend at a time as each is actually checked, with the `agcontext` recap
fallback covering any engine that hasn't been verified yet (or never gains an equivalent at all).

## Open items — not yet verified or resolved

- **Format is undocumented and already shows version drift** — the sample record read during
  verification was stamped `"version":"2.1.217"`; the CLI installed in the same environment is
  `2.1.220`. Resume failing loudly on a future incompatible format change (parse error, or the same
  `No conversation found`) is an acceptable failure mode, but there's currently no way to distinguish
  "the format changed under us" from "the blob is simply missing" — worth adding before this ships.
- **The exact `state.json` field(s)** for the session blob and session id aren't drafted — needs to
  fit alongside the existing `history`/`sandbox_image_kind` keys `agent.save()` already writes.
- **Mixed native/harness history within one agent's lifetime has no unified representation.** The
  native session file covers only that engine's span; whether/how `agcontext` should note "a harness
  session covers this range, unavailable in raw message form" for logging/webui purposes is
  undesigned.
- **Schema-retry turns issued against an existing `--resume`d session** (the reprompt seam described
  in Component 5 of the parent document) have only been verified for a single before/after copy —
  not for whether multiple retry turns compound cleanly against the same resumed session.
