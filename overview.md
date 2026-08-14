# Native engine execution trace — `examples/base_example.py` (current `master`)

This is a step-by-step trace of what happens when `base_example.py` runs its two skill calls
(`file_skill` then `qa_skill`) through the `native` engine, with `filename:line` citations against
the codebase as of commit `ead3698` on `master`. This supersedes an earlier trace done against the
`harness_abstration` branch — quite a bit changed in the merge (a profiler layer, shared-service
factory functions, and — most importantly — the sandbox now hibernates between skill calls instead
of staying warm).

## 0. Setup

`**base_example.py:86**`: `ag = agent(agconfig=cfg)` — no `engine=`. `**agent.py:296**`:
`self.engine = engine if engine is not None else _AgAgentFields(self.agconfig).engine`, and
`_AgAgentFields.engine`'s descriptor default is `"native"` (`**agent.py:50-56**`). The whole
construction runs inside `with agprof.span("agent:create")` (`**agent.py:248**`) — a no-op span
unless a profiler session is active (base_example.py never sets `AGENCY_PROFILE`, so this and every
other `agprof.span(...)` below is inert).

`file_skill`/`qa_skill` (`**base_example.py:55-83**`) are unchanged in shape from before —
`qa_skill` still sets `replace_tools=[]`.

## 1. `ag.run(file_skill, ...)` — `base_example.py:89`

`**agent.py:575-583**`: `run()` delegates to `skill.run(self, skill_input)` → `**agskill.py:234**`.

- **Lines 245-249**: `prev_ctx = ag.ctx`; builds `result_future`/`ctx_future`; tags both with
`agpause.tag_producer`.
- **Line 473**: `ag._set_ui_state("skill", skill=self.name)` is called **synchronously, before the
worker thread even starts** — new since the last trace, specifically to avoid a race where
`is_settled()` could see a stale `"inactive"` state for a run that hasn't started yet.
- **Lines 475-495**: `_traced_task()` wraps `_task()` in an `agprof.span(...)` (labeled
`run{N}:{skill_name}:{agname}`) that also annotates run/agent/parent-agent IDs and, once
`result_future` resolves, an outcome (`success`/`failure`). Started via
`agprof.spawn_traced(_traced_task).start()` (**line 495**) rather than a bare `threading.Thread` —
profiler-aware but otherwise the same non-blocking daemon-thread launch as before.
- **Lines 496-497**: `ag.ctx = agcontext(_future=ctx_future)`; returns `agdata(_future=result_future)`
— `ag.run()` is still non-blocking.

Inside `_task()` (`**agskill.py:252**`):

- **Lines 269-270**: `prev_ctx.resolve_prev_dependencies()` / `skill_input.resolve_input_dependencies()`,
inside `agprof.span("resolve")`.
- **Line 283**: defensive shallow copy of `skill_input`.
- **Line 287**: `ag._check_pause(self.name)`.
- **Lines 291-303**: sandbox provisioning, inside `agprof.span("sandbox:provision")` —
`ag.sandbox = agSandbox(ag.agname, agconfig=sb_cfg)` on first run.
- **Lines 308-309**: acquires `ag.sandbox._lock` for the whole skill call.
- **Lines 327-332**: `self.execute_harness(ag, prev_ctx, local_skill_input, max_steps)`.

## 2. Sandbox construction — `agsandbox.py:121`

`agSandbox.__init_`_ (inside `agprof.span("sandbox:create")`) resolves `base_image`/`mounts`
(**lines 170-175**), then unconditionally adds the same three bind mounts as before —
`_agharness_llm_gateway` (rw), `_agharness_bin_cache` (ro), `_agency_package` →
`AGENCY_PACKAGE_CONTAINER_MOUNT` (ro) (**lines 197-219**). **Line 221**:
`self._backend = agsandbox_backend.for_config(...)` — still just builds the facade; no real
container yet. Every subsequent sandbox operation (`exec`, `read_file`, `write_file`, `commit`,
`stop`, ...) is now wrapped in its own `agprof.span(...)` (**lines 341-393**).

## 3. `execute_harness()` — `agskill.py:522`

- **Lines 564-569**: input validation.
- **Lines 572-584**: `agprof.span("input:prepare")` around `prepare_inputs_in_sandbox(...)` — for
`file_skill`, `file_path: agpath` needs no sandbox handling (pass-through), `task: str` is too
short to offload.
- **Line 597**: `backend = agharness_backend.for_config(ag.engine, ag.agconfig)` →
`**base.py:112-133`** → `"native"` → `_NativeBackend(agconfig)` (**line 120-121**).
- **Lines 599-601**: `backend.execute(ag, prev_ctx, skill_input, max_steps, skill=self, extra_system=None)`.
- **Lines 602-603**: `finally: ag.sandbox.remove_files(_offloaded_paths)` (empty here).
- **Lines 604-640**: since the result isn't an `agerror` and `ag.engine == "native"`, calls
`agSandbox.wait_for_processes(ag.sandbox, ...)` (**line 629**) — no background jobs here, so
`_has_pending_background_work()` is already `False` and it returns immediately. Note: unlike
`execute_react()`'s old convention, **the returned status message from `wait_for_processes()` is
discarded here**, not injected into the conversation — not relevant for this example (no
background job was ever started), but a real, current gap. Then **line 640**:
`self.output_schema.recover_outputs(result, ag.sandbox)`.

## 4. `_NativeBackend.execute()` — `native.py:372`

- **Lines 386-416**: `suppress_builtins = skill.replace_tools is not None` → `False` for
`file_skill`; `custom_tool_objs = skill.add_tools or []` → `[]`; loop over `custom_tool_objs` does
nothing → `custom_tools_payload = []`.
- **Lines 425-438**: builds `sys_msg` (`skill._build_system_prompt(None)`, `**agskill.py:96`**,
unchanged shape) and `user_msg` (`skill._build_user_content(...)`, `**agskill.py:157**` — note
**line 188**: the plain-JSON fast path now prefixes with `"[HARNESS SYSTEM] New Skill Input:\n"`, a
small wording change from before). Since this is structured output (`_use_structured_output=True`),
**line 435**: `agharness.build_mcp_output_format_instruction(skill)` appends the `submit_output`
-tool instruction text.
- **Line 440**: `sock_path = _ensure_entrypoint(ag.sandbox)` → `**native.py:282`**. No
`sandbox._native_entrypoint_sock` yet → `**launch_in_container_entrypoint(sandbox)**` (**line
193**):
  - **Line 218**: `ensure_python_packages_in_container(sandbox, ["mcp", "html2text", "cloudpickle"], timeout_s=180)` — this call's own `sandbox.exec(...)` is what actually triggers the real
  `docker run` (lazy start, unchanged mechanism).
  - **Lines 223-256**: mints a UDS path, builds `echo $$ > pid_path; exec python3 <entrypoint> <container_sock_path>`, and calls `sandbox.exec_detached(...)`.
  - **Line 258**: `_wait_ready` polls `ping()`.
  - **Lines 260-264**: reads back the entrypoint's PID and calls
  `sandbox.release_daemon(entrypoint_pid)`.
- **Lines 442-464**: gets the three shared bridge singletons via the new factory functions —
`get_shared_terminus(ag.agconfig)`, `get_shared_mcp_server(ag.agconfig)`,
`get_shared_messenger(ag.agconfig)` — plus, new this round, `get_shared_profiler_ingest()`
(**line 452**). `token = uuid.uuid4().hex`; `profile_native_events = agprof.enabled()` → `False`
here, so `profiler_host_sock = None` (**line 460**) and the `AGENCY_PROFILE`-env-var warning at
**lines 455-459** doesn't fire (env var unset). **Lines 461-464**: registers the token against
`terminus`, `profiler_ingest`, `mcp_server`, `messenger`.
- **Lines 469-472**: starts the `_LiveTranscriptPusher` background thread (unchanged mechanism —
polls `terminus.transcript_for_token(token)` every 250ms).
- **Lines 474-493**: builds the request dict each loop iteration — now also carries
`"profiler_sock": None` and `"profiler_turn_offset": 0` (both inert here) alongside `token`/the
three bridge sock paths/`model`/`messages`/`max_steps`/`custom_tools=[]`/`suppress_builtins=False`.
- **Line 494**: `resp = run_react_loop(sock_path, request)` — blocks on the socket call.

## 5. Inside the container — the entrypoint

Same launch mechanism as before, but the request handling now flows through a small
profiler-aware wrapper:

- `**_run_react_loop(req)`** (`**_native_in_container_entrypoint.py:1062**`): constructs
`profiler = _agprof_emit.RemoteProfilerEmitter(req.get("profiler_sock"), req["token"])` — with
`profiler_sock=None`, this emitter is a no-op sink (every `profiler.span(...)` call below returns
a context manager that does nothing). Calls `**_run_react_loop_inner(req, profiler)**` inside a
`try/finally` that always calls `profiler.close()` and stamps
`response["profiler_dropped_events"]`.
- `**_run_react_loop_inner**` (**line 1074**): `context_limit = _fetch_context_limit(...)`; builds
the tool list (**lines 1109-1126**) — `suppress_builtins=False` → `dispatch`/`tools`/`_have_tool`
seeded from `_TOOL_DISPATCH`/`_BUILTIN_TOOL_SCHEMAS` (`bash`/`read`/`write`/`edit`/`glob`/`grep`/
`webfetch`/`todowrite`); MCP tools merged in via `_mcp_tool_schemas` (`reserve_cpu`/`cpu_release`/
`daemon_release`/`submit_output`/`ask_human`); `custom_tools` is empty → nothing added.
- Main loop (**lines 1129-1218**), one iteration per `for step in range(max_steps)`, each wrapped
in `profiler.span(f"turn{turn_index}", ...)` (a no-op here):
  - **Lines 1138-1139**: `_check_in(messenger_sock, token)` — no pending inbox.
  - **Lines 1140-1148**: `_maybe_compact(...)` — no-op (tiny history).
  - **Line 1152**: `_dispatch_via_terminus(terminus_sock, token, kwargs, profiler=profiler)` — the
  real LLM call, forwarded to the terminus.
  - The model calls `write` then `read` (dispatched to `_run_write_tool`/`_run_read_tool`,
  unchanged plain local file I/O), each result run through `_offload_if_oversized` (**line
  1198**) and, new this round, classified `success`/`failure` for the (no-op) profiler span
  annotation (**lines 1199-1209**) by sniffing an `"error"` key in the parsed JSON result.
  - Model then calls `submit_output` three times (`status`/`path`/`content`), each via
  `_make_mcp_tool_handler` → a real MCP round trip to `agmcp_server.py`'s `submit_output` tool
  (`**agmcp_server.py:219`**) — logic there is unchanged: validates via
  `output_schema.check_field`, accumulates into `self._collected_outputs[token]`.
  - Model finally responds with no tool calls → **lines 1169-1179**: returns `{"status": "done", "messages": ..., "final_text": ..., "usage": {...}, "turn_count": step+1}`.

## 6. Back in `native.py` — finishing up

- **Lines 494-513**: `resp["status"] == "done"` → accumulates `usage` into
`prev_ctx.total_input_tokens`/`total_output_tokens`; also increments `profiler_turn_offset`
(inert here).
- **Lines 517-524**: `collected_output = mcp_server.collected_output(token)` — all 3 fields present
→ `missing` empty → breaks the `while True:` loop.
- **Lines 536-548** (`finally:`): stops the live poller thread, one last synchronous
`pusher.poll_once()`, then unregisters the token from all **four** bridge services now
(`terminus`, `profiler_ingest`, `mcp_server`, `messenger` — the profiler-ingest unregister is
new).
- **Lines 552-568**: builds `result = agdata(**collected_output)`; mutates `prev_ctx.messages` in
place; returns `(result, prev_ctx, [sys_msg, user_msg] + new_since_call_start)`.

## 7. Unwinding through `agskill.py` — including the new hibernate step

Back in `execute_harness()` (`**agskill.py:604-640`**): `wait_for_processes` (no-op) and
`recover_outputs` (no-op for `agpath`) as described in step 3.

Back in `_task()`'s `finally:` block (`**agskill.py:340-393**`) — this is where the biggest
behavioral change from the earlier trace shows up:

- **Line 344**: `_had_error = False`.
- **Lines 366-393**: on success, `agprof.span("teardown:commit")`, then:
  ```python
  try:
      ag.sandbox.commit()
  finally:
      if not ag.sandbox._has_pending_background_work():
          ag.sandbox.stop()
  ```
  **The container is now stopped (hibernated) after every successful skill call**, not left
  running. The comment at **lines 372-381** explains why: `commit()` itself leaves the container
  running, and without an explicit `stop()` a finished/one-shot agent would hold its session
  keyring forever sitting idle. This means `ag.sandbox._native_entrypoint_sock` still points at a
  socket, but the container (and the entrypoint process inside it) is gone.
- **Line 452**: `result_future.set_result(outer_result)` — unblocks `r1.status`/`r1.path`/
`r1.content`.
- **Lines 455-464**: history pruning via `agllm._prune_tool_outputs`, inside
`agprof.span("prune")`.
- **Line 466**: `ctx_future.set_result(updated_ctx)` — unblocks `ag.ctx` for the next chained
call.

## 8. The second call — `ag.run(qa_skill, ...)` re-provisions from a stopped container

`qa_skill` still sets `replace_tools=[]`, so as before: `suppress_builtins=True`, zero custom
tools, and the container-side tool list starts empty (only MCP tools are present) — `qa_skill` can
only answer from shared history and `submit_output`.

The new wrinkle: `**agskill.py:291`**, `if ag.sandbox is None:` — this is `False` now (the sandbox
object still exists, just stopped), so the sandbox is *not* reconstructed. But
`_NativeBackend.execute()` (`**native.py:440*`*) calls `_ensure_entrypoint(ag.sandbox)` again
(**line 282**):

- **Lines 290-295**: `existing = sandbox._native_entrypoint_sock` is set (from call 1) → tries
`ping(existing, timeout_s=2)` → **fails**, since the container was stopped and the entrypoint
process is dead.
- Falls through to **line 296**: `launch_in_container_entrypoint(sandbox)` again — this
re-triggers a real backend-level resume/start of the (already-committed) container via the same
lazy-start path, and relaunches a fresh entrypoint process inside it.
`ensure_python_packages_in_container` re-checks `mcp`/`html2text`/`cloudpickle` (cheap — they're
already installed on the committed image, so the importability check short-circuits the pip
install).

So `qa_skill`'s call pays one more container resume + entrypoint relaunch than the first call did
— but crucially, `prev_ctx.messages` (the shared `agcontext` chain) already carries `file_skill`'s
full conversation, since that's independent of the sandbox's own lifecycle. That's what lets
`qa_skill`'s system prompt claim "You have access to prior conversation context" and actually be
correct, even though it's now running inside a brand-new container process from `file_skill`'s.

From there, steps 5-7 repeat identically, except the tool list is empty (`suppress_builtins=True`)
and the model's only path to a result is one `submit_output("answer", ...)` call before finishing.

## Open items worth double-checking if you want to dig further

- `**wait_for_processes()`'s return value is discarded** in `execute_harness()`
(`agskill.py:629-638`) — if a background job were actually running, the "still running" /
"completed" message it would otherwise inject into the conversation never reaches the model.
Confirm whether this is intentional or a regression from the `execute_react()` days.
- The commit+stop-every-call behavior (`agskill.py:366-393`) means **every** skill call after the
first pays a container-resume + entrypoint-relaunch cost, not just cross-agent or long-idle
cases — worth confirming this is the intended tradeoff versus, e.g., gating hibernation on some
idle threshold.
- `agSandboxConfig`'s `persistent` field (`agsandbox.py:31-43`) is documented as controlling
whether `dispatch_tools()` (deleted in the Phase 0 retirement) hibernates between tool calls —
that docstring looks stale now that `dispatch_tools()` no longer exists; worth checking whether
`persistent` still does anything for native at all.

