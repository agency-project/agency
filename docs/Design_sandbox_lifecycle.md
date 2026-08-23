# Execution: Process Control

> This document describes the **container backends** (`_DockerBackend`/`_PodmanBackend`, both subclassing `_ContainerBackendBase` in `sandbox/container.py` — see [sandbox/container.md](sandbox/container.md)) specifically. `agSandbox` itself (in `sandbox/agsandbox.py`) is a thin facade over a selectable `agsandbox_backend` — see [sandbox/base.md](sandbox/base.md) for backend selection and [sandbox/chroot.md](sandbox/chroot.md) for the lighter chroot-based alternative and how it differs (no PID namespace, so process tracking/background-job semantics below don't carry over the same way; no cgroup, so there's nothing analogous to the keyring/semaphore concurrency controls below). The chroot backend exposes the same three-operation `stop()`/`rm_container()`/`commit()` split described below, just with simpler mechanics (no runtime slot to release, and releasing the GPU on a mere `stop()` is safe there — see `chroot.md`).

This document traces what happens in each process-spawning scenario from the moment a bash tool call is made through to the end of the agent run.

---

## Historical host-side path: tool call → exec wrapper

The retired in-process ReAct loop sent bash tool calls through this path. It is
kept here to explain the container exec wrapper; current execution is sequenced
by `ExecutionBuilder`, and the selected sandbox-side harness reaches host-owned
tools through `HostServerManager`:

```
agskill.execute_react()
  └─ t(agdata(command="..."))          agtool.__call__
       └─ _run_sandboxed(arg)          make_bash closure
            └─ sandbox.exec(cmd)       agsandbox.py
                 └─ docker exec ... bash -c <wrapped>
```

Inside the container the wrapper does the following around every user command:

```sh
exec 2>&1                              # merge stderr into stdout

# 1. snapshot PIDs before
__AGENCY_BEFORE=$(for __d in /proc/[0-9]*; do
  [ -f "$__d/status" ] && echo "${__d##*/}"; done | tr '\n' ' ')
__AGENCY_SHELL=$$

# 2. run user command
<user command>
__AGENCY_RC=$?

# 3. diff /proc after — any new PID was spawned by the command
__AGENCY_BGPIDS=''
for __d in /proc/[0-9]*; do
  [ -f "$__d/status" ] || continue
  __p=${__d##*/}
  case " $__AGENCY_BEFORE $__AGENCY_SHELL " in
    *" $__p "*) ;;
    *) __AGENCY_BGPIDS="$__AGENCY_BGPIDS $__p" ;;
  esac
done

printf '\n__BGPIDS__:%s' "$__AGENCY_BGPIDS"
exit $__AGENCY_RC
```

Back in Python, `sandbox.exec()` strips the `__BGPIDS__` annotation, writes any found PIDs into `sandbox._watched_pids`, and returns `(clean_output, returncode)` to `_run_sandboxed`.

The tool then logs the call via `agtool.log()` (terminal + file) and returns the output to the LLM as a tool result.

---

## Execution-transaction container lifecycle

Sandbox lifecycle is deliberately coarser than tool dispatch. `SandboxProvisioner.acquire()` resolves or creates the durable facade, acquires its lock, and explicitly starts the physical backend before the host server or harness is launched. The sandbox then stays ready throughout prompt delivery, harness execution, all tool calls, completion waiting, and reverse service cleanup. `dispatch_tools()` does not call `ensure_started()`, `stop()`, `commit()`, or `rm_container()` as a lifecycle boundary.

After the harness manager and host server have stopped, `SandboxProvisioner.finalize()` makes the one durability decision for the execution: commit and optionally hibernate on success, or discard dirty live state after an attempted failure. Preparation failure before the harness-attempt mark unwinds without claiming a revert. Provisioner teardown follows and the sandbox lock is released last.

`run_in_subprocess` only selects where a tool function executes. It has no effect on physical startup, hibernation, checkpointing, or discard.

The grace period used when provisioner teardown does choose `docker/podman stop` is `0`: the sandbox's entrypoint is `tail -f /dev/null`, which never handles `SIGTERM`, so a nonzero grace period would only add a timeout.

The first physical preparation in an engine transaction is explicit and occurs
while the provisioner holds the sandbox lock. If a later safe point deliberately
hibernates the backend, sandbox operations retain the same idempotent readiness
check to resume it defensively:

```
SandboxProvisioner.acquire()
  sandbox._lock.acquire()
  agSandbox.ensure_started()
    backend._ensure_started()
  (running, status) = _inspect_container_state()   # ONE docker/podman inspect call,
                                                     # not two separate ones
  ├─ running?                      → reuse (worker-reuse fast path; does NOT acquire the runtime slot)
  ├─ status truthy, not running (hibernating)?
  │    _acquire_runtime_slot()        (blocks until a keyring slot is free)
  │    docker/podman start <name>     (resume in place — GPU re-acquired lazily on the next exec())
  └─ container absent entirely?
       _acquire_runtime_slot()        (blocks until a keyring slot is free)
       _lifecycle_image set → docker run from lifecycle image
       not set              → docker run from base_image (no checkpoint yet)
```

`exec()` and other operations may call the same internal ensure idempotently so
they remain safe after an intentional hibernate or when used outside the engine,
but an incidental command is not the normal initial-start mechanism.

`_inspect_container_state()` merges what used to be two separate `_container_running()`/`_container_status()` inspect round-trips into one (`{{.State.Running}}|{{.State.Status}}`), since `_ensure_started()` is the only caller that ever needs both facts together — `stop()`/`rm_container()`/`commit()`/`destroy()` still call the individual methods directly, since each only needs one fact.

**Three states, not two.** The old invariant — "after `stop()` the container does not exist," so anything not running is a zombie to be force-removed before a fresh `run` — no longer holds. `rm_container()` still guarantees the container is gone afterward, but a merely hibernated (`stop()`ped) container is *not* a zombie: it holds the last committed transaction state on its own writable layer, and the next transaction's explicit `ensure_started()` resumes it via `docker/podman start`. A hibernating container under this exact name is unambiguously this backend's own — container names embed `_RUN_ID` (a fresh UUID per process). State is preserved either through that live writable layer (running → hibernating → running) or through `_lifecycle_image` commits when an absent backend must be recreated after discard or for a new agent.

**"Created" state cleanup.** A container stuck in Docker's `"created"` state — from a `run` that partially succeeded (e.g. the container object got allocated, the name reserved, the overlay created, but the process never actually started because the session keyring quota was hit) — is a different, unchanged case from the hibernate-resume branch above: `_ensure_started()` never sees a `"created"` container as "hibernating" (it only recognizes cleanly-stopped containers that way), so this is still handled the old way, by `_run_with_conflict_retry()`'s name-conflict removal logic on the *next* fresh `run` attempt.

This keeps at most one physical backend active for a leased sandbox. Its session-keyring runtime slot remains held for the engine transaction and is released when provisioner finalization hibernates or discards the container. A GPU lease follows the same backend teardown cadence; permanent `destroy()` remains a separate lifetime cleanup (see "GPU device access" in `sandbox/container.md`).

---

## Skill-call boundary: commit or discard

`stop()` only ever hibernates — nothing durable happens to the sandbox's checkpoint until `SandboxProvisioner.finalize()` runs, still holding the provisioner-owned `ag.sandbox._lock` lease (see "Concurrency controls" below):

| Skill outcome | What happens |
|---|---|
| Success | After the harness manager and host server stop cleanly, `SandboxProvisioner.finalize()` calls `ag.sandbox.commit()` to checkpoint the current filesystem into `agency/lifecycle-<name>` without removing the container. It may then call `ag.sandbox.stop()` when no background work requires the backend to remain active. Provisioner teardown follows and the lock is released last. |
| Failure after `mark_execution_attempted()` | The builder first stops every harness/host service that started. The provisioner then calls `ag.sandbox.rm_container()`, discarding live state accumulated since the last successful commit, and queues `ag.inbox.put(...)`. It performs its teardown and releases the lock last. |
| Preparation failure before the attempt mark | Partially started services and provisioner state are unwound; the physical backend may be hibernated, but no dirty execution is claimed and no revert notice is queued. The lock is still released last. |

If a physical discard or hibernation fails, the facade retains a provisioner
recovery marker. The next lease retries `rm_container()` while holding the
sandbox lock and **before** `ensure_started()`; it cannot resume the ambiguous
writable layer. A failed-execution revert notice is delivered only after that
discard eventually succeeds, and it is delivered to the agent whose execution
failed even if another agent acquires the shared sandbox next.

Because a failing execution's own result is already final by the time teardown
runs, the revert notice cannot be attached to it the way an older
`workspace_reverted` key once was. `HarnessInteractionServer.check_inbox()`
exposes the agent inbox to the sandbox-side harness and converts queued strings
into user conversation turns through `agent._drain_inbox()`. The replacement
Harness Manager protocol must poll that endpoint as it resumes sandbox work;
the retired `execute_react()` loop performed the same drain directly. The
notice therefore belongs to the next execution, not the failed call's output.

This is now the **only** rollback boundary in the system. A single failed tool call inside an otherwise-successful skill does not, by itself, discard anything — the skill's own final outcome is what decides, once, at the very end.

---

## Container naming and run isolation

Each container is named `sandbox-{RUN_ID}-{agname}`, where `_RUN_ID` is a UUID prefix generated once at module import time (e.g. `r4a7f9c21`). This scopes all containers to the current process run:

- Two agents with the same `agname` in different runs get different container names — no cross-run collision even if a previous run crashed without cleanup.
- `_lifecycle_image` tags follow the same pattern: `agency/lifecycle-sandbox-{RUN_ID}-{agname}` (all lowercased by `_lifecycle_tag()`, since Docker requires lowercase repository names).
- Worker processes (spawned by `ProcessPoolExecutor`) inherit the parent's `_RUN_ID` because it is set at import time in the parent, so they use the same container names.

---

## Concurrency controls

Two semaphores gate Docker daemon calls:

| Semaphore | Limit | Guards |
|---|---|---|
| `_docker_semaphore` | 16 | All Docker/Podman daemon calls — held for the duration of each `_run()` invocation. The daemon serialises most operations internally (GPU init, overlay diff, container teardown), so more than ~16 concurrent calls increase contention without reducing wall-clock time. Replaces the former `_startup_semaphore` / `_commit_semaphore` / `_shutdown_semaphore` trio. |
| `_container_semaphore` | `maxkeys − 5` | Total simultaneously running containers, derived from `/proc/sys/kernel/keys/maxkeys`. Each running container holds one Linux session keyring; hitting the limit causes `docker run` (or `docker start`) to fail with "disk quota exceeded". Held while the provisioner-owned engine transaction keeps the container running, released by finalization's `stop()`/`rm_container()`, and re-acquired by the next transaction's explicit `ensure_started()`. |

`_docker_semaphore` limits *throughput* (concurrent daemon calls); `_container_semaphore` limits *capacity* (simultaneously running containers).

A third mechanism guards a different axis — not Docker daemon load, but **exclusive use of one `agSandbox` object**:

| Lock | Scope | Held by | Guards |
|---|---|---|---|
| `agSandbox._lock` (`threading.RLock`) | Per `agSandbox` instance | `SandboxProvisioner` lease, from before physical `ensure_started()` through service cleanup, finalization, and provisioner teardown | Two skill runs interleaving lifecycle operations against the same shared sandbox. |

This lock is reentrant and thread-local to whichever thread is running the engine transaction. The lock is *not* acquired automatically by `agSandbox` methods; the provisioner method that acquires the lease is also responsible for releasing it, establishing the "one skill run at a time" invariant. See `Design_architecture.md`'s "Per-sandbox mutex" section and `agsandbox.md`'s "Concurrent access" section.

---

## Dangling image accumulation and cleanup

Every successful *skill* call commits the container state at most once, from `SandboxProvisioner.finalize()`, using the same tag:

```
docker commit <container> agency/lifecycle-<agname>
```

When Docker retags an existing image, the old image loses its tag and becomes **dangling** — no name, not referenced by any container, but still occupying space in `/var/lib/docker/.../overlay2`, *unless* something deletes it. A container runtime refuses to delete an image while another image still depends on it as a parent (`docker rmi` fails with "has dependent child images") — under the OLD per-tool-call-teardown design, a plain commit's result WAS always a child of whatever it replaced (the container was recreated FROM each checkpoint), so two consecutive plain-commit cycles could never free the first one's image, and dangling images genuinely accumulated between squash points.

**Under the hibernate model this no longer holds, and cleanup happens on every commit now, not just at squash points.** The container is never recreated between successful commits — only `rm_container()` does that, and only on skill failure — so `docker/podman commit` always diffs against the container's *fixed* `run`-time ancestor, never against whatever the previous commit produced. Confirmed empirically: two consecutive `commit()` calls on the same never-recreated container produce **sibling** images of identical depth (each capturing the container's full cumulative diff, not just what changed since the last commit), never a parent/child chain. Since the previous commit is never a parent of the new one, it's immediately safe to delete once the tag moves off it — `commit()` does exactly this now, on *every* cycle:

```python
# runs at the START of every commit() call, before the plain commit itself
previous_image_id = ...  # whatever image the tag currently points to, if any
# ... plain commit runs, moving the tag to a new image ...
if previous_image_id and <no container still running from it>:
    self._rmi(previous_image_id)       # best-effort — warns rather than raises on failure
```

This is separate from, and in addition to, the squash's own cleanup of the transient plain-commit image it superseded:

```python
# only reached when this cycle's commit also crossed checkpoint_squash_max_depth
old_image_id = ...  # THIS cycle's own plain-commit result, about to be replaced by the squash
self._build_accumulator_for_squash(tag)  # lazy: only when squash is due
self._accumulator_squash_commit(tag)     # fast path, falls back to _squash_commit() (export/import)
# finally: _reset_accumulator() always clears the temp tar
if old_image_id and <no container still running from it>:
    self._rmi(old_image_id)            # best-effort — warns rather than raises on failure
```

- **Continuous, not squash-triggered**: the previous-sibling cleanup above reclaims space on every successful commit, not just in one lump sum at squash points — a real change from the old design, where nothing could be reclaimed between squashes at all.
- **Best-effort**: either delete is skipped (not raised) if a container — e.g. a fork still running from that exact tag — is confirmed still using the old image; any other failure to check/delete just warns, since a stray dangling image costs disk space, not correctness.
- **No background thread needed**: the prune thread and `_PRUNE_INTERVAL_S` constant remain removed.

`checkpoint_squash_max_depth` (default `100`) is a depth check on the chain's own actual layer count, not a fixed commit-count interval — a real base image was found to already carry 80 layers on its own, so a naive `base_depth + interval` count could cross the runtime's real cap before the interval ever fired. Squashing itself is triggered from inside `commit()` now (once per skill), rather than from the old per-tool-call `stop(commit=True)` — an earlier design also force-flattened the chain at every skill exit regardless of depth (`force_squash`), to guarantee bounded depth despite many uncontrolled per-tool-call commits landing at an arbitrary mid-chain point in between. That parameter no longer exists at all: since `commit()` now runs at most once per skill call, the depth check performed right after that one commit is sufficient on its own — there's nothing left for a forced, unconditional squash to guard against. See `sandbox/container.md`'s "Layer-depth squashing" and "Squashing and dangling images" sections for the full mechanism, including the fast incremental accumulator path (`_accumulator_squash_commit()`) that avoids re-serializing the whole image on every squash.

---

## `stop()`/`rm_container()`/`commit()` reliability

The old single `stop(commit, force_squash)` call bundled retry-and-raise behavior for both the optional commit and the removal into one method. Splitting it into three orthogonal operations split that behavior across them too:

- **`stop()`** — a single `docker/podman stop` attempt, not retried at all: the sandbox's entrypoint is always `tail -f /dev/null`, which never handles `SIGTERM` gracefully, so a retry wouldn't change the outcome (only the grace period — already fixed at `0` — would, and retrying doesn't touch that). Releases the runtime slot only once `_container_running()` confirms the container is actually stopped, then **raises** the underlying exception if the `stop` call itself failed.
- **`rm_container()`** — retries `docker/podman rm -f` up to `rm_retry_attempts` (3×) with a fixed backoff (`rm_retry_backoff_s`) between attempts. Releases the GPU only if it was actually held, and captures `had_container = self._container_running()` before removal so it releases the runtime slot only when this call actually transitioned a running container away. That guard also makes direct or recovery calls on an already-hibernating container safe: `multiprocessing.Semaphore.release()` does not reject over-release. The method raises the last exception if every removal attempt failed, because an unconfirmed removal may still hold real resources.
- **`commit()`** — retries the plain `docker/podman commit` up to `commit_retry_attempts` (3×) with its own backoff (`commit_retry_backoff_s`), and **raises** the last exception if every attempt failed. The backend itself never removes or stops the container on that error; at the transaction layer, `SandboxProvisioner.finalize()` responds by attempting `rm_container()` so dirty live state cannot leak into the next execution. A squash failure (fast accumulator path or the export/import fallback) is different: best-effort, warned rather than raised, since the plain commit that triggered the squash check already succeeded by that point.

All three follow the same "don't release until ground truth agrees" pattern: a resource (the runtime slot, the GPU) is only released once `_container_running()` independently confirms the state the release assumes. This replaced an earlier version that emitted a `WARNING` to stderr after exhausting retries and then silently continued — visible in logs, but not actionable, since the caller had no way to know teardown hadn't actually succeeded.

`destroy()` (called from `atexit`/`agSandbox.__del__` as a last-resort cleanup) now literally calls `self.rm_container()` for removal, rather than keeping its own separate copy of the retry-and-release logic — the duplication used to exist because the old combined `stop()` raised immediately on failure, but `rm_container()` doesn't need that workaround, and keeping a second copy in sync was exactly how the double-release bug above went unnoticed. `destroy()` wraps that call in a try/except so it still runs its own remaining steps regardless — a best-effort courtesy kill of `_watched_pids` beforehand (rm_container() doesn't do this; destroy() may be called on a sandbox that never went through a normal `stop()` first), and checkpoint-image/accumulator-dir cleanup afterward — then raises the `rm` failure, if any, at the end. A raise there is caught and logged by the atexit wrapper, not fatal, so raising is safe even in that path. The checkpoint-image cleanup remains best-effort (`WARNING` and continue), since a stray dangling image costs disk space, not correctness.

---

## Historical process monitoring inside `agskill`

The retired in-process ReAct loop embedded background-process monitoring
directly in `agskill.execute_react()`. The replacement sandbox-side harness
protocol must preserve the same completion gate; the implementation below is
historical and is not part of the execute-only engine facade:

```python
# inside agskill.execute_react(), after output schema validation passes:
result = agdata(**_collected_outputs)
proc_msg = agSandbox.wait_for_processes(
    ag.sandbox, self.name, ag.terminal, ag.log,
    str(ag.agname), type(ag).ping_interval_s, type(ag).poll_interval_s, ag._set_ui_state,
)
if proc_msg is not None:
    messages.append({"role": "user", "content": proc_msg})
    continue   # re-enter loop — LLM sees process status and acts

return (result, prev_ctx, delta_messages)   # sandbox clean → truly done
```

`agSandbox.wait_for_processes()` returns `None` immediately if `sandbox._has_pending_background_work()` is false (for this backend, that default check is `bool(self._watched_pids)`). Otherwise it polls — checking `_has_pending_background_work()` again each iteration, not `get_live_pids()`, which is used only for reporting (`pid_status_summary()`/the log payload) — until either it goes false or `ping_interval_s` elapses:

| Outcome | Return value | Log event |
|---|---|---|
| No watched PIDs | `None` | — |
| All PIDs exit before deadline | `"Background processes have completed. Read their output and act on the results."` | `procs_completed` |
| PIDs still alive at deadline | `"Background processes are still running: {summary}. ..."` | `procs_ping` |

The returned string is appended as a `{"role": "user"}` message — the LLM reads it and decides what to do next (read output, call more tools, wait, or call `daemon_release`). The ReAct loop then continues.

Crucially, the loop never lets a skill return while `_has_pending_background_work()` is still true — it keeps injecting the "still running" ping and looping. The container remains running throughout this monitoring because tool dispatch performs no lifecycle transition. At transaction finalization, the provisioner checks the same pending-work signal before optional hibernation.

**Cap:** `max_steps` (default `AGSKILL_REACT_MAX_STEPS = 4096`) bounds the total number of ReAct iterations including all process-monitoring continuations. Each monitoring ping consumes ~1 step (1 LLM turn to read the final-answer attempt + 1 more after the process message is injected).

---

## Case 1: Background job (`python train.py &`)

### Tool call → exec wrapper

The `&` means the shell starts `train.py` and continues without waiting.

1. `/proc` before-snapshot taken
2. `python train.py &` — shell starts the process and returns immediately
3. `/proc` after-diff — `train.py` is still alive → added to `__BGPIDS__`
4. `_watched_pids[train_pid] = now`
5. `exec()` returns `("", 0)` — the LLM sees empty output
6. Tool dispatch performs no lifecycle operation. The container keeps running, and so does `train.py` inside it.
7. LLM produces its final answer; output schema validation passes

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — train_pid is in _watched_pids
  ↳ container is still running under the provisioner-owned execution lease
  ↳ polls _has_pending_background_work() every poll_interval_s
  ↳ train.py exits before ping_interval_s elapses → _watched_pids cleared →
    returns "Background processes have completed..." message, and the loop truly ends, OR
  ↳ train.py still running at the deadline → returns a "still running" ping message,
    and the ReAct loop continues (LLM may read output, wait, or call more tools)
```

If the skill finishes successfully, `SandboxProvisioner.finalize()` calls `ag.sandbox.commit()` after harness and host cleanup and may then hibernate when no pending work requires the backend to remain active (see "Skill-call boundary: commit or discard" above).

> **Note**: a background process spawned inside one tool call survives into later tool calls because the physical backend is held for the complete engine transaction. It exits naturally, is removed from monitoring via `daemon_release`, or is terminated when execution-level finalization eventually stops or removes the backend.

---

## Case 1b: Multi-tool background job (monitoring within one agent turn)

When the agent uses multiple bash tool calls to monitor a long-running background process, every call uses the **same running container** held by the provisioner lease; no tool-level stop/start cycle occurs.

```
Tool call 1: bash("python train.py &; echo started")
  → exec(): train_pid added to _watched_pids
  → dispatch_tools(): no sandbox lifecycle operation
  → LLM sees "started"

Tool call 2: bash("cat train.log")   # agent polls the file instead of the PID
  → container is already running under the same execution lease
  → exec(): reads train.log, which reflects train.py's progress so far — same live
    process, same live container
  → dispatch_tools(): no sandbox lifecycle operation
  ...
```

The background job and writable state stay live across tool calls for the complete execution. Writing progress to `/workspace` files remains how the agent reads incremental output and how successful state becomes durable after the provisioner commits.

---

## Case 2: Foreground job (`python eval.py`)

### Tool call → exec wrapper

No `&` — the wrapper shell blocks on `python eval.py` until it exits.

1. `/proc` before-snapshot taken
2. `python eval.py` — **blocks** until eval.py exits
3. `/proc` after-diff — eval.py is already gone from `/proc`; diff is empty
4. `_watched_pids` unchanged
5. `exec()` returns `(full_stdout_of_eval, rc)` — LLM sees the output inline
6. Tool dispatch performs no lifecycle operation. The running container and `/workspace` state remain available to the harness for the rest of the execution.
7. LLM reads the result, produces final answer

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — _watched_pids is empty → returns None immediately
→ (result, prev_ctx, delta_messages) returned

# after harness and host cleanup:
SandboxProvisioner.finalize(..., succeeded=True)  # commit, then optional hibernate
result_future.set_result()   ← caller unblocks
```

The output was already in the tool result. No re-entry, no waiting. A later tool call in this execution uses the already-running backend. The next execution explicitly calls `ensure_started()`, which reuses a backend left active or resumes one hibernated by prior finalization.

---

## Case 3: Foreground job that spawns a long-running child

Example: `python launcher.py`, where launcher.py does `subprocess.Popen(['python', 'train.py'])` then exits.

### Tool call → exec wrapper

1. `/proc` before-snapshot taken
2. `python launcher.py` — **blocks** until launcher exits; launcher spawns `train.py` first
3. `/proc` after-diff — launcher.py is gone, but `train.py` is still alive → added to `__BGPIDS__`
4. `_watched_pids[train_pid] = now`
5. `exec()` returns `(launcher_stdout, rc)` — LLM sees whatever launcher printed
6. Tool dispatch performs no lifecycle operation. An orphaned child discovered by the `/proc` diff is tracked no differently from an explicitly backgrounded (`&`) process; the container remains active for the transaction.
7. LLM produces final answer

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — train_pid is in _watched_pids
  ↳ container is still running under the execution lease
  ↳ polls until train_pid exits or ping_interval_s elapses, exactly as in Case 1
```

The orphaned child is tracked and can outlive the tool call that spawned it, just like any other backgrounded process. It exits naturally or is terminated by execution-level finalization after monitoring no longer blocks completion.

---

## Case 4: Daemon job (`python server.py &`, then `daemon_release(pid)`)

`daemon_release` removes a PID from the set `wait_for_processes()` must see clear before the execution can complete. It does not create a separate lifetime for that process: the sandbox remains running through the transaction, but provisioner finalization can stop or remove it after harness and host cleanup.

### Tool call — start the server and immediately release it

The LLM issues a single bash command that starts the server and calls `daemon_release`:

```
bash({"command": "python server.py &\ndaemon_release " + server_pid})
```

Or more commonly, the LLM uses two tool calls in a single dispatch batch:
1. `bash({"command": "python server.py &"})` — server_pid added to `_watched_pids`
2. `daemon_release({"pid": server_pid})`:

```
make_daemon_release._run(arg)
  └─ sandbox.release_daemon(server_pid)
       ├─ _daemon_pids.add(server_pid)
       └─ _watched_pids.pop(server_pid)
```

After both tools complete, `_has_pending_background_work()` is **`False`** because `server_pid` moved out of `_watched_pids`. Tool dispatch performs no lifecycle operation, so the server remains alive until execution-level finalization.

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — _watched_pids is empty (server_pid was moved to
  _daemon_pids by release_daemon())
→ returns None immediately
→ (result, prev_ctx, delta_messages) returned

# after harness and host cleanup — assuming the execution succeeded:
SandboxProvisioner.finalize(..., succeeded=True)  # commit, then optional hibernate
result_future.set_result()   ← caller unblocks
```

> **Note**: `daemon_release` suppresses process-monitoring pings, but it does not make the process outlive execution finalization. A server that must genuinely outlive a skill call needs to live outside this sandbox entirely; `daemon_release` alone does not establish that lifetime.

---

## `get_live_pids()` — baseline diff approach

Called during `agSandbox.wait_for_processes`'s polling window. Reads the full `/proc` table in one pass and returns every PID that is:
- **not** in `_baseline_pids` (the container's process set at sandbox creation), and
- **not** in `_daemon_pids` (or descended from one), and
- **not** in zombie state (`State: Z` in `/proc/<pid>/status`)

Any newly discovered non-baseline PID is added to `_watched_pids` with the current timestamp. This handles the case where a watched process spawns new children after `exec()` returns — the children appear in `/proc` on the next `get_live_pids()` call and are automatically tracked.

---

## Summary

| Scenario | `docker exec` blocks? | `/proc` diff finds PIDs? | `_watched_pids` after exec | After tool call | `agSandbox.wait_for_processes` result |
|---|---|---|---|---|---|
| Background job (`&`), still running | No | Yes — background process | non-empty | No lifecycle transition; container keeps running | polls until it exits, or pings if still running |
| Background job (`&`), already finished by the next check | No | Yes (then exits) | non-empty → cleared once it exits | No lifecycle transition | `None` once it exits |
| Foreground job | Yes | No — process already exited | empty | No lifecycle transition | `None` immediately |
| Foreground spawns child, child still running | Yes | Yes — orphaned child | non-empty | No lifecycle transition | polls until it exits, or pings if still running |
| Daemon + `daemon_release` | No | Yes — server PID | moved to `_daemon_pids` (removed from `_watched_pids`) | No lifecycle transition; finalization controls teardown | `None` immediately |
| Failed tool call (error result or exception) | — | — | — | No lifecycle transition; the completed execution outcome controls commit/discard | — |
