# Execution: Process Control

> This document describes the **container backends** (`_DockerBackend`/`_PodmanBackend`, both subclassing `_ContainerBackendBase` in `sandbox/container.py` — see [sandbox/container.md](sandbox/container.md)) specifically. `agSandbox` itself (in `sandbox/agsandbox.py`) is a thin facade over a selectable `agsandbox_backend` — see [sandbox/base.md](sandbox/base.md) for backend selection and [sandbox/chroot.md](sandbox/chroot.md) for the lighter chroot-based alternative and how it differs (no PID namespace, so process tracking/background-job semantics below don't carry over the same way; no cgroup, so there's nothing analogous to the keyring/semaphore concurrency controls below). The chroot backend exposes the same three-operation `stop()`/`rm_container()`/`commit()` split described below, just with simpler mechanics (no runtime slot to release, and releasing the GPU on a mere `stop()` is safe there — see `chroot.md`).

This document traces what happens in each process-spawning scenario from the moment a bash tool call is made through to the end of the agent run.

---

## Shared path: tool call → exec wrapper

Every bash tool call follows the same path into the container:

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

## Per-tool-call container lifecycle

After every tool dispatch in `dispatch_tools()`, the container is *hibernated* — `sandbox.stop()` — unless the tool call left background work still running inside the sandbox (`sandbox._has_pending_background_work()` is true), in which case `stop()` is skipped entirely for that call. `stop()` kills every process inside the container (`docker/podman stop`, exactly as `rm -f` used to), so running it while something is still backgrounded (`cmd &`) would kill that work before the agent ever got a chance to check on it in a later tool call. Skipping just defers the hibernate to whichever later tool call finds nothing pending; this is not specific to tools flagged `run_in_subprocess=True` — that flag no longer gates this at all (see below).

Unlike the previous design, a tool call's own success or failure has **no bearing** on what happens to the sandbox — there is no more per-tool-call commit or rollback:

```
tool returns result
  ├─ sandbox._has_pending_background_work() is True
  │    stop() skipped entirely for this call — deferred to a later one
  │
  └─ otherwise (success OR failure — no distinction anymore)
       sandbox.stop()
         docker/podman stop -t 0 <container>   # kills everything inside; grace period is always 0
         # container object and its writable layer survive intact — no commit, no rm, no image
         # releases the runtime slot (session keyring) AND the GPU
```

The grace period passed to `docker/podman stop` is always `0`: the sandbox's entrypoint is always `tail -f /dev/null`, which never handles `SIGTERM`, so a nonzero grace period would only ever be wasted wall-clock time waiting out a timeout that always fires.

Checkpointing (`commit()`) and discarding (`rm_container()`) still exist, but no longer happen at this granularity — they run once per *skill* call, in `agskill.py`'s teardown, after the whole ReAct loop (every tool call inside it) has finished. See "Skill-call boundary: commit or discard" below and `Design_execution_loop.md`'s step 3d for the full teardown sequence.

**Not gated by `run_in_subprocess` anymore.** Earlier, this whole block only ran for tools with `run_in_subprocess=True` — a parameter that traces back to one literally named `need_sandbox`, renamed in an unrelated architecture refactor without revisiting whether the `stop()` gate still made sense under the new name. Since every built-in sandboxed tool (`bash`, `read`, `write`, `edit`, …) sets `run_in_subprocess=False` for unrelated reasons (they need to run synchronously against the same persistent object, not a disposable worker copy), that old gate meant `stop()` was never actually being called after any of them. `stop()` now runs after every tool call regardless of that flag, gated only on whether background work is pending.

Before the next tool call `_ensure_started()` resumes or (re)creates the container:

```
_ensure_started() (called lazily from exec())
  (running, status) = _inspect_container_state()   # ONE docker/podman inspect call,
                                                     # not two separate ones
  ├─ running?                      → reuse (worker-reuse fast path; does NOT acquire the runtime slot)
  ├─ status truthy, not running (hibernating)?
  │    _acquire_runtime_slot()        (blocks until a keyring slot is free)
  │    docker/podman start <name>     (resume in place — GPU re-acquired lazily on the next exec())
  └─ container absent entirely?
       _acquire_runtime_slot()        (blocks until a keyring slot is free)
       _lifecycle_image set → docker run from lifecycle image
       not set              → docker run from base_image (first tool call ever)
```

`_inspect_container_state()` merges what used to be two separate `_container_running()`/`_container_status()` inspect round-trips into one (`{{.State.Running}}|{{.State.Status}}`), since `_ensure_started()` is the only caller that ever needs both facts together — `stop()`/`rm_container()`/`commit()`/`destroy()` still call the individual methods directly, since each only needs one fact.

**Three states, not two.** The old invariant — "after `stop()` the container does not exist," so anything not running is a zombie to be force-removed before a fresh `run` — no longer holds. `rm_container()` still guarantees the container is gone afterward, but a merely-hibernated (`stop()`ped) container is *not* a zombie: it holds exactly the state the previous tool call left behind on its own writable layer, and `docker/podman start` resumes it in place at a fraction of the cost of a fresh `run`. A hibernating container found under this exact name is unambiguously this backend's own — container names embed `_RUN_ID` (a fresh UUID per process), so nothing else could have created one under it — there is no "leftover from someone else" case to force-remove here anymore. State is now preserved two different ways depending on which branch resumed the container: through the container's own writable layer (running → hibernating → running again, with no image involved at all), or through `_lifecycle_image` commits (absent → running, e.g. right after a skill-level `rm_container()`, or on a brand-new agent).

**"Created" state cleanup.** A container stuck in Docker's `"created"` state — from a `run` that partially succeeded (e.g. the container object got allocated, the name reserved, the overlay created, but the process never actually started because the session keyring quota was hit) — is a different, unchanged case from the hibernate-resume branch above: `_ensure_started()` never sees a `"created"` container as "hibernating" (it only recognizes cleanly-stopped containers that way), so this is still handled the old way, by `_run_with_conflict_retry()`'s name-conflict removal logic on the *next* fresh `run` attempt.

This keeps at most one container alive per agent during active tool execution. The session-keyring runtime slot is freed between tool calls (via `stop()`'s hibernate) while the LLM thinks, preventing the kernel keyring quota from being exhausted under high agent concurrency — but the GPU, once actually leased via `reserve_gpu`, stays leased across every hibernate/resume cycle for the rest of the container's life; only `rm_container()`/`destroy()` release it (see "GPU device access" in `sandbox/container.md`).

---

## Skill-call boundary: commit or discard

`stop()` only ever hibernates — nothing durable happens to the sandbox's checkpoint until the skill itself finishes, in `agskill.py`'s `_task()` `finally` block, still holding `ag.sandbox._lock` (see "Concurrency controls" below):

| Skill outcome | What happens |
|---|---|
| Success | `ag.sandbox.commit()` — checkpoints the container's current filesystem into its lifecycle image (`agency/lifecycle-<name>`), **without removing the container**. Then `ag.sandbox.stop()` hibernates it again (same pending-background-work deferral as per-tool `stop()`), releasing the session-keyring slot: `execute_react()`'s output path may have re-woken the container after the last tool-call hibernate, and `commit()` itself leaves it running. The next skill call resumes via `docker/podman start` on the same container object — no `run` needed. |
| Failure (the skill's own result carries an `"error"` key, or an exception escaped `execute_react()`) | `ag.sandbox.rm_container()` — force-removes the container outright, discarding every tool call's worth of state accumulated since the last successful skill's `commit()`. `ag.inbox.put(...)` queues a notice string onto the agent's `inbox` at the same time. |

Because a failing skill's own result is already final by the time teardown runs, the revert notice can't be attached to it the way an older `workspace_reverted` key once was. Instead it surfaces at the **start of the next skill call**: `execute_react()`'s loop calls `ag._drain_inbox(messages)` every iteration — including the first, before that skill's first LLM call — turning any queued string on `ag.inbox` (a plain `queue.Queue[str]`) into a `{"role": "user", ...}` conversation turn. The agent learns about the revert as it resumes sandbox work, not inside the failed call's own output.

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
| `_container_semaphore` | `maxkeys − 5` | Total simultaneously running containers, derived from `/proc/sys/kernel/keys/maxkeys`. Each running container holds one Linux session keyring; hitting the limit causes `docker run` (or `docker start`) to fail with "disk quota exceeded". Released by `stop()` on hibernate and re-acquired by `_ensure_started()` on resume — not just on removal — so it turns over on every tool-call boundary, not only when a container is actually destroyed. |

`_docker_semaphore` limits *throughput* (concurrent daemon calls); `_container_semaphore` limits *capacity* (simultaneously running containers).

A third mechanism guards a different axis — not Docker daemon load, but **exclusive use of one `agSandbox` object**:

| Lock | Scope | Held by | Guards |
|---|---|---|---|
| `agSandbox._lock` (`threading.RLock`) | Per `agSandbox` instance | `agskill.py`'s `_task()`, for the full duration of one skill run (acquired right after provisioning, released only after teardown's `commit()`/`rm_container()` call has finished) | Two skill runs interleaving `exec()` / `stop()` / `_ensure_started()` / `commit()` / `rm_container()` against the *same* container. There's no ownership flag anymore that ties a sandbox to exactly one agent, so a shared `agSandbox` (e.g. handed from one agent to another) needs this to stay safe. |

This lock is reentrant and thread-local to whichever thread is running the skill — every per-tool-call `stop()`/`_ensure_started()` described above, `wait_for_processes()`'s polling, and the single teardown `commit()`/`rm_container()` call all happen on that same thread, so they re-acquire the already-held lock at no cost. The lock is *not* acquired automatically by `agSandbox`'s methods themselves; only `agskill`'s session-scoped acquire/release around a whole skill run establishes the "one skill run at a time" invariant. See `Design_architecture.md`'s "Per-sandbox mutex" section and `agsandbox.md`'s "Concurrent access" section for the full rationale, including why the lock is excluded from pickling.

---

## Dangling image accumulation and cleanup

Every successful *skill* call (not tool call — `commit()` now runs at most once per skill, from `agskill.py`'s teardown; see "Skill-call boundary: commit or discard" above) commits the container state with the same tag:

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
- **`rm_container()`** — retries `docker/podman rm -f` up to `rm_retry_attempts` (3×) with a fixed backoff (`rm_retry_backoff_s`) between attempts. Releases the GPU only if it was actually held (`self._gpu_id is not None`, which is already `None` if a prior `stop()` released it) — self-guarding, since the GPU's own state variable tracks whether it's held. The runtime slot needs its own explicit guard: `had_container = self._container_running()` is captured *before* the rm attempt, and the slot is only released `if had_container and not self._container_running()` afterward. Without this, calling `rm_container()` on a container that's *already* hibernating (the normal skill-failure case — every prior tool call already hibernated it via `stop()`) would release the runtime slot a second time: `multiprocessing.Semaphore.release()` does not raise on over-release (unlike `threading.BoundedSemaphore`), so this was a real, previously-latent bug that silently over-credited the semaphore on every skill failure until it was found and fixed. Raises the last exception if every rm attempt failed — an unconfirmed removal means real resources (the keyring slot, the GPU) may still be held, which the caller must not silently ignore.
- **`commit()`** — retries the plain `docker/podman commit` up to `commit_retry_attempts` (3×) with its own backoff (`commit_retry_backoff_s`), and **raises** the last exception if every attempt failed. Never removes or stops the container either way, so a failed commit only means this cycle's state wasn't checkpointed forward — nothing about the container's own liveness changes. A squash failure (fast accumulator path or the export/import fallback) is different: best-effort, warned rather than raised, since the plain commit that triggered the squash check already succeeded by that point.

All three follow the same "don't release until ground truth agrees" pattern: a resource (the runtime slot, the GPU) is only released once `_container_running()` independently confirms the state the release assumes. This replaced an earlier version that emitted a `WARNING` to stderr after exhausting retries and then silently continued — visible in logs, but not actionable, since the caller had no way to know teardown hadn't actually succeeded.

`destroy()` (called from `atexit`/`agSandbox.__del__` as a last-resort cleanup) now literally calls `self.rm_container()` for removal, rather than keeping its own separate copy of the retry-and-release logic — the duplication used to exist because the old combined `stop()` raised immediately on failure, but `rm_container()` doesn't need that workaround, and keeping a second copy in sync was exactly how the double-release bug above went unnoticed. `destroy()` wraps that call in a try/except so it still runs its own remaining steps regardless — a best-effort courtesy kill of `_watched_pids` beforehand (rm_container() doesn't do this; destroy() may be called on a sandbox that never went through a normal `stop()` first), and checkpoint-image/accumulator-dir cleanup afterward — then raises the `rm` failure, if any, at the end. A raise there is caught and logged by the atexit wrapper, not fatal, so raising is safe even in that path. The checkpoint-image cleanup remains best-effort (`WARNING` and continue), since a stray dangling image costs disk space, not correctness.

---

## Process monitoring inside agskill

Background process monitoring is embedded directly in `agskill.execute_react()`'s ReAct loop (not in an outer caller loop). Each time the LLM produces a valid final answer, the framework calls `agSandbox.wait_for_processes()` before returning:

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

Crucially, the loop never lets a skill return while `_has_pending_background_work()` is still true — it keeps injecting the "still running" ping and looping. Since `dispatch_tools()`'s per-tool-call `stop()` is *also* deferred for exactly as long as that flag is true, this means the sandbox's container is never hibernated while genuinely pending background work exists — it only gets hibernated (by a later tool call) once that work has actually finished, been dropped via `daemon_release`, or the skill gives up at `max_steps`.

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
6. `sandbox._has_pending_background_work()` is **`True`** (`train_pid` is watched) → `stop()` is **skipped** for this call. The container keeps running, and so does `train.py` inside it.
7. LLM produces its final answer; output schema validation passes

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — train_pid is in _watched_pids
  ↳ container is still running — never hibernated (see step 6 above)
  ↳ polls _has_pending_background_work() every poll_interval_s
  ↳ train.py exits before ping_interval_s elapses → _watched_pids cleared →
    returns "Background processes have completed..." message, and the loop truly ends, OR
  ↳ train.py still running at the deadline → returns a "still running" ping message,
    and the ReAct loop continues (LLM may read output, wait, or call more tools)
```

Only once `train.py` has actually finished does the *next* tool call's `dispatch_tools()` find `_has_pending_background_work()` false and finally call `sandbox.stop()` — hibernating a container that, by then, has nothing left running inside it anyway. If the skill then finishes successfully, `agskill.py`'s teardown calls `ag.sandbox.commit()` (see "Skill-call boundary: commit or discard" above) — a step entirely separate from any of the per-tool-call `stop()` calls that came before it.

> **Note**: a background process spawned inside one tool call *does* survive into later tool calls, for as long as it's still tracked as pending — `stop()` is deferred, not forced, so the container backing it is left untouched while it's running. It only dies once it finishes on its own, or once something removes it from `_watched_pids` (a natural exit, or `daemon_release` — see Case 4) and a subsequent tool call's `stop()` actually runs and hibernates (killing everything still inside) the container.

---

## Case 1b: Multi-tool background job (monitoring within one agent turn)

When the agent uses multiple bash tool calls to monitor a long-running background process, every call reuses the **same** container for as long as `train.py` is still running and tracked — `stop()` keeps being deferred, so nothing about the container is lost between calls.

```
Tool call 1: bash("python train.py &; echo started")
  → exec(): train_pid added to _watched_pids
  → dispatch_tools(): _has_pending_background_work() True → stop() deferred
  → LLM sees "started"

Tool call 2: bash("cat train.log")   # agent polls the file instead of the PID
  → _ensure_started(): container already running — reused directly, no start/run at all
  → exec(): reads train.log, which reflects train.py's progress so far — same live
    process, same live container
  → dispatch_tools(): if train.py has since exited, _has_pending_background_work() is
    now False → stop() actually runs (hibernate); otherwise deferred again
  ...
```

The practical pattern for persistent work across tool calls no longer requires writing results to `/workspace` files and restarting from a checkpoint each time — a still-pending background job keeps its own container (and therefore its own live state) across as many tool calls as it takes to finish, for as long as nothing clears it from `_watched_pids` in between. Writing progress to `/workspace` files is still the way the agent *reads* incremental output, but it is no longer the only thing that survives between calls.

---

## Case 2: Foreground job (`python eval.py`)

### Tool call → exec wrapper

No `&` — the wrapper shell blocks on `python eval.py` until it exits.

1. `/proc` before-snapshot taken
2. `python eval.py` — **blocks** until eval.py exits
3. `/proc` after-diff — eval.py is already gone from `/proc`; diff is empty
4. `_watched_pids` unchanged
5. `exec()` returns `(full_stdout_of_eval, rc)` — LLM sees the output inline
6. `sandbox._has_pending_background_work()` is **`False`** (nothing new spawned) → **`sandbox.stop()`** runs — the container is hibernated (`docker/podman stop`, not removed): `/workspace` state stays exactly as `eval.py` left it, on the container's own writable layer, with no commit and no image involved.
7. LLM reads the result, produces final answer

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — _watched_pids is empty → returns None immediately
→ (result, prev_ctx, delta_messages) returned

# at skill teardown (agskill.py's finally block) — not per tool call:
ag.sandbox.commit()   # checkpoints the hibernating container's state; does not start it
result_future.set_result()   ← caller unblocks
```

The output was already in the tool result. No re-entry, no waiting. If a later tool call needs the sandbox again — later in this same skill, or the next skill call — `_ensure_started()` resumes this exact container via `docker/podman start` rather than running a fresh one from an image.

---

## Case 3: Foreground job that spawns a long-running child

Example: `python launcher.py`, where launcher.py does `subprocess.Popen(['python', 'train.py'])` then exits.

### Tool call → exec wrapper

1. `/proc` before-snapshot taken
2. `python launcher.py` — **blocks** until launcher exits; launcher spawns `train.py` first
3. `/proc` after-diff — launcher.py is gone, but `train.py` is still alive → added to `__BGPIDS__`
4. `_watched_pids[train_pid] = now`
5. `exec()` returns `(launcher_stdout, rc)` — LLM sees whatever launcher printed
6. `sandbox._has_pending_background_work()` is **`True`** (`train_pid` is watched) → `stop()` is **skipped**, exactly like Case 1 — an orphaned child discovered by the `/proc` diff is tracked no differently from an explicitly backgrounded (`&`) process. The LLM never explicitly launched `train.py`, but that makes no difference to how it's tracked.
7. LLM produces final answer

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — train_pid is in _watched_pids
  ↳ container is still running — never hibernated
  ↳ polls until train_pid exits or ping_interval_s elapses, exactly as in Case 1
```

The orphaned child is tracked and can outlive the tool call that spawned it, just like any other backgrounded process. It only dies once it finishes on its own, or once a later tool call's `stop()` actually runs after `_has_pending_background_work()` goes false.

---

## Case 4: Daemon job (`python server.py &`, then `daemon_release(pid)`)

`daemon_release` is still only meaningful when both the start and the release happen **within the same tool-call batch** (i.e., inside one `dispatch_tools()` call processing one LLM response), because `stop()` (hibernate) still kills every process inside the container the moment nothing is left pending — hibernating instead of removing doesn't change that: `docker/podman stop` kills everything inside regardless of whether the container survives afterward. `daemon_release` only removes a PID from the set `wait_for_processes()` must see clear before a skill can return; it does not protect that process from a subsequent `stop()`'s kill.

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

After both tools complete: `_has_pending_background_work()` is now **`False`** (`server_pid` moved out of `_watched_pids`) → **`sandbox.stop()`** runs — the container is hibernated, which kills the server just as `rm -f` used to.

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — _watched_pids is empty (server_pid was moved to
  _daemon_pids by release_daemon(), and the container itself was then hibernated by the
  stop() above)
→ returns None immediately
→ (result, prev_ctx, delta_messages) returned

# at skill teardown, not per tool call — assuming the skill succeeded:
ag.sandbox.commit()
result_future.set_result()   ← caller unblocks
```

> **Note**: `daemon_release` suppresses the process-monitoring ping within a single ReAct iteration — it tells the framework "don't wait on this PID before considering the skill done" — but it does not make the process outlive `stop()`. Once nothing is left in `_watched_pids`, the very next tool call's `stop()` hibernates (and thereby kills) the container just like it would for any tool call with no pending work at all. A server that must genuinely outlive a skill call needs to live outside this sandbox entirely (a separate host process, or a different container this sandbox merely talks to) — `daemon_release` on its own does not achieve that.

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
| Background job (`&`), still running | No | Yes — background process | non-empty | `stop()` deferred — container keeps running | polls until it exits, or pings if still running |
| Background job (`&`), already finished by the next check | No | Yes (then exits) | non-empty → cleared once it exits | `stop()` deferred until it exits; a later tool call's `stop()` then hibernates | `None` once it exits |
| Foreground job | Yes | No — process already exited | empty | `stop()` runs — container hibernated | `None` immediately |
| Foreground spawns child, child still running | Yes | Yes — orphaned child | non-empty | `stop()` deferred — same as a background job | polls until it exits, or pings if still running |
| Daemon + `daemon_release` | No | Yes — server PID | moved to `_daemon_pids` (removed from `_watched_pids`) | `stop()` runs — container hibernated, server killed | `None` immediately |
| Failed tool call (error result or exception) | — | — | — | `stop()` runs exactly the same as on success — no distinction anymore; commit vs. discard is decided only once, at skill end | — |
