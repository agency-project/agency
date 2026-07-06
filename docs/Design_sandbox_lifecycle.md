# Execution: Process Control

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

After every tool dispatch in `_dispatch_tools()`, the container is stopped:

```
tool returns result
  ├─ success (no "error" key in result)
  │    sandbox.stop(commit=True)
  │      docker commit → agency/lifecycle-<agname>   # snapshot /workspace (lowercased by _lifecycle_tag())
  │      docker rm -f <container>                    # release session keyring + GPU
  │      _lifecycle_image = "agency/lifecycle-<agname>"
  │
  └─ failure ("error" key in result, or exception raised)
       sandbox.stop(commit=False)
         docker rm -f <container>                    # discard dirty state, no commit
         # next start restores from previous _lifecycle_image
       result gains "workspace_reverted" note
```

Before the next tool call `_ensure_started()` recreates the container:

```
_ensure_started() (called lazily from exec())
  ├─ container running?  → reuse (worker-reuse fast path; does NOT acquire semaphore)
  └─ container absent (or stuck in any non-running state)?
       ├─ docker rm -f <name>   (no-op if absent; clears any "Created"/"Exited" zombie)
       ├─ _container_semaphore.acquire()   (blocks until a keyring slot is free)
       ├─ _lifecycle_image set → docker run from lifecycle image
       └─ not set              → docker run from base_image (first tool call ever)
```

**Two states only.** The "exited" fast-path (`docker start`) was removed. Any container that is not running is treated as a zombie and force-removed before a fresh `docker run`. State is preserved exclusively through `_lifecycle_image` commits, not through the container's overlay filesystem. This eliminates a class of zombie containers that accumulated when `docker stop` succeeded but `docker rm` later failed.

**"Created" state cleanup.** When the Linux session keyring is full, `docker run` can partially succeed — allocating the container object (name reserved, overlay created) but failing before starting any processes. This leaves the container in `"created"` state. `_ensure_started()` removes it with `docker rm -f` before retrying, preventing a spurious name-conflict error on the next attempt.

This keeps at most one container alive per agent during active tool execution. All session keyrings and GPU slots are freed between tool calls while the LLM thinks, preventing the kernel keyring quota from being exhausted under high agent concurrency.

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
| `_container_semaphore` | `maxkeys − 5` | Total simultaneously running containers, derived from `/proc/sys/kernel/keys/maxkeys`. Each running container holds one Linux session keyring; hitting the limit causes `docker run` to fail with "disk quota exceeded". |

`_docker_semaphore` limits *throughput* (concurrent daemon calls); `_container_semaphore` limits *capacity* (simultaneously running containers).

A third mechanism guards a different axis — not Docker daemon load, but **exclusive use of one `agSandbox` object**:

| Lock | Scope | Held by | Guards |
|---|---|---|---|
| `agSandbox._lock` (`threading.RLock`) | Per `agSandbox` instance | `agskill.py`'s `_task()`, for the full duration of one skill run (acquired right after provisioning, released after teardown's `stop()`) | Two skill runs interleaving `exec()` / `stop()` / `_ensure_started()` against the *same* container. There's no ownership flag anymore that ties a sandbox to exactly one agent, so a shared `agSandbox` (e.g. handed from one agent to another) needs this to stay safe. |

This lock is reentrant and thread-local to whichever thread is running the skill — every per-tool-call `stop()`/`_ensure_started()` described above (and `wait_for_processes()`'s polling) happens on that same thread, so they re-acquire the already-held lock at no cost. The lock is *not* acquired automatically by `agSandbox`'s methods themselves; only `agskill`'s session-scoped acquire/release around a whole skill run establishes the "one skill run at a time" invariant. See `Design_architecture.md`'s "Per-sandbox mutex" section and `agsandbox.md`'s "Concurrent access" section for the full rationale, including why the lock is excluded from pickling.

---

## Dangling image accumulation and eager cleanup

Every successful tool call commits the container state with the same tag:

```
docker commit <container> agency/lifecycle-<agname>
```

When Docker retags an existing image, the old image loses its tag and becomes **dangling** — no name, not referenced by any container, but still occupying space in `/var/lib/docker/.../overlay2`. With `MAX_CONCURRENT=80` agents each making dozens of tool calls this can consume tens of GB during a long run.

**Fix: delete old image immediately on commit**

Before committing, `stop(commit=True)` inspects the tag to record the current image ID. After the new commit succeeds, it deletes the now-unreferenced old image with `docker rmi`:

```python
# capture old ID before overwriting the tag
result = self._run([runtime, "inspect", "--format={{.Id}}", tag], check=False)
old_image_id = result.stdout.decode().strip() or None

# commit new snapshot
self._run([runtime, "commit", container, tag], check=True)

# delete the image that just lost its tag
if old_image_id:
    self._run([runtime, "rmi", old_image_id], check=False)
```

- **Eager, not deferred**: space is reclaimed at every tool-call boundary rather than on a periodic sweep.
- **Best-effort**: the `rmi` uses `check=False` — if it fails (e.g. race with another agent's inspect) the image becomes dangling as before, which is no worse than the old behaviour.
- **No background thread needed**: the prune thread and `_PRUNE_INTERVAL_S` constant have been removed.

---

## `stop()` reliability

`docker rm -f` is not always instantaneous: the Docker daemon can be slow under load, an overlay filesystem may have open file handles, or container namespaces may not have been fully released by the kernel. The original implementation swallowed all failures silently, causing zombie containers to accumulate.

The current `stop()` implementation:

1. Optionally commits the container state via `docker commit` (retried up to 3×) before removal.
2. Retries `docker rm -f` up to 3 times with a 1-second delay between attempts. Each attempt goes through `_run()`, which holds `_docker_semaphore` for the duration of the subprocess call.
3. Emits a `WARNING` to stderr after all retries are exhausted, then continues — `_started` is cleared and `_container_semaphore` released regardless, so the framework can keep running even if a zombie remains.

The warning makes accumulation visible rather than silent, and the retries handle transient daemon overload.

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

`agSandbox.wait_for_processes()` returns `None` immediately if `_watched_pids` is empty or `get_live_pids()` finds no active processes. Otherwise it polls until either all PIDs exit or `ping_interval_s` elapses:

| Outcome | Return value | Log event |
|---|---|---|
| No watched PIDs | `None` | — |
| All PIDs exit before deadline | `"Background processes have completed. Read their output and act on the results."` | `procs_completed` |
| PIDs still alive at deadline | `"Background processes are still running: {summary}. ..."` | `procs_ping` |

The returned string is appended as a `{"role": "user"}` message — the LLM reads it and decides what to do next (read output, call more tools, wait, or call `daemon_release`). The ReAct loop then continues.

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
6. **`sandbox.stop(commit=True)`** — container committed and removed; `train.py` dies with it (background processes do not survive across tool-call boundaries)
7. LLM produces its final answer; output schema validation passes

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — train_pid is in _watched_pids
  ↳ _ensure_started() recreates container from lifecycle image
  ↳ get_live_pids() reads /proc — train_pid absent (container is fresh) → returns {}
  ↳ _watched_pids cleared → returns None immediately
→ (result, prev_ctx, delta_messages) returned

ag.sandbox.stop(commit=True)   # commits container to lifecycle image and stops it
aglog._record()
result_future.set_result()   ← caller unblocks
```

> **Note**: background processes spawned inside a single tool call do not survive to the next tool call. The commit+remove cycle after each tool call terminates all container processes. Agents that need long-running background work (training, servers) should use a foreground exec per monitoring checkpoint, or launch the work via a daemon-release pattern scoped within a single tool call.

---

## Case 1b: Multi-tool background job (monitoring within one agent turn)

When the agent uses multiple bash tool calls to monitor a long-running background process, each call gets a fresh container from the lifecycle image, which does **not** preserve live processes from prior calls.

```
Tool call 1: bash("python train.py &; echo started")
  → exec(): train_pid added to _watched_pids
  → stop(commit=True): container committed (train.py killed); lifecycle image updated
  → LLM sees "started"

Tool call 2: bash("cat train.log")   # agent polls the file instead of the PID
  → _ensure_started(): docker run from lifecycle image (train.log absent — train never ran)
  → exec(): reads file
  → stop(commit=True)
  ...
```

The practical pattern for persistent work across tool calls is to write results to `/workspace` files during a **single** foreground exec, then read them in subsequent execs. Live processes do not persist.

---

## Case 2: Foreground job (`python eval.py`)

### Tool call → exec wrapper

No `&` — the wrapper shell blocks on `python eval.py` until it exits.

1. `/proc` before-snapshot taken
2. `python eval.py` — **blocks** until eval.py exits
3. `/proc` after-diff — eval.py is already gone from `/proc`; diff is empty
4. `_watched_pids` unchanged
5. `exec()` returns `(full_stdout_of_eval, rc)` — LLM sees the output inline
6. **`sandbox.stop(commit=True)`** — container committed and removed; `/workspace` state preserved in lifecycle image
7. LLM reads the result, produces final answer

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — _watched_pids is empty → returns None immediately
→ (result, prev_ctx, delta_messages) returned

ag.sandbox.stop(commit=True)   # commits container to lifecycle image and stops it
result_future.set_result()   ← caller unblocks
```

The output was already in the tool result. No re-entry, no waiting.

---

## Case 3: Foreground job that spawns a long-running child

Example: `python launcher.py`, where launcher.py does `subprocess.Popen(['python', 'train.py'])` then exits.

### Tool call → exec wrapper

1. `/proc` before-snapshot taken
2. `python launcher.py` — **blocks** until launcher exits; launcher spawns `train.py` first
3. `/proc` after-diff — launcher.py is gone, but `train.py` is still alive → added to `__BGPIDS__`
4. `_watched_pids[train_pid] = now`
5. `exec()` returns `(launcher_stdout, rc)` — LLM sees whatever launcher printed
6. **`sandbox.stop(commit=True)`** — container committed and removed; `train.py` killed by `docker rm -f`
7. LLM produces final answer

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — train_pid is in _watched_pids
  ↳ _ensure_started(): docker run from lifecycle image (fresh container, train.py absent)
  ↳ get_live_pids() → {} (process does not exist in new container)
  ↳ _watched_pids cleared → returns None immediately
→ (result, prev_ctx, delta_messages) returned

ag.sandbox.stop(commit=True)   # commits container to lifecycle image and stops it
result_future.set_result()
```

The orphaned child does not survive the commit+remove cycle. The LLM never explicitly launched `train.py` — the `/proc` diff is what detected it — but it is terminated with the container at tool-call boundary.

---

## Case 4: Daemon job (`python server.py &`, then `daemon_release(pid)`)

`daemon_release` is only meaningful when both the start and the release happen **within the same tool call** (i.e., inside a single `bash(...)` invocation), because the container is committed and removed at tool-call boundary regardless.

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

After both tools complete: **`sandbox.stop(commit=True)`** — container committed and removed; the server is killed by `docker rm -f`.

### Process monitoring (inside agskill)

```
agSandbox.wait_for_processes() called — _watched_pids is empty (server_pid was in _daemon_pids,
  which is cleared when container is removed)
→ returns None immediately
→ (result, prev_ctx, delta_messages) returned

ag.sandbox.stop(commit=True)   # commits container to lifecycle image and stops it
result_future.set_result()   ← caller unblocks
```

> **Note**: because the container is removed after each tool call, the server does not actually remain running across tool-call boundaries. The `daemon_release` tool is useful for suppressing the process-monitoring ping within a single ReAct iteration — it tells the framework "this PID is intentional and should not block the final answer" — but the process is terminated at tool-call boundary just like all others.

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
| Background job (`&`) | No | Yes — background process | non-empty | `stop(commit=True)` — process killed | `None` immediately (process gone in fresh container) |
| Foreground job | Yes | No — process already exited | empty | `stop(commit=True)` | `None` immediately |
| Foreground spawns child | Yes | Yes — orphaned child | non-empty | `stop(commit=True)` — child killed | `None` immediately (child gone in fresh container) |
| Daemon + `daemon_release` | No | Yes — server PID | moved to `_daemon_pids` | `stop(commit=True)` — server killed | `None` immediately |
| Failed tool (error result or exception) | — | — | — | `stop(commit=False)` — dirty state discarded; next start restores previous lifecycle image | — |
