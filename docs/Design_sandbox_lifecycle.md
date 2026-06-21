# Execution: Process Control

This document traces what happens in each process-spawning scenario from the moment a bash tool call is made through to the end of the agent run.

---

## Shared path: tool call → exec wrapper

Every bash tool call follows the same path into the container:

```
agskill.run()
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

## Process monitoring inside agskill

Background process monitoring is embedded directly in `agskill.run()`'s ReAct loop (not in an outer caller loop). Each time the LLM produces a valid final answer, the framework calls `_wait_for_processes()` before returning:

```python
# inside agskill.run(), after output schema validation passes:
if far.kind != "error" and sandbox is not None:
    proc_msg = _wait_for_processes(
        sandbox, self.name, term, log, agname,
        _ping_interval_s, _poll_interval_s, _state_fn,
    )
    if proc_msg is not None:
        messages.append({"role": "user", "content": proc_msg})
        continue   # re-enter loop — LLM sees process status and acts

return far.return_tuple   # sandbox clean → truly done
```

`_wait_for_processes()` returns `None` immediately if `_watched_pids` is empty or `get_live_pids()` finds no active processes. Otherwise it polls until either all PIDs exit or `_ping_interval_s` elapses:

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
6. LLM produces its final answer; output schema validation passes

### Process monitoring (inside agskill)

```
_wait_for_processes() called — train_pid is in _watched_pids

log: PROCS ▶  monitoring: PID <train_pid> (running 0m 0s)
log: procs_started  (lifecycle event)

poll get_live_pids() every poll_interval_s (5s) for up to ping_interval_s (5min)
  → train_pid alive each check, full window elapses

live_now = {train_pid}      ← still running after ping_interval_s

log: PROCS ⏳  still running: PID <train_pid> (running 5m 0s)
log: procs_ping  (lifecycle event)

proc_msg = "Background processes are still running: PID <train_pid>..."
→ appended as user message, loop continues

─── LLM re-entry ─────────────────────────────────────────────────────────────
LLM receives: "Background processes are still running: PID <train_pid>..."
LLM calls: bash({"command": "tail -20 train.log"})
LLM produces final answer

_wait_for_processes() called again
poll get_live_pids() ... train_pid exits mid-poll → break immediately

log: PROCS ✓  all processes completed, re-entering agent
log: procs_completed  (lifecycle event)

proc_msg = "Background processes have completed. Read their output..."
→ appended as user message, loop continues

─── LLM re-entry ─────────────────────────────────────────────────────────────
LLM receives: "Background processes have completed."
LLM calls: bash({"command": "cat results.json"})
LLM reads output, produces final result

_wait_for_processes() called — _watched_pids is empty → returns None
→ far.return_tuple returned

sandbox commit + destroy
aglog._record()
result_future.set_result()   ← caller unblocks
```

---

## Case 2: Foreground job (`python eval.py`)

### Tool call → exec wrapper

No `&` — the wrapper shell blocks on `python eval.py` until it exits.

1. `/proc` before-snapshot taken
2. `python eval.py` — **blocks** until eval.py exits
3. `/proc` after-diff — eval.py is already gone from `/proc`; diff is empty
4. `_watched_pids` unchanged
5. `exec()` returns `(full_stdout_of_eval, rc)` — LLM sees the output inline
6. LLM reads the result, produces final answer

### Process monitoring (inside agskill)

```
_wait_for_processes() called — _watched_pids is empty → returns None immediately
→ far.return_tuple returned

sandbox commit + destroy
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
6. LLM produces final answer

### Process monitoring (inside agskill)

Identical to Case 1 from this point. `_wait_for_processes` detects `train_pid` in `_watched_pids`, polls, pings the LLM with status messages, and re-enters until the process exits.

The LLM never explicitly launched `train.py` — it called a foreground script. The `/proc` diff is what captures the orphaned child.

---

## Case 4: Daemon job (`python server.py &`, then `daemon_release(pid)`)

### Tool call 1 — start the server

Same path as Case 1. Server PID added to `_watched_pids`. LLM sees empty output.

### Tool call 2 — `daemon_release(pid)`

LLM calls `daemon_release({"pid": <server_pid>})`:

```
make_daemon_release._run(arg)
  └─ sandbox.release_daemon(server_pid)
       ├─ _daemon_pids.add(server_pid)
       └─ _watched_pids.pop(server_pid)
```

Returns `agdata(message="PID X released as daemon — will not block skill completion")`. LLM produces final answer.

### Process monitoring (inside agskill)

```
_wait_for_processes() called — _watched_pids is empty (server_pid moved to _daemon_pids)
→ returns None immediately

sandbox commit + destroy
result_future.set_result()   ← caller unblocks
```

The server keeps running. If it later spawns worker processes, `get_live_pids()` reads their PPid from `/proc/status`, finds it traces to `_daemon_pids`, and propagates daemon status — workers are also excluded from monitoring.

The server and all its descendants are killed when `sandbox.destroy()` is called from the `finally` block in `_task`.

---

## `get_live_pids()` — baseline diff approach

Called during `_wait_for_processes`'s polling window. Reads the full `/proc` table in one pass and returns every PID that is:
- **not** in `_baseline_pids` (the container's process set at sandbox creation), and
- **not** in `_daemon_pids` (or descended from one), and
- **not** in zombie state (`State: Z` in `/proc/<pid>/status`)

Any newly discovered non-baseline PID is added to `_watched_pids` with the current timestamp. This handles the case where a watched process spawns new children after `exec()` returns — the children appear in `/proc` on the next `get_live_pids()` call and are automatically tracked.

---

## Summary

| Scenario | `docker exec` blocks? | `/proc` diff finds PIDs? | `_watched_pids` after exec | `_wait_for_processes` result |
|---|---|---|---|---|
| Background job (`&`) | No | Yes — background process | non-empty | procs_ping message(s) → procs_completed message → `None` |
| Foreground job | Yes | No — process already exited | empty | `None` immediately |
| Foreground spawns child | Yes | Yes — orphaned child | non-empty | Same as background job |
| Daemon + `daemon_release` | No | Yes — server PID | moved to `_daemon_pids` | `None` immediately |
