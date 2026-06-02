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

## Case 1: Background job (`python train.py &`)

### Tool call → exec wrapper

The `&` means the shell starts `train.py` and continues without waiting.

1. `/proc` before-snapshot taken
2. `python train.py &` — shell starts the process and returns immediately
3. `/proc` after-diff — `train.py` is still alive → added to `__BGPIDS__`
4. `_watched_pids[train_pid] = now`
5. `exec()` returns `("", 0)` — the LLM sees empty output
6. LLM produces its final answer; `agskill.run()` returns

### Outer loop

```
pids_at_end = {train_pid}   ← non-empty, don't break

log: PROCS ▶  monitoring: PID <train_pid> (running 0m 0s)
log: procs_started  (file)

poll get_live_pids() every poll_interval_s (5s) for up to ping_interval_s (5min)
  → train_pid alive each check, full window elapses

live_now = {train_pid}      ← still running after ping_interval_s

log: PROCS ⏳  still running: PID <train_pid> (running 5m 0s)
log: procs_ping  (file)
inject: _event="process_update"

─── re-enter ReAct loop ───────────────────────────────────────
LLM receives: "Background processes still running: PID <train_pid>..."
LLM calls: bash({"command": "tail -20 train.log"})
LLM decides to keep waiting
agskill.run() returns

poll get_live_pids() every poll_interval_s ... train_pid exits mid-poll → break immediately

log: PROCS ✓  all processes completed, re-entering agent
log: procs_completed  (file)
inject: _event="process_completed"

─── re-enter ReAct loop ───────────────────────────────────────
LLM receives: "Background processes have completed."
LLM calls: bash({"command": "cat results.json"})
LLM reads output, produces final result
agskill.run() returns

pids_at_end = {}   ← break

release_resources()
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
6. LLM reads the result, produces final answer; `agskill.run()` returns

### Outer loop

```
pids_at_end = {}   ← empty, break immediately

release_resources()
aglog._record()
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
6. LLM produces final answer; `agskill.run()` returns

### Outer loop

Identical to Case 1 from this point. The outer loop monitors `train_pid`, pings the agent with `process_update` while it runs, and re-enters with `process_completed` when it exits.

The LLM never explicitly launched train.py — it called a foreground script. The `/proc` diff is what captures the orphaned child.

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

Returns `agdata(message="PID X released as daemon — will not block skill completion")`. LLM produces final answer; `agskill.run()` returns.

### Outer loop

```
pids_at_end = set(_watched_pids)   ← empty (server_pid moved to _daemon_pids)
break immediately

release_resources()
result_future.set_result()   ← caller unblocks
```

The server keeps running. If it later spawns worker processes, `get_live_pids()` reads their PPid from `/proc/status`, finds it traces to `_daemon_pids`, and propagates daemon status — workers are also excluded from monitoring.

The server and all its descendants are killed when `sandbox.destroy()` is called from `agent.__del__`.

---

## `get_live_pids()` — baseline diff approach

Called during the outer loop's polling window. Reads the full `/proc` table in one pass and returns every PID that is:
- **not** in `_baseline_pids` (the container's process set at sandbox creation), and
- **not** in `_daemon_pids` (or descended from one), and
- **not** in zombie state (`State: Z` in `/proc/<pid>/status`)

Any newly discovered non-baseline PID is added to `_watched_pids` with the current timestamp. This handles the case where a watched process spawns new children after `exec()` returns — the children appear in `/proc` on the next `get_live_pids()` call and are automatically tracked.

---

## Summary

| Scenario | `docker exec` blocks? | `/proc` diff finds PIDs? | `_watched_pids` after exec | Outer loop |
|---|---|---|---|---|
| Background job (`&`) | No | Yes — background process | non-empty | `process_update` pings → `process_completed` re-entry |
| Foreground job | Yes | No — process already exited | empty | `break` immediately |
| Foreground spawns child | Yes | Yes — orphaned child | non-empty | Same as background job |
| Daemon + `daemon_release` | No | Yes — server PID | moved to `_daemon_pids` | `break` immediately |
