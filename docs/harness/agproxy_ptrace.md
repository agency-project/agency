# Syscall-level supervisor (`harness/ptrace/`)

> See [Design_harness_integration.md](../Design_harness_integration.md) ("Component 3") for the
> full design rationale — this doc covers the concrete implementation.

`agProxyPtrace.launch()` starts a target process the way a debugger does: `fork()`, the child
calls `PTRACE_TRACEME`, installs a seccomp filter that traps a small set of syscalls
(`SECCOMP_RET_TRACE`), then `execve()`s the real binary. The parent becomes that process's tracer
and, transitively, tracer of everything it forks/execs (`PTRACE_O_TRACEFORK`/`_VFORK`/`_CLONE`),
observing every trapped syscall before it runs — with no cooperation required from the traced
binary, unlike a harness's own hook system.

x86_64 Linux only — `seccomp`+`PTRACE_EVENT_SECCOMP` has no macOS/BSD equivalent. The
architecture-specific implementation is loaded only when a launch begins; `ptrace_available()`
returns `False` without importing it on unsupported hosts.

## Public API

```python
from agency.harness.ptrace.supervisor import agProxyPtrace, ptrace_available

class MyPolicy:
    def check(self, ag, event) -> bool | tuple[bool, str]:
        if event.argv and event.argv[0] == "/bin/rm":
            return (False, "no rm allowed")
        return True

px = agProxyPtrace()
handle = px.launch(["some-cli", "--flag"], envp={"PATH": "/usr/bin"}, cwd="/workspace", policy=MyPolicy())
stdout, stderr, returncode = handle.wait(timeout=300)
```

- `agProxyPtrace(agconfig=None)` — holds config only; reusable across multiple `launch()` calls.
- `.launch(argv, envp, *, cwd, policy, ag=None)` — forks, traces, and returns a
  `agProxyPtraceHandle` immediately (non-blocking; the traced process runs concurrently).
- `agProxyPtraceHandle.wait(timeout=None)` — blocks until the root process exits; returns
  `(stdout, stderr, returncode)`, deliberately in the same family as `agSandbox.exec()`'s
  `(str, int)` shape (see [agsandbox.md](../agsandbox.md)). A timeout is a polling result
  (`returncode == -1`), not termination: the handle may be waited on again and live agprof
  process spans remain open.
- `.pids()` — currently-live traced pids.
- `.on_spawn(callback)` / `.on_exec(callback)` / `.on_exit(callback)` — fire for every process
  the traced tree forks, successfully execs, or exits, not just the root. Exec callbacks receive
  `(pid, executable_path)` only after `PTRACE_EVENT_EXEC`; the staged exec pathname is preferred
  (so a shebang launcher is named for the requested script), with the stopped process's
  `/proc/<pid>/exe` link as fallback. It is never taken from attacker-controlled `argv[0]`, and
  credential-shaped basenames or names containing a registered launch credential are redacted
  before agprof records them. An exec
  candidate that fails produces no callback. `PTRACE_EVENT_CLONE` tracees are checked by thread-group ID;
  non-leader threads remain traced internally but do not reach these process callbacks. Registering
  after some events have already happened still replays them — see `TracerLoop`'s backlog lists.
  Pass `include_exit_code=True` to `.on_exit()` for `(pid, exit_code)`. These events also feed
  agprof lifecycle spans.
- Ptrace runs in the sandbox Harness Manager daemon, so process lifecycle timestamps are local
  to the tracer and carry `timing="exact"`; there is no separate relay protocol.
- `.kill()` — SIGKILLs every currently-known traced pid.
- `ptrace_available()` — process-lifetime-cached probe (mirrors
  `sandbox/chroot.py`'s `chroot_available()`): a live fork+`PTRACE_TRACEME` smoke test,
  not just an architecture/sysctl check, since a stricter LSM policy can deny ptrace even when
  `/proc/sys/kernel/yama/ptrace_scope` alone would suggest it's fine.

## `agsyscallevent`

```python
@dataclass
class agsyscallevent:
    syscall: str            # "execve", "openat", ...
    pid: int
    tid: int
    argv: list[str] | None  # resolved for execve/execveat
    envp: dict[str, str] | None
    path: str | None
    timestamp: float
```

`execve`/`execveat` argument and `openat`/`open` path resolution are implemented.

## Config

`agPtraceConfig` (`_OWNER = "agproxy_ptrace"`), following the same `_AgXxxFields` /
`agXxxConfig(_AgConfigViewBase)` pattern as every other backend family — see
[agconfig.md](../agconfig.md).

| Field | Tier | Default | Purpose |
|---|---|---|---|
| `syscalls` | dynamic | `("execve", "execveat")` | Which syscalls the seccomp filter traps. |
| `profiler` | dynamic | `None` | Reserved for a future heavyweight process profiler (for example `perf`). The low-cost agprof process-lifecycle spans are automatic whenever an agprof session is active and do not consume this selector. |
| `disable_harness_native_sandbox` | dynamic | `True` | Advisory only — see the design doc's "Design Tensions" on seccomp filter stacking. |
| `attach_timeout_s` | global | `30` | Ceiling on waiting for the traced process's initial post-`TRACEME` stop. |

## Implementation notes (`harness/ptrace/`)

- **`_ctypes_defs.py`** — raw `ptrace()`/`process_vm_readv()`/`process_vm_writev()` bindings.
  Every FFI call site sets `restype`/`argtypes` explicitly: an unconfigured `ctypes` foreign
  function defaults to a 32-bit `c_int` return, which silently truncates the 64-bit word
  `PTRACE_PEEKDATA` returns — this was hit and fixed during development, not a hypothetical.
  `read_bytes()` tries `process_vm_readv` first, falling back to word-at-a-time
  `PTRACE_PEEKDATA` if that fails (exercised directly in the test suite by monkeypatching
  `process_vm_readv` to raise).
- **`_seccomp_filter.py`** — thin wrapper over `pyseccomp.SyscallFilter(defaction=ALLOW)` +
  `TRACE(0)` rules. Must be installed in the traced child *after* the parent has already applied
  `PTRACE_O_TRACESECCOMP` (via `PTRACE_SETOPTIONS`) to that child — a filtered syscall fires
  before the tracer has set that option and it fails with `ENOSYS` instead of trapping. The child
  synchronizes by raising `SIGSTOP` on itself right after `PTRACE_TRACEME` and blocking until the
  parent's `PTRACE_CONT` resumes it (by which point `PTRACE_SETOPTIONS` has already run).
- **`_tracer_loop.py`** — the `waitpid`/ptrace-stop dispatch loop. Two load-bearing details:
  - **Runs entirely on one dedicated thread.** ptrace's tracer identity is per-*thread*, not
    per-process — only the thread that attaches (via `TRACEME`) may subsequently
    `ptrace()`/`waitpid()` that tracee. `fork()` and the dispatch loop that follows it must
    therefore happen on the same thread for the lifetime of a launch; `TracerLoop.start()`
    enforces this by doing the fork itself inside the thread it spawns, not on the caller's
    thread.
  - **Polls `waitpid(pid, WNOHANG)` per known pid, not `waitpid(-1, ...)`.** `waitpid(-1, ...)`
    reaps exit status for *any* child of the calling process — in a process that also spawns
    subprocesses elsewhere (a `ProcessPoolExecutor`, `subprocess.run` for `docker`/`podman`, ...),
    that would race with and could steal the exit status those other call sites are waiting on.
    Verified during development against a concurrent unrelated `subprocess.Popen` child, which
    was reaped correctly through `subprocess`'s own machinery, untouched by this loop.
  - A denied syscall is skipped (`orig_rax = -1`) and reports `EPERM`; an allowed syscall resumes
    normally. The current `agpolicy` contract is allow/deny only.
  - Output (stdout/stderr) is drained continuously by dedicated reader threads for the lifetime
    of the launch, not just inside `wait()` — a traced process that writes more than one pipe
    buffer's worth of output before anyone reads it would otherwise deadlock.

## Known gap surfaced during implementation

Error-reporting from inside the forked child (e.g. a failed `execve()`) writes directly to raw
fd 2 via `os.write(2, ...)`, not through `sys.stderr`/`print()`. A test runner that captures
output (pytest) monkeypatches `sys.stderr` to a Python-level buffer object *before* `fork()`; the
forked child inherits that same monkeypatched object, and writing through it never reaches the
real fd this process's stderr was `dup2`'d onto — the message would silently vanish under
capture instead of surfacing in `wait()`'s returned `stderr`. Any future code path that reports
an error from inside the traced child (before `execve()` replaces its image) must write to the
raw fd, not through Python's buffered `sys.stderr`.
