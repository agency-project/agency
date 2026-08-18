"""Standalone in-container ptrace supervisor entrypoint.

This file is deliberately self-contained -- stdlib + ctypes only, ZERO
imports from the `agency` package and ZERO third-party dependencies
(notably not `pyseccomp`, which the host-side `_seccomp_filter.py` uses --
that requires `libseccomp` to be present in whatever image the sandbox
container happens to be built from, which cannot be assumed). It is written
to a fixed path inside the sandbox container (via `agsandbox`'s existing
`write_file_bytes`) and invoked with `docker exec -i <container> python3
<path>` (or the podman equivalent) by the host-side launcher in
`_in_container_launcher.py`. Every Python 3 install ships `ctypes`, so this
has no dependency beyond "the sandbox image has a `python3` on PATH."

See docs/Design_harness_integration.md's "Prerequisites" (Component 3) for
why this exists: `agproxy_ptrace.launch()` forks from the *calling*
process's own PID namespace, which for a docker/podman-backed sandbox is
the host, not the container. A `docker exec`'d process is attached into the
container's namespaces by the container runtime itself, so a `fork()`
inside THIS process (not the host-side launcher) lands the traced child in
the container's namespace -- the actual fix; a parent cannot relocate an
already-running child into a different namespace after the fact.

Mechanically this replicates `_tracer_loop.py`'s fork/PTRACE_TRACEME/
seccomp-install/execve/waitpid-dispatch loop almost exactly (see that
module's docstrings for the "why" behind each step -- per-thread tracer
identity, WNOHANG polling instead of waitpid(-1, ...), the SIGSTOP
synchronization point, etc. -- none of that is re-explained here). The one
structural difference: instead of calling a same-process Python
`syscall_hook` callback synchronously, each interceptable stop is written
as a JSON line over a Unix domain socket connected back to the host-side
launcher, and this process blocks reading a JSON decision line back over
that same connection -- which is where the real `agpolicy.check()` call
(and the `agent`/sandbox objects it needs) actually lives. See "Protocol"
below.

This connection is NOT `docker exec -i`'s own stdio (an earlier version
used that): a `docker exec -i` pipe is relayed through several extra
hops -- the `docker` CLI client, the Docker daemon's own API connection,
and the container-runtime shim (containerd/runc) each re-buffer the same
bytes -- versus a UDS being one direct kernel-mediated hop between
exactly the two processes on each end. Confirmed empirically: routing
this same protocol over `docker exec -i` stdio stalled permanently on a
later event under real load (a real, heavily multi-threaded harness
process plus its own PreToolUse-hook subprocess churn) in a way a direct
UDS connection carrying the identical protocol did not reproduce.
`docker exec -i` is still used to actually *start* this process inside
the container (there is no way around that -- it is what attaches a new
process into the container's namespaces at all), but nothing needed for
correctness travels over its stdio anymore; that's only drained
best-effort for startup-failure diagnostics on the host side now.

The socket path is bind-mounted into the container the same way
`agproxy_llm`'s own LLM-traffic UDS gateway is (`agsandbox.py` attaches
that directory into every container-backed sandbox unconditionally) --
this entrypoint is handed the *container-side* path into that same
directory as its one command-line argument and connects to it itself,
rather than the host writing anything to this process's stdin.

Protocol (newline-delimited JSON, one object per line, over the UDS
connection):

  host -> entrypoint:
    first line:  {"argv": [...], "envp": {...}, "cwd": "...", "syscalls": [...]}
    thereafter:  {"type": "decision", "kind": "allow"|"deny"|"rewrite",
                  "new_args": [...] | null}
                 -- exactly one decision line per "event" line this process
                 emits, in order; nothing else is ever read from this
                 connection.

  entrypoint -> host:
    {"type": "spawn", "pid": N}
    {"type": "exec", "pid": N, "path": "..." | null}
    {"type": "exit", "pid": N, "code": N}
    {"type": "event", "pid": N, "syscall": "...", "argv": [...] | null,
     "envp": {...} | null, "path": "..." | null, "timestamp": T}
      -- blocks until the matching "decision" line arrives back.
    {"type": "result", "stdout": "...", "stderr": "...", "returncode": N}
      -- always the last line written; process exits immediately after.
    {"type": "error", "message": "..."}
      -- written instead of "result" if launch failed before any process
      ever ran (e.g. fork() itself failed); process exits immediately after.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import signal
import socket
import struct
import sys
import threading
import time

# ---------------------------------------------------------------------------
# ptrace(2) ctypes bindings -- x86_64 only, mirrors
# agproxy_ptrace_internal/_ctypes_defs.py's subset actually needed here.
# Kept as a private copy (not imported) because this file must survive being
# copied alone into an arbitrary container with no `agency` package present.
# ---------------------------------------------------------------------------

if platform.machine() not in ("x86_64", "AMD64"):
    raise RuntimeError(
        f"in-container ptrace entrypoint only supports x86_64 (running on {platform.machine()!r})"
    )

libc = ctypes.CDLL(None, use_errno=True)
libc.ptrace.restype = ctypes.c_long
libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
libc.process_vm_readv.restype = ctypes.c_ssize_t
libc.process_vm_writev.restype = ctypes.c_ssize_t
libc.prctl.restype = ctypes.c_int

PTRACE_TRACEME = 0
PTRACE_PEEKDATA = 2
PTRACE_CONT = 7
PTRACE_GETREGS = 12
PTRACE_SETREGS = 13
PTRACE_SETOPTIONS = 0x4200
PTRACE_GETEVENTMSG = 0x4201

PTRACE_O_TRACEFORK = 0x00000002
PTRACE_O_TRACEVFORK = 0x00000004
PTRACE_O_TRACECLONE = 0x00000008
PTRACE_O_TRACEEXEC = 0x00000010
PTRACE_O_TRACEEXIT = 0x00000040
PTRACE_O_TRACESECCOMP = 0x00000080
ALL_TRACE_OPTIONS = (
    PTRACE_O_TRACEFORK
    | PTRACE_O_TRACEVFORK
    | PTRACE_O_TRACECLONE
    | PTRACE_O_TRACEEXEC
    | PTRACE_O_TRACEEXIT
    | PTRACE_O_TRACESECCOMP
)

PTRACE_EVENT_FORK = 1
PTRACE_EVENT_VFORK = 2
PTRACE_EVENT_CLONE = 3
PTRACE_EVENT_EXEC = 4
PTRACE_EVENT_EXIT = 6
PTRACE_EVENT_SECCOMP = 7

# Same table as _ctypes_defs.SYSCALL_NUMBERS -- kept in sync by hand; see
# that module's comment: verify against
# /usr/include/x86_64-linux-gnu/asm/unistd_64.h, never guess.
SYSCALL_NUMBERS = {
    "execve": 59,
    "execveat": 322,
    "open": 2,
    "openat": 257,
    "connect": 42,
    "unlink": 87,
    "unlinkat": 263,
    "rename": 82,
    "renameat2": 316,
}
SYSCALL_NAMES_BY_NUMBER = {v: k for k, v in SYSCALL_NUMBERS.items()}

_EPERM = 1
REWRITE_SCRATCH_SIZE = 8192


class UserRegsStruct(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "r15",
            "r14",
            "r13",
            "r12",
            "rbp",
            "rbx",
            "r11",
            "r10",
            "r9",
            "r8",
            "rax",
            "rcx",
            "rdx",
            "rsi",
            "rdi",
            "orig_rax",
            "rip",
            "cs",
            "eflags",
            "rsp",
            "ss",
            "fs_base",
            "gs_base",
            "ds",
            "es",
            "fs",
            "gs",
        )
    ]


class _IoVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


def ptrace(request: int, pid: int, addr: int = 0, data: int = 0) -> int:
    ctypes.set_errno(0)
    result = libc.ptrace(request, pid, ctypes.c_void_p(addr), ctypes.c_void_p(data))
    if result == -1 and ctypes.get_errno() != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"ptrace(request={request}, pid={pid}) failed: {os.strerror(errno)}")
    return result


def get_regs(pid: int) -> UserRegsStruct:
    regs = UserRegsStruct()
    ptrace(PTRACE_GETREGS, pid, 0, ctypes.addressof(regs))
    return regs


def set_regs(pid: int, regs: UserRegsStruct) -> None:
    ptrace(PTRACE_SETREGS, pid, 0, ctypes.addressof(regs))


def get_eventmsg(pid: int) -> int:
    msg = ctypes.c_ulong()
    ptrace(PTRACE_GETEVENTMSG, pid, 0, ctypes.addressof(msg))
    return msg.value


def read_bytes(pid: int, addr: int, length: int) -> bytes:
    try:
        buf = ctypes.create_string_buffer(length)
        local = _IoVec(ctypes.cast(buf, ctypes.c_void_p), length)
        remote = _IoVec(ctypes.c_void_p(addr), length)
        ctypes.set_errno(0)
        n = libc.process_vm_readv(pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
        if n < 0:
            raise OSError(ctypes.get_errno(), "process_vm_readv failed")
        return buf.raw[:n]
    except OSError:
        out = bytearray()
        a = addr
        while len(out) < length:
            ctypes.set_errno(0)
            word = libc.ptrace(PTRACE_PEEKDATA, pid, ctypes.c_void_p(a), None)
            if word == -1 and ctypes.get_errno() != 0:
                raise OSError(ctypes.get_errno(), "PTRACE_PEEKDATA failed")
            out += struct.pack("<q", word)
            a += 8
        return bytes(out[:length])


def read_cstring(pid: int, addr: int, max_len: int = 4096) -> str:
    if addr == 0:
        return ""
    raw = read_bytes(pid, addr, max_len)
    return raw.split(b"\x00", 1)[0].decode(errors="replace")


def resolve_argv(pid: int, argv_ptr: int, max_entries: int = 4096) -> "list[str]":
    if argv_ptr == 0:
        return []
    pointers = []
    addr = argv_ptr
    for _ in range(max_entries):
        raw = read_bytes(pid, addr, 8)
        (ptr,) = struct.unpack("<Q", raw)
        if ptr == 0:
            break
        pointers.append(ptr)
        addr += 8
    return [read_cstring(pid, p) for p in pointers]


def resolve_envp(pid: int, envp_ptr: int, max_entries: int = 8192) -> "dict[str, str]":
    entries = resolve_argv(pid, envp_ptr, max_entries)
    result = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if sep:
            result[key] = value
    return result


def write_bytes(pid: int, addr: int, data: bytes) -> None:
    buf = ctypes.create_string_buffer(data, len(data))
    local = _IoVec(ctypes.cast(buf, ctypes.c_void_p), len(data))
    remote = _IoVec(ctypes.c_void_p(addr), len(data))
    ctypes.set_errno(0)
    n = libc.process_vm_writev(pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
    if n != len(data):
        raise OSError(ctypes.get_errno(), f"process_vm_writev wrote {n}/{len(data)} bytes")


def inject_argv(pid: int, rsp: int, path: str, argv: "list[str]") -> "tuple[int, int]":
    scratch = (rsp - REWRITE_SCRATCH_SIZE) & ~0xF
    blob = bytearray()
    str_addrs = []
    for s in argv:
        str_addrs.append(scratch + len(blob))
        blob += s.encode() + b"\x00"
    path_addr = scratch + len(blob)
    blob += path.encode() + b"\x00"
    ptrarr_addr = (scratch + len(blob) + 7) & ~0x7
    if ptrarr_addr - scratch + (len(argv) + 1) * 8 > REWRITE_SCRATCH_SIZE:
        raise ValueError("rewritten argv too large for scratch space")
    ptrs = b"".join(struct.pack("<Q", a) for a in str_addrs) + struct.pack("<Q", 0)
    write_bytes(pid, scratch, bytes(blob))
    write_bytes(pid, ptrarr_addr, ptrs)
    return path_addr, ptrarr_addr


def _resolve_syscall_args(pid: int, regs: UserRegsStruct, syscall_nr: int):
    if syscall_nr == SYSCALL_NUMBERS["execve"]:
        path_ptr, argv_ptr, envp_ptr = regs.rdi, regs.rsi, regs.rdx
        return resolve_argv(pid, argv_ptr), resolve_envp(pid, envp_ptr), read_cstring(pid, path_ptr)
    if syscall_nr == SYSCALL_NUMBERS["execveat"]:
        path_ptr, argv_ptr, envp_ptr = regs.rsi, regs.rdx, regs.r10
        return resolve_argv(pid, argv_ptr), resolve_envp(pid, envp_ptr), read_cstring(pid, path_ptr)
    if syscall_nr == SYSCALL_NUMBERS["open"]:
        return None, None, read_cstring(pid, regs.rdi)
    if syscall_nr == SYSCALL_NUMBERS["openat"]:
        return None, None, read_cstring(pid, regs.rsi)
    return None, None, None


# ---------------------------------------------------------------------------
# Raw BPF seccomp filter -- NOT pyseccomp/libseccomp (see module docstring
# for why: cannot assume libseccomp is installed in an arbitrary sandbox
# image). Hand-built classic BPF program, same shape libseccomp itself
# would generate for "TRACE these syscall numbers, ALLOW everything else":
#
#   [0] LD  nr                      -- load seccomp_data.nr (offset 0)
#   [1..N] JEQ syscalls[i], jt=T    -- jt jumps to the RET TRACE instruction
#                                       at the end if this syscall matches;
#                                       jf=0 falls through to the next check
#   [N+1] RET ALLOW                 -- fallthrough: no syscall matched
#   [N+2] RET TRACE                 -- jump target for every JEQ match
#
# BPF jump offsets are counted from the NEXT instruction, so for a JEQ at
# 0-indexed position i (i = 1..N), jt = (N+2) - (i+1) = N+1-i.
# ---------------------------------------------------------------------------

_BPF_LD, _BPF_W, _BPF_ABS = 0x00, 0x00, 0x20
_BPF_JMP, _BPF_JEQ, _BPF_K = 0x05, 0x10, 0x00
_BPF_RET = 0x06
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_TRACE = 0x7FF00000
_SECCOMP_DATA_NR_OFFSET = 0  # offsetof(struct seccomp_data, nr)

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def install_trace_filter(syscall_names: "list[str]") -> None:
    nrs = [SYSCALL_NUMBERS[name] for name in syscall_names]
    n = len(nrs)
    instrs = [_SockFilter(_BPF_LD | _BPF_W | _BPF_ABS, 0, 0, _SECCOMP_DATA_NR_OFFSET)]
    for i, nr in enumerate(nrs, start=1):
        jt = (n + 1) - i
        instrs.append(_SockFilter(_BPF_JMP | _BPF_JEQ | _BPF_K, jt, 0, nr))
    instrs.append(_SockFilter(_BPF_RET | _BPF_K, 0, 0, _SECCOMP_RET_ALLOW))
    instrs.append(_SockFilter(_BPF_RET | _BPF_K, 0, 0, _SECCOMP_RET_TRACE))

    ArrayType = _SockFilter * len(instrs)
    prog_array = ArrayType(*instrs)
    fprog = _SockFprog(len(instrs), ctypes.cast(prog_array, ctypes.POINTER(_SockFilter)))

    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")


# ---------------------------------------------------------------------------
# UDS protocol -- see module docstring's "Protocol" section. _conn_file is
# set once, in main(), before any of this is used.
# ---------------------------------------------------------------------------

_send_lock = threading.Lock()
_conn_file = None  # socket.makefile("rw"), assigned in main() after connect()


def _send(obj: dict) -> None:
    with _send_lock:
        _conn_file.write(json.dumps(obj) + "\n")
        _conn_file.flush()


def _recv_decision() -> dict:
    # Only ever called from the single tracer thread, immediately after
    # _send({"type": "event", ...}) -- one decision line per event line,
    # in order, so no correlation id is needed on the wire.
    line = _conn_file.readline()
    if not line:
        raise EOFError("host closed the UDS connection while awaiting a decision")
    return json.loads(line)


def _is_thread_group_leader(pid: int) -> bool:
    """Distinguish clone-created processes from CLONE_THREAD threads."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("Tgid:"):
                    return int(line.split(":", 1)[1].strip()) == pid
    except (OSError, ValueError):
        # Losing a genuine subprocess is worse than one conservative false
        # positive on a host with an unexpectedly hidden procfs.
        return True
    return True


def _kernel_executable_path(pid: int) -> "str | None":
    """Return the image installed by the kernel at PTRACE_EVENT_EXEC."""
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Tracer loop -- same shape as agproxy_ptrace_internal/_tracer_loop.py,
# adapted to synchronous stdio instead of an in-process Python callback.
# ---------------------------------------------------------------------------


class _Tracer:
    def __init__(self, syscalls: "list[str]") -> None:
        self._syscalls = syscalls
        self.root_pid: "int | None" = None
        self._known_pids: "set[int]" = set()
        self._process_pids: "set[int]" = set()
        self._pending_clone_pids: "set[int]" = set()
        self._pending_exec_paths: "dict[int, str | None]" = {}
        self._options_applied: "set[int]" = set()
        self._returncode = None
        self._stdout_buf = bytearray()
        self._stderr_buf = bytearray()
        self._buf_lock = threading.Lock()

    def run(self, argv: "list[str]", envp: "dict[str, str]", cwd: str) -> None:
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()

        pid = os.fork()
        if pid == 0:
            self._child_exec(argv, envp, cwd, stdout_r, stdout_w, stderr_r, stderr_w)
            os._exit(127)  # unreachable

        os.close(stdout_w)
        os.close(stderr_w)
        self.root_pid = pid
        self._remember_spawn(pid)

        threading.Thread(
            target=self._drain_pipe, args=(stdout_r, self._stdout_buf), daemon=True
        ).start()
        threading.Thread(
            target=self._drain_pipe, args=(stderr_r, self._stderr_buf), daemon=True
        ).start()

        _, status = os.waitpid(pid, 0)
        assert os.WIFSTOPPED(status), f"expected initial stop, got status={status:#x}"
        ptrace(PTRACE_SETOPTIONS, pid, 0, ALL_TRACE_OPTIONS)
        self._options_applied.add(pid)
        ptrace(PTRACE_CONT, pid, 0, 0)

        self._wait_loop()

    def _child_exec(self, argv, envp, cwd, stdout_r, stdout_w, stderr_r, stderr_w) -> None:
        os.close(stdout_r)
        os.close(stderr_r)
        os.dup2(stdout_w, 1)
        os.dup2(stderr_w, 2)
        os.close(stdout_w)
        os.close(stderr_w)
        # The traced target always receives its prompt via argv, never
        # stdin -- but without this, it inherits this entrypoint's own fd 0,
        # which is `docker exec -i`'s pipe (no longer used for anything
        # since the switch to the UDS control channel, but still open and
        # unfed). Newer Claude Code CLI builds detect that non-tty stdin
        # and stall for a few seconds waiting for data that will never
        # arrive before giving up (confirmed against the real CLI).
        devnull_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull_fd, 0)
        os.close(devnull_fd)
        if cwd:
            os.chdir(cwd)
        ptrace(PTRACE_TRACEME, 0, 0, 0)
        os.kill(os.getpid(), signal.SIGSTOP)
        install_trace_filter(self._syscalls)
        try:
            os.execve(argv[0], argv, dict(envp))
        except BaseException as exc:
            os.write(2, f"in_container_entrypoint: execve({argv[0]!r}) failed: {exc!r}\n".encode())
            os._exit(126)

    def _drain_pipe(self, fd: int, buf: bytearray) -> None:
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            with self._buf_lock:
                buf += chunk
        try:
            os.close(fd)
        except OSError:
            pass

    def _wait_loop(self) -> None:
        # Wildcard-reap (waitpid(-1, ...)), not per-known-pid polling: a
        # ptrace tracer sees status changes for every tracee via -1
        # regardless of direct-parentage (a fork/clone-attached grandchild
        # included), and draining -1 until it returns 0 guarantees every
        # pending zombie is reaped before the kernel could ever hand its
        # number to a new process -- individually polling waitpid(pid,
        # WNOHANG) per known pid instead left a window where a just-exited
        # pid's own slot hadn't been reaped yet while the tracer was
        # blocked elsewhere (in _handle_seccomp_stop's decision round
        # trip), so a *different*, newly-forked process could be assigned
        # that exact recycled pid number before this loop ever saw the
        # original's exit -- then got misdispatched as if it were the old,
        # already-`_options_applied` process continuing, instead of a
        # brand-new one needing PTRACE_SETOPTIONS + its own initial
        # PTRACE_CONT, leaving it permanently stuck in tracing-stop.
        while self._known_pids:
            made_progress = False
            while True:
                try:
                    got_pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    self._known_pids.clear()
                    break
                if got_pid == 0:
                    break
                made_progress = True
                self._dispatch(got_pid, status)
            if not made_progress:
                time.sleep(0.002)

    def _dispatch(self, pid: int, status: int) -> None:
        if os.WIFEXITED(status):
            self._forget(pid, os.WEXITSTATUS(status))
            return
        if os.WIFSIGNALED(status):
            self._forget(pid, -os.WTERMSIG(status))
            return
        assert os.WIFSTOPPED(status), (pid, status)
        sig = os.WSTOPSIG(status)
        event = (status >> 16) & 0xFF
        if os.environ.get("AGENCY_DEBUG_PTRACE_EVENTS"):
            try:
                sig_name = signal.Signals(sig).name if sig else sig
            except ValueError:
                sig_name = sig
            try:
                with open("/tmp/.agency_ptrace_debug.log", "a") as _f:
                    _f.write(f"pid={pid} status={status:#x} sig={sig_name} event={event}\n")
            except OSError:
                pass

        if pid not in self._options_applied:
            self._classify_pending_clone(pid)
            ptrace(PTRACE_SETOPTIONS, pid, 0, ALL_TRACE_OPTIONS)
            self._options_applied.add(pid)
            ptrace(PTRACE_CONT, pid, 0, 0)
            return

        if sig == signal.SIGTRAP and event == PTRACE_EVENT_SECCOMP:
            self._handle_seccomp_stop(pid)
            return
        if sig == signal.SIGTRAP and event in (
            PTRACE_EVENT_FORK,
            PTRACE_EVENT_VFORK,
            PTRACE_EVENT_CLONE,
        ):
            new_pid = get_eventmsg(pid)
            self._remember_spawn(
                new_pid,
                is_process=None if event == PTRACE_EVENT_CLONE else True,
            )
            ptrace(PTRACE_CONT, pid, 0, 0)
            return
        if sig == signal.SIGTRAP and event == PTRACE_EVENT_EXEC:
            self._commit_exec(pid)
            ptrace(PTRACE_CONT, pid, 0, 0)
            return
        if sig == signal.SIGTRAP and event == PTRACE_EVENT_EXIT:
            ptrace(PTRACE_CONT, pid, 0, 0)
            return
        forward = 0 if sig == signal.SIGTRAP else sig
        ptrace(PTRACE_CONT, pid, 0, forward)

    def _handle_seccomp_stop(self, pid: int) -> None:
        regs = get_regs(pid)
        nr = regs.orig_rax
        name = SYSCALL_NAMES_BY_NUMBER.get(nr, f"nr:{nr}")
        argv, envp, path = _resolve_syscall_args(pid, regs, nr)

        if os.environ.get("AGENCY_DEBUG_PTRACE_EVENTS"):
            try:
                with open("/tmp/.agency_ptrace_debug.log", "a") as _f:
                    _f.write(
                        f"SECCOMP pid={pid} syscall={name} argv={argv} path={path} -- sending event...\n"
                    )
            except OSError:
                pass
        _send(
            {
                "type": "event",
                "pid": pid,
                "syscall": name,
                "argv": argv,
                "envp": envp,
                "path": path,
                "timestamp": time.time(),
            }
        )
        decision = _recv_decision()
        if os.environ.get("AGENCY_DEBUG_PTRACE_EVENTS"):
            try:
                with open("/tmp/.agency_ptrace_debug.log", "a") as _f:
                    _f.write(f"SECCOMP pid={pid} syscall={name} -- got decision={decision}\n")
            except OSError:
                pass
        kind = decision.get("kind", "allow")
        is_exec = nr in (SYSCALL_NUMBERS["execve"], SYSCALL_NUMBERS["execveat"])

        if kind == "deny":
            regs.orig_rax = ctypes.c_ulonglong(-1).value
            set_regs(pid, regs)
            regs2 = get_regs(pid)
            regs2.rax = ctypes.c_ulonglong((-_EPERM) & 0xFFFFFFFFFFFFFFFF).value
            set_regs(pid, regs2)
        elif kind == "rewrite" and decision.get("new_args"):
            if nr in (SYSCALL_NUMBERS["execve"], SYSCALL_NUMBERS["execveat"]):
                new_args = decision["new_args"]
                path_addr, argv_addr = inject_argv(pid, regs.rsp, new_args[0], new_args)
                if nr == SYSCALL_NUMBERS["execve"]:
                    regs.rdi = path_addr
                    regs.rsi = argv_addr
                else:
                    regs.rsi = path_addr
                    regs.rdx = argv_addr
                set_regs(pid, regs)
        if is_exec:
            # Stage the kernel pathname at syscall entry, but do not report it
            # until PTRACE_EVENT_EXEC proves the image was installed.
            if kind == "deny":
                self._pending_exec_paths.pop(pid, None)
            else:
                new_args = decision.get("new_args")
                self._pending_exec_paths[pid] = (
                    new_args[0] if kind == "rewrite" and new_args else path
                )
        ptrace(PTRACE_CONT, pid, 0, 0)

    def _remember_spawn(self, pid: int, *, is_process: "bool | None" = True) -> None:
        self._known_pids.add(pid)
        if is_process is None:
            self._pending_clone_pids.add(pid)
        elif is_process:
            self._process_pids.add(pid)
            _send({"type": "spawn", "pid": pid})

    def _classify_pending_clone(self, pid: int) -> None:
        if pid not in self._pending_clone_pids:
            return
        self._pending_clone_pids.discard(pid)
        if _is_thread_group_leader(pid):
            self._process_pids.add(pid)
            _send({"type": "spawn", "pid": pid})

    def _commit_exec(self, pid: int) -> None:
        procfs_path = _kernel_executable_path(pid)
        staged_path = self._pending_exec_paths.pop(pid, None)
        executable_path = staged_path or procfs_path
        if pid in self._process_pids:
            _send({"type": "exec", "pid": pid, "path": executable_path})

    def _forget(self, pid: int, exit_code: int) -> None:
        self._known_pids.discard(pid)
        self._pending_clone_pids.discard(pid)
        self._pending_exec_paths.pop(pid, None)
        self._options_applied.discard(pid)
        if pid == self.root_pid:
            self._returncode = exit_code
        if pid in self._process_pids:
            self._process_pids.discard(pid)
            _send({"type": "exit", "pid": pid, "code": exit_code})


def main() -> None:
    global _conn_file

    if len(sys.argv) < 2:
        # No connection to _send an error over yet -- best-effort stderr,
        # matching the "always report, never crash silently" intent as far
        # as it can go without a channel to report it on.
        os.write(2, b"in_container_entrypoint: missing UDS socket path argument\n")
        return
    sock_path = sys.argv[1]

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + 10
    last_exc: "Exception | None" = None
    connected = False
    while time.monotonic() < deadline:
        try:
            sock.connect(sock_path)
            connected = True
            break
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            last_exc = exc
            time.sleep(0.05)
    if not connected:
        os.write(
            2,
            f"in_container_entrypoint: could not connect to {sock_path!r}: {last_exc!r}\n".encode(),
        )
        return
    _conn_file = sock.makefile("rw")

    first_line = _conn_file.readline()
    if not first_line:
        _send({"type": "error", "message": "no launch spec received over the UDS connection"})
        return
    spec = json.loads(first_line)

    tracer = _Tracer(syscalls=spec.get("syscalls") or ["execve", "execveat"])
    try:
        tracer.run(spec["argv"], spec.get("envp") or {}, spec.get("cwd") or "")
    except BaseException as exc:  # noqa: BLE001 -- always report, never crash silently
        _send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return

    stdout = bytes(tracer._stdout_buf).decode(errors="replace")
    stderr = bytes(tracer._stderr_buf).decode(errors="replace")
    _send(
        {
            "type": "result",
            "stdout": stdout,
            "stderr": stderr,
            "returncode": tracer._returncode or 0,
        }
    )


if __name__ == "__main__":
    main()
