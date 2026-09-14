"""Raw ctypes bindings for ptrace(2) on Linux/x86_64.

x86_64-only for v1 -- `user_regs_struct`'s field layout and the syscall
numbers in `SYSCALL_NUMBERS` are both architecture-specific. ARM64 (or other
architectures) would need a parallel struct/table here; `_arch_guard()`
raises a clear error rather than silently reading garbage registers on an
unsupported architecture.

Every raw ptrace() call site sets `restype`/`argtypes` explicitly -- ctypes
defaults an unconfigured foreign function's return type to `c_int` (32-bit),
which silently truncates the 64-bit word returned by `PTRACE_PEEKDATA` (the
actual data read from tracee memory), corrupting any string or
pointer read back through it. This was caught by hand during development
(see the design doc's implementation notes) and is exactly the kind of bug
that reproduces silently rather than raising, so it is asserted here once at
import time via `_configure_libc()` instead of trusting each call site.
"""

from __future__ import annotations

import ctypes
import platform
import socket
import struct


def _arch_guard() -> None:
    if platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError(
            f"agproxy_ptrace only supports x86_64 in this version "
            f"(running on {platform.machine()!r})"
        )


_arch_guard()

libc = ctypes.CDLL(None, use_errno=True)


def _configure_libc() -> None:
    libc.ptrace.restype = ctypes.c_long
    libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
    libc.process_vm_readv.restype = ctypes.c_ssize_t
    libc.process_vm_writev.restype = ctypes.c_ssize_t


_configure_libc()

# ---------------------------------------------------------------------------
# ptrace(2) request / option / event constants (Linux, from <sys/ptrace.h>)
# ---------------------------------------------------------------------------

PTRACE_TRACEME = 0
PTRACE_PEEKDATA = 2
PTRACE_CONT = 7
PTRACE_SYSCALL = 24
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

PTRACE_EVENT_FORK = 1
PTRACE_EVENT_VFORK = 2
PTRACE_EVENT_CLONE = 3
PTRACE_EVENT_EXEC = 4
PTRACE_EVENT_EXIT = 6
PTRACE_EVENT_SECCOMP = 7

# Linux waitpid flags are not exposed by Python's os module.
WAIT_ALL = 0x40000000
WAIT_NOTHREAD = 0x20000000

# The full set of PTRACE_O_TRACE* lifecycle options agproxy_ptrace always
# requests -- every new thread/process a traced tree creates gets
# auto-attached, and PTRACE_O_TRACESECCOMP is what turns a SECCOMP_RET_TRACE
# action into a PTRACE_EVENT_SECCOMP stop instead of failing the syscall with
# ENOSYS (see seccomp(2)).
ALL_TRACE_OPTIONS = (
    PTRACE_O_TRACEFORK
    | PTRACE_O_TRACEVFORK
    | PTRACE_O_TRACECLONE
    | PTRACE_O_TRACEEXEC
    | PTRACE_O_TRACEEXIT
    | PTRACE_O_TRACESECCOMP
)

# x86_64 syscall numbers for the syscalls agproxy_ptrace knows how to filter
# on and resolve arguments for. Extend this table (and the resolver in
# _tracer_loop.py) to add more -- do not guess a number, verify against
# /usr/include/x86_64-linux-gnu/asm/unistd_64.h or the kernel's syscall table.
SYSCALL_NUMBERS: dict[str, int] = {
    "read": 0,
    "pread64": 17,
    "readv": 19,
    "execve": 59,
    "execveat": 322,
    "open": 2,
    "openat": 257,
    "connect": 42,
    "bind": 49,
    "sendto": 44,
    "unlink": 87,
    "unlinkat": 263,
    "rename": 82,
    "renameat2": 316,
}
SYSCALL_NAMES_BY_NUMBER: dict[int, str] = {v: k for k, v in SYSCALL_NUMBERS.items()}


class UserRegsStruct(ctypes.Structure):
    """Mirrors x86_64 Linux's `struct user_regs_struct`
    (<sys/user.h>) -- field order matters, it is a raw memory layout."""

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


class PtraceError(OSError):
    """Raised when a ptrace(2) call fails; carries the raw errno."""


def ptrace(request: int, pid: int, addr: int = 0, data: int = 0) -> int:
    ctypes.set_errno(0)
    result = libc.ptrace(request, pid, ctypes.c_void_p(addr), ctypes.c_void_p(data))
    if result == -1 and ctypes.get_errno() != 0:
        errno = ctypes.get_errno()
        raise PtraceError(
            errno, f"ptrace(request={request}, pid={pid}) failed: {os_strerror(errno)}"
        )
    return result


def os_strerror(errno: int) -> str:
    import os

    return os.strerror(errno)


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
    """Read *length* bytes from the tracee's memory at *addr*. Tries
    process_vm_readv first (no ptrace-stop overhead); falls back to
    word-at-a-time PTRACE_PEEKDATA if a stricter LSM denies it."""
    try:
        return _read_bytes_vm_readv(pid, addr, length)
    except OSError:
        return _read_bytes_peekdata(pid, addr, length)


def _read_bytes_vm_readv(pid: int, addr: int, length: int) -> bytes:
    buf = ctypes.create_string_buffer(length)
    local = _IoVec(ctypes.cast(buf, ctypes.c_void_p), length)
    remote = _IoVec(ctypes.c_void_p(addr), length)
    ctypes.set_errno(0)
    n = libc.process_vm_readv(pid, ctypes.byref(local), 1, ctypes.byref(remote), 1, 0)
    if n < 0:
        raise OSError(ctypes.get_errno(), "process_vm_readv failed")
    return buf.raw[:n]


def _read_bytes_peekdata(pid: int, addr: int, length: int) -> bytes:
    out = bytearray()
    a = addr
    while len(out) < length:
        ctypes.set_errno(0)
        word = libc.ptrace(PTRACE_PEEKDATA, pid, ctypes.c_void_p(a), None)
        if word == -1 and ctypes.get_errno() != 0:
            raise PtraceError(ctypes.get_errno(), "PTRACE_PEEKDATA failed")
        out += struct.pack("<q", word)
        a += 8
    return bytes(out[:length])


def read_cstring(pid: int, addr: int, max_len: int = 4096) -> str:
    """Read a NUL-terminated C string from the tracee's memory."""
    if addr == 0:
        return ""
    raw = read_bytes(pid, addr, max_len)
    return raw.split(b"\x00", 1)[0].decode(errors="replace")


# sizeof(struct sockaddr_in6) -- the largest of the two shapes this decodes,
# so a single capped read covers either one.
_SOCKADDR_MAX_LEN = 28


def read_sockaddr(pid: int, addr: int, addrlen: int) -> "tuple[str | None, int | None]":
    """Decode a `struct sockaddr*` argument into (ip, port) -- IPv4/IPv6
    only. Deliberately reads only the address struct, never any data
    buffer a caller (e.g. sendto()) might pass alongside it: this is for
    connection metadata (who a syscall is talking to), not payload
    content. Returns (None, None) for a null pointer, an unreadable
    address, or any other address family (AF_UNIX, AF_NETLINK, ...)."""
    if not addr or addrlen <= 0:
        return None, None
    try:
        raw = read_bytes(pid, addr, min(addrlen, _SOCKADDR_MAX_LEN))
    except OSError:
        return None, None
    if len(raw) < 8:
        return None, None
    # sa_family is native byte order; the port that follows it is always
    # network (big-endian) byte order regardless of host endianness.
    (family,) = struct.unpack_from("<H", raw, 0)
    port = struct.unpack_from(">H", raw, 2)[0]
    if family == socket.AF_INET:
        return socket.inet_ntop(socket.AF_INET, raw[4:8]), port
    if family == socket.AF_INET6 and len(raw) >= 24:
        return socket.inet_ntop(socket.AF_INET6, raw[8:24]), port
    return None, None


def resolve_argv(pid: int, argv_ptr: int, max_entries: int = 4096) -> list[str]:
    """Resolve a NULL-terminated `char *const argv[]` pointer array (as
    passed to execve(2)) into a list of strings."""
    if argv_ptr == 0:
        return []
    pointers: list[int] = []
    addr = argv_ptr
    for _ in range(max_entries):
        raw = read_bytes(pid, addr, 8)
        (ptr,) = struct.unpack("<Q", raw)
        if ptr == 0:
            break
        pointers.append(ptr)
        addr += 8
    return [read_cstring(pid, p) for p in pointers]


def resolve_envp(pid: int, envp_ptr: int, max_entries: int = 8192) -> dict[str, str]:
    """Resolve a NULL-terminated `char *const envp[]` array into a dict."""
    entries = resolve_argv(pid, envp_ptr, max_entries)
    result: dict[str, str] = {}
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


# Bytes of scratch space claimed below the tracee's stack pointer for
# argv/path injection on a `rewrite` decision. The kernel guarantees the
# region immediately below rsp is unmapped-but-growable stack that nothing
# else is using at a syscall-entry stop, so writing there and repointing
# rdi/rsi at it is safe as long as the injected payload comfortably fits
# (a handful of short strings plus their pointer array never approaches this).
REWRITE_SCRATCH_SIZE = 8192


def inject_argv(pid: int, rsp: int, path: str, argv: list[str]) -> tuple[int, int]:
    """Write *path* and *argv* into scratch stack memory and return
    `(path_addr, argv_addr)` -- new values for the tracee's rdi/rsi to make
    the pending execve() run the rewritten command instead."""
    scratch = (rsp - REWRITE_SCRATCH_SIZE) & ~0xF
    blob = bytearray()
    str_addrs = []
    for s in argv:
        str_addrs.append(scratch + len(blob))
        blob += s.encode() + b"\x00"
    path_addr = scratch + len(blob)
    blob += path.encode() + b"\x00"
    ptrarr_addr = scratch + len(blob)
    ptrarr_addr = (ptrarr_addr + 7) & ~0x7  # 8-byte align the pointer array
    if ptrarr_addr - scratch + (len(argv) + 1) * 8 > REWRITE_SCRATCH_SIZE:
        raise ValueError("rewritten argv too large for scratch space")
    ptrs = b"".join(struct.pack("<Q", a) for a in str_addrs) + struct.pack("<Q", 0)
    write_bytes(pid, scratch, bytes(blob))
    write_bytes(pid, ptrarr_addr, ptrs)
    return path_addr, ptrarr_addr


# A separate scratch region further below rsp than inject_argv's own, so an
# envp rewrite and an argv rewrite never overlap even if a single decision
# ever needed both at once (today nothing does -- GPU env scoping is the
# only rewrite user, and it never also rewrites argv).
ENVP_REWRITE_SCRATCH_SIZE = 8192


def inject_envp(pid: int, rsp: int, envp: "dict[str, str]") -> int:
    """Write *envp* into its own scratch stack memory and return the new
    envp array address -- the tracee's rdx (execve) / r10 (execveat) value
    to make the pending exec see this environment instead."""
    scratch = (rsp - REWRITE_SCRATCH_SIZE - ENVP_REWRITE_SCRATCH_SIZE) & ~0xF
    blob = bytearray()
    str_addrs = []
    for key, value in envp.items():
        str_addrs.append(scratch + len(blob))
        blob += f"{key}={value}".encode() + b"\x00"
    ptrarr_addr = scratch + len(blob)
    ptrarr_addr = (ptrarr_addr + 7) & ~0x7  # 8-byte align the pointer array
    if ptrarr_addr - scratch + (len(envp) + 1) * 8 > ENVP_REWRITE_SCRATCH_SIZE:
        raise ValueError("rewritten envp too large for scratch space")
    ptrs = b"".join(struct.pack("<Q", a) for a in str_addrs) + struct.pack("<Q", 0)
    write_bytes(pid, scratch, bytes(blob))
    write_bytes(pid, ptrarr_addr, ptrs)
    return ptrarr_addr
