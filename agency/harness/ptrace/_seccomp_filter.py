"""Seccomp filter construction for agproxy_ptrace, via the `pyseccomp`
library (pure-Python ctypes bindings to libseccomp, API-compatible with
libseccomp's own Python bindings).

The filter's default action is ALLOW -- agproxy_ptrace does not sandbox the
traced process by denying syscalls wholesale, it only asks the kernel to
stop-and-report the specific syscalls the caller wants observed
(`SECCOMP_RET_TRACE`, i.e. pyseccomp's `TRACE` action). Every other syscall
runs untouched, at native speed, with no ptrace-stop overhead at all --
that's the whole point of installing a filter instead of full
PTRACE_SYSCALL single-stepping.
"""

from __future__ import annotations

import pyseccomp as seccomp


def install_trace_filter(syscalls: "list[str] | tuple[str, ...]") -> None:
    """Install a seccomp filter in the CALLING process (must be the traced
    child, after PTRACE_TRACEME and before execve -- see agProxyPtrace.launch()
    for the exact ordering and why it matters: the tracer must already have
    PTRACE_O_TRACESECCOMP set on this tracee by the time any filtered syscall
    fires, or the kernel fails it with ENOSYS instead of generating a trace
    stop. That's why launch() synchronizes via SIGSTOP before calling this."""
    filt = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    for name in syscalls:
        filt.add_rule(seccomp.TRACE(0), name)
    filt.load()
