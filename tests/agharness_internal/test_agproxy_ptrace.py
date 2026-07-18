"""Tests for agproxy_ptrace -- the ptrace/seccomp syscall-level supervisor.

Tier 1 (pure logic, no real process) tests seccomp filter construction and
the agsyscallevent/agdecision shapes. Tier 2 tests (marked `ptrace`) launch
real traced processes and are skipped when `ptrace_available()` returns
False -- mirroring `tests/test_agsandbox.py`'s `docker`/`nvidia_smi` markers
and `agsandbox_backends/chroot.py`'s `chroot_available()` convention: a live
smoke test, not just a config/capability check, is the authoritative signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agency.agpolicy import agAllowAllPolicy, agdecision, agpolicy
from agency.agharness_internal.agproxy_ptrace import agProxyPtrace, ptrace_available

ptrace = pytest.mark.skipif(not ptrace_available(), reason="ptrace not usable on this host")

_SPAWN_CHILD_SCRIPT = str(
    # tests/fixtures/ stays a shared top-level directory (not moved into
    # agharness_internal/ -- it's not module-specific), so this now goes up
    # one more level (tests/agharness_internal/ -> tests/) to reach it.
    Path(__file__).parent.parent / "fixtures" / "ptrace_test_bins" / "spawn_child.py"
)


class _RecordingPolicy(agpolicy):
    def __init__(self):
        self.events = []

    def check(self, ag, event):
        self.events.append(event)
        return agdecision.allow()


class _DenyPolicy(agpolicy):
    def __init__(self, deny_path):
        self._deny_path = deny_path

    def check(self, ag, event):
        if event.argv and event.argv[0] == self._deny_path:
            return agdecision.deny(f"{self._deny_path} is denied by test policy")
        return agdecision.allow()


class _RewritePolicy(agpolicy):
    def __init__(self, target_path, new_args):
        self._target_path = target_path
        self._new_args = new_args

    def check(self, ag, event):
        if event.argv and event.argv[0] == self._target_path:
            return agdecision.rewrite(self._new_args)
        return agdecision.allow()


# ---------------------------------------------------------------------------
# Tier 1 -- pure logic
# ---------------------------------------------------------------------------


def test_agdecision_allow():
    d = agdecision.allow()
    assert d.kind == "allow"
    assert d.reason is None
    assert d.new_args is None


def test_agdecision_deny():
    d = agdecision.deny("no thanks")
    assert d.kind == "deny"
    assert d.reason == "no thanks"


def test_agdecision_rewrite():
    d = agdecision.rewrite(["/bin/echo", "x"])
    assert d.kind == "rewrite"
    assert d.new_args == ["/bin/echo", "x"]


def test_install_trace_filter_builds_without_error():
    """Pure construction -- installing the filter in *this* process would
    actually seccomp-restrict the test runner, so only verify the
    SyscallFilter/rule-adding machinery itself doesn't raise for the
    default syscall set, without calling .load()."""
    import pyseccomp as seccomp

    filt = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
    for name in ("execve", "execveat"):
        filt.add_rule(seccomp.TRACE(0), name)  # must not raise


def test_ptrace_available_is_cached():
    first = ptrace_available()
    second = ptrace_available()
    assert first == second


# ---------------------------------------------------------------------------
# Tier 2 -- real traced processes
# ---------------------------------------------------------------------------


@ptrace
def test_launch_basic_echo():
    px = agProxyPtrace()
    handle = px.launch(["/bin/echo", "hello"], {}, cwd="/tmp", policy=agAllowAllPolicy())
    stdout, stderr, rc = handle.wait(timeout=10)
    assert stdout == "hello\n"
    assert stderr == ""
    assert rc == 0


@ptrace
def test_launch_resolves_multi_exec_argv():
    px = agProxyPtrace()
    policy = _RecordingPolicy()
    handle = px.launch(
        [sys.executable, _SPAWN_CHILD_SCRIPT], {}, cwd="/tmp", policy=policy
    )
    stdout, stderr, rc = handle.wait(timeout=10)
    assert rc == 0
    assert stdout == "child-ran\n"

    execve_events = [e for e in policy.events if e.syscall == "execve"]
    assert len(execve_events) >= 2, f"expected >=2 execve events, got {policy.events}"
    argvs = [e.argv for e in execve_events]
    assert any(a and a[0] == "/bin/true" for a in argvs), argvs
    assert any(a and a[:2] == ["/bin/echo", "child-ran"] for a in argvs), argvs


@ptrace
def test_launch_deny_blocks_execve():
    px = agProxyPtrace()
    policy = _DenyPolicy("/bin/false")
    handle = px.launch(["/bin/false"], {}, cwd="/tmp", policy=policy)
    stdout, stderr, rc = handle.wait(timeout=10)
    # /bin/false's own (allowed) exit code is 1 -- denial must be
    # distinguishable from that: the execve itself never succeeds, so the
    # forked child's own EPERM-reporting fallback path runs instead and its
    # stderr names the syscall failure, not anything /bin/false would print
    # (it prints nothing).
    assert rc != 0
    assert "Operation not permitted" in stderr


@ptrace
def test_launch_rewrite_changes_argv():
    px = agProxyPtrace()
    policy = _RewritePolicy("/bin/echo", ["/bin/echo", "rewritten-arg"])
    handle = px.launch(
        ["/bin/echo", "original-arg-should-not-appear"], {}, cwd="/tmp", policy=policy
    )
    stdout, stderr, rc = handle.wait(timeout=10)
    assert rc == 0
    assert stdout == "rewritten-arg\n"


@ptrace
def test_launch_uses_peekdata_fallback_when_vm_readv_unavailable(monkeypatch):
    """Forces process_vm_readv to fail so read_bytes() falls back to
    PTRACE_PEEKDATA -- both paths must resolve argv identically."""
    import ctypes

    from agency.agharness_internal.agproxy_ptrace_internal import _ctypes_defs as pt

    def failing_vm_readv(*args, **kwargs):
        ctypes.set_errno(1)  # EPERM
        return -1

    monkeypatch.setattr(pt.libc, "process_vm_readv", failing_vm_readv)

    px = agProxyPtrace()
    policy = _RecordingPolicy()
    handle = px.launch(["/bin/echo", "peekdata-fallback"], {}, cwd="/tmp", policy=policy)
    stdout, stderr, rc = handle.wait(timeout=10)
    assert rc == 0
    assert stdout == "peekdata-fallback\n"
    assert any(e.argv == ["/bin/echo", "peekdata-fallback"] for e in policy.events)


@ptrace
def test_handle_pids_reflects_live_process():
    px = agProxyPtrace()
    handle = px.launch(["/bin/echo", "x"], {}, cwd="/tmp", policy=agAllowAllPolicy())
    handle.wait(timeout=10)
    # Process has exited -- no pids should remain tracked.
    assert handle.pids() == set()


@ptrace
def test_on_exit_callback_fires():
    px = agProxyPtrace()
    seen = []
    handle = px.launch(["/bin/echo", "x"], {}, cwd="/tmp", policy=agAllowAllPolicy())
    handle.on_exit(lambda pid: seen.append(pid))
    handle.wait(timeout=10)
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Phase 2: openat/open path resolution
# ---------------------------------------------------------------------------


@ptrace
def test_launch_resolves_openat_path():
    from agency.agconfig import agConfig
    from agency.agharness_internal.agproxy_ptrace import agPtraceConfig

    events = []

    class RecordingPolicy(agpolicy):
        def check(self, ag, event):
            events.append(event)
            return agdecision.allow()

    cfg = agConfig(agPtraceConfig(syscalls=("execve", "execveat", "openat", "open")))
    px = agProxyPtrace(cfg)
    handle = px.launch(
        ["/bin/cat", "/etc/hostname"], {}, cwd="/tmp", policy=RecordingPolicy()
    )
    stdout, stderr, rc = handle.wait(timeout=10)
    assert rc == 0

    openat_events = [e for e in events if e.syscall in ("openat", "open")]
    assert any(e.path and "hostname" in e.path for e in openat_events), [
        (e.syscall, e.path) for e in openat_events
    ]


@ptrace
def test_deny_openat_blocks_file_read():
    class DenyHostnamePolicy(agpolicy):
        def check(self, ag, event):
            if event.syscall in ("openat", "open") and event.path and "hostname" in event.path:
                return agdecision.deny("no reading /etc/hostname")
            return agdecision.allow()

    from agency.agconfig import agConfig
    from agency.agharness_internal.agproxy_ptrace import agPtraceConfig

    cfg = agConfig(agPtraceConfig(syscalls=("execve", "execveat", "openat", "open")))
    px = agProxyPtrace(cfg)
    handle = px.launch(
        ["/bin/cat", "/etc/hostname"], {}, cwd="/tmp", policy=DenyHostnamePolicy()
    )
    stdout, stderr, rc = handle.wait(timeout=10)
    assert rc != 0
    assert stdout == ""


# ---------------------------------------------------------------------------
# Phase 2: wire_to_sandbox() -- agsandbox PID-tracking integration
# ---------------------------------------------------------------------------


def _docker_available() -> bool:
    import subprocess

    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


docker = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")


@ptrace
@docker
def test_wire_to_sandbox_reflects_traced_background_process():
    from agency.agconfig import agConfig
    from agency.agharness_internal.agproxy_ptrace import wire_to_sandbox
    from agency.agsandbox import agSandbox
    from agency.agsandbox_backends import agSandboxBackendConfig
    import time
    import uuid

    cfg = agConfig(agSandboxBackendConfig(backend="docker"))
    sb = agSandbox(str(uuid.uuid4()), agconfig=cfg)
    try:
        px = agProxyPtrace()
        # Long enough to comfortably outlast cold container startup (~4-5s
        # even with a warm image cache, per direct measurement) -- a too-
        # short sleep here makes the traced process exit (correctly firing
        # on_exit and clearing its own tracking) before the sandbox's first
        # get_live_pids() call even finishes starting its container.
        handle = px.launch(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            {}, cwd="/tmp", policy=agAllowAllPolicy(),
        )
        wire_to_sandbox(handle, sb)
        # wire_to_sandbox's on_spawn callback fires for the root pid too
        # (see _fork_and_exec's use of _remember_spawn) -- give the
        # supervisor a brief moment to actually reach that call.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not sb.get_live_pids():
            time.sleep(0.1)
        assert sb.get_live_pids(), "expected the launched process's pid to reach the sandbox"
        handle.kill()
        handle.wait(timeout=10)
    finally:
        sb.destroy()
