"""Tests for agproxy_ptrace -- the ptrace/seccomp syscall-level supervisor.

Tier 1 (pure logic, no real process) tests seccomp filter construction and
policy handling. Tier 2 tests (marked `ptrace`) launch
real traced processes and are skipped when `ptrace_available()` returns
False -- mirroring `tests/test_agsandbox.py`'s `docker`/`nvidia_smi` markers
and `sandbox/chroot.py`'s `chroot_available()` convention: a live
smoke test, not just a config/capability check, is the authoritative signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agency.harness.ptrace.supervisor import agProxyPtrace, ptrace_available

ptrace = pytest.mark.skipif(not ptrace_available(), reason="ptrace not usable on this host")

_SPAWN_CHILD_SCRIPT = str(
    # tests/fixtures/ stays a shared top-level directory (not moved into
    # harness/ -- it's not module-specific), so this now goes up
    # one more level (tests/harness/ -> tests/) to reach it.
    Path(__file__).parent.parent / "fixtures" / "ptrace_test_bins" / "spawn_child.py"
)


class _AllowPolicy:
    def check(self, _ag, _event):
        return True


class _RecordingPolicy:
    def __init__(self):
        self.events = []

    def check(self, ag, event):
        self.events.append(event)
        return True


class _DenyPolicy:
    def __init__(self, deny_path):
        self._deny_path = deny_path

    def check(self, ag, event):
        if event.argv and event.argv[0] == self._deny_path:
            return (False, f"{self._deny_path} is denied by test policy")
        return True


# ---------------------------------------------------------------------------
# Tier 1 -- pure logic
# ---------------------------------------------------------------------------


@ptrace
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


def test_process_lifecycle_profiler_uses_safe_exec_name_and_exit_status(monkeypatch):
    """Only the kernel-confirmed path, never attacker-controlled argv, is named."""
    from agency.harness.ptrace import supervisor as ptrace_module
    from agency.observability.profiler import agprof

    secret = "AGPROF_TOKEN_must-not-reach-the-trace"
    parent_context = object()

    class FakeExternalSpan:
        def __init__(self, name, fields):
            self.name = name
            self.fields = fields
            self.updates = []
            self.end_fields = None

        def update(self, name=None, **metadata):
            self.name = name or self.name
            self.updates.append(metadata)

        def end(self, **fields):
            self.end_fields = fields

    captured = []

    monkeypatch.setattr(agprof, "enabled", lambda: True)
    monkeypatch.setattr(agprof, "current_span_context", lambda: parent_context)
    monkeypatch.setattr(
        agprof,
        "current_span_attributes",
        lambda: {"agency.run_id": "run9", "agency.agent_id": "tester"},
    )
    monkeypatch.setattr(
        agprof,
        "start_external_span",
        lambda name, **kwargs: captured.append(FakeExternalSpan(name, kwargs)) or captured[-1],
    )

    class FakeLoop:
        def __init__(self, syscalls, syscall_hook):
            self.syscall_hook = syscall_hook
            self.spawn_callbacks = []
            self.exec_callbacks = []
            self.exit_callbacks = []

        def on_spawn(self, callback):
            self.spawn_callbacks.append(callback)

        def on_exit(self, callback):
            self.exit_callbacks.append(callback)

        def on_exec(self, callback):
            self.exec_callbacks.append(callback)

        def start(self, argv, envp, cwd):
            for callback in self.spawn_callbacks:
                callback(4321)
            self.syscall_hook(
                SimpleNamespace(
                    pid=4321,
                    syscall="execve",
                    syscall_nr=59,
                    argv=[secret, "--mcp-config", secret, "full user prompt"],
                    envp=None,
                    path=f"/tmp/{secret}/claude",
                    timestamp=0,
                )
            )
            for callback in self.exec_callbacks:
                callback(4321, f"/tmp/tool-{secret}")
                callback(4321, "/usr/bin/claude")
            for callback in self.exit_callbacks:
                callback(4321, 17)

        def join(self, timeout=None):
            return 17

        def read_output(self):
            return "", ""

        def live_pids(self):
            return set()

        def kill(self):
            pass

    monkeypatch.setattr(ptrace_module, "TracerLoop", FakeLoop)

    handle = ptrace_module.agProxyPtrace().launch(
        [secret, "initial prompt containing " + secret],
        {"AGPROF_TOKEN": secret},
        policy=_AllowPolicy(),
    )
    assert handle.wait() == ("", "", 17)

    assert len(captured) == 1
    external_span = captured[0]
    assert external_span.name == "process:claude"
    assert external_span.fields["parent_context"] is parent_context
    assert external_span.fields["metadata"] == {
        "agency.run_id": "run9",
        "agency.agent_id": "tester",
        "timing": "exact",
        "provenance": "ptrace",
        "pid": 4321,
        "executable": "<unknown>",
    }
    assert external_span.updates == [
        {"executable": "<redacted>"},
        {"executable": "claude"},
    ]
    assert external_span.end_fields["metadata"] == {
        "executable": "claude",
        "exit_code": 17,
        "outcome": "failure",
    }
    assert secret not in json.dumps(
        (
            external_span.name,
            external_span.fields["metadata"],
            external_span.updates,
            external_span.end_fields["metadata"],
        )
    )


@pytest.mark.parametrize("failure_phase", ["start", "update", "end"])
def test_profiler_callback_failure_cannot_abort_ptrace_lifecycle(monkeypatch, failure_phase):
    from agency.harness.ptrace import supervisor as ptrace_module
    from agency.observability.profiler import agprof

    reached = []

    class FaultingExternalSpan:
        def update(self, *args, **kwargs):
            if failure_phase == "update":
                raise RuntimeError("injected update failure")

        def end(self, **kwargs):
            if failure_phase == "end":
                raise RuntimeError("injected end failure")

    def start_external_span(*args, **kwargs):
        if failure_phase == "start":
            raise RuntimeError("injected start failure")
        return FaultingExternalSpan()

    monkeypatch.setattr(agprof, "enabled", lambda: True)
    monkeypatch.setattr(agprof, "current_span_context", lambda: None)
    monkeypatch.setattr(agprof, "current_span_attributes", lambda: {})
    monkeypatch.setattr(agprof, "start_external_span", start_external_span)

    class FakeLoop:
        def __init__(self, syscalls, syscall_hook):
            self.spawn_callbacks = []
            self.exec_callbacks = []
            self.exit_callbacks = []

        def on_spawn(self, callback):
            self.spawn_callbacks.append(callback)

        def on_exec(self, callback):
            self.exec_callbacks.append(callback)

        def on_exit(self, callback):
            self.exit_callbacks.append(callback)

        def start(self, argv, envp, cwd):
            for callback in self.spawn_callbacks:
                callback(4501)
            reached.append("spawn")
            for callback in self.exec_callbacks:
                callback(4501, "/usr/bin/true")
            reached.append("exec")
            for callback in self.exit_callbacks:
                callback(4501, 0)
            reached.append("exit")

        def join(self, timeout=None):
            return 0

        def read_output(self):
            return "", ""

        def live_pids(self):
            return set()

        def kill(self):
            return None

    monkeypatch.setattr(ptrace_module, "TracerLoop", FakeLoop)

    handle = ptrace_module.agProxyPtrace().launch(["/usr/bin/true"], {}, policy=_AllowPolicy())

    assert reached == ["spawn", "exec", "exit"]
    assert handle.wait() == ("", "", 0)


@pytest.mark.parametrize(
    ("exec_path", "display_name"),
    [
        ("/tmp/credential-in-parent/claude", "claude"),
        ("/usr/bin/python3.12", "python3.12"),
        ("/tmp/tool (deleted)", "tool"),
        ("/tmp/tool?token=credential", "<redacted>"),
        ("/tmp/unsafe\nname", "<redacted>"),
        ("/tmp/" + "x" * 65, "<redacted>"),
        ("/tmp/0123456789abcdef0123456789abcdef", "<redacted>"),
        (None, "<unknown>"),
    ],
)
def test_executable_display_name_sanitizes_confirmed_exec_path(exec_path, display_name):
    from agency.harness.ptrace.supervisor import _executable_display_name

    assert _executable_display_name(exec_path) == display_name


def test_executable_display_name_redacts_sensitive_substrings():
    from agency.harness.ptrace.supervisor import _executable_display_name

    secret = "opaqueSafeAlphabetToken"
    assert (
        _executable_display_name(f"/tmp/tool-{secret}", sensitive_values=frozenset({secret}))
        == "<redacted>"
    )


def test_process_lifecycle_finalize_interrupts_live_children(monkeypatch):
    from agency.harness.ptrace import supervisor as ptrace_module
    from agency.observability.profiler import agprof

    external_span = object()
    interrupted = []
    monkeypatch.setattr(agprof, "start_external_span", lambda *args, **kwargs: external_span)
    monkeypatch.setattr(
        agprof,
        "interrupt_external_span",
        lambda span, **kwargs: interrupted.append((span, kwargs)),
    )
    recorder = ptrace_module._ProcessLifecycleProfiler(
        parent_context=None,
        span_attributes={"agency.run_id": "run0"},
    )

    recorder.on_spawn(701)
    recorder.finalize()
    recorder.finalize()
    recorder.on_spawn(702)

    assert len(interrupted) == 1
    assert interrupted[0][0] is external_span
    assert interrupted[0][1]["ended_perf_ns"] >= 0


def test_handle_on_exit_can_include_ptrace_exit_status():
    from agency.harness.ptrace.supervisor import agProxyPtraceHandle

    class FakeLoop:
        def on_exec(self, callback):
            callback(123, "/usr/bin/true")

        def on_exit(self, callback):
            callback(123, -9)

    legacy = []
    statuses = []
    execs = []
    handle = agProxyPtraceHandle(FakeLoop())
    handle.on_exec(lambda pid, path: execs.append((pid, path)))
    handle.on_exit(lambda pid: legacy.append(pid))
    handle.on_exit(lambda pid, code: statuses.append((pid, code)), include_exit_code=True)

    assert execs == [(123, "/usr/bin/true")]
    assert legacy == [123]
    assert statuses == [(123, -9)]


def test_handle_timeout_is_polling_and_later_wait_can_succeed():
    from agency.harness.ptrace.supervisor import agProxyPtraceHandle

    class FakeLoop:
        def __init__(self):
            self.returncodes = iter((None, 0))

        def join(self, timeout=None):
            return next(self.returncodes)

        def read_output(self):
            return "partial", ""

    class FakeProfiler:
        finalized = False

        def finalize(self):
            self.finalized = True

    profiler = FakeProfiler()
    handle = agProxyPtraceHandle(FakeLoop(), profiler)

    assert handle.wait(timeout=0) == ("partial", "", -1)
    assert not profiler.finalized
    assert handle.wait(timeout=1) == ("partial", "", 0)
    assert not profiler.finalized


@ptrace
def test_clone_threads_do_not_reach_process_lifecycle_callbacks(monkeypatch):
    from agency.harness.ptrace import _tracer_loop

    loop = _tracer_loop.TracerLoop(
        syscalls=("execve",),
        syscall_hook=lambda _stop: _tracer_loop.StopDecision(kind="allow"),
    )
    spawned = []
    exited = []
    loop.on_spawn(spawned.append)
    loop.on_exit(lambda pid, code: exited.append((pid, code)))

    monkeypatch.setattr(
        _tracer_loop,
        "_is_thread_group_leader",
        lambda pid: pid == 9002,
    )
    loop._remember_spawn(9001, is_process=None)  # clone(CLONE_THREAD)
    loop._classify_pending_clone(9001)
    loop._forget(9001, 0)
    loop._remember_spawn(9002, is_process=None)  # clone without CLONE_THREAD
    loop._classify_pending_clone(9002)
    loop._forget(9002, 7)

    assert spawned == [9002]
    assert exited == [(9002, 7)]


@ptrace
def test_exec_callback_is_success_only_kernel_named_and_replay_safe(monkeypatch):
    from types import SimpleNamespace

    from agency.harness.ptrace import _tracer_loop

    loop = _tracer_loop.TracerLoop(
        syscalls=("execve",),
        syscall_hook=lambda _stop: _tracer_loop.StopDecision(kind="allow"),
    )
    pid = 9201
    loop._remember_spawn(pid)
    monkeypatch.setattr(
        _tracer_loop.pt,
        "get_regs",
        lambda _pid: SimpleNamespace(orig_rax=_tracer_loop.pt.SYSCALL_NUMBERS["execve"]),
    )
    requested_path = ["/tmp/credential-parent/old-image"]
    attacker_argv0 = "AGPROF_TOKEN_attacker-controlled-argv-zero"
    monkeypatch.setattr(
        _tracer_loop,
        "_resolve_syscall_args",
        lambda *_args: ([attacker_argv0], None, requested_path[0]),
    )
    monkeypatch.setattr(_tracer_loop.pt, "ptrace", lambda *_args: None)

    kernel_path = ["/usr/bin/dash"]
    monkeypatch.setattr(_tracer_loop, "_kernel_executable_path", lambda _pid: kernel_path[0])
    loop._handle_seccomp_stop(pid)
    loop._commit_exec(pid)

    # Registration after a successful exec replays the committed image.
    execs = []
    loop.on_exec(lambda seen_pid, path: execs.append((seen_pid, path)))
    assert execs == [(pid, "/tmp/credential-parent/old-image")]

    # A later candidate can fail after its seccomp stop. Without an EXEC
    # event there is no callback, so consumers retain the old image.
    requested_path[0] = "/does/not/exist"
    loop._handle_seccomp_stop(pid)
    assert loop._pending_exec_paths[pid] == "/does/not/exist"
    assert execs == [(pid, "/tmp/credential-parent/old-image")]

    # A subsequent success overwrites the stale candidate. The confirmed
    # syscall pathname wins over attacker-controlled argv[0] and keeps a
    # shebang launcher named for the requested command rather than interpreter.
    requested_path[0] = "/tmp/credential-parent/symlink"
    kernel_path[0] = "/usr/bin/true"
    loop._handle_seccomp_stop(pid)
    loop._commit_exec(pid)
    assert execs == [
        (pid, "/tmp/credential-parent/old-image"),
        (pid, "/tmp/credential-parent/symlink"),
    ]
    # If exec syscall trapping is disabled (or execveat has an empty path),
    # the successful event still gets a procfs-derived fallback identity.
    kernel_path[0] = "/usr/bin/fallback"
    loop._commit_exec(pid)
    assert execs[-1] == (pid, "/usr/bin/fallback")
    assert attacker_argv0 not in json.dumps(execs)


# ---------------------------------------------------------------------------
# Tier 2 -- real traced processes
# ---------------------------------------------------------------------------


@ptrace
def test_launch_basic_echo():
    px = agProxyPtrace()
    handle = px.launch(["/bin/echo", "hello"], {}, cwd="/tmp", policy=_AllowPolicy())
    stdout, stderr, rc = handle.wait(timeout=10)
    assert stdout == "hello\n"
    assert stderr == ""
    assert rc == 0


@ptrace
def test_launch_resolves_multi_exec_argv():
    px = agProxyPtrace()
    policy = _RecordingPolicy()
    handle = px.launch([sys.executable, _SPAWN_CHILD_SCRIPT], {}, cwd="/tmp", policy=policy)
    stdout, stderr, rc = handle.wait(timeout=10)
    assert rc == 0
    assert stdout == "child-ran\n"

    execve_events = [e for e in policy.events if e.syscall == "execve"]
    assert len(execve_events) >= 2, f"expected >=2 execve events, got {policy.events}"
    argvs = [e.argv for e in execve_events]
    assert any(a and a[0] == "/bin/true" for a in argvs), argvs
    assert any(a and a[:2] == ["/bin/echo", "child-ran"] for a in argvs), argvs


@ptrace
def test_launch_records_process_lifecycles_under_current_span(tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    from agency.observability.profiler import agprof

    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        with agprof.span("run0:ptrace:test"):
            handle = agProxyPtrace().launch(
                [sys.executable, _SPAWN_CHILD_SCRIPT],
                {},
                cwd="/tmp",
                policy=_AllowPolicy(),
            )
            _stdout, _stderr, rc = handle.wait(timeout=10)
            assert rc == 0

    records = agprof.profile_records()
    run_record = next(record for record in records if record[1] == "run0:ptrace:test")
    process_records = [record for record in records if record[1].startswith("process:")]
    assert {record[1] for record in process_records} >= {"process:true", "process:echo"}
    assert all(record[8] == run_record[7] for record in process_records)
    assert all(record[6]["timing"] == "exact" for record in process_records)
    assert all(record[6]["provenance"] == "ptrace" for record in process_records)
    assert all(record[6]["exit_code"] == 0 for record in process_records)


@ptrace
def test_traced_thread_does_not_create_process_span(tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    from agency.observability.profiler import agprof

    script = (
        "import threading; "
        "thread = threading.Thread(target=lambda: None); "
        "thread.start(); thread.join()"
    )
    with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
        handle = agProxyPtrace().launch(
            [sys.executable, "-c", script],
            {},
            cwd="/tmp",
            policy=_AllowPolicy(),
        )
        _stdout, _stderr, rc = handle.wait(timeout=10)
        assert rc == 0

    process_records = [
        record for record in agprof.profile_records() if record[1].startswith("process:")
    ]
    assert [record[1] for record in process_records] == [f"process:{Path(sys.executable).name}"]


@ptrace
def test_live_traced_process_is_incomplete_when_profiler_stops(tmp_path):
    pytest.importorskip("opentelemetry.sdk.trace")
    from agency.observability.profiler import agprof

    handle = None
    try:
        with agprof.session(tmp_path, sample_hz=0, sample_gpu=False):
            handle = agProxyPtrace().launch(
                ["/bin/sleep", "30"],
                {},
                cwd="/tmp",
                policy=_AllowPolicy(),
            )
        summary = json.loads((tmp_path / "summary.json").read_text())
        incomplete = next(
            span for span in summary["incomplete_spans"] if span["label"] == "process:sleep"
        )
        assert incomplete["outcome"] == "interrupted"
        assert incomplete["executable"] == "sleep"
    finally:
        if handle is not None:
            handle.kill()
            handle.wait(timeout=10)


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
def test_launch_uses_peekdata_fallback_when_vm_readv_unavailable(monkeypatch):
    """Forces process_vm_readv to fail so read_bytes() falls back to
    PTRACE_PEEKDATA -- both paths must resolve argv identically."""
    import ctypes

    from agency.harness.ptrace import _ctypes_defs as pt

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
    handle = px.launch(["/bin/echo", "x"], {}, cwd="/tmp", policy=_AllowPolicy())
    handle.wait(timeout=10)
    # Process has exited -- no pids should remain tracked.
    assert handle.pids() == set()


@ptrace
def test_on_exit_callback_fires():
    px = agProxyPtrace()
    seen = []
    handle = px.launch(["/bin/echo", "x"], {}, cwd="/tmp", policy=_AllowPolicy())
    handle.on_exit(lambda pid: seen.append(pid))
    handle.wait(timeout=10)
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Phase 2: openat/open path resolution
# ---------------------------------------------------------------------------


@ptrace
def test_launch_resolves_openat_path():
    from agency.agconfig import agConfig
    from agency.harness.ptrace.supervisor import agPtraceConfig

    events = []

    class RecordingPolicy:
        def check(self, ag, event):
            events.append(event)
            return True

    cfg = agConfig(agPtraceConfig(syscalls=("execve", "execveat", "openat", "open")))
    px = agProxyPtrace(cfg)
    handle = px.launch(["/bin/cat", "/etc/hostname"], {}, cwd="/tmp", policy=RecordingPolicy())
    stdout, stderr, rc = handle.wait(timeout=10)
    assert rc == 0

    openat_events = [e for e in events if e.syscall in ("openat", "open")]
    assert any(e.path and "hostname" in e.path for e in openat_events), [
        (e.syscall, e.path) for e in openat_events
    ]


@ptrace
def test_deny_openat_blocks_file_read():
    class DenyHostnamePolicy:
        def check(self, ag, event):
            if event.syscall in ("openat", "open") and event.path and "hostname" in event.path:
                return (False, "no reading /etc/hostname")
            return True

    from agency.agconfig import agConfig
    from agency.harness.ptrace.supervisor import agPtraceConfig

    cfg = agConfig(agPtraceConfig(syscalls=("execve", "execveat", "openat", "open")))
    px = agProxyPtrace(cfg)
    handle = px.launch(["/bin/cat", "/etc/hostname"], {}, cwd="/tmp", policy=DenyHostnamePolicy())
    stdout, stderr, rc = handle.wait(timeout=10)
    assert rc != 0
    assert stdout == ""
