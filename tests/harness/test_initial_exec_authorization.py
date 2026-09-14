from __future__ import annotations

from types import SimpleNamespace

import pytest

from agency.harness.ptrace import supervisor


class _DenyPolicy:
    def __init__(self) -> None:
        self.events = []

    def check(self, _agent, event):
        self.events.append(event)
        return False


class _FakeLoop:
    decisions = []

    def __init__(self, *, syscalls, syscall_hook, syscall_exit_hook=None) -> None:
        self.syscalls = syscalls
        self.syscall_hook = syscall_hook
        self.syscall_exit_hook = syscall_exit_hook

    def start(self, argv, envp, cwd, *, stdin_data=None) -> None:
        del cwd, stdin_data
        self.root_pid = 101
        root = SimpleNamespace(
            syscall="execve",
            pid=101,
            argv=list(argv),
            envp=dict(envp),
            path=argv[0],
            timestamp=1.0,
            address=None,
            port=None,
        )
        child = SimpleNamespace(
            syscall="execve",
            pid=102,
            argv=["/bin/tool", "arg"],
            envp={},
            path="/bin/tool",
            timestamp=2.0,
            address=None,
            port=None,
        )
        matching_child = SimpleNamespace(**vars(root))
        matching_child.pid = 102
        matching_child.timestamp = 0.5
        altered_root = SimpleNamespace(**vars(root))
        altered_root.argv = [argv[0], "--different-arguments"]
        altered_root.timestamp = 2.5
        repeated_root = SimpleNamespace(**vars(root))
        repeated_root.timestamp = 3.0
        type(self).decisions = [
            self.syscall_hook(matching_child).kind,
            self.syscall_hook(root).kind,
            self.syscall_hook(altered_root).kind,
            self.syscall_hook(child).kind,
            self.syscall_hook(repeated_root).kind,
        ]


def test_initial_exec_authorization_is_exact_and_one_shot(monkeypatch):
    monkeypatch.setattr(supervisor, "TracerLoop", _FakeLoop)
    policy = _DenyPolicy()

    supervisor.agProxyPtrace(allow_initial_exec=True).launch(
        ["/opt/agency_harness_bin/claude", "-p", "prompt"],
        {"PATH": "/usr/bin:/bin"},
        policy=policy,
    )

    assert _FakeLoop.decisions == ["deny", "allow", "deny", "deny", "deny"]
    assert [event.path for event in policy.events] == [
        "/opt/agency_harness_bin/claude",
        "/opt/agency_harness_bin/claude",
        "/bin/tool",
        "/opt/agency_harness_bin/claude",
    ]


def test_initial_exec_remains_policy_controlled_without_opt_in(monkeypatch):
    monkeypatch.setattr(supervisor, "TracerLoop", _FakeLoop)
    policy = _DenyPolicy()

    supervisor.agProxyPtrace().launch(
        ["/bin/false"],
        {},
        policy=policy,
    )

    assert _FakeLoop.decisions == ["deny", "deny", "deny", "deny", "deny"]
    assert [event.path for event in policy.events] == [
        "/bin/false",
        "/bin/false",
        "/bin/false",
        "/bin/tool",
        "/bin/false",
    ]


@pytest.mark.skipif(not supervisor.ptrace_available(), reason="ptrace unavailable")
def test_live_deny_policy_allows_only_selected_root_executable():
    root_policy = _DenyPolicy()
    root = supervisor.agProxyPtrace(allow_initial_exec=True).launch(
        ["/bin/true"],
        {},
        policy=root_policy,
    )
    _stdout, _stderr, root_rc = root.wait(timeout=10)

    descendant_policy = _DenyPolicy()
    descendant = supervisor.agProxyPtrace(allow_initial_exec=True).launch(
        ["/bin/sh", "-c", "exec /bin/true"],
        {},
        policy=descendant_policy,
    )
    _stdout, descendant_stderr, descendant_rc = descendant.wait(timeout=10)

    assert root_rc == 0
    assert root_policy.events == []
    assert descendant_rc != 0
    assert "Operation not permitted" in descendant_stderr
    assert [event.path for event in descendant_policy.events] == ["/bin/true"]
