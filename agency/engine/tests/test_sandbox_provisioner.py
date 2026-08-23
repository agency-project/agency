from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agency.agconfig import agConfig
from agency.engine.sandbox_provisioner import SandboxProvisioner
from agency.profiler import agprof
from agency.sandbox.agsandbox import agSandboxConfig


class _Inbox:
    def __init__(self, events):
        self.events = events
        self.messages = []

    def put(self, message):
        self.events.append("inbox.put")
        self.messages.append(message)


class _Agent:
    output_dir = None

    def __init__(self, *, sandbox=None, agconfig=None, agname="test-agent", events=None):
        self.sandbox = sandbox
        self.agconfig = agconfig
        self.agname = agname
        self.inbox = _Inbox(events if events is not None else [])


class _TrackingLock:
    def __init__(self, events):
        self.events = events
        self.balance = 0
        self.release_calls = 0
        self.acquire_error = None
        self.release_error = None
        self.on_acquire_attempt = None
        self._lock = threading.RLock()

    def acquire(self):
        self.events.append("lock.acquire")
        if self.on_acquire_attempt is not None:
            self.on_acquire_attempt()
        if self.acquire_error is not None:
            raise self.acquire_error
        acquired = self._lock.acquire()
        if acquired:
            self.balance += 1
        return acquired

    def release(self):
        self.events.append("lock.release")
        self.release_calls += 1
        if self.release_error is not None:
            raise self.release_error
        self._lock.release()
        self.balance -= 1


class _Sandbox:
    def __init__(self, events):
        self.events = events
        self._lock = _TrackingLock(events)
        self._provisioner_discard_required = False
        self._provisioner_revert_notice_agent = None
        self.start_error = None
        self.commit_error = None
        self.commit_result = True
        self.discard_error = None
        self.pending_error = None
        self.pending = False
        self.stop_error = None

    def ensure_started(self):
        self.events.append("sandbox.ensure_started")
        if self.start_error is not None:
            raise self.start_error

    def commit(self):
        self.events.append("sandbox.commit")
        if self.commit_error is not None:
            raise self.commit_error
        return self.commit_result

    def rm_container(self):
        self.events.append("sandbox.discard")
        if self.discard_error is not None:
            raise self.discard_error

    def _has_pending_background_work(self):
        self.events.append("sandbox.pending")
        if self.pending_error is not None:
            raise self.pending_error
        return self.pending

    def stop(self):
        self.events.append("sandbox.stop")
        if self.stop_error is not None:
            raise self.stop_error


class _RecordingProvisioner(SandboxProvisioner):
    def __init__(self, events, **kwargs):
        super().__init__(**kwargs)
        self.events = events
        self.teardown_error = None

    def _teardown(self, lease):
        try:
            super()._teardown(lease)
        finally:
            self.events.append("provisioner.teardown")
        if self.teardown_error is not None:
            raise self.teardown_error


def _transaction_inputs():
    events = []
    sandbox = _Sandbox(events)
    agent = _Agent(sandbox=sandbox, events=events)
    provisioner = _RecordingProvisioner(events)
    return provisioner, agent, sandbox, events


def test_get_or_create_reuses_the_agents_existing_sandbox():
    existing = object()
    factory_calls = []
    provisioner = SandboxProvisioner(
        sandbox_factory=lambda *args, **kwargs: factory_calls.append((args, kwargs))
    )

    agent = _Agent(sandbox=existing)

    assert provisioner.get_or_create(agent) is existing
    assert factory_calls == []


def test_get_or_create_constructs_and_attaches_a_sandbox():
    created = SimpleNamespace()
    calls = []

    def factory(agname, *, agconfig):
        calls.append((agname, agconfig))
        return created

    agent = _Agent(agconfig=agConfig())
    provisioner = SandboxProvisioner(sandbox_factory=factory)

    assert provisioner.get_or_create(agent) is created
    assert agent.sandbox is created
    assert calls == [("test-agent", agent.agconfig)]


def test_get_or_create_adds_the_agent_output_mount_without_mutating_agent_config(tmp_path):
    original_config = agConfig({"agent": {"output_dir": str(tmp_path)}})
    captured = {}

    def factory(_agname, *, agconfig):
        captured["agconfig"] = agconfig
        return SimpleNamespace()

    agent = _Agent(agconfig=original_config, agname="worker")
    SandboxProvisioner(sandbox_factory=factory).get_or_create(agent)

    sandbox_config = captured["agconfig"]
    assert sandbox_config is not original_config
    assert original_config.get("agSandbox", "mounts") is None
    assert agSandboxConfig(sandbox_config).mounts["agent_output"] == (
        str(Path(tmp_path) / "worker"),
        "/agent_output",
        "rw",
    )


def test_get_or_create_builds_a_mount_config_from_class_output_dir(tmp_path):
    class _OutputAgent(_Agent):
        output_dir = tmp_path

    captured = {}

    def factory(_agname, *, agconfig):
        captured["agconfig"] = agconfig
        return SimpleNamespace()

    agent = _OutputAgent(agconfig=None, agname="worker")
    SandboxProvisioner(sandbox_factory=factory).get_or_create(agent)

    assert agent.agconfig is None
    assert agSandboxConfig(captured["agconfig"]).mounts["agent_output"] == (
        str(tmp_path / "worker"),
        "/agent_output",
        "rw",
    )


def test_factory_failure_does_not_attach_a_partial_sandbox():
    agent = _Agent(sandbox=None, agconfig=agConfig())

    def factory(*_args, **_kwargs):
        raise RuntimeError("factory failed")

    with pytest.raises(RuntimeError, match="factory failed"):
        SandboxProvisioner(sandbox_factory=factory).get_or_create(agent)

    assert agent.sandbox is None


def test_acquire_locks_before_explicit_physical_start():
    provisioner, agent, sandbox, events = _transaction_inputs()

    lease = provisioner.acquire(agent)

    assert events == ["lock.acquire", "sandbox.ensure_started"]
    assert lease.sandbox is sandbox
    assert lease.lock_acquired is True
    assert lease.physical_start_attempted is True
    assert lease.provisioned is True

    provisioner.finalize(lease, succeeded=False)


def test_profiler_span_exit_failure_unwinds_the_prepared_lease(monkeypatch):
    provisioner, agent, sandbox, events = _transaction_inputs()

    class _FailingProvisionSpan:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise RuntimeError("profiler exit failed")

    def fake_span(name):
        if name == "sandbox:provision":
            return _FailingProvisionSpan()

        class _NoopSpan:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        return _NoopSpan()

    monkeypatch.setattr(agprof, "span", fake_span)

    with pytest.raises(RuntimeError, match="profiler exit failed"):
        provisioner.acquire(agent)

    assert events == [
        "lock.acquire",
        "sandbox.ensure_started",
        "sandbox.discard",
        "provisioner.teardown",
        "lock.release",
    ]
    assert sandbox._lock.balance == 0


def test_lock_acquisition_failure_never_releases_an_unacquired_lock():
    provisioner, agent, sandbox, events = _transaction_inputs()
    sandbox._lock.acquire_error = RuntimeError("lock failed")

    with pytest.raises(RuntimeError, match="lock failed"):
        provisioner.acquire(agent)

    assert events == ["lock.acquire"]
    assert sandbox._lock.release_calls == 0


def test_start_failure_discards_partial_state_then_tears_down_and_releases_last():
    provisioner, agent, sandbox, events = _transaction_inputs()
    sandbox.start_error = RuntimeError("start failed")

    with pytest.raises(RuntimeError, match="start failed"):
        provisioner.acquire(agent)

    assert events == [
        "lock.acquire",
        "sandbox.ensure_started",
        "sandbox.discard",
        "provisioner.teardown",
        "lock.release",
    ]
    assert sandbox._lock.balance == 0
    assert agent.inbox.messages == []


def test_start_error_remains_primary_when_every_cleanup_stage_also_fails():
    provisioner, agent, sandbox, events = _transaction_inputs()
    sandbox.start_error = RuntimeError("start failed")
    sandbox.discard_error = RuntimeError("discard failed")
    sandbox.stop_error = RuntimeError("stop failed")
    sandbox._lock.release_error = RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="start failed") as raised:
        provisioner.acquire(agent)

    notes = "\n".join(raised.value.__notes__)
    assert "discard failed" in notes
    assert "stop failed" in notes
    assert "release failed" in notes
    assert events[-1] == "lock.release"
    assert sandbox._lock.release_calls == 1


def test_failed_start_discard_forces_stop_even_when_background_work_is_pending():
    provisioner, agent, sandbox, events = _transaction_inputs()
    sandbox.start_error = RuntimeError("start failed")
    sandbox.discard_error = RuntimeError("discard failed")
    sandbox.pending = True

    with pytest.raises(RuntimeError, match="start failed") as raised:
        provisioner.acquire(agent)

    assert "discard failed" in "\n".join(raised.value.__notes__)
    assert "sandbox.pending" not in events
    assert events[-3:] == ["sandbox.stop", "provisioner.teardown", "lock.release"]
    assert sandbox._provisioner_discard_required is True
    assert sandbox._lock.balance == 0


def test_success_commits_hibernates_tears_down_and_releases_last():
    provisioner, agent, sandbox, events = _transaction_inputs()
    lease = provisioner.acquire(agent)
    provisioner.mark_execution_attempted(lease)

    provisioner.finalize(lease, succeeded=True)

    assert events == [
        "lock.acquire",
        "sandbox.ensure_started",
        "sandbox.commit",
        "sandbox.pending",
        "sandbox.stop",
        "provisioner.teardown",
        "lock.release",
    ]
    assert lease.committed is True
    assert lease.hibernated is True
    assert lease.discarded is False
    assert lease.finalized is True
    assert sandbox._lock.balance == 0
    assert agent.inbox.messages == []


def test_success_defers_hibernation_while_background_work_is_pending():
    provisioner, agent, sandbox, events = _transaction_inputs()
    sandbox.pending = True
    lease = provisioner.acquire(agent)
    provisioner.mark_execution_attempted(lease)

    provisioner.finalize(lease, succeeded=True)

    assert "sandbox.commit" in events
    assert "sandbox.stop" not in events
    assert events[-2:] == ["provisioner.teardown", "lock.release"]
    assert lease.hibernated is False


def test_failed_hibernation_forces_physical_cleanup_before_the_next_start():
    provisioner, first_agent, sandbox, events = _transaction_inputs()
    second_agent = _Agent(sandbox=sandbox, agname="second", events=events)
    sandbox.stop_error = RuntimeError("stop failed")
    first_lease = provisioner.acquire(first_agent)
    provisioner.mark_execution_attempted(first_lease)

    with pytest.raises(RuntimeError, match="stop failed"):
        provisioner.finalize(first_lease, succeeded=True)

    assert first_lease.committed is True
    assert sandbox._provisioner_discard_required is True

    sandbox.stop_error = None
    events.clear()
    second_lease = provisioner.acquire(second_agent)

    assert events[:3] == [
        "lock.acquire",
        "sandbox.discard",
        "sandbox.ensure_started",
    ]
    assert first_agent.inbox.messages == []
    assert second_agent.inbox.messages == []

    provisioner.finalize(second_lease, succeeded=False)


def test_execution_failure_discards_notifies_tears_down_and_releases_last():
    provisioner, agent, sandbox, events = _transaction_inputs()
    lease = provisioner.acquire(agent)
    provisioner.mark_execution_attempted(lease)

    provisioner.finalize(lease, succeeded=False)

    assert events == [
        "lock.acquire",
        "sandbox.ensure_started",
        "sandbox.discard",
        "inbox.put",
        "provisioner.teardown",
        "lock.release",
    ]
    assert lease.discarded is True
    assert lease.committed is False
    assert len(agent.inbox.messages) == 1
    assert "reverted" in agent.inbox.messages[0]


def test_failed_discard_stops_dirty_live_state_but_still_surfaces_the_failure():
    provisioner, agent, sandbox, events = _transaction_inputs()
    sandbox.discard_error = RuntimeError("discard failed")
    lease = provisioner.acquire(agent)
    provisioner.mark_execution_attempted(lease)

    with pytest.raises(RuntimeError, match="discard failed"):
        provisioner.finalize(lease, succeeded=False)

    assert events[-4:] == [
        "sandbox.discard",
        "sandbox.stop",
        "provisioner.teardown",
        "lock.release",
    ]
    assert lease.discard_attempted is True
    assert lease.discarded is False
    assert lease.committed is False
    assert lease.hibernated is True
    assert agent.inbox.messages == []


def test_next_lease_retries_failed_discard_before_starting_dirty_state():
    provisioner, first_agent, sandbox, events = _transaction_inputs()
    second_agent = _Agent(sandbox=sandbox, agname="second", events=events)
    sandbox.discard_error = RuntimeError("discard failed")
    first_lease = provisioner.acquire(first_agent)
    provisioner.mark_execution_attempted(first_lease)

    with pytest.raises(RuntimeError, match="discard failed"):
        provisioner.finalize(first_lease, succeeded=False)

    assert sandbox._provisioner_discard_required is True
    assert first_agent.inbox.messages == []

    sandbox.discard_error = None
    events.clear()
    second_lease = provisioner.acquire(second_agent)

    assert events[:4] == [
        "lock.acquire",
        "sandbox.discard",
        "inbox.put",
        "sandbox.ensure_started",
    ]
    assert sandbox._provisioner_discard_required is False
    assert second_lease.discard_attempted is False
    assert second_lease.discarded is False
    assert len(first_agent.inbox.messages) == 1
    assert second_agent.inbox.messages == []

    provisioner.finalize(second_lease, succeeded=False)


def test_pre_execution_failure_hibernates_without_discard_or_revert_notice():
    provisioner, agent, sandbox, events = _transaction_inputs()
    lease = provisioner.acquire(agent)

    provisioner.finalize(lease, succeeded=False)

    assert events == [
        "lock.acquire",
        "sandbox.ensure_started",
        "sandbox.pending",
        "sandbox.stop",
        "provisioner.teardown",
        "lock.release",
    ]
    assert lease.discarded is False
    assert lease.hibernated is True
    assert agent.inbox.messages == []


def test_success_requires_an_execution_attempt_and_still_releases_the_lock():
    provisioner, agent, sandbox, events = _transaction_inputs()
    lease = provisioner.acquire(agent)

    with pytest.raises(RuntimeError, match="execution was not attempted"):
        provisioner.finalize(lease, succeeded=True)

    assert "sandbox.commit" not in events
    assert "sandbox.discard" not in events
    assert events[-1] == "lock.release"
    assert sandbox._lock.balance == 0


@pytest.mark.parametrize(
    ("commit_result", "commit_error", "message"),
    [
        (False, None, "no physical sandbox existed"),
        (True, RuntimeError("commit failed"), "commit failed"),
    ],
)
def test_commit_failure_attempts_discard_before_teardown_and_release(
    commit_result, commit_error, message
):
    provisioner, agent, sandbox, events = _transaction_inputs()
    sandbox.commit_result = commit_result
    sandbox.commit_error = commit_error
    lease = provisioner.acquire(agent)
    provisioner.mark_execution_attempted(lease)

    with pytest.raises(RuntimeError, match=message):
        provisioner.finalize(lease, succeeded=True)

    assert events[-5:] == [
        "sandbox.commit",
        "sandbox.discard",
        "inbox.put",
        "provisioner.teardown",
        "lock.release",
    ]
    assert lease.discarded is True
    assert len(agent.inbox.messages) == 1


def test_commit_error_remains_primary_when_discard_and_teardown_fail():
    provisioner, agent, sandbox, events = _transaction_inputs()
    sandbox.commit_error = RuntimeError("commit failed")
    sandbox.discard_error = RuntimeError("discard failed")
    provisioner.teardown_error = RuntimeError("teardown failed")
    lease = provisioner.acquire(agent)
    provisioner.mark_execution_attempted(lease)

    with pytest.raises(RuntimeError, match="commit failed") as raised:
        provisioner.finalize(lease, succeeded=True)

    notes = "\n".join(raised.value.__notes__)
    assert "discard failed" in notes
    assert "teardown failed" in notes
    assert events[-1] == "lock.release"


def test_teardown_failure_still_releases_the_lock_last():
    provisioner, agent, sandbox, events = _transaction_inputs()
    provisioner.teardown_error = RuntimeError("teardown failed")
    lease = provisioner.acquire(agent)
    provisioner.mark_execution_attempted(lease)

    with pytest.raises(RuntimeError, match="teardown failed"):
        provisioner.finalize(lease, succeeded=True)

    assert events[-2:] == ["provisioner.teardown", "lock.release"]
    assert sandbox._lock.balance == 0


def test_finalize_is_idempotent_and_releases_exactly_once():
    provisioner, agent, sandbox, events = _transaction_inputs()
    lease = provisioner.acquire(agent)
    provisioner.mark_execution_attempted(lease)
    provisioner.finalize(lease, succeeded=True)
    first_events = list(events)

    provisioner.finalize(lease, succeeded=False)

    assert events == first_events
    assert events.count("sandbox.commit") == 1
    assert events.count("sandbox.discard") == 0
    assert sandbox._lock.release_calls == 1


def test_mark_execution_attempted_requires_an_active_provisioned_lease():
    provisioner, agent, _, _ = _transaction_inputs()
    lease = provisioner.acquire(agent)
    provisioner.finalize(lease, succeeded=False)

    with pytest.raises(RuntimeError, match="finalized"):
        provisioner.mark_execution_attempted(lease)


def test_shared_sandbox_serializes_the_complete_provisioner_transaction():
    events = []
    sandbox = _Sandbox(events)
    first_agent = _Agent(sandbox=sandbox, agname="first", events=events)
    second_agent = _Agent(sandbox=sandbox, agname="second", events=events)
    first_provisioner = _RecordingProvisioner(events)
    second_provisioner = _RecordingProvisioner(events)
    second_acquire_attempted = threading.Event()
    second_finished = threading.Event()
    errors = []

    first_lease = first_provisioner.acquire(first_agent)

    def observe_second_attempt():
        if threading.current_thread().name == "second-sandbox-transaction":
            second_acquire_attempted.set()

    sandbox._lock.on_acquire_attempt = observe_second_attempt

    def run_second_transaction():
        try:
            second_lease = second_provisioner.acquire(second_agent)
            second_provisioner.finalize(second_lease, succeeded=False)
        except BaseException as exc:
            errors.append(exc)
        finally:
            second_finished.set()

    second_thread = threading.Thread(
        target=run_second_transaction,
        name="second-sandbox-transaction",
    )
    second_thread.start()
    assert second_acquire_attempted.wait(timeout=2)
    assert second_thread.is_alive()
    assert events.count("sandbox.ensure_started") == 1

    first_provisioner.finalize(first_lease, succeeded=False)

    assert second_finished.wait(timeout=2)
    second_thread.join(timeout=2)
    assert not second_thread.is_alive()
    assert errors == []

    first_release = events.index("lock.release")
    second_start = [
        index for index, event in enumerate(events) if event == "sandbox.ensure_started"
    ][1]
    assert first_release < second_start
    assert sandbox._lock.balance == 0
