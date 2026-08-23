from __future__ import annotations

from types import SimpleNamespace

import pytest

from agency.agdata import agdata, agerror
from agency.engine.execution_builder import ExecutionBuilder
from agency.engine.types import CompletedResult


class _Provisioner:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.sandbox = SimpleNamespace(name="sandbox")
        self.lease = SimpleNamespace(sandbox=self.sandbox, execution_attempted=False)
        self.acquire_error: BaseException | None = None
        self.mark_error: BaseException | None = None
        self.finalize_error: BaseException | None = None
        self.finalize_outcomes: list[bool] = []

    def acquire(self, agent):
        self.events.extend(["sandbox.resolve", "lock.acquire", "sandbox.start"])
        if self.acquire_error is not None:
            raise self.acquire_error
        return self.lease

    def mark_execution_attempted(self, lease):
        assert lease is self.lease
        self.events.append("sandbox.execution_attempted")
        if self.mark_error is not None:
            raise self.mark_error
        lease.execution_attempted = True

    def finalize(self, lease, *, succeeded):
        assert lease is self.lease
        self.events.append(f"sandbox.finalize:{succeeded}")
        self.finalize_outcomes.append(succeeded)
        if self.finalize_error is not None:
            raise self.finalize_error


class _HostServer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.start_error: BaseException | None = None
        self.stop_error: BaseException | None = None

    def start(self):
        self.events.append("host.start:/actual/host.sock")
        if self.start_error is not None:
            raise self.start_error
        return "/actual/host.sock"

    def stop(self):
        self.events.append("host.stop")
        if self.stop_error is not None:
            raise self.stop_error


class _Bridge:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.launch_error: BaseException | None = None
        self.run_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        self.stop_calls: list[int | None] = []

    def ensure_launched(self, sandbox, host_uds_path):
        self.events.append(f"harness.launch:{host_uds_path}")
        if self.launch_error is not None:
            raise self.launch_error
        return 42

    def run_attempt(self, agconfig, skill, harness_manager_pid, prompt):
        self.events.append(f"harness.run:{harness_manager_pid}:{prompt}")
        if self.run_error is not None:
            raise self.run_error
        return "attempt"

    def wait(self, harness_manager_pid):
        self.events.append(f"bridge.wait:{harness_manager_pid}")

    def stop(self, harness_manager_pid):
        self.events.append(f"harness.stop:{harness_manager_pid}")
        self.stop_calls.append(harness_manager_pid)
        if self.stop_error is not None:
            raise self.stop_error


class _Skill:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.prompt_error: BaseException | None = None

    def build_prompt_payload(self, skill_input):
        self.events.append("prompt.build")
        if self.prompt_error is not None:
            raise self.prompt_error
        return "compiled-prompt"


class _Builder(ExecutionBuilder):
    def __init__(
        self,
        provisioner: _Provisioner,
        bridge: _Bridge,
        host: _HostServer,
        result: object,
    ) -> None:
        super().__init__(sandbox_provisioner=provisioner, harness_bridge=bridge)
        self._test_events = provisioner.events
        self._test_host = host
        self.test_skill = _Skill(provisioner.events)
        self.wait_result = result
        self.host_build_error: BaseException | None = None
        self.wait_error: BaseException | None = None

    def _build_host_side_server(self, agent, skill, sandbox, resource_pool):
        self._test_events.append("host.build")
        if self.host_build_error is not None:
            raise self.host_build_error
        return self._test_host

    def _wait_for_completion(self, harness_manager_pid, attempt):
        self._test_events.append(f"harness.wait:{harness_manager_pid}:{attempt}")
        if self.wait_error is not None:
            raise self.wait_error
        return self.wait_result


def _completed(output=None, *, ok=True, error_message="") -> CompletedResult:
    return CompletedResult(
        output=output if output is not None else agdata(result="ok"),
        context="context",
        delta=[],
        ok=ok,
        error_message=error_message,
    )


def _inputs(result: object | None = None):
    events: list[str] = []
    provisioner = _Provisioner(events)
    host = _HostServer(events)
    bridge = _Bridge(events)
    builder = _Builder(provisioner, bridge, host, result or _completed())
    agent = SimpleNamespace(agconfig="config")
    return builder, provisioner, host, bridge, agent, events


def _execute(builder: ExecutionBuilder, agent):
    return builder.execute(
        agent=agent,
        context="context",
        skill=builder.test_skill,
        skill_input="input",
        resource_pool="pool",
    )


def test_execute_sequences_preparation_harness_cleanup_and_success_finalization():
    builder, provisioner, _, bridge, agent, events = _inputs()

    result = _execute(builder, agent)

    assert result.output.result == "ok"
    assert bridge.stop_calls == [42]
    assert provisioner.finalize_outcomes == [True]
    assert events == [
        "sandbox.resolve",
        "lock.acquire",
        "sandbox.start",
        "host.build",
        "host.start:/actual/host.sock",
        "prompt.build",
        "sandbox.execution_attempted",
        "harness.launch:/actual/host.sock",
        "harness.run:42:compiled-prompt",
        "harness.wait:42:attempt",
        "harness.stop:42",
        "host.stop",
        "sandbox.finalize:True",
    ]


@pytest.mark.parametrize(
    ("phase", "message"),
    [
        ("launch", "launch failed"),
        ("run", "run failed"),
        ("wait", "wait failed"),
    ],
)
def test_harness_phase_failures_stop_services_then_roll_back(phase, message):
    builder, provisioner, _, bridge, agent, events = _inputs()
    error = RuntimeError(message)
    if phase == "launch":
        bridge.launch_error = error
    elif phase == "run":
        bridge.run_error = error
    else:
        builder.wait_error = error

    with pytest.raises(RuntimeError, match=message):
        _execute(builder, agent)

    expected_pid = None if phase == "launch" else 42
    assert bridge.stop_calls == [expected_pid]
    assert events.index(f"harness.stop:{expected_pid}") < events.index("host.stop")
    assert events.index("host.stop") < events.index("sandbox.finalize:False")
    assert provisioner.finalize_outcomes == [False]


def test_host_start_failure_stops_partial_host_before_pre_execution_finalization():
    builder, provisioner, host, bridge, agent, events = _inputs()
    host.start_error = RuntimeError("host start failed")

    with pytest.raises(RuntimeError, match="host start failed"):
        _execute(builder, agent)

    assert bridge.stop_calls == []
    assert "sandbox.execution_attempted" not in events
    assert events[-2:] == ["host.stop", "sandbox.finalize:False"]
    assert provisioner.finalize_outcomes == [False]


def test_prompt_failure_stops_host_without_launching_the_harness():
    builder, provisioner, _, bridge, agent, events = _inputs()
    builder.test_skill.prompt_error = RuntimeError("prompt failed")

    with pytest.raises(RuntimeError, match="prompt failed"):
        _execute(builder, agent)

    assert bridge.stop_calls == []
    assert not any(event.startswith("harness.launch") for event in events)
    assert events[-2:] == ["host.stop", "sandbox.finalize:False"]
    assert provisioner.finalize_outcomes == [False]


def test_host_cleanup_failure_prevents_commit_and_causes_rollback():
    builder, provisioner, host, _, agent, events = _inputs()
    host.stop_error = RuntimeError("host cleanup failed")

    with pytest.raises(RuntimeError, match="host cleanup failed"):
        _execute(builder, agent)

    assert events.index("harness.stop:42") < events.index("host.stop")
    assert events.index("host.stop") < events.index("sandbox.finalize:False")
    assert events.count("host.stop") == 2
    assert provisioner.finalize_outcomes == [False]


def test_transient_host_cleanup_failure_is_retried_before_commit():
    builder, provisioner, host, _, agent, events = _inputs()
    stop_calls = 0

    def transient_stop():
        nonlocal stop_calls
        stop_calls += 1
        events.append("host.stop")
        if stop_calls == 1:
            raise RuntimeError("transient host cleanup failure")

    host.stop = transient_stop

    result = _execute(builder, agent)

    assert result.output.result == "ok"
    assert stop_calls == 2
    assert events[-1] == "sandbox.finalize:True"
    assert provisioner.finalize_outcomes == [True]


def test_error_output_is_returned_but_the_sandbox_is_not_committed():
    result = _completed(agerror("harness failed"))
    builder, provisioner, _, _, agent, _ = _inputs(result)

    assert _execute(builder, agent) is result
    assert provisioner.finalize_outcomes == [False]


def test_error_output_remains_primary_when_cleanup_also_fails():
    result = _completed(agerror("harness failed"))
    builder, provisioner, host, _, agent, _ = _inputs(result)
    host.stop_error = RuntimeError("host cleanup failed")
    provisioner.finalize_error = RuntimeError("discard failed")

    with pytest.raises(
        RuntimeError, match="execution result reported an error: harness failed"
    ) as raised:
        _execute(builder, agent)

    notes = "\n".join(raised.value.__notes__)
    assert "host cleanup failed" in notes
    assert "discard failed" in notes
    assert provisioner.finalize_outcomes == [False]


def test_unsuccessful_result_without_error_output_becomes_primary_exception():
    builder, provisioner, _, _, agent, _ = _inputs(
        _completed(ok=False, error_message="manager failed")
    )

    with pytest.raises(RuntimeError, match="manager failed"):
        _execute(builder, agent)

    assert provisioner.finalize_outcomes == [False]


def test_malformed_completed_result_is_rejected_and_rolled_back():
    builder, provisioner, _, _, agent, _ = _inputs(SimpleNamespace(ok=True))

    with pytest.raises(TypeError, match="must return CompletedResult"):
        _execute(builder, agent)

    assert provisioner.finalize_outcomes == [False]


def test_cleanup_failures_do_not_replace_the_primary_execution_error():
    builder, provisioner, host, bridge, agent, events = _inputs()
    builder.wait_error = RuntimeError("primary wait failure")
    bridge.stop_error = RuntimeError("harness stop failure")
    host.stop_error = RuntimeError("host stop failure")
    provisioner.finalize_error = RuntimeError("discard/teardown failure")

    with pytest.raises(RuntimeError, match="primary wait failure") as raised:
        _execute(builder, agent)

    notes = "\n".join(raised.value.__notes__)
    assert "harness stop failure" in notes
    assert "host stop failure" in notes
    assert "discard/teardown failure" in notes
    assert events[-1] == "sandbox.finalize:False"


def test_nested_cleanup_notes_are_preserved_on_the_primary_execution_error():
    builder, provisioner, _, _, agent, _ = _inputs()
    builder.wait_error = RuntimeError("primary wait failure")
    finalization_error = RuntimeError("discard failed")
    finalization_error.add_note("teardown also failed: stop failed")
    provisioner.finalize_error = finalization_error

    with pytest.raises(RuntimeError, match="primary wait failure") as raised:
        _execute(builder, agent)

    notes = "\n".join(raised.value.__notes__)
    assert "discard failed" in notes
    assert "teardown also failed: stop failed" in notes


def test_acquisition_failure_is_already_unwound_and_not_finalized_again():
    builder, provisioner, _, bridge, agent, events = _inputs()
    provisioner.acquire_error = RuntimeError("start failed")

    with pytest.raises(RuntimeError, match="start failed"):
        _execute(builder, agent)

    assert bridge.stop_calls == []
    assert provisioner.finalize_outcomes == []
    assert events == ["sandbox.resolve", "lock.acquire", "sandbox.start"]
