"""A local traced CLI uses the existing policy and telemetry boundaries."""

from agency.agpolicy import agpolicy
from agency.engine.host_servers.interaction_policy import InteractionSyscallPolicy

from .test_host_interaction_server import _FakeDataLogger, _FakeSandbox, _make_server, _make_syscall


def test_admission_keeps_policy_decision_environment_and_completion_correlation():
    observed = []
    logger = _FakeDataLogger()
    server = _make_server(
        policy=agpolicy(syscall_hooks={"execve": lambda event: observed.append(event) or True}),
        data_logger=logger,
        sandbox=_FakeSandbox(gpu_count_requested=1, gpu_ids=[3]),
    )
    bridge = InteractionSyscallPolicy(server)
    event = _make_syscall(syscall="execve", path="/bin/true", argv=["true"])

    allowed, reason, call_id, environment = bridge.check(None, event)
    assert allowed and reason is None
    assert observed == [event]
    assert environment == {"CUDA_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "3"}
    assert logger.events[0][1]["call_id"] == call_id
    bridge.check_completion(None, call_id, 0)
    bridge.check_completion(None, call_id, 0)

    assert [record[0] for record in logger.events] == [
        "syscall_call",
        "agent_state",
        "syscall_result",
    ]
    assert logger.events[-1][1]["call_id"] == call_id
    assert logger.events[-1][1]["return_value"] == 0
    assert len(logger.spans) == 1


def test_denial_uses_shared_policy_and_does_not_create_a_completed_call():
    logger = _FakeDataLogger()
    server = _make_server(
        policy=agpolicy(syscall_hooks={"connect": lambda event: (False, "blocked destination")}),
        data_logger=logger,
    )
    bridge = InteractionSyscallPolicy(server)

    allowed, reason, call_id, environment = bridge.check(None, _make_syscall(syscall="connect"))
    assert not allowed and reason == "blocked destination" and environment is None
    bridge.check_completion(None, call_id, -1)
    bridge.check_completion(None, None, -1)
    assert [record[0] for record in logger.events] == ["syscall_call", "agent_state"]
    assert not logger.spans


def test_file_access_uses_the_existing_observation_event():
    logger = _FakeDataLogger()
    bridge = InteractionSyscallPolicy(_make_server(data_logger=logger))
    payload = {"operation": "read", "path": "/workspace/source.py", "pid": 123}

    bridge.record_file_access(payload)

    assert len(logger.events) == 1
    kind, recorded, *_ = logger.events[0]
    assert kind == "checkpoint_file_access" and recorded == payload
    assert logger.events[0][-1] is False
