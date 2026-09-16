from unittest.mock import Mock

from agency.configs.agconfig import agconfig
from agency.harness.clients.host_services_client import HostServicesClient
from agency.harness.daemon import HarnessManager
from agency.harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload


class _Execution:
    def __init__(self):
        self.handle = SimpleHandle()
        self.runs = []
        self.closed = False

    def run(self, prompt, *, keep_alive=False):
        self.runs.append((prompt, keep_alive))
        return prompt

    def close(self):
        self.closed = True


class SimpleHandle:
    returncode = None


def test_persistent_runner_reuses_matching_idle_execution():
    manager = HarnessManager.__new__(HarnessManager)
    manager._persistent = True
    manager._live_execution = None
    manager._live_execution_key = None
    manager._live_local_token = None
    first = _Execution()
    second_factory = Mock(return_value=_Execution())

    assert manager._run_pty_execution("same", lambda: first, "one") == "one"
    assert manager._run_pty_execution("same", second_factory, "two") == "two"

    second_factory.assert_not_called()
    assert first.runs == [("one", True), ("two", True)]
    assert first.closed is False


def test_persistent_runner_retires_incompatible_execution():
    manager = HarnessManager.__new__(HarnessManager)
    manager._persistent = True
    manager._live_execution = None
    manager._live_execution_key = None
    manager._live_local_token = None
    first, second = _Execution(), _Execution()

    manager._run_pty_execution("first", lambda: first, "one")
    manager._run_pty_execution("second", lambda: second, "two")

    assert first.closed is True
    assert second.runs == [("two", True)]


def test_restored_cli_token_maps_to_current_host_attempt():
    client = HostServicesClient("/tmp/nonexistent-agency-fast-resume.sock")
    try:
        client.register_attempt_token("host-attempt-2", local_token="process-attempt-1")
        assert client.validate_token("process-attempt-1") is True
        assert client.validate_token("host-attempt-2") is False
        assert client._attempt_headers("process-attempt-1") == {
            "X-Agency-Attempt-Token": "host-attempt-2"
        }
        assert client.clear_attempt_token("host-attempt-2") is True
        assert client.validate_token("process-attempt-1") is False
    finally:
        client.close()


def test_persistent_adapter_uses_the_restored_process_token(monkeypatch):
    manager = HarnessManager.__new__(HarnessManager)
    manager._agconfig = agconfig()
    manager._harness = "codex"
    manager._bootstrapped = True
    manager._engine_name = "agent-1"
    manager._persistent = True
    manager._current_attempt_token = "host-attempt-2"
    manager._live_local_token = "process-attempt-1"
    manager._register_control_handle = Mock()
    manager._register_redirect = Mock()
    manager._run_pty_execution = Mock()
    seen = []

    class HarnessApi:
        base_url = "http://127.0.0.1:8766"

        def resolve_model(self, token):
            seen.append(("model", token))
            return "model"

        def syscall_policy(self, token, **_kwargs):
            seen.append(("policy", token))
            return object()

    manager._harness_api = HarnessApi()

    def run_adapter(*args):
        seen.append(("adapter", args[-1]))
        return HarnessAttemptResult(ok=True)

    monkeypatch.setattr("agency.harness.daemon._run_adapter_attempt", run_adapter)
    result = manager._run_adapter_request(
        HarnessAttemptRequest(
            prompt=PromptPayload("", "continue"),
            harness="codex",
            attempt_token="host-attempt-2",
        )
    )

    assert result.ok
    assert seen == [
        ("model", "process-attempt-1"),
        ("policy", "process-attempt-1"),
        ("adapter", "process-attempt-1"),
    ]
