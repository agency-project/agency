"""Local facts must be observed on the exact dispatched attempt, never inferred."""

from types import SimpleNamespace
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from agency.engine.clients import HarnessInteractionClient
from agency.harness.control_observation import ControlObservations
from agency.harness.daemon import HarnessManager
from agency.harness.ptrace._tracer_loop import TracerLoop
from agency.harness.servers.harness_interaction_server import HarnessInteractionServer


def test_pause_intent_resume_queue_and_root_exit_are_not_application():
    loop = TracerLoop((), lambda _: None)
    loop._known_pids = {10, 11}
    loop._held_pids = {10, 11}
    loop._parked_pids = {10}
    assert loop._control_snapshot()["local_state"] == "pending"
    loop._parked_pids.add(11)
    assert loop._control_snapshot()["local_state"] == "suspended"
    loop._pending_clone_pids.add(12)
    assert loop._control_snapshot()["local_state"] == "pending"
    loop._pending_clone_pids.clear()
    loop.resume()
    assert loop._control_snapshot()["local_state"] == "pending"
    loop._resume_requests.clear()
    loop._resume_in_progress.add(10)
    assert loop._control_snapshot()["local_state"] == "pending"
    loop._resume_in_progress.clear()
    loop._resume_unobserved.add(11)
    assert loop._control_snapshot()["local_state"] == "pending"
    loop._resume_unobserved.clear()
    assert loop._control_snapshot()["local_state"] == "continued"
    loop._returncode = 0  # Root exit does not establish child exit.
    loop._known_pids = {11}
    assert loop._control_snapshot()["local_state"] != "exited"
    loop._known_pids.clear()
    assert loop._control_snapshot()["local_state"] == "pending"
    loop._finished.set()
    assert loop._control_snapshot()["local_state"] == "exited"
    loop._exit_observation_lost = True
    assert loop._control_snapshot()["local_state"] == "unknown"
    loop._exit_observation_lost = False
    loop._checkpoint_detached = True
    assert loop._control_snapshot()["local_state"] == "unknown"


@pytest.mark.parametrize(
    ("action", "desired"), [("pause", "suspended"), ("resume", "continued"), ("cancel", "exited")]
)
def test_observation_retains_exact_attempt_and_terminal_evidence(action, desired):
    facts = {"local_state": "pending", "task_count": 2, "parked_count": 1}
    handle = SimpleNamespace(_control_snapshot=lambda: facts)
    registry = ControlObservations()
    initial = registry.add(action, handle, "invocation-A", "attempt-A")
    assert initial["state"] == "pending"
    facts["local_state"] = desired
    result = registry.get(initial["id"])
    assert result["state"] == "confirmed"
    assert result["request_id"] == "invocation-A" and result["attempt_token"] == "attempt-A"
    assert result["observed_at"] > 0 and result["task_count"] == 2
    facts["local_state"] = "unknown"
    assert registry.get(initial["id"]) == result
    assert registry.rows[initial["id"]]["handle"] is None


def test_missing_superseded_and_evicted_observations_stay_unknown_or_unavailable():
    registry = ControlObservations()
    assert registry.add("pause", None, None, None)["state"] == "unavailable"
    handle = SimpleNamespace(_control_snapshot=lambda: {"local_state": "pending"})
    old = registry.add("pause", handle, "A", "a")
    registry.add("resume", handle, "A", "a")
    assert registry.get(old["id"])["state"] == "unknown"
    assert registry.rows[old["id"]]["handle"] is None
    for _ in range(70):
        registry.add("pause", None, None, None)
    assert len(registry.rows) == 64
    assert registry.get(old["id"])["state"] == "unknown"
    assert (
        registry.add(
            "pause", SimpleNamespace(_control_snapshot=lambda: {"local_state": "exited"}), "B", "b"
        )["state"]
        == "target_exited"
    )

    def broken():
        raise OSError("disconnected observer")

    assert (
        registry.add("pause", SimpleNamespace(_control_snapshot=broken), "C", "c")["state"]
        == "unknown"
    )


def test_request_fence_and_read_only_endpoint_never_probe_replacement_handle():
    calls = []
    facts = {"local_state": "pending"}
    manager = HarnessManager.__new__(HarnessManager)
    manager._control_lock = threading.Lock()
    manager._current_request_id = "A"
    manager._current_attempt_token = "attempt-A"
    manager._current_request_cancelled = False
    manager._current_cancel_force = True
    manager._agent_paused = False
    manager._current_control_handle = SimpleNamespace(
        pause=lambda: calls.append("pause-A"),
        terminate=lambda: calls.append("term-A"),
        kill=lambda: calls.append("kill-A"),
        _control_snapshot=lambda: facts,
    )
    server = HarnessInteractionServer(
        "/unused",
        lambda _: None,
        control_handler=manager.control,
        observation_handler=manager._control_observation,
    )
    with TestClient(server.build_app()) as client:
        initial = client.post("/control/pause").json()["observation"]
        assert initial["state"] == "pending" and calls == ["pause-A"]
        manager._current_request_id = "B"
        manager._current_attempt_token = "attempt-B"
        manager._current_control_handle = SimpleNamespace(
            _control_snapshot=lambda: pytest.fail("Replacement must not be probed")
        )
        assert client.post("/control/cancel", json={"request_id": "A"}).json() == {"ok": True}
        facts["local_state"] = "suspended"
        observed = client.get("/control_observation/" + initial["id"]).json()
        assert observed["state"] == "confirmed" and observed["request_id"] == "A"
        assert calls == ["pause-A"]
        assert client.get("/control_observation/bad").status_code == 400


def test_client_public_none_and_old_daemon_compatibility():
    client = HarnessInteractionClient("/unused")
    responses = [{"ok": True, "observation": {"id": "a" * 32, "state": "pending"}}, {"ok": True}]
    paths = []

    def respond(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=responses.pop(0))

    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(respond), base_url="http://local")
    try:
        assert client.pause_harness() is None
        assert client._last_control_observation["uds_path"] == "/unused"
        assert client.resume_harness() is None
        assert client._last_control_observation is None
        assert paths == ["/control/pause", "/control/resume"]
    finally:
        client.close()


@pytest.mark.parametrize("force", [False, True])
def test_engine_cancellation_captures_exact_request_observation_and_clears_stale_receipts(force):
    from agency.engine.engine import AgentEngine

    called = []
    observation = {"id": "a" * 32, "request_id": "A", "attempt_token": "attempt-A"}
    client = SimpleNamespace(
        cancel_harness=lambda request, **kwargs: called.append((request, kwargs)),
        _last_control_observation=observation,
    )
    engine = AgentEngine.__new__(AgentEngine)
    engine._services_lock = threading.RLock()
    engine._services_closed = False
    engine._sandbox_interaction_client = client
    engine._request_id = "A"
    assert engine.cancel(force=force) is None
    assert called == [("A", {"force": force})]
    assert engine._last_control_receipt == observation
    engine._services_closed = True
    engine.cancel(force=force)
    assert engine._last_control_receipt is None and len(called) == 1
