# The synchronous pause acknowledgement (see docs/Message_Patches.md,
# "Pause makes it synchronous").

from __future__ import annotations

import threading
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from agency.engine.clients import HarnessInteractionClient
from agency.harness.daemon import HarnessManager
from agency.harness.servers import HarnessInteractionServer


# -- daemon: control("pause") ----------------------------------------------------------


def _manager(handle) -> HarnessManager:
    manager = HarnessManager("/unused/sandbox", "/unused/host", "agent", "claude_code")
    manager._current_control_handle = handle
    return manager


def test_pause_without_a_wait_just_requests_the_stop():
    calls = []
    handle = SimpleNamespace(pause=lambda: calls.append("pause"))
    assert _manager(handle).control("pause") is None
    assert calls == ["pause"]


def test_synchronous_pause_returns_parked_once_the_harness_is_parked():
    order = []
    handle = SimpleNamespace(
        pause=lambda: order.append("pause"),
        wait_paused=lambda timeout: order.append(("wait", timeout)) or True,
    )
    manager = _manager(handle)

    ack = manager.control("pause", wait_timeout_s=3)

    assert ack == {"parked": True}
    assert order == ["pause", ("wait", 3)]


def test_synchronous_pause_reports_a_harness_that_would_not_stop():
    handle = SimpleNamespace(pause=lambda: None, wait_paused=lambda timeout: False)
    assert _manager(handle).control("pause", wait_timeout_s=0.1) == {"parked": False}


def test_synchronous_pause_with_no_harness_process_is_immediately_parked():
    manager = _manager(None)
    ack = manager.control("pause", wait_timeout_s=5)
    assert ack == {"parked": True}
    assert manager._agent_paused is True  # the next process starts stopped


def test_the_wait_does_not_hold_the_control_lock():
    """cancel and resume must stay deliverable while a pause is waiting."""
    waiting, release = threading.Event(), threading.Event()

    def wait_paused(timeout):
        waiting.set()
        assert release.wait(3)
        return True

    killed = []
    handle = SimpleNamespace(
        pause=lambda: None,
        wait_paused=wait_paused,
        resume=lambda: None,
        kill=lambda: killed.append(1),
    )
    manager = _manager(handle)
    manager._current_request_id = "run"
    pauser = threading.Thread(target=manager.control, args=("pause",), kwargs={"wait_timeout_s": 3})
    pauser.start()
    try:
        assert waiting.wait(2)
        manager.control("cancel", request_id="run")  # would deadlock if the lock were held
        assert killed == [1]
    finally:
        release.set()
        pauser.join(3)


# -- the route and client -----------------------------------------------------------------


def test_control_route_round_trips_the_ack_and_the_wait_timeout():
    seen = {}

    def control(action, **kwargs):
        seen[action] = kwargs
        return {"parked": True} if action == "pause" else None

    server = HarnessInteractionServer("/unused.sock", lambda request: None, control_handler=control)
    client = TestClient(server.build_app())

    waited = client.post("/control/pause", json={"wait_timeout_s": 4})
    assert waited.json() == {"ok": True, "parked": True}
    assert seen["pause"] == {"wait_timeout_s": 4.0}

    plain = client.post("/control/pause")
    assert plain.json() == {"ok": True, "parked": True}
    assert seen["pause"] == {}

    assert client.post("/control/resume").json() == {"ok": True}


def test_client_asks_for_a_wait_and_outlives_it():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "parked": True})

    client = HarnessInteractionClient("/unused.sock")
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")

    assert client.pause_harness(wait_timeout_s=7) == {"ok": True, "parked": True}
    assert b'"wait_timeout_s"' in requests[0].content
    assert requests[0].extensions["timeout"]["read"] == 7 + 5

    client.pause_harness()
    assert requests[1].content == b""
