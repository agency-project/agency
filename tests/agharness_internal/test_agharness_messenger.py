"""Tests for agharness_messenger.py -- the shared host-side bridge for an
agent's pause/inbox state, reached by native.py's in-container loop before
every turn (mirrors what execute_react() already does in-process via
ag._check_pause()/ag._drain_inbox()).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from agency.agharness_internal.agharness_messenger import agHarnessMessenger


def _make_agent_with_inbox(*pending_messages):
    """A minimal object exposing exactly the (_check_pause, _drain_inbox)
    surface the messenger's route reads -- real queue semantics, not a
    full `agent`, since _drain_inbox's own draining logic is what's under
    test here, not agent.py's queue plumbing."""
    ag = MagicMock()

    def _drain_inbox(messages):
        had = False
        for m in pending_messages:
            messages.append({"role": "user", "content": m})
            had = True
        return had

    ag._drain_inbox.side_effect = _drain_inbox
    return ag


def _client_for(messenger):
    return TestClient(messenger._app)


def test_check_in_unknown_token_returns_401():
    messenger = agHarnessMessenger()
    client = _client_for(messenger)
    resp = client.post("/internal/check_in", json={"token": "nope"})
    assert resp.status_code == 401


def test_check_in_calls_check_pause_and_returns_drained_inbox():
    messenger = agHarnessMessenger()
    ag = _make_agent_with_inbox("hello from the human")
    token = "tok"
    messenger.register(token, ag)
    try:
        client = _client_for(messenger)
        resp = client.post("/internal/check_in", json={"token": token})
        assert resp.status_code == 200
        ag._check_pause.assert_called_once()
        messages = resp.json()["messages"]
        assert messages == [{"role": "user", "content": "hello from the human"}]
    finally:
        messenger.unregister(token)


def test_check_in_returns_empty_list_when_inbox_is_empty():
    messenger = agHarnessMessenger()
    ag = _make_agent_with_inbox()  # nothing pending
    token = "tok-empty"
    messenger.register(token, ag)
    try:
        client = _client_for(messenger)
        resp = client.post("/internal/check_in", json={"token": token})
        assert resp.status_code == 200
        assert resp.json()["messages"] == []
    finally:
        messenger.unregister(token)


def test_unregister_removes_token():
    messenger = agHarnessMessenger()
    ag = _make_agent_with_inbox()
    messenger.register("tok", ag)
    messenger.unregister("tok")
    client = _client_for(messenger)
    resp = client.post("/internal/check_in", json={"token": "tok"})
    assert resp.status_code == 401


def test_real_tcp_roundtrip():
    messenger = agHarnessMessenger()
    base_url = messenger.start(timeout_s=10)
    try:
        ag = _make_agent_with_inbox("via real tcp")
        token = "tok-tcp"
        messenger.register(token, ag)
        try:
            import httpx

            resp = httpx.post(f"{base_url}/internal/check_in", json={"token": token}, timeout=10)
            assert resp.status_code == 200
            assert resp.json()["messages"][0]["content"] == "via real tcp"
        finally:
            messenger.unregister(token)
    finally:
        messenger.stop()


def test_real_uds_roundtrip():
    """The transport native.py's in-container entrypoint actually uses --
    a TCP-only listener isn't reachable from inside a container."""
    messenger = agHarnessMessenger()
    sock_path = messenger.ensure_uds_started(timeout_s=10)
    try:
        ag = _make_agent_with_inbox("via real uds")
        token = "tok-uds"
        messenger.register(token, ag)
        try:
            import httpx

            transport = httpx.HTTPTransport(uds=sock_path)
            with httpx.Client(transport=transport, base_url="http://agharness-messenger") as client:
                resp = client.post("/internal/check_in", json={"token": token}, timeout=10)
            assert resp.status_code == 200
            assert resp.json()["messages"][0]["content"] == "via real uds"
        finally:
            messenger.unregister(token)
    finally:
        messenger.stop_uds()
