# Tests for host_server_base.py -- the shared contract every host server subclasses.

from __future__ import annotations

import pytest

from agency.engine.host_servers.host_server_base import HostServerBase


# ---------------------------------------------------------------------------
# Default (unoverridden) contract
# ---------------------------------------------------------------------------


def test_set_config_raises_not_implemented():
    server = HostServerBase()
    with pytest.raises(NotImplementedError):
        server.set_config(object())


def test_build_app_raises_not_implemented():
    server = HostServerBase()
    with pytest.raises(NotImplementedError):
        server.build_app()


def test_start_is_a_noop_by_default():
    server = HostServerBase()
    assert server.start() is None


def test_stop_is_a_noop_by_default():
    server = HostServerBase()
    assert server.stop() is None


# ---------------------------------------------------------------------------
# Subclass override
# ---------------------------------------------------------------------------


def test_subclass_can_override_every_method():
    calls = []

    class _FakeServer(HostServerBase):
        def set_config(self, agconfig):
            calls.append(("set_config", agconfig))

        def start(self):
            calls.append(("start",))

        def stop(self):
            calls.append(("stop",))

        def build_app(self):
            calls.append(("build_app",))
            return "app"

    server = _FakeServer()
    server.set_config("cfg")
    server.start()
    server.stop()
    assert server.build_app() == "app"
    assert calls == [("set_config", "cfg"), ("start",), ("stop",), ("build_app",)]
