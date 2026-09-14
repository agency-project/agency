"""Regression tests for stale CPU-only PID tracking at the commit boundary."""

from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.sandbox.base import agsandbox_backend


def backend(table, rc=0):
    sb = object.__new__(agsandbox_backend)
    sb._agconfig = agconfig()
    sb._watched_pids = {42: 1.0}
    sb._baseline_pids = {1}
    sb._daemon_pids = set()
    sb._ptrace_managed_pids = set()
    sb._read_proc_table = Mock(return_value=(table, rc))
    return sb


@pytest.mark.parametrize(
    "table",
    [
        "1 0 S init\n",
        "1 0 S init\n42 1 Z short_helper\n",
        "1 0 S init\n99 1 S harness_daemon\n",
    ],
)
def test_finished_work_does_not_prevent_hibernation_or_adopt_unrelated_processes(table):
    sb = backend(table)
    sb._daemon_pids.add(99)
    assert sb._has_pending_background_work() is False
    assert sb._watched_pids == {}


def test_live_descendant_is_adopted_after_its_tracked_parent_exits():
    sb = backend("1 0 S init\n99 1 S background_child\n")
    assert sb._has_pending_background_work() is True
    assert set(sb._watched_pids) == {99}


def test_harness_exemption_does_not_hide_its_user_workload_descendants():
    from agency.sandbox.pid_diagnostics import MARKER

    stat = "50 (python) S 1 " + "0 " * 17 + "1234 0"
    sb = backend(
        f"1 0 S init\n50 1 S python\n99 50 S user_background\n{MARKER}50|harness_daemon|{stat}\n"
    )
    sb._register_harness_pid(50, 1234)
    assert sb._has_pending_background_work() is True
    assert set(sb._watched_pids) == {99}


def test_reused_harness_pid_is_not_exempted_as_infrastructure():
    from agency.sandbox.pid_diagnostics import MARKER

    stat = "50 (user_work) S 1 " + "0 " * 17 + "9999 0"
    sb = backend(f"1 0 S init\n50 1 S user_work\n{MARKER}50|unknown|{stat}\n")
    sb._register_harness_pid(50, 1234)
    assert sb._has_pending_background_work() is True
    assert 50 in sb._watched_pids


def test_live_background_work_still_prevents_hibernation():
    sb = backend("1 0 S init\n42 1 S sleep\n")
    assert sb._has_pending_background_work() is True
    assert 42 in sb._watched_pids


def test_failed_process_read_preserves_background_work():
    sb = backend("", rc=1)
    assert sb._has_pending_background_work() is True
    assert 42 in sb._watched_pids


def test_ptrace_exit_events_remain_authoritative_across_pid_namespaces():
    sb = backend("1 0 S init\n")
    sb._ptrace_managed_pids.add(42)
    assert sb._has_pending_background_work() is True


def test_empty_tracking_does_not_start_a_process_scan():
    sb = backend("1 0 S init\n")
    sb._watched_pids = {}
    assert sb._has_pending_background_work() is False
    sb._read_proc_table.assert_not_called()


def test_health_payload_is_compatible_and_reports_process_identity():
    import os
    from fastapi.testclient import TestClient
    from agency.harness.servers.harness_interaction_server import HarnessInteractionServer

    server = HarnessInteractionServer("/unused", Mock())
    with TestClient(server.build_app()) as client:
        response = client.get("/health")
    assert response.json() == {"ready": True}
    assert int(response.headers["X-Agency-Daemon-Pid"]) == os.getpid()


def test_launcher_registers_only_the_reported_harness_identity():
    from agency.engine.harness_daemon_launcher import _register_daemon

    sandbox = Mock()
    handle = Mock()
    client = Mock()
    client.daemon_identity.return_value = (50, 1234)
    from unittest.mock import MagicMock

    handle.client.return_value = MagicMock(__enter__=Mock(return_value=client))
    _register_daemon(sandbox, handle)
    sandbox._register_harness_pid.assert_called_once_with(50, 1234)
    sandbox.release_daemon.assert_not_called()
