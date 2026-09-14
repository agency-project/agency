import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig, sandboxconfig
from agency.sandbox import pid_diagnostics as diag


def test_stat_parser_preserves_names_with_spaces_and_parentheses():
    stat = "42 (a ) b) S 1 " + "0 " * 17 + "1234 0"
    assert diag.parse_stat(stat) == {"comm": "a ) b", "state": "S", "ppid": 1, "start_ticks": 1234}


@pytest.mark.parametrize(
    "current,expected",
    [
        (None, "absent_at_decision_probe"),
        ({"start_ticks": 2, "state": "S"}, "pid_reused"),
        ({"start_ticks": 1, "state": "Z"}, "zombie"),
        ({"start_ticks": 1, "state": "S", "harness_identity": "harness_daemon"}, "infrastructure"),
        ({"start_ticks": 1, "state": "S"}, "live_process_identity_verified"),
    ],
)
def test_classification_uses_identity_and_liveness(current, expected):
    s = {
        "watched": {42: 1},
        "processes": {"42": current} if current else {},
        "registration_history": {42: [{"identity": {"start_ticks": 1}}]},
        "baseline_pids": [],
        "daemon_pids": [],
        "ptrace_managed_pids": [],
        "probe_status": "observed",
    }
    assert diag.classify_watched(s)[0]["classification"] == expected


def test_probe_does_not_prune_watched_pids_or_launch_a_stopped_container():
    backend = SimpleNamespace(
        _watched_pids={42: 1},
        _baseline_pids={1},
        _daemon_pids=set(),
        _ptrace_managed_pids=set(),
        _runtime="docker",
        _name="test",
        _run=Mock(return_value=SimpleNamespace(stdout=json.dumps({"processes": {}}))),
    )
    original = copy.deepcopy(backend._watched_pids)
    result = diag.decision_snapshot(backend, True)
    assert result["pending_background_work"] is True
    assert result["watched_evidence"][0]["classification"] == "absent_at_decision_probe"
    assert backend._watched_pids == original
    assert backend._run.call_args.args[0][:3] == ["docker", "exec", "test"]


def test_failed_probe_does_not_claim_pids_are_absent():
    backend = SimpleNamespace(
        _watched_pids={42: 1},
        _baseline_pids=set(),
        _daemon_pids=set(),
        _ptrace_managed_pids=set(),
        _runtime="docker",
        _name="test",
        _run=Mock(side_effect=TimeoutError),
    )
    result = diag.decision_snapshot(backend, True)
    assert result["watched_evidence"][0]["classification"] == "unresolved"
    assert backend._watched_pids == {42: 1}


def test_exec_retains_original_pid_tracking_and_records_registration_identity():
    from agency.sandbox.base import agsandbox_backend

    backend = object.__new__(agsandbox_backend)
    backend._agconfig = agconfig(sandboxconfig(hibernation_diagnostics=True))
    backend._gpu_count_requested = 0
    backend._gpu_ids = []
    backend._watched_pids = {}
    stat = "42 (short helper) S 1 " + "0 " * 17 + "1234 0"
    backend._exec_with_pid_tracking = Mock(
        return_value=(f"hello\n{diag.MARKER}42|unknown|{stat}\n__BGPIDS__:42", 0)
    )
    output, rc = backend.exec("irrelevant")
    assert (output, rc) == ("hello", 0)
    assert 42 in backend._watched_pids
    entry = backend._pid_registration_history[42][0]
    assert entry["source"] == "exec_proc_diff"
    assert entry["identity"]["start_ticks"] == 1234


def test_engine_preserves_skip_even_when_probe_finds_stale_pid(monkeypatch):
    from agency.engine.engine import AgentEngine
    from agency.agdata import agdata

    cfg = agconfig(sandboxconfig(hibernation_diagnostics=True))
    agent = SimpleNamespace(agconfig=cfg, agname="test", data_logger=Mock())
    engine = AgentEngine(agent)
    engine._execute_harness = Mock(return_value=agdata())
    engine._commit_pending_session_update = Mock()
    sandbox = SimpleNamespace(
        _lock=threading.RLock(),
        _backend=object(),
        commit=Mock(),
        _has_pending_background_work=Mock(return_value=True),
        stop=Mock(),
        rm_container=Mock(),
    )
    monkeypatch.setattr(
        diag,
        "decision_snapshot",
        lambda backend, pending: {
            "decision_epoch": 1,
            "watched_evidence": [{"classification": "absent_at_decision_probe"}],
        },
    )
    engine.execute(None, None, None, None, sandbox)
    sandbox.stop.assert_not_called()
    sandbox.rm_container.assert_not_called()
    assert (
        agent.data_logger.record_event.call_args.kwargs["payload"]["outcome"]
        == "skipped_pending_work"
    )
