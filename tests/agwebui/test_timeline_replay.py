"""Temporal regression coverage shared by trajectory and timeline views."""

from fastapi.testclient import TestClient
from agency.observability.agwebui import server
from agency.observability.agwebui.trajectory import Trajectory
from tests.agwebui.test_trajectory import event, write_rows


def receive_seek(socket, at):
    socket.send_json({"type": "seek", "clock": at})
    for _ in range(30):
        frame = socket.receive_json()
        if frame["type"] == "snapshot" and frame["replay"]["clock"] == at:
            return frame
    raise AssertionError("No seek acknowledgement")


def test_live_replay_seeks_paused_resumed_failed_and_late_events(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_run_dir", tmp_path)
    rows = [
        event(0, "workload_started", ts=100),
        event(1, "tool_call", ts=101, call_id="one", tool="read", arguments={"nested": [1, 2]}),
        event(2, "agent_paused", ts=103),
        event(3, "resource_sample", ts=104, values={"workload:cpu_pct": 30}),
        event(4, "agent_resumed", ts=106),
        event(5, "tool_result", ts=108, call_id="one", result="failed", error="failure"),
        event(6, "done", ts=110, status="failed"),
    ]
    write_rows(tmp_path / "global_data.sqlite3", rows)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/ws/trajectory?mode=replay&run=live") as ws,
    ):
        assert ws.receive_json()["run"]["actions"] == []
        paused = receive_seek(ws, 4)
        assert paused["run"]["agents"][0]["status"] == "paused"
        assert paused["run"]["actions"][0]["result"] == ""
        assert paused["run"]["counters"]["workload:cpu_pct"] == [[4, 30]]
        ended = receive_seek(ws, 10)
        assert ended["run"]["actions"][0]["outcome"] == "failed"
        before = receive_seek(ws, 2)
        assert before["run"]["actions"][0]["outcome"] == "running"
        assert before["run"]["counters"] == {}
        write_rows(
            tmp_path / "global_data.sqlite3",
            [event(7, "resource_sample", ts=101.5, values={"workload:cpu_pct": 12})],
        )
        late = receive_seek(ws, 2)
        for _ in range(10):
            if late.get("run", {}).get("counters", {}).get("workload:cpu_pct"):
                break
            late = ws.receive_json()
        assert late["run"]["counters"]["workload:cpu_pct"] == [[1.5, 12]]
        assert late["replay"]["clock"] == 2


def test_saved_seek_restores_resource_prefix_and_completed_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_run_dir", tmp_path)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/ws/trajectory?mode=replay&run=demo-baseline") as ws,
    ):
        ws.receive_json()
        full = receive_seek(ws, 142)
        assert full["replay"]["finished"]
        partial = receive_seek(ws, 20)
        assert all(p[0] <= 20 for points in partial["run"]["counters"].values() for p in points)
        assert all(not a["result"] for a in partial["run"]["actions"] if a["outcome"] == "running")
        assert len(partial["run"]["actions"]) < len(full["run"]["actions"])
        assert len(receive_seek(ws, 142)["run"]["actions"]) == len(full["run"]["actions"])


def test_missing_timestamp_and_counter_patches_are_explicit():
    model = Trajectory("test", "test", mode="replay")
    model.apply(event(0, "workload_started", ts=0))
    model.apply({**event(1, "tool_call", tool="read"), "ts": None})
    assert model.run["actions"][0]["start"] == 0
    assert "timestamp_missing" in model.run["actions"][0]["source"]
    model.apply(event(2, "resource_sample", values={"cpu_pct": 42, "bad": float("nan")}))
    assert model.patch()["counter_samples"] == {"cpu_pct": [[2, 42]]}
    assert model.patch()["counter_samples"] == {}


def test_sampler_persists_measured_samples_without_second_collection(monkeypatch):
    from agency.observability.profiler import agprof
    from types import SimpleNamespace

    sampler = agprof._Sampler(1, False)
    records = []
    monkeypatch.setattr(agprof, "_samples", [])
    monkeypatch.setattr(
        agprof,
        "_profile_data_logger",
        SimpleNamespace(record_event=lambda kind, payload: records.append((kind, payload))),
    )
    monkeypatch.setattr(sampler, "_tick_processes", lambda t: None)
    monkeypatch.setattr(sampler, "_tick_cgroups", lambda t: None)
    monkeypatch.setattr(
        sampler,
        "_tick_workload_cgroup",
        lambda t: agprof._samples.append((t, "workload:memory_mb", 64)),
    )
    sampler._tick()
    assert records[0][0] == "resource_sample"
    assert records[0][1]["values"] == {"workload:memory_mb": 64}
    assert records[0][1]["sampled_at"] > 0


def test_reconnect_preserves_time_between_events(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_run_dir", tmp_path)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/ws/trajectory?mode=replay&run=demo-baseline&clock=20.5") as ws,
    ):
        frame = ws.receive_json()
        assert frame["replay"]["clock"] == 20.5
        assert not frame["replay"]["playing"]
        assert all(a["start"] <= 20.5 for a in frame["run"]["actions"])


def test_replay_history_bounds_payloads_but_keeps_original_evidence(tmp_path):
    from agency.observability.agwebui.trajectory import LiveSource

    result = {"stdout": "x" * 250000, "exit_code": 0}
    write_rows(
        tmp_path / "global_data.sqlite3",
        [
            event(0, "workload_started"),
            event(1, "tool_call", call_id="long", tool="read"),
            event(2, "tool_result", call_id="long", result=result),
        ],
    )
    source = LiveSource(tmp_path)
    source.refresh(force=True)
    retained = source.events[-1]
    assert len(retained["payload"]["result"]) == 8000
    assert retained["payload"]["output_truncated"]
    assert source.raw_event(retained["id"])["payload"]["result"] == result
    replay = Trajectory("live", "replay", mode="replay")
    for row in source.events:
        replay.apply(row)
    assert replay.run["actions"][0]["output_truncated"]
    assert (
        replay.run["actions"][0]["result_preview"]
        == source.model.run["actions"][0]["result_preview"]
    )
