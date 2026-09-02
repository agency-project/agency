from __future__ import annotations

import json
import sqlite3
import threading
import time
from types import SimpleNamespace

from agency.agdatacollector import agDataCollector, agDataCollectorConfigs
from agency.agcollector import GlobalDataCollector
from agency.agent import agent
from agency.agconfig import agConfig
from agency.orchestrator import get_orchestrator
from agency.agteam import agteam
from agency.agwebui.server import _fetch_agent_detail


def _rows(path, query, params=()):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(query, params).fetchall()
    finally:
        connection.close()


def test_global_collector_persists_scoped_events_spans_and_latest_values(tmp_path):
    path = tmp_path / "agency.sqlite3"
    collector = GlobalDataCollector(path, flush_batch_size=100, flush_interval_s=60)
    team = collector.scoped(
        source="team", scope_key="research", attributes={"team": "research"}
    )

    team.record_event("team_registered", {"agents": ["a"]}, overwrite=True)
    team.record_span("team:run", 10.0, 12.0, {"outcome": "success"})
    collector.flush(timeout_s=2)

    event = _rows(path, "SELECT * FROM lifecycle_events")[0]
    assert event["source"] == "team"
    assert json.loads(event["payload"]) == {"team": "research", "agents": ["a"]}
    span = _rows(path, "SELECT * FROM spans")[0]
    assert span["source"] == "team"
    assert span["name"] == "team:run"
    latest = _rows(
        path,
        "SELECT scope_key,payload FROM latest_values WHERE type='team_registered'",
    )[0]
    assert latest["scope_key"] == "research"
    collector.shutdown(timeout_s=2)


def test_global_schema_keeps_only_global_projections(tmp_path):
    path = tmp_path / "agency.sqlite3"
    collector = GlobalDataCollector(path, flush_interval_s=60)
    collector.record_event(
        "agent_registered",
        {"db_path": "/tmp/a.sqlite3", "team": "research"},
        source="catalog",
        agname="a",
        overwrite=True,
    )
    collector.record_event(
        "team_registered",
        {"team_name": "research", "agents": ["a"]},
        source="team",
        scope_key="research",
        overwrite=True,
    )
    collector.record_event(
        "resource_update",
        {"cpus_acquired": 1, "cpus_total": 8},
        source="resources",
        scope_key="resource_pool",
        overwrite=True,
    )
    collector.flush(timeout_s=2)

    table_names = {
        row["name"]
        for row in _rows(path, "SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"agent_registry", "team_registry", "resource_state"} <= table_names
    assert {
        "agent_tokens",
        "agent_messages",
        "agent_state",
        "agent_config_state",
    }.isdisjoint(table_names)
    catalog = _rows(path, "SELECT agname,db_path FROM agent_registry")[0]
    assert tuple(catalog) == ("a", "/tmp/a.sqlite3")
    assert len(_rows(path, "SELECT * FROM team_registry")) == 1
    assert len(_rows(path, "SELECT * FROM resource_state")) == 1
    collector.shutdown(timeout_s=2)


def test_concurrent_publishers_receive_one_global_order(tmp_path):
    path = tmp_path / "agency.sqlite3"
    collector = GlobalDataCollector(path, flush_batch_size=17, flush_interval_s=60)

    def publish(worker: int):
        for index in range(25):
            collector.record_event("sample", {"worker": worker, "index": index})

    threads = [threading.Thread(target=publish, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    collector.flush(timeout_s=2)

    rows = _rows(path, "SELECT sequence FROM lifecycle_events ORDER BY id")
    sequences = [row["sequence"] for row in rows]
    assert sequences == list(range(1, 201))
    collector.shutdown(timeout_s=2)


def test_batch_and_interval_flush_in_background(tmp_path):
    batch_path = tmp_path / "batch.sqlite3"
    batch = GlobalDataCollector(batch_path, flush_batch_size=2, flush_interval_s=60)
    batch.record_event("sample", {"n": 1})
    batch.record_event("sample", {"n": 2})

    deadline = time.time() + 2
    while time.time() < deadline:
        if len(_rows(batch_path, "SELECT * FROM lifecycle_events")) == 2:
            break
        time.sleep(0.01)
    assert len(_rows(batch_path, "SELECT * FROM lifecycle_events")) == 2
    batch.shutdown(timeout_s=2)

    interval_path = tmp_path / "interval.sqlite3"
    interval = GlobalDataCollector(
        interval_path, flush_batch_size=100, flush_interval_s=0.02
    )
    interval.record_event("sample", {"n": 1})
    deadline = time.time() + 2
    while time.time() < deadline:
        if _rows(interval_path, "SELECT * FROM lifecycle_events"):
            break
        time.sleep(0.01)
    assert len(_rows(interval_path, "SELECT * FROM lifecycle_events")) == 1
    interval.shutdown(timeout_s=2)


def test_payload_is_copied_and_metadata_cannot_be_overridden(tmp_path):
    path = tmp_path / "agency.sqlite3"
    collector = GlobalDataCollector(path, flush_interval_s=60)
    payload = {"nested": {"value": 1}, "type": "spoofed", "source": "spoofed"}
    collector.record_event("scheduler_state", payload, source="orchestrator", flush=True)
    payload["nested"]["value"] = 2

    lifecycle = _rows(path, "SELECT type,source,payload FROM lifecycle_events")[0]
    transport = json.loads(_rows(path, "SELECT data FROM events")[0]["data"])
    assert lifecycle["type"] == "scheduler_state"
    assert lifecycle["source"] == "orchestrator"
    assert json.loads(lifecycle["payload"])["nested"]["value"] == 1
    assert transport["type"] == "scheduler_state"
    assert transport["source"] == "orchestrator"
    collector.shutdown(timeout_s=2)


def test_subscriber_and_database_failures_do_not_break_publication(tmp_path):
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("x", encoding="utf-8")
    collector = GlobalDataCollector(blocked_parent / "agency.sqlite3")

    def fail(_event):
        raise RuntimeError("subscriber broke")

    collector.subscribe(fail)
    sequence = collector.record_event("sample", {"ok": True})
    collector.flush(timeout_s=2)
    snapshot = collector.snapshot()
    assert sequence == 1
    assert snapshot["persistence_error"]
    assert "subscriber broke" in snapshot["telemetry_error"]
    collector.shutdown(timeout_s=2)


def test_webui_reads_selected_agent_database_on_demand(tmp_path):
    global_path = tmp_path / "agency.sqlite3"
    agent_path = tmp_path / "researcher_data.sqlite3"
    agent_collector = agDataCollector(
        SimpleNamespace(
            agDataCollectorConfigs=agDataCollectorConfigs(db_path=str(agent_path))
        )
    )
    agent_collector.start()
    agent_collector.record_event(
        "agent_state", {"state": "inactive", "skill": None}, overwrite=True
    )
    agent_collector.record_event(
        "agent_config", {"temperature": 0.2}, overwrite=True
    )
    agent_collector.record_event(
        "live_messages", {"messages": [{"role": "assistant", "content": "done"}]},
        overwrite=True, flush=True,
    )

    global_collector = GlobalDataCollector(global_path, flush_interval_s=60)
    global_collector.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": "research"},
        source="catalog",
        agname="researcher",
        overwrite=True,
        flush=True,
    )

    detail = _fetch_agent_detail(global_path, "researcher")
    assert detail["state"]["state"] == "inactive"
    assert detail["config"] == {"temperature": 0.2}
    assert detail["messages"] == [{"role": "assistant", "content": "done"}]
    assert _fetch_agent_detail(global_path, "missing")["error"] == "unknown agent"

    global_collector.shutdown(timeout_s=2)
    agent_collector.stop()


def test_team_uses_global_collector_instead_of_own_database(tmp_path):
    class EmptyTeam(agteam):
        def setup(self):
            return None

    agent.log_dir = tmp_path
    team = EmptyTeam()
    collector = team.data_collector._collector
    collector.flush(timeout_s=2)

    event_types = {
        row["type"]
        for row in _rows(
            collector.db_path,
            "SELECT type FROM lifecycle_events WHERE source='team'",
        )
    }
    assert {"team_created", "team_registered"} <= event_types
    assert not (tmp_path / f"{team.team_name}_data.sqlite3").exists()


def test_resource_pool_uses_global_collector_instead_of_own_database(tmp_path):
    path = tmp_path / "agency.sqlite3"
    orchestrator = get_orchestrator(
        agConfig({"agorchestrator": {"db_path": str(path)}})
    )
    pool = orchestrator.agresource_pool
    assert pool._data_collector._collector is orchestrator.data_collector

    pool._emit_resource()
    orchestrator.flush(timeout_s=2)

    resource = json.loads(
        _rows(path, "SELECT data FROM resource_state WHERE id=1")[0]["data"]
    )
    assert resource["cpus_total"] == pool.total_cpus
    assert resource["memory_total_mb"] == pool.total_memory_mb
    assert not (tmp_path / "resources_data.sqlite3").exists()
