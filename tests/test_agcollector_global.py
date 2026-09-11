from __future__ import annotations

import json
import sqlite3

from agency.observability.agdatalogger import agDataLogger
from agency.agent import agent
from agency.configs.agconfig import agconfig as agconfig_cls, dataloggerconfig, orchestratorconfig
from agency.orchestrator import get_orchestrator
from agency.agteam import agteam
from agency.observability.agwebui.server import _fetch_agent_detail


def _rows(path, query, params=()):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(query, params).fetchall()
    finally:
        connection.close()


def test_webui_reads_selected_agent_database_on_demand(tmp_path):
    """_fetch_agent_detail() reads the orchestrator's real global agDataLogger
    database -- the one get_orchestrator() actually writes to."""
    global_path = tmp_path / "global_data.sqlite3"
    agent_path = tmp_path / "researcher_data.sqlite3"
    agent_logger = agDataLogger(agconfig_cls(dataloggerconfig(db_path=str(agent_path))))
    agent_logger.start()
    agent_logger.record_event(
        "agent_state", {"state": "inactive", "skill": None}, update_latest_snapshot=True
    )
    agent_logger.record_event("agent_config", {"temperature": 0.2}, update_latest_snapshot=True)
    agent_logger.record_event(
        "llm_block",
        {"role": "assistant", "content": "done"},
        flush=True,
    )
    agent_logger.stop()

    global_logger = agDataLogger(agconfig_cls(dataloggerconfig(db_path=str(global_path))))
    global_logger.start()
    global_logger.record_event(
        "agent_registered",
        {"db_path": str(agent_path), "team": "research"},
        name="researcher",
        object="agent",
        update_latest_snapshot=True,
        flush=True,
    )
    global_logger.stop()

    detail = _fetch_agent_detail(global_path, "researcher")
    assert detail["state"]["state"] == "inactive"
    assert detail["config"] == {"temperature": 0.2}
    # _compute_agent_messages() stamps every message with a best-effort `ts`
    # (real wall-clock time, not something to assert an exact literal for).
    (message,) = detail["messages"]
    assert isinstance(message.pop("ts", None), (int, float))
    assert message == {"role": "assistant", "blocks": [{"content": "done"}]}
    assert _fetch_agent_detail(global_path, "missing")["error"] == "unknown agent"


def test_team_uses_global_logger_instead_of_own_database(tmp_path):
    class EmptyTeam(agteam):
        def setup(self):
            return None

    agent.log_dir = tmp_path
    team = EmptyTeam()
    logger = team.data_logger
    logger.flush()

    event_types = {
        row["type"]
        for row in _rows(
            logger.db_path,
            "SELECT type FROM events WHERE object='agteam' AND name=?",
            (team.name,),
        )
    }
    assert {"team_created", "team_registered"} <= event_types
    assert not (tmp_path / f"{team.name}_data.sqlite3").exists()


def test_resource_pool_uses_global_logger_instead_of_own_database(tmp_path):
    path = tmp_path / "agency.sqlite3"
    orchestrator = get_orchestrator(agconfig_cls(orchestratorconfig(db_path=str(path))))
    pool = orchestrator.agresource_pool
    assert pool._data_logger is orchestrator.data_logger

    pool._emit_resource()
    orchestrator.flush()

    resource = json.loads(
        _rows(
            path,
            "SELECT payload FROM latest_values WHERE type='resource_update' AND name='resource_pool'",
        )[0]["payload"]
    )
    assert resource["cpus_total"] == pool.total_cpus
    assert resource["memory_total_mb"] == pool.total_memory_mb
    assert not (tmp_path / "resources_data.sqlite3").exists()
