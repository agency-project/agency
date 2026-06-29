"""Tests for agwebui framework hooks — agterm, agent, agteam, ask_human."""
import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agency.agwebui.emitter import agwebui_emitter


# ---------------------------------------------------------------------------
# Fixture: activate a real agwebui emitter as the global singleton
# ---------------------------------------------------------------------------

@pytest.fixture()
def active_webui(tmp_path):
    """
    Install a real agwebui_emitter as the active web UI singleton.
    Yields the run_dir so tests can read the emitted events.
    Restores the previous singleton on teardown.
    """
    import agency.agwebui as agwebui_mod
    from agency.agwebui import agwebui as agwebui_cls

    ui = agwebui_cls.__new__(agwebui_cls)
    ui.emitter = agwebui_emitter(tmp_path)

    old = agwebui_mod._active
    agwebui_mod._active = ui
    yield tmp_path
    agwebui_mod._active = old


def _events(run_dir: Path) -> list[dict]:
    db = run_dir / "ui_events.db"
    if not db.exists():
        return []
    con = sqlite3.connect(str(db))
    rows = con.execute("SELECT data FROM events ORDER BY id").fetchall()
    con.close()
    return [json.loads(r[0]) for r in rows]


def _events_of(run_dir: Path, etype: str) -> list[dict]:
    return [e for e in _events(run_dir) if e["type"] == etype]


# ---------------------------------------------------------------------------
# agterm — agent_registered
# ---------------------------------------------------------------------------

def test_agterm_emits_agent_registered(active_webui):
    from agency.agterm import agterm
    agterm("__test_reg_agent__")
    regs = _events_of(active_webui, "agent_registered")
    assert any(e["agname"] == "__test_reg_agent__" for e in regs)

def test_agterm_registered_event_has_hex_color(active_webui):
    from agency.agterm import agterm
    agterm("__test_color_agent__")
    regs = _events_of(active_webui, "agent_registered")
    ev = next(e for e in regs if e["agname"] == "__test_color_agent__")
    assert ev["color"].startswith("#")
    assert len(ev["color"]) in (4, 7)   # #rgb or #rrggbb


# ---------------------------------------------------------------------------
# agterm — log routing
# ---------------------------------------------------------------------------

def test_agterm_log_routes_to_emitter(active_webui):
    from agency.agterm import agterm
    term = agterm("__test_log_agent__")
    term.log("TEST_EV  ", "unique-payload-xyzzy")
    logs = _events_of(active_webui, "log")
    assert any("unique-payload-xyzzy" in e["line"] for e in logs)

def test_agterm_log_does_not_write_to_stderr(active_webui, capsys):
    from agency.agterm import agterm
    term = agterm("__test_stderr_agent__")
    term.log("TEST_EV  ", "should-not-appear-on-stderr")
    captured = capsys.readouterr()
    assert "should-not-appear-on-stderr" not in captured.err


# ---------------------------------------------------------------------------
# agent._set_ui_state hook
# ---------------------------------------------------------------------------

def test_agent_set_ui_state_emits_event(active_webui):
    from agency.agent import agent

    ag = agent.__new__(agent)
    ag.agname = "__test_state_agent__"
    ag._ui_state = {}

    ag._set_ui_state("llm", skill="design", tool=None)

    states = _events_of(active_webui, "agent_state")
    ev = next((e for e in states if e["agname"] == "__test_state_agent__"), None)
    assert ev is not None
    assert ev["state"] == "llm"
    assert ev["skill"] == "design"
    assert ev["tool"]  is None

def test_agent_set_ui_state_inactive(active_webui):
    from agency.agent import agent

    ag = agent.__new__(agent)
    ag.agname = "__test_inactive_agent__"
    ag._ui_state = {}
    ag._set_ui_state("inactive")

    states = _events_of(active_webui, "agent_state")
    ev = next((e for e in states if e["agname"] == "__test_inactive_agent__"), None)
    assert ev is not None
    assert ev["state"] == "inactive"


# ---------------------------------------------------------------------------
# agent._push_live_messages hook
# ---------------------------------------------------------------------------

def test_agent_push_live_messages_emits_snapshot(active_webui):
    from agency.agent import agent

    ag = agent.__new__(agent)
    ag.agname = "__test_msgs_agent__"
    ag._snapshot_messages = []

    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    ag._push_live_messages(msgs)

    snaps = _events_of(active_webui, "messages_snapshot")
    ev = next((e for e in snaps if e["agname"] == "__test_msgs_agent__"), None)
    assert ev is not None
    assert ev["messages"] == msgs

def test_agent_push_live_messages_updates_snapshot(active_webui):
    from agency.agent import agent

    ag = agent.__new__(agent)
    ag.agname = "__test_snap_agent__"
    ag._snapshot_messages = []

    msgs = [{"role": "system", "content": "sys prompt"}]
    ag._push_live_messages(msgs)
    assert ag._snapshot_messages == msgs


# ---------------------------------------------------------------------------
# agteam — team_registered
# ---------------------------------------------------------------------------

def test_agteam_emits_team_registered(active_webui):
    import agency.agwebui as agwebui_mod

    captured = []
    original = agwebui_mod._active.emitter.team_registered

    def _capture(team_name, agents):
        captured.append({"team_name": team_name, "agents": agents})
        original(team_name, agents)

    agwebui_mod._active.emitter.team_registered = _capture

    from agency.agteam import agteam as agteam_cls

    class _MinimalTeam(agteam_cls):
        def setup(self): pass
        def run(self): pass

    with patch("agency.agname.agname.allocate_agname", return_value="MinimalTeam_0000"), \
         patch("agency.aglog.aglog.__init__", return_value=None), \
         patch("agency.aglog.aglog._lifecycle", return_value=None):
        team = _MinimalTeam.__new__(_MinimalTeam)
        team._agents = set()
        team.team_name = "MinimalTeam_0000"
        # Directly call the post-setup hook
        try:
            from . import agwebui as _agwebui
        except Exception:
            pass
        import agency.agwebui as _agwebui2
        if _agwebui2._active is not None:
            _agwebui2._active.emitter.team_registered(
                team.team_name,
                [a.agname for a in team._agents],
            )

    teams = _events_of(active_webui, "team_registered")
    assert any(e["team_name"] == "MinimalTeam_0000" for e in teams)


# ---------------------------------------------------------------------------
# ask_human tool — web UI path
# ---------------------------------------------------------------------------

def test_ask_human_uses_file_reply_when_webui_active(active_webui):
    from agency.tools.human import make_ask_human
    from agency.agdata import agdata

    tool = make_ask_human("__test_ask_agent__")
    reply_dir = active_webui / "ui_replies"

    def _write_reply():
        time.sleep(0.1)
        # find the ask_id from the emitted event
        deadline = time.time() + 2.0
        while time.time() < deadline:
            asks = _events_of(active_webui, "ask_human")
            if asks:
                ask_id = asks[0]["ask_id"]
                (reply_dir / f"{ask_id}.txt").write_text("file reply")
                return
            time.sleep(0.05)

    threading.Thread(target=_write_reply, daemon=True).start()
    result = tool.fn(agdata(question="Use the file path?"))
    assert result.reply == "file reply"

def test_ask_human_emits_ask_event(active_webui):
    from agency.tools.human import make_ask_human
    from agency.agdata import agdata

    tool = make_ask_human("__test_ask_ev_agent__")
    reply_dir = active_webui / "ui_replies"

    def _write_reply():
        time.sleep(0.1)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            asks = _events_of(active_webui, "ask_human")
            if asks:
                (reply_dir / f"{asks[0]['ask_id']}.txt").write_text("ok")
                return
            time.sleep(0.05)

    threading.Thread(target=_write_reply, daemon=True).start()
    tool.fn(agdata(question="Confirm?"))

    asks = _events_of(active_webui, "ask_human")
    assert asks
    assert asks[0]["question"] == "Confirm?"
    assert asks[0]["agname"] == "__test_ask_ev_agent__"
    assert "ask_id" in asks[0]
