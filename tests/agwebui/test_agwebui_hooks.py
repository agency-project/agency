"""Tests for agwebui framework hooks — agterm, agent, agteam, ask_human."""

import json
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import patch

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
    assert len(ev["color"]) in (4, 7)  # #rgb or #rrggbb


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

    from agency.agent import agent_state

    ag = agent.__new__(agent)
    ag.agname = "__test_state_agent__"
    ag._state = agent_state(ag.agname)

    ag._set_ui_state("llm", skill="design", tool=None)

    states = _events_of(active_webui, "agent_state")
    ev = next((e for e in states if e["agname"] == "__test_state_agent__"), None)
    assert ev is not None
    assert ev["state"] == "llm"
    assert ev["skill"] == "design"
    assert ev["tool"] is None


def test_agent_set_ui_state_inactive(active_webui):
    from agency.agent import agent

    from agency.agent import agent_state

    ag = agent.__new__(agent)
    ag.agname = "__test_inactive_agent__"
    ag._state = agent_state(ag.agname)
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
# agent._emit_config — pushes dynamic_snapshot() to the webui
# ---------------------------------------------------------------------------


def test_agent_construction_emits_config(active_webui):
    from agency.agent import agent
    from agency.agconfig import agConfig

    ag = agent(
        agconfig=agConfig(
            {"agllm_backend": {"api_key": "k", "model": ""}, "agskill": {"react_max_steps": 3}}
        )
    )

    configs = _events_of(active_webui, "agent_config")
    ev = next((e for e in configs if e["agname"] == ag.agname), None)
    assert ev is not None
    assert ev["config"]["agskill"]["react_max_steps"] == 3


def test_change_config_re_emits_config(active_webui):
    from agency.agent import agent
    from agency.agconfig import agConfig

    ag = agent(agconfig=agConfig({"agllm_backend": {"api_key": "k", "model": ""}}))
    ag.change_config(agConfig({"agskill": {"react_max_steps": 42}}))

    configs = [e for e in _events_of(active_webui, "agent_config") if e["agname"] == ag.agname]
    assert configs[-1]["config"]["agskill"]["react_max_steps"] == 42


# ---------------------------------------------------------------------------
# agwebui command dispatch — pause/resume/pause_all/resume_all
# ---------------------------------------------------------------------------


def _make_agent():
    from agency.agent import agent
    from agency.agconfig import agConfig

    return agent(agconfig=agConfig({"agllm_backend": {"api_key": "k", "model": ""}}))


def test_dispatch_pause_command_pauses_named_agent():
    from agency.agwebui import _dispatch_command

    ag = _make_agent()
    _dispatch_command({"type": "pause", "agname": ag.agname})
    assert not ag._state.run_allowed.is_set()


def test_dispatch_resume_command_resumes_named_agent():
    from agency.agwebui import _dispatch_command

    ag = _make_agent()
    ag.pause()
    _dispatch_command({"type": "resume", "agname": ag.agname})
    assert ag._state.run_allowed.is_set()


def test_dispatch_pause_command_ignores_unknown_agname():
    from agency.agwebui import _dispatch_command

    ag = _make_agent()
    _dispatch_command({"type": "pause", "agname": "__no_such_agent__"})
    assert ag._state.run_allowed.is_set()  # untouched


def test_dispatch_pause_all_pauses_every_live_agent():
    from agency.agwebui import _dispatch_command

    a, b = _make_agent(), _make_agent()
    _dispatch_command({"type": "pause_all"})
    assert not a._state.run_allowed.is_set()
    assert not b._state.run_allowed.is_set()


def test_dispatch_resume_all_resumes_every_live_agent():
    from agency.agwebui import _dispatch_command

    a, b = _make_agent(), _make_agent()
    a.pause()
    b.pause()
    _dispatch_command({"type": "resume_all"})
    assert a._state.run_allowed.is_set()
    assert b._state.run_allowed.is_set()


def test_dispatch_update_config_applies_to_named_agent():
    from agency.agwebui import _dispatch_command

    ag = _make_agent()
    _dispatch_command(
        {
            "type": "update_config",
            "agname": ag.agname,
            "config": {"agskill": {"react_max_steps": 7}},
        }
    )
    assert ag.agconfig.get("agskill", "react_max_steps") == 7


def test_dispatch_update_config_ignores_unknown_agname():
    from agency.agwebui import _dispatch_command

    ag = _make_agent()
    before = ag.agconfig.get("agskill", "react_max_steps")
    _dispatch_command(
        {
            "type": "update_config",
            "agname": "__no_such_agent__",
            "config": {"agskill": {"react_max_steps": 999}},
        }
    )
    assert ag.agconfig.get("agskill", "react_max_steps") == before


def test_dispatch_update_config_all_applies_to_every_agent():
    from agency.agwebui import _dispatch_command

    a, b = _make_agent(), _make_agent()
    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"agskill": {"react_max_steps": 11}},
        }
    )
    assert a.agconfig.get("agskill", "react_max_steps") == 11
    assert b.agconfig.get("agskill", "react_max_steps") == 11


def test_dispatch_update_config_preserves_sandbox_mounts():
    """Webui editor payloads are dynamic_snapshot() only. Replacing the whole
    agconfig would drop agSandbox.mounts; forks after that bake HF weights
    into lifecycle images instead of using the shared host cache bind."""
    from agency.agwebui import _dispatch_command
    from agency.agent import agent
    from agency.agconfig import agConfig
    from agency.agsandbox import agSandboxConfig

    cfg = agConfig({"agllm_backend": {"api_key": "k", "model": "", "base_url": "http://old"}})
    agSandboxConfig(cfg).add_mount("hf_cache", "/tmp/hf-cache", "/root/.cache/huggingface")
    ag = agent(agconfig=cfg)

    _dispatch_command(
        {
            "type": "update_config",
            "agname": ag.agname,
            "config": {"agllm_backend": {"base_url": "http://new"}},
        }
    )

    assert ag.agconfig.get("agllm_backend", "base_url") == "http://new"
    mounts = ag.agconfig.get("agSandbox", "mounts") or {}
    assert "hf_cache" in mounts
    assert mounts["hf_cache"][1] == "/root/.cache/huggingface"


def test_dispatch_update_config_all_preserves_sandbox_mounts():
    import copy

    from agency.agwebui import _all_agteam_subclasses, _dispatch_command
    from agency.agent import agent
    from agency.agconfig import agConfig
    from agency.agsandbox import agSandboxConfig
    from agency.agteam import agteam

    cfg = agConfig({"agllm_backend": {"api_key": "k", "model": "", "base_url": "http://old"}})
    agSandboxConfig(cfg).add_mount("hf_cache", "/tmp/hf-cache", "/root/.cache/huggingface")

    class _MountPreserveTeam(agteam):
        agconfig = cfg

        def setup(self):
            self.ag = agent()

        def run(self):
            pass

    # update_config_all merges into every agteam subclass's class-level
    # agconfig in place -- snapshot/restore so we don't leak base_url into
    # unrelated suites (e.g. test_agteam's _EchoTeam).
    saved_class_configs = {
        team_cls: copy.deepcopy(team_cls.agconfig.data)
        for team_cls in _all_agteam_subclasses(agteam)
        if team_cls.agconfig is not None
    }
    try:
        team = _MountPreserveTeam()
        assert "hf_cache" in (team.ag.agconfig.get("agSandbox", "mounts") or {})

        _dispatch_command(
            {
                "type": "update_config_all",
                "config": {"agllm_backend": {"base_url": "http://new"}},
            }
        )

        assert team.ag.agconfig.get("agllm_backend", "base_url") == "http://new"
        assert "hf_cache" in (team.ag.agconfig.get("agSandbox", "mounts") or {})
        assert "hf_cache" in (team.agconfig.get("agSandbox", "mounts") or {})
        assert "hf_cache" in (_MountPreserveTeam.agconfig.get("agSandbox", "mounts") or {})
    finally:
        for team_cls, data in saved_class_configs.items():
            team_cls.agconfig.data.clear()
            for owner, fields in data.items():
                team_cls.agconfig.data[owner] = copy.deepcopy(fields)


def test_dispatch_update_config_all_mutates_default_agconfig():
    """A bare agent() with no team context falls back to agent.default_agconfig
    -- update_config_all must mutate it in place so a future such agent
    clones fresh data, not just push into agents that already exist."""
    from agency.agwebui import _dispatch_command
    from agency.agent import agent
    from agency.agconfig import agConfig

    saved = agent.default_agconfig
    try:
        agent.default_agconfig = agConfig({"agllm_backend": {"api_key": "k", "model": ""}})
        _dispatch_command(
            {
                "type": "update_config_all",
                "config": {"agskill": {"react_max_steps": 123}},
            }
        )
        assert agent.default_agconfig.get("agskill", "react_max_steps") == 123
    finally:
        agent.default_agconfig = saved


def test_dispatch_update_config_all_mutates_team_class_attr_for_future_construction():
    """A team class's own agconfig class attribute (e.g. a user script's
    `agconfig = LLM_CONFIG`) must be reached via __subclasses__() and mutated
    in place, so a team constructed AFTER the update clones fresh data --
    not just teams/agents that already exist."""
    from agency.agwebui import _dispatch_command
    from agency.agteam import agteam
    from agency.agconfig import agConfig

    class _CfgAllTeamA(agteam):
        agconfig = agConfig({"agllm_backend": {"api_key": "k", "model": ""}})

        def setup(self):
            pass

        def run(self):
            pass

    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"agskill": {"react_max_steps": 77}},
        }
    )
    assert _CfgAllTeamA.agconfig.get("agskill", "react_max_steps") == 77

    # Constructed AFTER the update -- clones the now-updated class attribute.
    team = _CfgAllTeamA()
    assert team.agconfig.get("agskill", "react_max_steps") == 77


def test_dispatch_update_config_all_updates_live_team_and_cascades_to_its_agents():
    """A team instance that already exists (already cloned its own agconfig
    at construction) must be reached directly, and that update must cascade
    to every agent the team already tracks."""
    from agency.agwebui import _dispatch_command
    from agency.agteam import agteam
    from agency.agconfig import agConfig
    from agency.agent import agent as agent_cls

    class _CfgAllTeamB(agteam):
        agconfig = agConfig({"agllm_backend": {"api_key": "k", "model": ""}})

        def setup(self):
            self.ag = agent_cls()

        def run(self):
            pass

    team = _CfgAllTeamB()  # constructed before the update -- already cloned

    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"agskill": {"react_max_steps": 55}},
        }
    )

    assert team.agconfig.get("agskill", "react_max_steps") == 55
    assert team.ag.agconfig.get("agskill", "react_max_steps") == 55


def test_dispatch_update_config_all_reaches_grandchild_team_class():
    """_all_agteam_subclasses() must recurse -- a team class that subclasses
    another team class (not agteam directly) still has to be reached, since
    __subclasses__() alone only returns direct subclasses."""
    from agency.agwebui import _dispatch_command
    from agency.agteam import agteam
    from agency.agconfig import agConfig

    class _CfgAllTeamMid(agteam):
        agconfig = agConfig({"agllm_backend": {"api_key": "k", "model": ""}})

        def setup(self):
            pass

        def run(self):
            pass

    class _CfgAllTeamGrandchild(_CfgAllTeamMid):
        agconfig = agConfig({"agllm_backend": {"api_key": "k", "model": ""}})

        def setup(self):
            pass

        def run(self):
            pass

    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"agskill": {"react_max_steps": 88}},
        }
    )

    assert _CfgAllTeamMid.agconfig.get("agskill", "react_max_steps") == 88
    assert _CfgAllTeamGrandchild.agconfig.get("agskill", "react_max_steps") == 88


def test_dispatch_update_config_all_skips_team_class_with_no_agconfig():
    """A team subclass that never overrides agconfig (still None, inherited
    from the agteam base) must be safely skipped -- not crash, and not
    somehow acquire a config of its own."""
    from agency.agwebui import _dispatch_command
    from agency.agteam import agteam

    class _CfgAllTeamNoConfig(agteam):
        def setup(self):
            pass

        def run(self):
            pass

    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"agskill": {"react_max_steps": 99}},
        }
    )  # must not raise

    assert _CfgAllTeamNoConfig.agconfig is None


def test_poll_commands_applies_and_deletes_command_files(tmp_path):
    from agency.agwebui import _poll_commands

    ag = _make_agent()
    cmd_dir = tmp_path / "ui_commands"
    (cmd_dir).mkdir()
    (cmd_dir / "c1.json").write_text(json.dumps({"type": "pause", "agname": ag.agname}))

    stop = threading.Event()
    t = threading.Thread(target=_poll_commands, args=(cmd_dir, stop), daemon=True)
    t.start()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and ag._state.run_allowed.is_set():
            time.sleep(0.02)
        assert not ag._state.run_allowed.is_set()
        assert not list(cmd_dir.glob("*.json"))  # consumed
    finally:
        stop.set()
        t.join(timeout=1.0)


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
        def setup(self):
            pass

        def run(self):
            pass

    with (
        patch("agency.agname.agname.allocate_agname", return_value="MinimalTeam_0000"),
        patch("agency.aglog.aglog.__init__", return_value=None),
        patch("agency.aglog.aglog._lifecycle", return_value=None),
    ):
        team = _MinimalTeam.__new__(_MinimalTeam)
        team._agents = set()
        team.team_name = "MinimalTeam_0000"
        # Directly call the post-setup hook
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
