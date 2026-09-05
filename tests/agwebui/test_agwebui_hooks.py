"""Tests for agwebui's own command-dispatch hooks (webui -> agent direction:
pause/resume/update_config commands applied to real agent objects)."""

import json
import threading
import time


# ---------------------------------------------------------------------------
# agwebui command dispatch — pause/resume/pause_all/resume_all
# ---------------------------------------------------------------------------


def _make_agent():
    from agency.agent import agent
    from agency.configs.agconfig import agconfig

    return agent(agconfig=agconfig(provider="mock", api_key="k", model=""))


def test_dispatch_pause_command_pauses_named_agent():
    from agency.observability.agwebui import _dispatch_command

    ag = _make_agent()
    _dispatch_command({"type": "pause", "agname": ag.agname})
    assert ag.is_suspended() is True


def test_dispatch_resume_command_resumes_named_agent():
    from agency.observability.agwebui import _dispatch_command

    ag = _make_agent()
    ag.suspend()
    _dispatch_command({"type": "resume", "agname": ag.agname})
    assert ag.is_suspended() is False


def test_dispatch_pause_command_ignores_unknown_agname():
    from agency.observability.agwebui import _dispatch_command

    ag = _make_agent()
    _dispatch_command({"type": "pause", "agname": "__no_such_agent__"})
    assert ag.is_suspended() is False


def test_dispatch_pause_all_pauses_every_live_agent():
    from agency.observability.agwebui import _dispatch_command

    a, b = _make_agent(), _make_agent()
    _dispatch_command({"type": "pause_all"})
    assert a.is_suspended() is True
    assert b.is_suspended() is True


def test_dispatch_resume_all_resumes_every_live_agent():
    from agency.observability.agwebui import _dispatch_command

    a, b = _make_agent(), _make_agent()
    a.suspend()
    b.suspend()
    _dispatch_command({"type": "resume_all"})
    assert a.is_suspended() is False
    assert b.is_suspended() is False


def test_dispatch_update_config_applies_to_named_agent():
    from agency.observability.agwebui import _dispatch_command

    ag = _make_agent()
    _dispatch_command(
        {
            "type": "update_config",
            "agname": ag.agname,
            "config": {"react_max_steps": 7},
        }
    )
    assert ag.agconfig.react_max_steps == 7


def test_dispatch_update_config_ignores_unknown_agname():
    from agency.observability.agwebui import _dispatch_command

    ag = _make_agent()
    before = ag.agconfig.react_max_steps
    _dispatch_command(
        {
            "type": "update_config",
            "agname": "__no_such_agent__",
            "config": {"react_max_steps": 999},
        }
    )
    assert ag.agconfig.react_max_steps == before


def test_dispatch_update_config_all_applies_to_every_agent():
    from agency.observability.agwebui import _dispatch_command

    a, b = _make_agent(), _make_agent()
    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"react_max_steps": 11},
        }
    )
    assert a.agconfig.react_max_steps == 11
    assert b.agconfig.react_max_steps == 11


def test_dispatch_update_config_preserves_sandbox_mounts():
    """Webui editor payloads are safe_snapshot() only. Replacing the whole
    agconfig would drop agSandbox.mounts; forks after that bake HF weights
    into lifecycle images instead of using the shared host cache bind."""
    from agency.observability.agwebui import _dispatch_command
    from agency.agent import agent
    from agency.configs.agconfig import agconfig

    cfg = agconfig(provider="mock", api_key="k", model="", base_url="http://old")
    cfg.add_mount("hf_cache", "/tmp/hf-cache", "/root/.cache/huggingface")
    ag = agent(agconfig=cfg)

    _dispatch_command(
        {
            "type": "update_config",
            "agname": ag.agname,
            "config": {"base_url": "http://new"},
        }
    )

    assert ag.agconfig.base_url == "http://new"
    mounts = ag.agconfig.mounts or {}
    assert "hf_cache" in mounts
    assert mounts["hf_cache"][1] == "/root/.cache/huggingface"


def test_dispatch_update_config_all_preserves_sandbox_mounts():
    import copy

    from agency.observability.agwebui import _all_agteam_subclasses, _dispatch_command
    from agency.agent import agent
    from agency.configs.agconfig import agconfig
    from agency.agteam import agteam

    cfg = agconfig(provider="mock", api_key="k", model="", base_url="http://old")
    cfg.add_mount("hf_cache", "/tmp/hf-cache", "/root/.cache/huggingface")

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
        team_cls: copy.deepcopy(team_cls.agconfig)
        for team_cls in _all_agteam_subclasses(agteam)
        if team_cls.agconfig is not None
    }
    try:
        team = _MountPreserveTeam()
        assert "hf_cache" in (team.ag.agconfig.mounts or {})

        _dispatch_command(
            {
                "type": "update_config_all",
                "config": {"base_url": "http://new"},
            }
        )

        assert team.ag.agconfig.base_url == "http://new"
        assert "hf_cache" in (team.ag.agconfig.mounts or {})
        assert "hf_cache" in (team.agconfig.mounts or {})
        assert "hf_cache" in (_MountPreserveTeam.agconfig.mounts or {})
    finally:
        for team_cls, saved in saved_class_configs.items():
            team_cls.agconfig = saved


def test_dispatch_update_config_all_mutates_default_agconfig():
    """A bare agent() with no team context falls back to agent.default_agconfig
    -- update_config_all must mutate it in place so a future such agent
    clones fresh data, not just push into agents that already exist."""
    from agency.observability.agwebui import _dispatch_command
    from agency.agent import agent
    from agency.configs.agconfig import agconfig

    saved = agent.default_agconfig
    try:
        agent.default_agconfig = agconfig(provider="mock", api_key="k", model="")
        _dispatch_command(
            {
                "type": "update_config_all",
                "config": {"react_max_steps": 123},
            }
        )
        assert agent.default_agconfig.react_max_steps == 123
    finally:
        agent.default_agconfig = saved


def test_dispatch_update_config_all_mutates_team_class_attr_for_future_construction():
    """A team class's own agconfig class attribute (e.g. a user script's
    `agconfig = LLM_CONFIG`) must be reached via __subclasses__() and mutated
    in place, so a team constructed AFTER the update clones fresh data --
    not just teams/agents that already exist."""
    from agency.observability.agwebui import _dispatch_command
    from agency.agteam import agteam
    from agency.configs.agconfig import agconfig as agconfig_cls

    class _CfgAllTeamA(agteam):
        agconfig = agconfig_cls(provider="mock", api_key="k", model="")

        def setup(self):
            pass

        def run(self):
            pass

    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"react_max_steps": 77},
        }
    )
    assert _CfgAllTeamA.agconfig.react_max_steps == 77

    # Constructed AFTER the update -- clones the now-updated class attribute.
    team = _CfgAllTeamA()
    assert team.agconfig.react_max_steps == 77


def test_dispatch_update_config_all_updates_live_team_and_cascades_to_its_agents():
    """A team instance that already exists (already cloned its own agconfig
    at construction) must be reached directly, and that update must cascade
    to every agent the team already tracks."""
    from agency.observability.agwebui import _dispatch_command
    from agency.agteam import agteam
    from agency.configs.agconfig import agconfig as agconfig_cls
    from agency.agent import agent as agent_cls

    class _CfgAllTeamB(agteam):
        agconfig = agconfig_cls(provider="mock", api_key="k", model="")

        def setup(self):
            self.ag = agent_cls()

        def run(self):
            pass

    team = _CfgAllTeamB()  # constructed before the update -- already cloned

    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"react_max_steps": 55},
        }
    )

    assert team.agconfig.react_max_steps == 55
    assert team.ag.agconfig.react_max_steps == 55


def test_dispatch_update_config_all_reaches_grandchild_team_class():
    """_all_agteam_subclasses() must recurse -- a team class that subclasses
    another team class (not agteam directly) still has to be reached, since
    __subclasses__() alone only returns direct subclasses."""
    from agency.observability.agwebui import _dispatch_command
    from agency.agteam import agteam
    from agency.configs.agconfig import agconfig as agconfig_cls

    class _CfgAllTeamMid(agteam):
        agconfig = agconfig_cls(provider="mock", api_key="k", model="")

        def setup(self):
            pass

        def run(self):
            pass

    class _CfgAllTeamGrandchild(_CfgAllTeamMid):
        agconfig = agconfig_cls(provider="mock", api_key="k", model="")

        def setup(self):
            pass

        def run(self):
            pass

    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"react_max_steps": 88},
        }
    )

    assert _CfgAllTeamMid.agconfig.react_max_steps == 88
    assert _CfgAllTeamGrandchild.agconfig.react_max_steps == 88


def test_dispatch_update_config_all_skips_team_class_with_no_agconfig():
    """A team subclass that never overrides agconfig (still None, inherited
    from the agteam base) must be safely skipped -- not crash, and not
    somehow acquire a config of its own."""
    from agency.observability.agwebui import _dispatch_command
    from agency.agteam import agteam

    class _CfgAllTeamNoConfig(agteam):
        def setup(self):
            pass

        def run(self):
            pass

    _dispatch_command(
        {
            "type": "update_config_all",
            "config": {"react_max_steps": 99},
        }
    )  # must not raise

    assert _CfgAllTeamNoConfig.agconfig is None


def test_poll_commands_applies_and_deletes_command_files(tmp_path):
    from agency.observability.agwebui import _poll_commands

    ag = _make_agent()
    cmd_dir = tmp_path / "ui_commands"
    (cmd_dir).mkdir()
    (cmd_dir / "c1.json").write_text(json.dumps({"type": "pause", "agname": ag.agname}))

    stop = threading.Event()
    t = threading.Thread(target=_poll_commands, args=(cmd_dir, stop), daemon=True)
    t.start()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and not ag.is_suspended():
            time.sleep(0.02)
        assert ag.is_suspended() is True
        # Wait for the unlink on its own deadline rather than asserting it
        # immediately: _poll_commands() dispatches first and unlinks after
        # (in its `finally`), so the pause landing above says nothing about
        # whether the file is gone yet -- on a loaded host the poll thread
        # can be descheduled in exactly that window.
        while time.time() < deadline and list(cmd_dir.glob("*.json")):
            time.sleep(0.02)
        assert not list(cmd_dir.glob("*.json"))  # consumed
    finally:
        stop.set()
        t.join(timeout=1.0)
