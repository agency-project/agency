"""Tests for agteam base class."""

import pytest
from agency.agteam import agteam
from agency.agskill import agskill
from agency.agdata import agdata
from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig


def _llm_agconfig(d: dict) -> agconfig_cls:
    return agconfig_cls(llmconfig(**d))


def _llm_view(cfg: agconfig_cls, ref: dict) -> dict:
    return {k: getattr(cfg.llm, k) for k in ref}


_ECHO_LLM = {"api_key": "k", "model": "m"}


# ---------------------------------------------------------------------------
# Minimal concrete subclass used across tests
# ---------------------------------------------------------------------------


class _EchoTeam(agteam):
    agconfig = _llm_agconfig(_ECHO_LLM)

    def setup(self):
        self.skill = agskill(name="echo", prompt="Echo.")
        from agency.agent import agent

        self.agent = agent()

    def run(self):
        return agdata(done=True)


# ---------------------------------------------------------------------------
# Construction and config kwargs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,attr,value",
    [
        ({"topic": "flash attention"}, "topic", "flash attention"),
        ({"max_papers": 5}, "max_papers", 5),
        ({"output_path": "/tmp/out.md"}, "output_path", "/tmp/out.md"),
        ({"enabled": True}, "enabled", True),
        ({"tags": ["a", "b"]}, "tags", ["a", "b"]),
        ({"config": {"k": "v"}}, "config", {"k": "v"}),
    ],
)
def test_init_sets_arbitrary_config_kwargs(kwargs, attr, value):
    team = _EchoTeam(**kwargs)
    assert getattr(team, attr) == value


def test_init_multiple_kwargs_all_become_attributes():
    team = _EchoTeam(topic="x", max_papers=3, output_path="/tmp", enabled=False)
    assert team.topic == "x"
    assert team.max_papers == 3
    assert team.output_path == "/tmp"
    assert team.enabled is False


@pytest.mark.parametrize(
    "llm_cfg",
    [
        {"api_key": "x", "model": "gpt-4"},
        {"api_key": "y", "model": "claude-3", "base_url": "https://api.example.com"},
        {"api_key": "z", "model": "llama-3", "temperature": 0.7},
        {"api_key": "a", "model": "mistral"},
    ],
)
def test_init_agconfig_instance_override_does_not_affect_class(llm_cfg):
    cfg = _llm_agconfig(llm_cfg)
    team = _EchoTeam(agconfig=cfg)
    # team.agconfig is its own clone of cfg, not cfg itself.
    assert team.agconfig is not cfg
    assert _llm_view(team.agconfig, llm_cfg) == llm_cfg
    assert _llm_view(_EchoTeam.agconfig, _ECHO_LLM) == _ECHO_LLM
    other = _EchoTeam()
    assert _llm_view(other.agconfig, _ECHO_LLM) == _ECHO_LLM


def test_init_agconfig_none_falls_back_to_class_attr():
    team = _EchoTeam(agconfig=None)
    assert _llm_view(team.agconfig, _ECHO_LLM) == _ECHO_LLM


def test_init_calls_setup_before_returning():
    team = _EchoTeam()
    assert hasattr(team, "agent")
    assert hasattr(team, "skill")


def test_setup_runs_exactly_once_at_construction():
    call_count = []

    class _CountTeam(agteam):
        def setup(self):
            call_count.append(1)

        def run(self):
            pass

    _CountTeam()
    assert len(call_count) == 1


def test_setup_runs_before_run():
    calls = []

    class _OrderTeam(agteam):
        def setup(self):
            calls.append("setup")

        def run(self):
            calls.append("run")

    from agency.utils.agsync import agsync

    t = _OrderTeam()
    t.run()
    agsync(t)
    assert calls == ["setup", "run"]


def test_base_agteam_setup_is_noop():
    t = agteam.__new__(agteam)
    t._agents = __import__("weakref").WeakSet()
    t.agconfig = None
    t.setup()


# ---------------------------------------------------------------------------
# Auto-tracked agents
# ---------------------------------------------------------------------------


def test_agent_created_in_setup_is_registered():
    from agency.agent import agent

    team = _EchoTeam()
    assert isinstance(team.agent, agent)
    assert team.agent in team.agents


def test_agent_created_in_setup_registered_count():
    team = _EchoTeam()
    assert len(team.agents) == 1


@pytest.mark.parametrize(
    "llm_cfg",
    [
        {"api_key": "a", "model": "m1"},
        {"api_key": "b", "model": "m2", "base_url": "https://x.com"},
        {"api_key": "c", "model": "m3", "temperature": 0.5},
    ],
)
def test_agent_inherits_team_llm_config(llm_cfg):
    from agency.agent import agent

    class _T(agteam):
        def setup(self):
            self.ag = agent()

        def run(self):
            pass

    team = _T(agconfig=_llm_agconfig(llm_cfg))
    assert _llm_view(team.ag.agconfig, llm_cfg) == llm_cfg


def test_multiple_agents_in_setup_all_registered():
    from agency.agent import agent

    class _MultiTeam(agteam):
        def setup(self):
            self.a1 = agent()
            self.a2 = agent()
            self.a3 = agent()

        def run(self):
            pass

    team = _MultiTeam(agconfig=_llm_agconfig(_ECHO_LLM))
    assert len(team.agents) == 3
    assert team.a1 in team.agents
    assert team.a2 in team.agents
    assert team.a3 in team.agents
    assert team.a1 is not team.a2
    assert team.a2 is not team.a3


def test_agent_name_kwarg_accepted():
    from agency.agent import agent

    class _T(agteam):
        def setup(self):
            self.ag = agent(name="my-custom-agent")

        def run(self):
            pass

    team = _T(agconfig=_llm_agconfig(_ECHO_LLM))
    assert team.ag.agname == "agent_my-custom-agent_0000"


def test_agents_property_returns_copy_not_live_list():
    team = _EchoTeam()
    snapshot = team.agents
    snapshot.clear()
    assert len(team.agents) == 1


def test_agents_property_contains_all_setup_agents():
    from agency.agent import agent

    class _T(agteam):
        def setup(self):
            self.first = agent()
            self.second = agent()
            self.third = agent()

        def run(self):
            pass

    team = _T(agconfig=_llm_agconfig(_ECHO_LLM))
    agents = team.agents
    assert team.first in agents
    assert team.second in agents
    assert team.third in agents
    assert len(agents) == 3


# ---------------------------------------------------------------------------
# run() — non-blocking, returns pending agdata
# ---------------------------------------------------------------------------


def test_run_not_implemented_on_base():
    with pytest.raises(NotImplementedError):
        agteam().run()


def test_run_returns_pending_data_before_work_finishes():
    import threading

    release = threading.Event()

    class WaitingTeam(agteam):
        def run(self):
            assert release.wait(2)
            return agdata(done=True)

    result = WaitingTeam().run()
    try:
        assert isinstance(result, agdata)
        assert result.is_pending()
    finally:
        release.set()
    assert result.wait(timeout=2).done


@pytest.mark.parametrize(
    "return_val,field,expected",
    [
        (agdata(done=True), "done", True),
        (agdata(result="ok"), "result", "ok"),
        (agdata(count=3), "count", 3),
        (agdata(papers=["p1"]), "papers", ["p1"]),
        (agdata(flag=False), "flag", False),
    ],
)
def test_run_result_fields_resolve(return_val, field, expected):
    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            return return_val

    assert getattr(_T().run(), field) == expected


def test_run_wraps_non_agdata_return_in_result_field():
    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            return 99

    assert _T().run().result == 99


def test_run_exception_raises_on_field_access():
    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        _ = _T().run().anything


# ---------------------------------------------------------------------------
# Parallel fan-out — [t.run() for t in teams]
# ---------------------------------------------------------------------------


def test_parallel_runs_reach_the_same_barrier():
    import threading

    barrier = threading.Barrier(5)

    class ParallelTeam(agteam):
        def run(self):
            barrier.wait(timeout=2)
            return agdata(ok=True)

    results = [ParallelTeam().run() for _ in range(4)]
    try:
        barrier.wait(timeout=2)
        assert all(result.wait(timeout=2).ok for result in results)
    finally:
        barrier.abort()


def test_parallel_run_mixed_success_and_failure():
    class _Good(agteam):
        def setup(self):
            pass

        def run(self):
            return agdata(ok=True)

    class _Bad(agteam):
        def setup(self):
            pass

        def run(self):
            raise ValueError("fail")

    teams = [_Good(), _Bad(), _Good()]
    results = [t.run() for t in teams]
    assert results[0].ok is True
    with pytest.raises(ValueError, match="fail"):
        _ = results[1].anything
    assert results[2].ok is True


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_agents,expect_in_repr",
    [
        (0, "0"),
        (1, "1"),
        (3, "3"),
    ],
)
def test_repr_contains_class_name_and_agent_count(n_agents, expect_in_repr):
    from agency.agent import agent

    class _T(agteam):
        def setup(self):
            self._ags = [agent() for _ in range(n_agents)]

        def run(self):
            pass

    r = repr(_T(agconfig=_llm_agconfig(_ECHO_LLM)))
    assert "_T" in r
    assert expect_in_repr in r


# ---------------------------------------------------------------------------
# Instance isolation
# ---------------------------------------------------------------------------


def test_instances_have_independent_agents():
    t1 = _EchoTeam()
    t2 = _EchoTeam()
    assert t1.agent is not t2.agent
    assert t1.agents[0] is not t2.agents[0]


@pytest.mark.parametrize(
    "key,vals",
    [
        ("topic", ["A", "B", "C"]),
        ("max_papers", [1, 5, 10]),
        ("flag", [True, False, True]),
    ],
)
def test_config_kwargs_are_independent_per_instance(key, vals):
    teams = [_EchoTeam(**{key: v}) for v in vals]
    for team, expected in zip(teams, vals):
        assert getattr(team, key) == expected


def test_agconfig_overrides_are_independent_per_instance():
    llm_dicts = [
        {"api_key": "a", "model": "m1"},
        {"api_key": "b", "model": "m2"},
        {"api_key": "c", "model": "m3"},
    ]
    cfgs = [_llm_agconfig(d) for d in llm_dicts]
    teams = [_EchoTeam(agconfig=c) for c in cfgs]
    for team, cfg, d in zip(teams, cfgs, llm_dicts):
        # Each team's agconfig is its own clone, not the source cfg itself.
        assert team.agconfig is not cfg
        assert _llm_view(team.agconfig, d) == d
    assert _llm_view(_EchoTeam.agconfig, _ECHO_LLM) == _ECHO_LLM


def test_many_instances_each_have_own_agent_list():
    teams = [_EchoTeam() for _ in range(6)]
    agent_ids = [id(t.agents[0]) for t in teams]
    assert len(set(agent_ids)) == 6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_kwargs_can_shadow_non_reserved_names():
    """`name` itself is reserved now (see test_name_kwarg_seeds_team_name below)
    -- this exercises the same **config passthrough mechanism (setattr for
    every kwarg not otherwise used) via a kwarg that isn't."""
    team = _EchoTeam(custom_field="custom_value")
    assert team.custom_field == "custom_value"


def test_name_kwarg_seeds_team_name():
    """`name` is reserved: it seeds the auto-suffixed identity name (see
    agteam.__init__'s `_base = config.get("name") or ...`), not a literal
    passthrough attribute."""
    team = _EchoTeam(name="custom_name")
    assert team.name == "team_custom_name_0000"


def test_setup_exception_propagates_from_init():
    class _BrokenTeam(agteam):
        def setup(self):
            raise RuntimeError("bad setup")

        def run(self):
            pass

    with pytest.raises(RuntimeError, match="bad setup"):
        _BrokenTeam()


def test_agent_created_outside_team_requires_explicit_llm_config():
    from agency.agent import agent

    with pytest.raises(TypeError):
        agent()


# ---------------------------------------------------------------------------
# change_config / get_config_copy
# ---------------------------------------------------------------------------


def test_team_change_config_replaces_agconfig():
    team = _EchoTeam()
    team.change_config(_llm_agconfig({"api_key": "k", "model": "m", "temperature": 0.2}))
    assert team.agconfig.llm.temperature == 0.2


def test_team_change_config_clones_given_agconfig():
    team = _EchoTeam()
    new_cfg = _llm_agconfig({"api_key": "k", "model": "m", "temperature": 0.2})
    team.change_config(new_cfg)
    new_cfg.llm.temperature = 0.9
    assert team.agconfig.llm.temperature == 0.2


def test_team_change_config_propagates_to_spawned_agents():
    team = _EchoTeam()
    team.change_config(_llm_agconfig({"api_key": "k", "model": "m", "temperature": 0.2}))
    assert team.agent.agconfig.llm.temperature == 0.2


def test_team_get_config_copy_returns_clone_not_same_object():
    team = _EchoTeam()
    copy = team.get_config_copy()
    assert copy is not team.agconfig


def test_team_get_config_copy_reflects_current_values():
    team = _EchoTeam()
    assert team.get_config_copy().llm.model == "m"


# ---------------------------------------------------------------------------
# team_registered snapshot
# ---------------------------------------------------------------------------


def test_team_registered_is_refreshed_after_run_builds_its_agents(tmp_path):
    """__init__'s team_registered snapshot is taken right after setup(),
    before run() ever executes -- a team with no setup() override (agents
    built directly in run(), the common ad-hoc pattern -- see main.py-style
    scripts) always has an empty _agents set at that point. The wrapped
    run() must log team_registered again once run() completes so the
    snapshot reflects the agents run() actually created."""
    import json
    import sqlite3

    from agency.agent import agent
    from agency.utils.agsync import agsync

    class RunBuiltTeam(agteam):
        agconfig = _llm_agconfig(_ECHO_LLM)

        def run(self):
            self.worker = agent(name="run_built_worker")
            return agdata(done=True)

    agent.log_dir = tmp_path
    team = RunBuiltTeam()
    team.run()
    agsync(team)
    team.data_logger.flush()

    con = sqlite3.connect(team.data_logger.db_path)
    try:
        rows = con.execute(
            "SELECT payload FROM events WHERE object='agteam' AND name=? "
            "AND type='team_registered' ORDER BY id",
            (team.name,),
        ).fetchall()
    finally:
        con.close()

    payloads = [json.loads(row[0]) for row in rows]
    assert payloads[0]["agents"] == []
    assert payloads[-1]["agents"] == [team.worker.agname]


def test_mutating_team_get_config_copy_does_not_affect_team():
    team = _EchoTeam()
    copy = team.get_config_copy()
    copy.llm.temperature = 0.9
    assert team.agconfig.llm.temperature is None
