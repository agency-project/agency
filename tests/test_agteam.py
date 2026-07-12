"""Tests for agteam base class."""

import pytest
from agency.agteam import agteam
from agency.agskill import agskill
from agency.agdata import agdata
from agency.agconfig import agConfig


def _llm_agconfig(d: dict) -> agConfig:
    return agConfig({"agllm_backend": dict(d)})


_ECHO_LLM = {"api_key": "k", "model": "m"}


# ---------------------------------------------------------------------------
# Minimal concrete subclass used across tests
# ---------------------------------------------------------------------------


class _EchoTeam(agteam):
    agconfig = _llm_agconfig(_ECHO_LLM)

    def setup(self):
        self.skill = agskill(name="echo", system_prompt="Echo.")
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
        {"api_key": "x", "model": ""},
        {"api_key": "y", "model": "claude-3", "base_url": "https://api.example.com"},
        {"api_key": "z", "model": "llama-3", "temperature": 0.7},
        {"api_key": "a", "model": "mistral"},
    ],
)
def test_init_agconfig_instance_override_does_not_affect_class(llm_cfg):
    cfg = _llm_agconfig(llm_cfg)
    team = _EchoTeam(agconfig=cfg)
    # team.agconfig is its own clone of cfg, not cfg itself -- see
    # docs/Design_configuration.md ("Changing a Dynamic field live").
    assert team.agconfig is not cfg
    assert team.agconfig.data.get("agllm_backend") == llm_cfg
    assert _EchoTeam.agconfig.data.get("agllm_backend") == _ECHO_LLM
    other = _EchoTeam()
    assert other.agconfig.data.get("agllm_backend") == _ECHO_LLM


def test_init_agconfig_none_falls_back_to_class_attr():
    team = _EchoTeam(agconfig=None)
    assert team.agconfig.data.get("agllm_backend") == _ECHO_LLM


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

    from agency.agsync import agsync

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
    assert team.ag.llm.backend.as_dict() == llm_cfg


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


def test_agent_agname_kwarg_accepted():
    from agency.agent import agent

    class _T(agteam):
        def setup(self):
            self.ag = agent(agname="my-custom-agent")

        def run(self):
            pass

    team = _T(agconfig=_llm_agconfig(_ECHO_LLM))
    assert team.ag.agname == "my-custom-agent_0000"


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


def test_run_returns_pending_agdata():
    result = _EchoTeam().run()
    assert isinstance(result, agdata)


def test_run_is_nonblocking():
    import time

    class _SlowTeam(agteam):
        def setup(self):
            pass

        def run(self):
            time.sleep(0.2)
            return agdata(done=True)

    t0 = time.perf_counter()
    result = _SlowTeam().run()
    assert time.perf_counter() - t0 < 0.1
    assert result.done is True  # blocks here


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


def test_parallel_run_all_results_resolve():
    teams = [_EchoTeam() for _ in range(4)]
    results = [t.run() for t in teams]
    assert all(r.done is True for r in results)


def test_parallel_run_runs_concurrently():
    import time

    class _SlowTeam(agteam):
        def setup(self):
            pass

        def run(self):
            time.sleep(0.2)
            return agdata(ok=True)

    teams = [_SlowTeam() for _ in range(4)]
    t0 = time.perf_counter()
    results = [t.run() for t in teams]
    _ = [r.ok for r in results]
    assert time.perf_counter() - t0 < 0.6  # 4×0.2s sequential = 0.8s


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_parallel_run_scales_to_n_teams(n):
    teams = [_EchoTeam() for _ in range(n)]
    results = [t.run() for t in teams]
    assert all(r.done is True for r in results)


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
    cfgs = [
        _llm_agconfig({"api_key": "a", "model": "m1"}),
        _llm_agconfig({"api_key": "b", "model": "m2"}),
        _llm_agconfig({"api_key": "c", "model": "m3"}),
    ]
    teams = [_EchoTeam(agconfig=c) for c in cfgs]
    for team, cfg in zip(teams, cfgs):
        # Each team's agconfig is its own clone, not the source cfg itself.
        assert team.agconfig is not cfg
        assert team.agconfig.data.get("agllm_backend") == cfg.data.get("agllm_backend")
    assert _EchoTeam.agconfig.data.get("agllm_backend") == _ECHO_LLM


def test_many_instances_each_have_own_agent_list():
    teams = [_EchoTeam() for _ in range(6)]
    agent_ids = [id(t.agents[0]) for t in teams]
    assert len(set(agent_ids)) == 6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_kwargs_can_shadow_non_reserved_names():
    team = _EchoTeam(name="custom_name")
    assert team.name == "custom_name"


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
    assert team.agconfig.get("agllm_backend", "temperature") == 0.2


def test_team_change_config_clones_given_agconfig():
    team = _EchoTeam()
    new_cfg = _llm_agconfig({"api_key": "k", "model": "m", "temperature": 0.2})
    team.change_config(new_cfg)
    new_cfg.agllm_backend.temperature = 0.9
    assert team.agconfig.get("agllm_backend", "temperature") == 0.2


def test_team_change_config_propagates_to_spawned_agents():
    team = _EchoTeam()
    team.change_config(_llm_agconfig({"api_key": "k", "model": "m", "temperature": 0.2}))
    assert team.agent.llm.backend.temperature == 0.2


def test_team_get_config_copy_returns_clone_not_same_object():
    team = _EchoTeam()
    copy = team.get_config_copy()
    assert copy is not team.agconfig


def test_team_get_config_copy_reflects_current_values():
    team = _EchoTeam()
    assert team.get_config_copy().agllm_backend.model == "m"


def test_mutating_team_get_config_copy_does_not_affect_team():
    team = _EchoTeam()
    copy = team.get_config_copy()
    copy.agllm_backend.temperature = 0.9
    assert team.agconfig.get("agllm_backend", "temperature") is None
