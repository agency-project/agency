"""Tests for agteam base class."""
import pytest
from agency.agteam import agteam
from agency.agskill import agskill
from agency.agdata import agdata, AgError


# ---------------------------------------------------------------------------
# Minimal concrete subclass used across tests
# ---------------------------------------------------------------------------

class _EchoTeam(agteam):
    llm_config = {"api_key": "k", "model": "m"}

    def setup(self):
        self.skill = agskill(name="echo", system_prompt="Echo.")
        self.agent = self.make_agent([self.skill])

    def run(self):
        return agdata(done=True)


# ---------------------------------------------------------------------------
# Construction and config kwargs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,attr,value", [
    ({"topic": "flash attention"},       "topic",       "flash attention"),
    ({"max_papers": 5},                  "max_papers",  5),
    ({"output_path": "/tmp/out.md"},     "output_path", "/tmp/out.md"),
    ({"enabled": True},                  "enabled",     True),
    ({"tags": ["a", "b"]},              "tags",        ["a", "b"]),
    ({"config": {"k": "v"}},            "config",      {"k": "v"}),
])
def test_init_sets_arbitrary_config_kwargs(kwargs, attr, value):
    team = _EchoTeam(**kwargs)
    assert getattr(team, attr) == value


def test_init_multiple_kwargs_all_become_attributes():
    team = _EchoTeam(topic="x", max_papers=3, output_path="/tmp", enabled=False)
    assert team.topic == "x"
    assert team.max_papers == 3
    assert team.output_path == "/tmp"
    assert team.enabled is False


@pytest.mark.parametrize("llm_cfg", [
    {"api_key": "x", "model": "gpt-4o"},
    {"api_key": "y", "model": "claude-3", "base_url": "https://api.example.com"},
    {"api_key": "z", "model": "llama-3", "temperature": 0.7},
    {"api_key": "a", "model": "mistral"},
    {},
])
def test_init_llm_config_instance_override_does_not_affect_class(llm_cfg):
    team = _EchoTeam(llm_config=llm_cfg)
    assert team.llm_config is llm_cfg
    # Class attribute unchanged for all other instances
    assert _EchoTeam.llm_config == {"api_key": "k", "model": "m"}
    other = _EchoTeam()
    assert other.llm_config == {"api_key": "k", "model": "m"}


def test_init_llm_config_none_falls_back_to_class_attr():
    team = _EchoTeam(llm_config=None)
    assert team.llm_config == {"api_key": "k", "model": "m"}


def test_init_calls_setup_before_returning():
    # setup() creates self.agent; if it wasn't called the attribute wouldn't exist
    team = _EchoTeam()
    assert hasattr(team, "agent")
    assert hasattr(team, "skill")


def test_setup_runs_exactly_once_at_construction():
    call_count = []

    class _CountTeam(agteam):
        def setup(self): call_count.append(1)
        def run(self): pass

    _CountTeam()
    assert len(call_count) == 1


def test_setup_runs_before_run():
    calls = []

    class _OrderTeam(agteam):
        def setup(self): calls.append("setup")
        def run(self): calls.append("run")

    t = _OrderTeam()
    t.run()
    assert calls == ["setup", "run"]


def test_base_agteam_setup_is_noop():
    # Base agteam.setup() does nothing — must not raise
    t = agteam.__new__(agteam)
    t._agents = []
    t.llm_config = {}
    t.setup()   # should not raise


# ---------------------------------------------------------------------------
# make_agent
# ---------------------------------------------------------------------------

def test_make_agent_returns_agent_instance():
    from agency.agent import agent
    team = _EchoTeam()
    assert isinstance(team.agent, agent)


def test_make_agent_registers_in_agents_list():
    team = _EchoTeam()
    assert team.agent in team.agents
    assert len(team.agents) == 1


@pytest.mark.parametrize("llm_cfg", [
    {"api_key": "a", "model": "m1"},
    {"api_key": "b", "model": "m2", "base_url": "https://x.com"},
    {"api_key": "c", "model": "m3", "temperature": 0.5},
])
def test_make_agent_uses_team_llm_config(llm_cfg):
    class _T(agteam):
        def setup(self):
            self.agent = self.make_agent([agskill(name="s", system_prompt="")])
        def run(self): pass

    team = _T(llm_config=llm_cfg)
    assert team.agent.llm_config == llm_cfg


def test_make_agent_multiple_agents_all_registered():
    class _MultiTeam(agteam):
        def setup(self):
            s = agskill(name="s", system_prompt="")
            self.a1 = self.make_agent([s])
            self.a2 = self.make_agent([s])
            self.a3 = self.make_agent([s])
        def run(self): pass

    team = _MultiTeam()
    assert len(team.agents) == 3
    assert team.a1 in team.agents
    assert team.a2 in team.agents
    assert team.a3 in team.agents
    # All distinct objects
    assert team.a1 is not team.a2
    assert team.a2 is not team.a3


def test_make_agent_agname_kwarg_forwarded():
    class _T(agteam):
        def setup(self):
            self.agent = self.make_agent(
                [agskill(name="s", system_prompt="")],
                agname="my-custom-agent",
            )
        def run(self): pass

    team = _T(llm_config={"api_key": "k", "model": "m"})
    assert team.agent.agname == "my-custom-agent"


def test_agents_property_returns_copy_not_live_list():
    team = _EchoTeam()
    snapshot = team.agents
    snapshot.clear()
    assert len(team.agents) == 1  # internal list unmodified


def test_agents_property_ordering_matches_make_agent_calls():
    class _T(agteam):
        def setup(self):
            s = agskill(name="s", system_prompt="")
            self.first  = self.make_agent([s])
            self.second = self.make_agent([s])
            self.third  = self.make_agent([s])
        def run(self): pass

    team = _T()
    agents = team.agents
    assert agents[0] is team.first
    assert agents[1] is team.second
    assert agents[2] is team.third


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_not_implemented_on_base():
    with pytest.raises(NotImplementedError) as exc_info:
        agteam().run()
    assert "run" in str(exc_info.value).lower() or "NotImplementedError" in type(exc_info.value).__name__


@pytest.mark.parametrize("return_val", [
    agdata(done=True),
    agdata(result="ok", count=3),
    agdata(papers=["p1", "p2"]),
    None,
    42,
    "hello",
])
def test_run_can_return_any_value(return_val):
    class _T(agteam):
        def setup(self): pass
        def run(self): return return_val

    assert _T().run() == return_val


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_agents,expect_in_repr", [
    (0, "0"),
    (1, "1"),
    (3, "3"),
])
def test_repr_contains_class_name_and_agent_count(n_agents, expect_in_repr):
    class _T(agteam):
        def setup(self):
            s = agskill(name="s", system_prompt="")
            for _ in range(n_agents):
                self.make_agent([s])
        def run(self): pass

    r = repr(_T())
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


@pytest.mark.parametrize("key,vals", [
    ("topic",      ["A",  "B",  "C"]),
    ("max_papers", [1,    5,    10]),
    ("flag",       [True, False, True]),
])
def test_config_kwargs_are_independent_per_instance(key, vals):
    teams = [_EchoTeam(**{key: v}) for v in vals]
    for team, expected in zip(teams, vals):
        assert getattr(team, key) == expected


def test_llm_config_overrides_are_independent_per_instance():
    cfgs = [
        {"api_key": "a", "model": "m1"},
        {"api_key": "b", "model": "m2"},
        {"api_key": "c", "model": "m3"},
    ]
    teams = [_EchoTeam(llm_config=c) for c in cfgs]
    for team, cfg in zip(teams, cfgs):
        assert team.llm_config is cfg
    # Class attr still unchanged
    assert _EchoTeam.llm_config == {"api_key": "k", "model": "m"}


def test_many_instances_each_have_own_agent_list():
    teams = [_EchoTeam() for _ in range(6)]
    agent_ids = [id(t.agents[0]) for t in teams]
    assert len(set(agent_ids)) == 6  # all distinct


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_kwargs_can_shadow_non_reserved_names():
    # Passing a kwarg with any name sets it as an attribute
    team = _EchoTeam(name="custom_name")
    assert team.name == "custom_name"


def test_setup_exception_propagates_from_init():
    class _BrokenTeam(agteam):
        def setup(self): raise RuntimeError("bad setup")
        def run(self): pass

    with pytest.raises(RuntimeError, match="bad setup"):
        _BrokenTeam()


def test_run_exception_propagates_to_caller():
    class _FailTeam(agteam):
        def setup(self): pass
        def run(self): raise ValueError("run failed")

    with pytest.raises(ValueError, match="run failed"):
        _FailTeam().run()
