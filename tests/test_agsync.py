"""Tests for agsync — barrier synchronisation over agents and teams."""

import time
import pytest
from agency.agsync import agsync
from agency.agteam import agteam
from agency.agdata import agdata
from agency.agent import agent
from agency._context import _active_team
from agency.agconfig import agConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

LLM_CFG = {"api_key": "k", "model": "m"}
LLM_AGCONFIG = agConfig({"agllm_backend": LLM_CFG})


class _SimpleTeam(agteam):
    """One agent, run() returns immediately."""

    agconfig = LLM_AGCONFIG

    def setup(self):
        self.ag = agent()

    def run(self):
        return agdata(done=True)


def _slow_team(delay: float = 0.15):
    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            time.sleep(delay)
            return agdata(done=True)

    return _T()


# ---------------------------------------------------------------------------
# Input normalisation — single objects and lists
# ---------------------------------------------------------------------------


def test_agsync_single_agent_does_not_raise():
    agsync(_SimpleTeam().ag)


def test_agsync_single_team_does_not_raise():
    agsync(_SimpleTeam())


def test_agsync_empty_list_is_noop():
    agsync([])


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_agsync_list_of_agents(n):
    teams = [_SimpleTeam() for _ in range(n)]
    agsync([t.ag for t in teams])


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_agsync_list_of_teams(n):
    agsync([_SimpleTeam() for _ in range(n)])


def test_agsync_mixed_agents_and_teams():
    t1 = _SimpleTeam()
    t2 = _SimpleTeam()
    t3 = _SimpleTeam()
    agsync([t1.ag, t2, t3.ag, t1])


def test_agsync_duplicate_entries_are_safe():
    team = _SimpleTeam()
    # Same agent or team listed multiple times — must not raise or deadlock
    agsync([team, team, team.ag, team.ag])


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [42, "string", 3.14, b"bytes", object(), None])
def test_agsync_raises_on_wrong_scalar_type(bad):
    with pytest.raises(TypeError, match="agsync"):
        agsync(bad)


@pytest.mark.parametrize("bad", [42, "string", 3.14, None])
def test_agsync_raises_on_wrong_type_inside_list(bad):
    team = _SimpleTeam()
    with pytest.raises(TypeError, match="agsync"):
        agsync([team, bad])


def test_agsync_raises_on_nested_list():
    team = _SimpleTeam()
    with pytest.raises(TypeError):
        agsync([[team]])


# ---------------------------------------------------------------------------
# Multi-agent teams
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_agents", [1, 2, 3, 5])
def test_agsync_resolves_all_agents_in_team(n_agents):
    class _T(agteam):
        agconfig = LLM_AGCONFIG

        def setup(self):
            self.ags = [agent() for _ in range(n_agents)]

        def run(self):
            pass

    agsync(_T())  # must not raise or deadlock


# ---------------------------------------------------------------------------
# Barrier semantics — tasks must be finished before agsync returns
# ---------------------------------------------------------------------------


def test_agsync_waits_for_submitted_team():
    team = _slow_team(0.15)
    pending = team.run()
    assert pending.is_pending()
    agsync(team)
    assert not pending.is_pending()


def test_agsync_waits_for_multiple_submitted_teams():
    teams = [_slow_team(0.15) for _ in range(4)]
    pending = [t.run() for t in teams]
    agsync(teams)
    for p in pending:
        assert not p.is_pending()


def test_agsync_is_a_real_barrier_not_early_return():
    """agsync must not return before the slowest team finishes."""
    finished = []

    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            time.sleep(0.15)
            finished.append(1)
            return agdata(done=True)

    teams = [_T() for _ in range(3)]
    [t.run() for t in teams]
    agsync(teams)
    assert len(finished) == 3


def test_agsync_already_finished_team_returns_immediately():
    team = _slow_team(0.0)
    team.run()
    agsync(team)  # fully resolved already
    t0 = time.perf_counter()
    agsync(team)
    assert time.perf_counter() - t0 < 0.1


def test_agsync_team_never_run_returns_immediately():
    team = _SimpleTeam()  # run() never called
    t0 = time.perf_counter()
    agsync(team)
    assert time.perf_counter() - t0 < 0.1


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


def test_agsync_reraises_exception_from_submitted_team():
    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            raise ValueError("team exploded")

    team = _T()
    team.run()
    with pytest.raises(ValueError, match="team exploded"):
        agsync(team)


def test_agsync_reraises_single_exception_directly():
    """One failing team → original exception type raised, not ExceptionGroup."""

    class _Good(agteam):
        def setup(self):
            pass

        def run(self):
            return agdata(ok=True)

    class _Bad(agteam):
        def setup(self):
            pass

        def run(self):
            raise RuntimeError("bad team")

    teams = [_Good(), _Bad(), _Good()]
    [t.run() for t in teams]
    with pytest.raises(RuntimeError, match="bad team"):
        agsync(teams)


def test_agsync_raises_exception_group_for_multiple_failures():
    """Multiple failing teams → ExceptionGroup containing all their exceptions."""

    class _Bad(agteam):
        def setup(self):
            pass

        def run(self):
            raise ValueError("failed")

    teams = [_Bad(), _Bad(), _Bad()]
    [t.run() for t in teams]
    with pytest.raises(ExceptionGroup) as exc_info:
        agsync(teams)
    assert len(exc_info.value.exceptions) == 3
    assert all(isinstance(e, ValueError) for e in exc_info.value.exceptions)


def test_agsync_joins_all_teams_before_raising():
    """A fast-failing team must not cause slow teams to be abandoned."""
    finished = []

    class _Fast(agteam):
        def setup(self):
            pass

        def run(self):
            raise RuntimeError("fast failure")

    class _Slow(agteam):
        def setup(self):
            pass

        def run(self):
            time.sleep(0.15)
            finished.append(1)
            return agdata(ok=True)

    teams = [_Fast(), _Slow(), _Slow()]
    [t.run() for t in teams]
    with pytest.raises(RuntimeError):
        agsync(teams)
    # Both slow teams must have completed despite the fast failure.
    assert len(finished) == 2


def test_agsync_does_not_raise_for_idle_team_that_had_no_error():
    team = _SimpleTeam()
    team.run()
    agsync(team)  # no exception


# ---------------------------------------------------------------------------
# Dynamic fork-agent tracking
# ---------------------------------------------------------------------------


def test_fork_agents_created_in_run_are_auto_registered():
    class _T(agteam):
        agconfig = LLM_AGCONFIG

        def setup(self):
            self.parent = agent()

        def run(self):
            self.f1 = agent.fork(self.parent)
            self.f2 = agent.fork(self.parent)
            self.f3 = agent.fork(self.parent)
            return agdata(done=True)

    team = _T()
    n_before = len(team.agents)
    team.run()
    agsync(team)
    assert len(team.agents) == n_before + 3
    assert team.f1 in team.agents
    assert team.f2 in team.agents
    assert team.f3 in team.agents


def test_agents_in_setup_are_tracked():
    """All agents created in setup() are auto-tracked — setup runs inside the team context."""

    class _T(agteam):
        agconfig = LLM_AGCONFIG

        def setup(self):
            self.parent = agent()
            self.fork = agent.fork(self.parent)  # fork in setup — also tracked

        def run(self):
            return agdata(done=True)

    team = _T()
    assert team.parent in team.agents
    assert team.fork in team.agents
    assert len(team.agents) == 2


def test_fork_agents_tracked_in_run_thread():
    """Auto-registration must work in the background thread used by run()."""

    class _T(agteam):
        agconfig = LLM_AGCONFIG

        def setup(self):
            self.parent = agent()

        def run(self):
            self.fork = agent.fork(self.parent)
            return agdata(done=True)

    team = _T()
    team.run()
    agsync(team)
    assert team.fork in team.agents


@pytest.mark.parametrize("n_forks", [1, 2, 5, 8])
def test_fork_agents_scale_correctly(n_forks):
    class _T(agteam):
        agconfig = LLM_AGCONFIG

        def setup(self):
            self.parent = agent()

        def run(self):
            self.forks = [agent.fork(self.parent) for _ in range(n_forks)]
            return agdata(done=True)

    team = _T()
    n_before = len(team.agents)
    team.run()
    agsync(team)
    assert len(team.agents) == n_before + n_forks


def test_no_duplicate_registration_on_multiple_run_calls():
    """An agent created in setup() must not appear twice after run() is called."""

    class _T(agteam):
        agconfig = LLM_AGCONFIG

        def setup(self):
            self.ag = agent()

        def run(self):
            return agdata(done=True)

    team = _T()
    team.run()
    agsync(team)
    assert team.agents.count(team.ag) == 1


# ---------------------------------------------------------------------------
# Context var correctness
# ---------------------------------------------------------------------------


def test_context_var_is_set_inside_run():
    captured = []

    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            captured.append(_active_team.get(None))
            return agdata(ok=True)

    team = _T()
    team.run()
    agsync(team)
    assert captured == [team]


def test_context_var_cleared_after_run_returns():
    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            return agdata(ok=True)

    t = _T()
    t.run()
    agsync(t)
    assert _active_team.get(None) is None


def test_context_var_cleared_after_run_raises():
    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            raise RuntimeError("boom")

    t = _T()
    t.run()
    try:
        agsync(t)
    except RuntimeError:
        pass
    assert _active_team.get(None) is None


def test_context_var_set_to_correct_team_for_each_instance():
    """Each team instance sees itself as _active_team, not another instance."""
    seen = {}

    class _T(agteam):
        def setup(self):
            pass

        def run(self):
            seen[id(self)] = _active_team.get(None)
            return agdata(ok=True)

    t1, t2, t3 = _T(), _T(), _T()
    t1.run()
    t2.run()
    t3.run()
    agsync([t1, t2, t3])
    assert seen[id(t1)] is t1
    assert seen[id(t2)] is t2
    assert seen[id(t3)] is t3


def test_context_var_restored_after_nested_teams():
    """Nested run() calls from inside run() restore context correctly."""
    outer_saw_inner = []

    class _Inner(agteam):
        def setup(self):
            pass

        def run(self):
            return agdata(ok=True)

    class _Outer(agteam):
        def setup(self):
            pass

        def run(self):
            before = _active_team.get(None)
            inner = _Inner()
            inner.run()
            agsync(inner)
            after = _active_team.get(None)
            outer_saw_inner.append((before is self, after is self))
            return agdata(ok=True)

    outer = _Outer()
    outer.run()
    agsync(outer)
    assert outer_saw_inner == [(True, True)]
