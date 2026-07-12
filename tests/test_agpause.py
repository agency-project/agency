"""Tests for pause/resume execution control (agpause) — including deadlock-safety
of wait_all_paused() across agents blocked on each other's futures at various
points in the ReAct loop / dependency chain, and randomized fuzz coverage."""

import random
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agency import agpause
from agency.agdata import agdata
from agency.agskill import agskill
from agency.agent import agent
from agency.agconfig import agConfig


def _llm_agconfig(d: dict) -> agConfig:
    return agConfig({"agllm_backend": dict(d)})


def make_agent() -> agent:
    return agent(agconfig=_llm_agconfig({"api_key": "k", "model": ""}))


# ---------------------------------------------------------------------------
# agpause primitives — tag_producer / producer_of / note_blocked_on
# ---------------------------------------------------------------------------


def test_producer_of_returns_none_when_untagged():
    from concurrent.futures import Future

    assert agpause.producer_of(Future()) is None


def test_tag_producer_and_producer_of_roundtrip():
    from concurrent.futures import Future

    f = Future()
    ag = make_agent()
    agpause.tag_producer(f, ag)
    assert agpause.producer_of(f) is ag


def test_producer_of_returns_none_after_producer_gc():
    """The tag is a weakref -- it must not keep the producer agent alive."""
    from concurrent.futures import Future
    import gc

    f = Future()

    def _make_and_tag():
        ag = make_agent()
        agpause.tag_producer(f, ag)

    _make_and_tag()
    gc.collect()
    assert agpause.producer_of(f) is None


def test_note_blocked_on_is_noop_without_worker_thread():
    """No thread-local worker set -> null context, no state mutation."""
    ag = make_agent()
    with agpause.note_blocked_on(ag):
        pass  # must not raise
    assert ag._state.blocked_on is None


def test_note_blocked_on_is_noop_for_same_agent():
    """An agent waiting on its own future needs no extra tagging."""
    ag = make_agent()
    agpause.set_current_worker_agent(ag)
    try:
        with agpause.note_blocked_on(ag):
            assert ag._state.state != "blocked_on_dependency"
    finally:
        agpause.set_current_worker_agent(None)


def test_note_blocked_on_tags_and_restores_state():
    waiter = make_agent()
    producer = make_agent()
    waiter._set_ui_state("skill", skill="s")
    agpause.set_current_worker_agent(waiter)
    try:
        with agpause.note_blocked_on(producer):
            assert waiter._state.state == "blocked_on_dependency"
            assert waiter._state.blocked_on is producer
        assert waiter._state.state == "skill"
        assert waiter._state.blocked_on is None
    finally:
        agpause.set_current_worker_agent(None)


# ---------------------------------------------------------------------------
# agent.pause() / resume() / is_paused()
# ---------------------------------------------------------------------------


def test_pause_clears_run_allowed():
    ag = make_agent()
    ag.pause()
    assert not ag._state.run_allowed.is_set()


def test_resume_sets_run_allowed():
    ag = make_agent()
    ag.pause()
    ag.resume()
    assert ag._state.run_allowed.is_set()


def test_is_paused_false_until_checkpoint_actually_hit():
    ag = make_agent()
    ag.pause()
    assert ag.is_paused() is False  # requested, not yet honored


def test_check_pause_noop_when_not_paused():
    ag = make_agent()
    ag._check_pause("s")  # must return immediately, no hang
    assert ag._state.state != "paused"


def test_check_pause_blocks_until_resumed():
    ag = make_agent()
    ag.pause()
    reached = threading.Event()

    def worker():
        ag._check_pause("myskill")
        reached.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert ag._state.paused_ack.wait(timeout=1.0)
    assert ag.is_paused() is True
    assert not reached.is_set()  # still blocked

    ag.resume()
    assert reached.wait(timeout=1.0)
    t.join(timeout=1.0)
    assert ag.is_paused() is False


# ---------------------------------------------------------------------------
# agent.is_settled() — recursive settle-check
# ---------------------------------------------------------------------------


class _FakeAgent:
    is_settled = agent.is_settled

    def __init__(self, name, state="skill", blocked_on=None, paused=False):
        self.agname = name
        self._state = SimpleNamespace(
            state=state, blocked_on=blocked_on, paused_ack=threading.Event()
        )
        if paused:
            self._state.paused_ack.set()
        self.terminal = MagicMock()


@pytest.mark.parametrize("state", ["inactive", "finished", "error", "paused"])
def test_is_settled_true_for_leaf_states(state):
    assert _FakeAgent("a", state=state).is_settled() is True


def test_is_settled_false_while_actively_running():
    assert _FakeAgent("a", state="skill").is_settled() is False


def test_is_settled_false_while_pausing_but_not_yet_paused():
    assert _FakeAgent("a", state="pausing").is_settled() is False


def test_is_settled_recurses_through_paused_upstream():
    upstream = _FakeAgent("up", state="paused")
    downstream = _FakeAgent("down", state="blocked_on_dependency", blocked_on=upstream)
    assert downstream.is_settled() is True


def test_is_settled_false_when_upstream_still_running():
    upstream = _FakeAgent("up", state="skill")
    downstream = _FakeAgent("down", state="blocked_on_dependency", blocked_on=upstream)
    assert downstream.is_settled() is False


def test_is_settled_multi_hop_chain():
    a = _FakeAgent("a", state="paused")
    b = _FakeAgent("b", state="blocked_on_dependency", blocked_on=a)
    c = _FakeAgent("c", state="blocked_on_dependency", blocked_on=b)
    assert c.is_settled() is True


def test_is_settled_cycle_guard_does_not_recurse_forever():
    """A blocked_on cycle must resolve (not infinite-loop) via the seen-set guard."""
    x = _FakeAgent("x", state="blocked_on_dependency")
    y = _FakeAgent("y", state="blocked_on_dependency")
    x._state.blocked_on = y
    y._state.blocked_on = x
    # Must return promptly (bounded recursion), not hang.
    assert x.is_settled() is True
    assert y.is_settled() is True


def test_is_settled_self_loop_guard():
    x = _FakeAgent("x", state="blocked_on_dependency")
    x._state.blocked_on = x
    assert x.is_settled() is True


def test_is_settled_blocked_with_no_producer_is_not_settled():
    """Defensive case: blocked_on_dependency state but no producer recorded."""
    x = _FakeAgent("x", state="blocked_on_dependency", blocked_on=None)
    assert x.is_settled() is False


# ---------------------------------------------------------------------------
# wait_all_paused / wait_all_resumed — API-level behavior
# ---------------------------------------------------------------------------


def test_wait_all_paused_empty_list_returns_immediately():
    assert agpause.wait_all_paused([], timeout=1) == []


def test_wait_all_paused_already_settled_returns_fast():
    agents = [_FakeAgent(f"a{i}", state="paused") for i in range(5)]
    t0 = time.monotonic()
    assert agpause.wait_all_paused(agents, timeout=5, poll_interval=0.01) == []
    assert time.monotonic() - t0 < 1.0


def test_wait_all_paused_times_out_with_pending_names():
    running = _FakeAgent("stuck", state="skill")
    pending = agpause.wait_all_paused(
        [running], timeout=0.15, poll_interval=0.02, warn_after_s=None
    )
    assert pending == ["stuck"]


def test_wait_all_resumed_empty_list_returns_immediately():
    assert agpause.wait_all_resumed([], timeout=1) == []


def test_wait_all_resumed_times_out_while_paused():
    a = _FakeAgent("a", paused=True)
    assert agpause.wait_all_resumed([a], timeout=0.1, poll_interval=0.02) == ["a"]


def test_wait_all_resumed_succeeds_once_state_leaves_paused():
    a = _FakeAgent("a", paused=True)

    def flip():
        time.sleep(0.05)
        a._state.paused_ack.clear()

    threading.Thread(target=flip, daemon=True).start()
    assert agpause.wait_all_resumed([a], timeout=2, poll_interval=0.01) == []


def test_wait_all_paused_warns_when_upstream_not_in_requested_set():
    upstream = _FakeAgent("up", state="skill")
    downstream = _FakeAgent("down", state="blocked_on_dependency", blocked_on=upstream)
    pending = agpause.wait_all_paused(
        [downstream], timeout=0.2, poll_interval=0.02, warn_after_s=0.05
    )
    assert pending == ["down"]
    assert downstream.terminal.log.called
    logged = " ".join(str(c) for c in downstream.terminal.log.call_args_list)
    assert "up" in logged


# ---------------------------------------------------------------------------
# Integration: real agskill.run() background threads (fake execute_react,
# following the same convention as tests/test_agent.py's fork/concurrency
# tests) -- exercises the real _task() checkpoint plumbing end to end.
# ---------------------------------------------------------------------------


def make_looping_skill(name: str, counter: list, steps: int = 200, step_delay: float = 0.001):
    """A skill whose execute_react calls the real ag._check_pause() every
    iteration -- lets us pause/resume a genuine background _task() thread
    without choreographing LLM tool-call responses."""
    sk = agskill(name, "")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        for _ in range(steps):
            ag._check_pause(name)
            counter.append(1)
            time.sleep(step_delay)
        return agdata(done=True), prev_ctx, []

    sk.execute_react = fake_execute_react
    return sk


def test_pause_freezes_progress_mid_loop_and_resume_continues():
    counter: list = []
    ag = make_agent()
    skill = make_looping_skill("loop", counter, steps=300, step_delay=0.001)

    result = ag.run(skill, agdata())
    time.sleep(0.03)
    ag.pause()

    assert agpause.wait_all_paused([ag], timeout=2.0) == []
    assert ag.is_paused() is True
    frozen_at = len(counter)
    # Actually stopped early, not run to completion. (Real sandbox
    # provisioning latency before the loop starts is variable enough that
    # pause can legitimately land before iteration 0 ever runs -- frozen_at
    # can be 0; what matters is it never reached 300.)
    assert frozen_at < 300

    time.sleep(0.05)
    assert len(counter) == frozen_at  # no progress while paused

    ag.resume()
    assert agpause.wait_all_resumed([ag], timeout=2.0) == []
    assert result.wait().done is True
    assert len(counter) == 300


def test_pause_before_run_blocks_before_execute_react_starts():
    """pause() called on an idle agent before .run() -- the checkpoint added
    right after dependency resolution (before sandbox provisioning) must stop
    it before execute_react is ever invoked."""
    started = []
    ag = make_agent()
    skill = agskill("s", "")

    def fake_execute_react(a, prev_ctx, inp, max_steps=None, **_):
        started.append(1)
        return agdata(ok=True), prev_ctx, []

    skill.execute_react = fake_execute_react

    ag.pause()
    result = ag.run(skill, agdata())

    assert agpause.wait_all_paused([ag], timeout=2.0) == []
    assert started == []
    assert result.is_pending()

    ag.resume()
    assert result.wait().ok is True
    assert started == [1]


def test_wait_all_paused_settles_immediately_for_finished_agent():
    ag = make_agent()
    skill = agskill("s", "")
    skill.execute_react = lambda a, prev_ctx, inp, max_steps=None, **_: (
        agdata(ok=True),
        prev_ctx,
        [],
    )
    result = ag.run(skill, agdata())
    result.wait()
    ag.pause()  # pausing an already-finished agent is a no-op in practice
    assert agpause.wait_all_paused([ag], timeout=1.0) == []


def test_pause_avoids_deadlock_on_cross_agent_dependency():
    """The core scenario: B's skill_input carries A's still-pending agdata.
    Pausing A stops it before it produces a result, so B's own worker thread
    can never reach its own checkpoint -- it's parked inside
    resolve_input_dependencies() forever (until resumed). wait_all_paused
    must recognize B as settled via the transitive blocked_on chain instead
    of hanging."""
    calls: list = []
    a = make_agent()
    b = make_agent()

    skill_a = agskill("skillA", "")

    def fake_a(ag, prev_ctx, inp, max_steps=None, **_):
        calls.append("a")
        return agdata(val=1), prev_ctx, []

    skill_a.execute_react = fake_a

    skill_b = agskill("skillB", "")

    def fake_b(ag, prev_ctx, inp, max_steps=None, **_):
        calls.append("b")
        return agdata(val=inp.dep.val + 1), prev_ctx, []

    skill_b.execute_react = fake_b

    a.pause()
    pending_a = a.run(skill_a, agdata())
    time.sleep(0.1)
    assert a._state.state == "paused"
    assert calls == []

    pending_b = b.run(skill_b, agdata(dep=pending_a))
    time.sleep(0.1)
    assert b._state.state == "blocked_on_dependency"
    assert b._state.blocked_on is a

    still_pending = agpause.wait_all_paused([a, b], timeout=1.5)
    assert still_pending == []
    assert pending_a.is_pending()
    assert pending_b.is_pending()

    a.resume()
    agdata.wait_all([pending_a, pending_b])
    assert pending_a.val == 1
    assert pending_b.val == 2
    assert calls == ["a", "b"]
    assert agpause.wait_all_resumed([a, b], timeout=1.0) == []


def test_wait_all_paused_does_not_falsely_settle_when_upstream_genuinely_running():
    """If only B is asked about (A never paused and genuinely still running),
    wait_all_paused must NOT report done -- it should keep waiting until A
    either pauses or finishes, never claim a false 'all stopped'."""
    a = make_agent()
    b = make_agent()

    skill_a = agskill("skillA", "")

    def slow_a(ag, prev_ctx, inp, max_steps=None, **_):
        time.sleep(0.3)
        return agdata(val=1), prev_ctx, []

    skill_a.execute_react = slow_a

    skill_b = agskill("skillB", "")

    def fake_b(ag, prev_ctx, inp, max_steps=None, **_):
        return agdata(val=inp.dep.val + 1), prev_ctx, []

    skill_b.execute_react = fake_b

    pending_a = a.run(skill_a, agdata())
    time.sleep(0.05)
    assert a._state.state == "skill"  # genuinely running, not paused

    pending_b = b.run(skill_b, agdata(dep=pending_a))
    time.sleep(0.05)
    assert b._state.state == "blocked_on_dependency"

    pending = agpause.wait_all_paused([b], timeout=0.15, poll_interval=0.02, warn_after_s=None)
    assert pending == [b.agname]  # correctly NOT reported as settled

    # A was never paused -- it finishes on its own, unblocking B.
    assert pending_b.val == 2
    assert agpause.wait_all_paused([b], timeout=1.0) == []


# ---------------------------------------------------------------------------
# Fuzz: synthetic random dependency graphs — is_settled()/wait_all_paused()
# must always terminate quickly and match a direct is_settled() computation,
# including graphs with cycles and self-loops.
# ---------------------------------------------------------------------------

_STATES = ["skill", "paused", "finished", "error", "inactive", "blocked_on_dependency", "pausing"]


def test_fuzz_is_settled_never_hangs_on_random_graphs():
    """wait_all_paused's own timeout is honored per call, so a naive fuzz
    loop that expects many trials to have non-empty pending sets would burn
    trials * timeout seconds just waiting each one out -- that's not a hang,
    it's the mechanism working as designed. Use a tiny timeout (static state
    here never changes, so the very first poll already gives the final
    answer) and bound total trials accordingly."""
    rng = random.Random(0xA9E17)
    for trial in range(150):
        n = rng.randint(2, 16)
        nodes = [_FakeAgent(f"t{trial}_n{i}") for i in range(n)]
        for node in nodes:
            state = rng.choice(_STATES)
            node._state.state = state
            if state == "blocked_on_dependency":
                node._state.blocked_on = rng.choice(nodes)  # may be itself -> self-loop

        expected_pending = {node.agname for node in nodes if not node.is_settled()}

        t0 = time.monotonic()
        pending = agpause.wait_all_paused(
            nodes, timeout=0.01, poll_interval=0.005, warn_after_s=None
        )
        elapsed = time.monotonic() - t0

        assert set(pending) == expected_pending, (
            f"trial {trial} mismatch: {pending} vs {expected_pending}"
        )
        assert elapsed < 0.5, f"trial {trial}: wait_all_paused took {elapsed:.2f}s -- possible hang"


# ---------------------------------------------------------------------------
# Fuzz: real threaded agents forming random dependency DAGs, paused in random
# order/timing -- proves the actual production mechanism (real Futures, real
# background threads, real checkpoints) never deadlocks regardless of shape
# or race timing.
# ---------------------------------------------------------------------------


def _make_dag_skill(name: str, idx: int, rng: random.Random):
    sk = agskill(name, "")

    def fake_execute_react(ag, prev_ctx, inp, max_steps=None, **_):
        ag._check_pause(name)
        time.sleep(rng.uniform(0, 0.02))
        return agdata(val=idx), prev_ctx, []

    sk.execute_react = fake_execute_react
    return sk


@pytest.mark.parametrize("trial", range(15))
def test_fuzz_random_dag_pause_never_deadlocks(trial):
    rng = random.Random(1000 + trial)
    n = rng.randint(2, 4)
    agents = [make_agent() for _ in range(n)]
    pendings: list = [None] * n

    for i in range(n):
        deps = [j for j in range(i) if rng.random() < 0.5]
        dep_fields = {f"dep{j}": pendings[j] for j in deps}
        skill = _make_dag_skill(f"skill{trial}_{i}", i, rng)
        pendings[i] = agents[i].run(skill, agdata(**dep_fields))

    # Pause every agent, in random order, after a small random delay each --
    # deliberately racing against the agents' own completion.
    order = list(range(n))
    rng.shuffle(order)
    pausers = []
    for i in order:

        def _pause_after(idx=i):
            time.sleep(rng.uniform(0, 0.02))
            agents[idx].pause()

        th = threading.Thread(target=_pause_after, daemon=True)
        th.start()
        pausers.append(th)
    for th in pausers:
        th.join(timeout=2.0)

    still_pending = agpause.wait_all_paused(agents, timeout=3.0, poll_interval=0.01)
    assert still_pending == [], f"trial {trial}: deadlocked / never settled: {still_pending}"

    for ag in agents:
        ag.resume()

    agdata.wait_all([p for p in pendings if p is not None])
    for i, p in enumerate(pendings):
        assert p.val == i

    assert agpause.wait_all_resumed(agents, timeout=2.0) == []
