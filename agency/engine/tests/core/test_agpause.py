"""Tests for agpause -- cross-agent dependency tagging (tag_producer/
producer_of/note_blocked_on) and agent.is_settled()'s recursive settle-check.

wait_all_paused()/wait_all_resumed() and the run_allowed/paused_ack/
_check_pause() machinery they depended on were retired along with the
host-side in-process pause checkpoint -- pausing is now a message delivered
through the harness manager (see agent.pause()/resume()/is_paused() and
HarnessInteractionServer.update_state())."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agency import agpause
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
# agent.is_settled() — recursive settle-check
# ---------------------------------------------------------------------------


class _FakeAgent:
    is_settled = agent.is_settled

    def __init__(self, name, state="skill", blocked_on=None):
        self.agname = name
        self._state = SimpleNamespace(state=state, blocked_on=blocked_on)
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
