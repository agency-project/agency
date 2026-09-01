"""Tests for agcontext — conversation state container."""

from concurrent.futures import Future

from agency.agcontext import agcontext


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


def test_default_construction():
    context = agcontext()
    assert context.recent_transcript == []
    assert context.harness_sessions == {}
    assert context._future is None


def test_construction_with_values():
    msgs = [{"role": "user", "content": "hi"}]
    sessions = {"claude_code": {"session_id": "s1", "blob_b64": "abc"}}
    context = agcontext(recent_transcript=msgs, harness_sessions=sessions)
    assert context.recent_transcript is msgs
    assert context.harness_sessions is sessions


def test_recent_transcript_default_is_empty_list_not_shared():
    ctx1 = agcontext()
    ctx2 = agcontext()
    ctx1.recent_transcript.append({"role": "user", "content": "x"})
    assert ctx2.recent_transcript == []


def test_harness_sessions_default_is_empty_dict_not_shared():
    ctx1 = agcontext()
    ctx2 = agcontext()
    ctx1.harness_sessions["claude_code"] = {"session_id": "s1"}
    assert ctx2.harness_sessions == {}


# ---------------------------------------------------------------------------
# is_pending
# ---------------------------------------------------------------------------


def test_is_pending_false_when_no_future():
    assert agcontext().is_pending() is False


def test_is_pending_true_when_future_set():
    f: Future = Future()
    context = agcontext(_future=f)
    assert context.is_pending() is True


def test_is_pending_false_after_resolve():
    f: Future[agcontext] = Future()
    context = agcontext(_future=f)
    resolved = agcontext(recent_transcript=[{"role": "user", "content": "resolved"}])
    f.set_result(resolved)
    context.resolve_prev_dependencies()
    assert context.is_pending() is False


# ---------------------------------------------------------------------------
# resolve_prev_dependencies
# ---------------------------------------------------------------------------


def test_resolve_no_op_when_not_pending():
    context = agcontext(recent_transcript=[{"role": "user", "content": "x"}])
    context.resolve_prev_dependencies()
    assert context.recent_transcript == [{"role": "user", "content": "x"}]


def test_resolve_merges_future_state():
    f: Future[agcontext] = Future()
    placeholder = agcontext(_future=f)
    resolved = agcontext(
        recent_transcript=[{"role": "assistant", "content": "done"}],
        harness_sessions={"claude_code": {"session_id": "s1"}},
    )
    f.set_result(resolved)
    placeholder.resolve_prev_dependencies()

    assert placeholder.recent_transcript == [{"role": "assistant", "content": "done"}]
    assert placeholder.harness_sessions == {"claude_code": {"session_id": "s1"}}
    assert placeholder._future is None


def test_resolve_clears_future():
    f: Future[agcontext] = Future()
    context = agcontext(_future=f)
    f.set_result(agcontext())
    context.resolve_prev_dependencies()
    assert context._future is None


def test_resolve_blocks_until_future_set():
    import threading

    f: Future[agcontext] = Future()
    context = agcontext(_future=f)

    def setter():
        import time

        time.sleep(0.05)
        f.set_result(agcontext(harness_sessions={"claude_code": {"session_id": "from-setter"}}))

    t = threading.Thread(target=setter, daemon=True)
    t.start()
    context.resolve_prev_dependencies()
    t.join()
    assert context.harness_sessions == {"claude_code": {"session_id": "from-setter"}}


def test_resolve_is_idempotent():
    f: Future[agcontext] = Future()
    context = agcontext(_future=f)
    f.set_result(agcontext(harness_sessions={"claude_code": {"session_id": "s"}}))
    context.resolve_prev_dependencies()
    context.resolve_prev_dependencies()  # second call must not raise
    assert context.harness_sessions == {"claude_code": {"session_id": "s"}}


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_returns_new_instance():
    context = agcontext(recent_transcript=[{"role": "user", "content": "a"}])
    c = context.copy()
    assert c is not context


def test_copy_deep_copies_recent_transcript():
    msgs = [{"role": "user", "content": "original"}]
    context = agcontext(recent_transcript=msgs)
    c = context.copy()
    c.recent_transcript[0]["content"] = "mutated"
    assert context.recent_transcript[0]["content"] == "original"


def test_copy_deep_copies_harness_sessions():
    sessions = {"claude_code": {"session_id": "s1"}}
    context = agcontext(harness_sessions=sessions)
    c = context.copy()
    c.harness_sessions["claude_code"]["session_id"] = "s2"
    assert context.harness_sessions["claude_code"]["session_id"] == "s1"


def test_copy_resolves_pending_future():
    f: Future[agcontext] = Future()
    context = agcontext(_future=f)
    f.set_result(agcontext(recent_transcript=[{"role": "user", "content": "from future"}]))
    c = context.copy()
    assert c.recent_transcript == [{"role": "user", "content": "from future"}]
    assert c._future is None


def test_copy_does_not_carry_future():
    f: Future[agcontext] = Future()
    context = agcontext(_future=f)
    f.set_result(agcontext())
    c = context.copy()
    assert c._future is None
    assert c.is_pending() is False


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_not_pending():
    context = agcontext(
        recent_transcript=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ],
    )
    r = repr(context)
    assert "recent_transcript=2" in r
    assert "harnesses=[]" in r
    assert "pending" not in r


def test_repr_shows_harness_names():
    context = agcontext(harness_sessions={"claude_code": {"session_id": "s1"}})
    assert "harnesses=['claude_code']" in repr(context)


def test_repr_pending():
    f: Future[agcontext] = Future()
    context = agcontext(_future=f)
    assert "(pending)" in repr(context)
