"""Tests for agcontext — conversation state container."""

from concurrent.futures import Future

from agency.agcontext import agcontext


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


def test_default_construction():
    ctx = agcontext()
    assert ctx.recent_transcript == []
    assert ctx.harness_sessions == {}
    assert ctx._future is None


def test_construction_with_values():
    msgs = [{"role": "user", "content": "hi"}]
    sessions = {"claude_code": {"session_id": "s1", "blob_b64": "abc"}}
    ctx = agcontext(recent_transcript=msgs, harness_sessions=sessions)
    assert ctx.recent_transcript is msgs
    assert ctx.harness_sessions is sessions


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
    ctx = agcontext(_future=f)
    assert ctx.is_pending() is True


def test_is_pending_false_after_resolve():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    resolved = agcontext(recent_transcript=[{"role": "user", "content": "resolved"}])
    f.set_result(resolved)
    ctx.resolve_prev_dependencies()
    assert ctx.is_pending() is False


# ---------------------------------------------------------------------------
# resolve_prev_dependencies
# ---------------------------------------------------------------------------


def test_resolve_no_op_when_not_pending():
    ctx = agcontext(recent_transcript=[{"role": "user", "content": "x"}])
    ctx.resolve_prev_dependencies()
    assert ctx.recent_transcript == [{"role": "user", "content": "x"}]


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
    ctx = agcontext(_future=f)
    f.set_result(agcontext())
    ctx.resolve_prev_dependencies()
    assert ctx._future is None


def test_resolve_blocks_until_future_set():
    import threading

    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)

    def setter():
        import time

        time.sleep(0.05)
        f.set_result(agcontext(harness_sessions={"claude_code": {"session_id": "from-setter"}}))

    t = threading.Thread(target=setter, daemon=True)
    t.start()
    ctx.resolve_prev_dependencies()
    t.join()
    assert ctx.harness_sessions == {"claude_code": {"session_id": "from-setter"}}


def test_resolve_is_idempotent():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    f.set_result(agcontext(harness_sessions={"claude_code": {"session_id": "s"}}))
    ctx.resolve_prev_dependencies()
    ctx.resolve_prev_dependencies()  # second call must not raise
    assert ctx.harness_sessions == {"claude_code": {"session_id": "s"}}


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_returns_new_instance():
    ctx = agcontext(recent_transcript=[{"role": "user", "content": "a"}])
    c = ctx.copy()
    assert c is not ctx


def test_copy_deep_copies_recent_transcript():
    msgs = [{"role": "user", "content": "original"}]
    ctx = agcontext(recent_transcript=msgs)
    c = ctx.copy()
    c.recent_transcript[0]["content"] = "mutated"
    assert ctx.recent_transcript[0]["content"] == "original"


def test_copy_deep_copies_harness_sessions():
    sessions = {"claude_code": {"session_id": "s1"}}
    ctx = agcontext(harness_sessions=sessions)
    c = ctx.copy()
    c.harness_sessions["claude_code"]["session_id"] = "s2"
    assert ctx.harness_sessions["claude_code"]["session_id"] == "s1"


def test_copy_resolves_pending_future():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    f.set_result(agcontext(recent_transcript=[{"role": "user", "content": "from future"}]))
    c = ctx.copy()
    assert c.recent_transcript == [{"role": "user", "content": "from future"}]
    assert c._future is None


def test_copy_does_not_carry_future():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    f.set_result(agcontext())
    c = ctx.copy()
    assert c._future is None
    assert c.is_pending() is False


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_not_pending():
    ctx = agcontext(
        recent_transcript=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ],
    )
    r = repr(ctx)
    assert "recent_transcript=2" in r
    assert "harnesses=[]" in r
    assert "pending" not in r


def test_repr_shows_harness_names():
    ctx = agcontext(harness_sessions={"claude_code": {"session_id": "s1"}})
    assert "harnesses=['claude_code']" in repr(ctx)


def test_repr_pending():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    assert "(pending)" in repr(ctx)
