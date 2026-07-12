"""Tests for agcontext — conversation state container."""

from concurrent.futures import Future

from agency.agcontext import agcontext


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


def test_default_construction():
    ctx = agcontext()
    assert ctx.messages == []
    assert ctx.total_input_tokens == 0
    assert ctx.total_output_tokens == 0
    assert ctx.compaction_summary is None
    assert ctx._future is None


def test_construction_with_values():
    msgs = [{"role": "user", "content": "hi"}]
    ctx = agcontext(
        messages=msgs, total_input_tokens=10, total_output_tokens=5, compaction_summary="summary"
    )
    assert ctx.messages is msgs
    assert ctx.total_input_tokens == 10
    assert ctx.total_output_tokens == 5
    assert ctx.compaction_summary == "summary"


def test_messages_default_is_empty_list_not_shared():
    ctx1 = agcontext()
    ctx2 = agcontext()
    ctx1.messages.append({"role": "user", "content": "x"})
    assert ctx2.messages == []


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
    resolved = agcontext(
        messages=[{"role": "user", "content": "resolved"}],
        total_input_tokens=7,
        total_output_tokens=3,
    )
    f.set_result(resolved)
    ctx.resolve_prev_dependencies()
    assert ctx.is_pending() is False


# ---------------------------------------------------------------------------
# resolve_prev_dependencies
# ---------------------------------------------------------------------------


def test_resolve_no_op_when_not_pending():
    ctx = agcontext(messages=[{"role": "user", "content": "x"}], total_input_tokens=1)
    ctx.resolve_prev_dependencies()
    assert ctx.messages == [{"role": "user", "content": "x"}]
    assert ctx.total_input_tokens == 1


def test_resolve_merges_future_state():
    f: Future[agcontext] = Future()
    placeholder = agcontext(_future=f)
    resolved = agcontext(
        messages=[{"role": "assistant", "content": "done"}],
        total_input_tokens=42,
        total_output_tokens=17,
        compaction_summary="compact",
    )
    f.set_result(resolved)
    placeholder.resolve_prev_dependencies()

    assert placeholder.messages == [{"role": "assistant", "content": "done"}]
    assert placeholder.total_input_tokens == 42
    assert placeholder.total_output_tokens == 17
    assert placeholder.compaction_summary == "compact"
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
        f.set_result(agcontext(total_input_tokens=99))

    t = threading.Thread(target=setter, daemon=True)
    t.start()
    ctx.resolve_prev_dependencies()
    t.join()
    assert ctx.total_input_tokens == 99


def test_resolve_is_idempotent():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    f.set_result(agcontext(total_input_tokens=5))
    ctx.resolve_prev_dependencies()
    ctx.resolve_prev_dependencies()  # second call must not raise
    assert ctx.total_input_tokens == 5


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_returns_new_instance():
    ctx = agcontext(messages=[{"role": "user", "content": "a"}], total_input_tokens=3)
    c = ctx.copy()
    assert c is not ctx


def test_copy_deep_copies_messages():
    msgs = [{"role": "user", "content": "original"}]
    ctx = agcontext(messages=msgs)
    c = ctx.copy()
    c.messages[0]["content"] = "mutated"
    assert ctx.messages[0]["content"] == "original"


def test_copy_preserves_token_counts():
    ctx = agcontext(total_input_tokens=10, total_output_tokens=20)
    c = ctx.copy()
    assert c.total_input_tokens == 10
    assert c.total_output_tokens == 20


def test_copy_preserves_compaction_summary():
    ctx = agcontext(compaction_summary="the summary")
    c = ctx.copy()
    assert c.compaction_summary == "the summary"


def test_copy_resolves_pending_future():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    f.set_result(agcontext(messages=[{"role": "user", "content": "from future"}]))
    c = ctx.copy()
    assert c.messages == [{"role": "user", "content": "from future"}]
    assert c._future is None


def test_copy_does_not_carry_future():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    f.set_result(agcontext(total_input_tokens=1))
    c = ctx.copy()
    assert c._future is None
    assert c.is_pending() is False


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_not_pending():
    ctx = agcontext(
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}],
        total_input_tokens=10,
        total_output_tokens=5,
    )
    r = repr(ctx)
    assert "msgs=2" in r
    assert "in=10" in r
    assert "out=5" in r
    assert "compact=no" in r
    assert "pending" not in r


def test_repr_with_compaction_summary():
    ctx = agcontext(compaction_summary="summary text")
    assert "compact=yes" in repr(ctx)


def test_repr_pending():
    f: Future[agcontext] = Future()
    ctx = agcontext(_future=f)
    assert "(pending)" in repr(ctx)
