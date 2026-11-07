"""Tests for agcontext — conversation state container."""

from concurrent.futures import Future

import pytest

from agency.agcontext import agcontext


def test_concurrent_resolution_uses_the_captured_future():
    import threading

    captured = threading.Event()
    resolved = threading.Event()
    errors = []

    class InterleavedContext(agcontext):
        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            if name == "_future" and threading.current_thread().name == "slow-resolver":
                captured.set()
                assert resolved.wait(2)
            return value

    future = Future()
    future.set_result(agcontext.from_messages([{"content": "done"}]))
    context = InterleavedContext(_future=future)

    def resolve():
        try:
            context.resolve_prev_dependencies()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=resolve, name="slow-resolver")
    worker.start()
    try:
        assert captured.wait(2)
        context.resolve_prev_dependencies()
    finally:
        resolved.set()
        worker.join(2)
    assert not worker.is_alive()
    assert errors == []
    assert context.get_resolved_transcript() == [{"content": "done"}]


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


def test_default_construction():
    context = agcontext()
    assert context.get_resolved_transcript() == []
    assert context.harness_sessions == {}
    assert context.retained_messages == []
    assert context.harness_message_cursors == {}
    assert context._future is None


def test_construction_with_values():
    msgs = [{"role": "user", "content": "hi"}]
    sessions = {"claude_code": {"session_id": "s1", "blob_b64": "abc"}}
    context = agcontext.from_messages(msgs, harness_sessions=sessions)
    assert context.get_resolved_transcript() == msgs
    assert context.harness_sessions is sessions


def test_message_lookups_are_not_shared_between_contexts():
    ctx1 = agcontext()
    ctx2 = agcontext()
    ctx1.set_transcript([{"role": "user", "content": "x"}])
    assert ctx2.get_resolved_transcript() == []


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
    resolved = agcontext.from_messages([{"role": "user", "content": "resolved"}])
    f.set_result(resolved)
    context.resolve_prev_dependencies()
    assert context.is_pending() is False


# ---------------------------------------------------------------------------
# resolve_prev_dependencies
# ---------------------------------------------------------------------------


def test_resolve_no_op_when_not_pending():
    context = agcontext.from_messages([{"role": "user", "content": "x"}])
    context.resolve_prev_dependencies()
    assert context.get_resolved_transcript() == [{"role": "user", "content": "x"}]


def test_resolve_merges_future_state():
    f: Future[agcontext] = Future()
    placeholder = agcontext(_future=f)
    resolved = agcontext.from_messages(
        [{"role": "assistant", "content": "done"}],
        harness_sessions={"claude_code": {"session_id": "s1"}},
        retained_messages=[
            {
                "sequence": 1,
                "type": "message",
                "role": "user",
                "content": "remember",
            }
        ],
        harness_message_cursors={"claude_code": 1},
    )
    f.set_result(resolved)
    placeholder.resolve_prev_dependencies()

    assert placeholder.get_resolved_transcript() == [{"role": "assistant", "content": "done"}]
    assert placeholder.harness_sessions == {"claude_code": {"session_id": "s1"}}
    assert placeholder.retained_messages[0]["content"] == "remember"
    assert placeholder.harness_message_cursors == {"claude_code": 1}
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
    context = agcontext.from_messages([{"role": "user", "content": "a"}])
    c = context.copy()
    assert c is not context


def test_copy_isolates_the_rebuilt_messages():
    msgs = [{"role": "user", "content": "original"}]
    context = agcontext.from_messages(msgs)
    c = context.copy()
    c.get_resolved_transcript()[0]["content"] = "mutated"
    assert context.get_resolved_transcript()[0]["content"] == "original"


def test_copy_deep_copies_harness_sessions():
    sessions = {"claude_code": {"session_id": "s1"}}
    context = agcontext(harness_sessions=sessions)
    c = context.copy()
    c.harness_sessions["claude_code"]["session_id"] = "s2"
    assert context.harness_sessions["claude_code"]["session_id"] == "s1"


def test_retained_messages_are_validated_selected_and_cursor_advanced_monotonically():
    context = agcontext()
    context.append_retained_message(
        {
            "sequence": 1,
            "type": "message",
            "role": "user",
            "content": "first",
            "source": "queue_message",
        }
    )
    context.append_retained_message(
        {
            "sequence": 2,
            "type": "message",
            "role": "system",
            "content": "second",
        }
    )

    pending = context.pending_retained_messages("claude_code")
    assert [entry["content"] for entry in pending] == ["first", "second"]
    pending[0]["content"] = "mutated copy"
    assert context.retained_messages[0]["content"] == "first"

    context.advance_retained_cursor("claude_code", 1)
    assert [entry["content"] for entry in context.pending_retained_messages("claude_code")] == [
        "second"
    ]
    with pytest.raises(ValueError, match="cannot move backwards"):
        context.advance_retained_cursor("claude_code", 0)


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"sequence": 0, "type": "message", "role": "user", "content": "x"}, "positive"),
        ({"sequence": 1, "type": "event", "role": "user", "content": "x"}, "type"),
        ({"sequence": 1, "type": "message", "role": "assistant", "content": "x"}, "role"),
        ({"sequence": 1, "type": "message", "role": "user", "content": 1}, "content"),
    ],
)
def test_retained_message_validation(entry, message):
    with pytest.raises(ValueError, match=message):
        agcontext().append_retained_message(entry)


def test_copy_deep_copies_retained_state():
    context = agcontext(
        retained_messages=[
            {"sequence": 1, "type": "message", "role": "user", "content": "original"}
        ],
        harness_message_cursors={"native": 1},
    )
    copied = context.copy()
    copied.retained_messages[0]["content"] = "changed"
    copied.harness_message_cursors["native"] = 2
    assert context.retained_messages[0]["content"] == "original"
    assert context.harness_message_cursors == {"native": 1}


def test_copy_resolves_pending_future():
    f: Future[agcontext] = Future()
    context = agcontext(_future=f)
    f.set_result(agcontext.from_messages([{"role": "user", "content": "from future"}]))
    c = context.copy()
    assert c.get_resolved_transcript() == [{"role": "user", "content": "from future"}]
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
    context = agcontext.from_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ],
    )
    r = repr(context)
    assert "message_hash_list=2" in r
    assert "harnesses=[]" in r
    assert "retained=0" in r
    assert "pending" not in r


def test_repr_shows_harness_names():
    context = agcontext(harness_sessions={"claude_code": {"session_id": "s1"}})
    assert "harnesses=['claude_code']" in repr(context)


def test_repr_pending():
    f: Future[agcontext] = Future()
    context = agcontext(_future=f)
    assert "(pending)" in repr(context)


# Checkpoints written before block hashes stopped including their index.
def _indexed_checkpoint(messages, patches=()):
    import copy
    from agency.llm.message_store import content_hash

    blocks, structures, hashes, edits = {}, {}, [], {}

    def intern(role, block):
        payload = {"role": role, **copy.deepcopy(block)}
        digest = content_hash(payload)
        blocks[digest] = payload
        return digest

    for message in messages:
        record = {
            "role": message["role"],
            "blocks": [intern(message["role"], b) for b in message["blocks"]],
        }
        extra = {k: v for k, v in message.items() if k not in ("role", "blocks")}
        if extra:
            record["extra"] = extra
        digest = content_hash(record)
        structures[digest] = record
        hashes.append(digest)
    for message_index, block_index, replacements in patches:
        record = structures[hashes[message_index]]
        edits[record["blocks"][block_index]] = [intern(record["role"], b) for b in replacements]
    return {
        "message_hash_list": hashes,
        "message_block_lookup": blocks,
        "message_structure_lookup": structures,
        "message_patch_lookup": edits,
    }


def _indexed_text(value, index):
    return {"type": "text", "text": value, "index": index}


def test_legacy_checkpoint_patch_survives_resent_history_and_can_be_reverted():
    import copy
    from agency.llm.message_store import MessageBlock

    original = [{"role": "user", "name": "request", "blocks": [_indexed_text("original", 7)]}]
    state = _indexed_checkpoint(original, [(0, 0, [_indexed_text("replacement", 3)])])
    before = copy.deepcopy(state)
    restored = agcontext(**agcontext.message_state_fields(state))
    store = restored.message_store
    assert store.materialize(restored.message_hash_list)[0]["blocks"][0]["text"] == "replacement"
    resent = store.add_messages(original)
    assert resent == restored.message_hash_list
    assert store.materialize(resent)[0]["blocks"][0]["text"] == "replacement"
    shifted = [{"role": "user", "blocks": [_indexed_text("original", 9)]}]
    assert store.materialize(store.add_messages(shifted))[0]["blocks"][0]["text"] == "replacement"
    key = MessageBlock("user", original[0]["blocks"][0]).hash
    assert store.patch_origin(key) == "operator"
    assert store.remove_patch(key)
    assert restored.get_resolved_transcript() == original
    assert state == before
    saved = restored.message_state()
    assert saved["message_store_version"] == 2
    assert agcontext(**agcontext.message_state_fields(saved)).message_state() == saved


def test_legacy_checkpoint_keeps_deletions_multi_block_edits_and_origins():
    original = [{"role": "user", "blocks": [_indexed_text("a", 4), _indexed_text("b", 8)]}]
    state = _indexed_checkpoint(
        original,
        [
            (0, 0, []),
            (0, 1, [_indexed_text("summary", 0), _indexed_text("detail", 1)]),
        ],
    )
    second = state["message_structure_lookup"][state["message_hash_list"][0]]["blocks"][1]
    state["message_patch_origin"] = {second: "model"}
    restored = agcontext(**agcontext.message_state_fields(state))
    blocks = restored.message_store.materialize(restored.message_store.add_messages(original))[0][
        "blocks"
    ]
    assert blocks == [_indexed_text("summary", 0), _indexed_text("detail", 1)]
    assert sorted(
        restored.message_store.patch_origin(k) for k in restored.message_store.message_patch_lookup
    ) == ["model", "operator"]
    restored.message_store.clear_patches()
    assert restored.get_resolved_transcript() == original


@pytest.mark.parametrize("second_patch", [None, "other"])
def test_legacy_checkpoint_rejects_edits_that_would_change_scope(second_patch):
    original = [{"role": "user", "blocks": [_indexed_text("same", 0), _indexed_text("same", 1)]}]
    patches = [(0, 0, [_indexed_text("first", 0)])]
    if second_patch is not None:
        patches.append((0, 1, [_indexed_text(second_patch, 1)]))
    with pytest.raises(ValueError, match="conflicting patches"):
        agcontext.message_state_fields(_indexed_checkpoint(original, patches))


def test_legacy_checkpoint_merges_equivalent_patches():
    original = [{"role": "user", "blocks": [_indexed_text("same", 0), _indexed_text("same", 1)]}]
    state = _indexed_checkpoint(
        original,
        [
            (0, 0, [_indexed_text("edited", 0)]),
            (0, 1, [_indexed_text("edited", 1)]),
        ],
    )
    restored = agcontext(**agcontext.message_state_fields(state))
    assert len(restored.message_store.message_patch_lookup) == 1
    assert restored.get_resolved_transcript() == [
        {"role": "user", "blocks": [_indexed_text("edited", 0), _indexed_text("edited", 1)]}
    ]


def test_legacy_checkpoint_rejects_a_patch_with_no_original_payload():
    state = _indexed_checkpoint([])
    state["message_patch_lookup"]["missing"] = []
    with pytest.raises(ValueError, match="original blocks are missing"):
        agcontext.message_state_fields(state)


def test_unversioned_current_checkpoint_is_normalized_without_changing_it():
    from agency.llm.message_store import MessageBlock

    original = [{"role": "user", "blocks": [_indexed_text("original", 7)]}]
    context = agcontext.from_messages(original)
    store = context.message_store
    key = store.add_block(MessageBlock("user", original[0]["blocks"][0]))
    store.add_patch(key, [], origin="model")
    state = context.message_state()
    state.pop("message_store_version")
    restored = agcontext(**agcontext.message_state_fields(state))
    assert restored.message_state() == {"message_store_version": 2, **state}


def test_checkpoint_rejects_an_unknown_message_store_version():
    state = agcontext().message_state()
    state["message_store_version"] = 999
    with pytest.raises(ValueError, match="unsupported checkpoint message_store_version"):
        agcontext.message_state_fields(state)
