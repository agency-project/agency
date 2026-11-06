# agent-level behaviour of the history hash tables: the ordered patch API,
# checkpoints (new format and old), and what a context carries into a run.

from __future__ import annotations

import io
import json
import tarfile

import pytest

from agency import agdata, agent
from agency.agcontext import agcontext
from agency.agname import agname as _agname
from agency.configs.agconfig import agconfig, llmconfig
from agency.llm.message_store import MessageBlock


def _cfg():
    return agconfig(llmconfig(provider="bedrock", model="m", api_key="k"))


def _text(role, value):
    return {"role": role, "blocks": [{"type": "text", "index": 0, "text": value}]}


def _block(value):
    return {"type": "text", "index": 0, "text": value}


def patch(store, role, old, new_blocks):
    """Add one patch: dicts in the agency block format in, a hash-to-hashes link out."""
    with store:
        return store.add_patch(
            MessageBlock(role, old).hash,
            [store.add_block(MessageBlock(role, b)) for b in new_blocks],
        )


def patch_history(ag, role, old, new_blocks):
    return ag.patch_history(MessageBlock(role, old), [MessageBlock(role, b) for b in new_blocks])


def _texts(messages):
    return [m["blocks"][0]["text"] for m in messages if m["blocks"]]


def _release(ag):
    name = ag.agname
    del ag
    _agname._allocated.discard(name)


@pytest.fixture
def ag():
    instance = agent(agconfig=_cfg())
    instance.history = agdata(
        messages=[_text("user", "task"), _text("user", "secret"), _text("assistant", "ok")]
    )
    yield instance
    _release(instance)


def _state(path):
    with tarfile.open(path, "r:gz") as tar:
        return json.loads(tar.extractfile("state.json").read())


# -- the patch API --------------------------------------------------------------


def test_patch_history_rewrites_the_rebuilt_history_and_can_be_withdrawn(ag):
    key = patch_history(ag, "user", _block("secret"), [_block("[redacted]")])
    assert _texts(ag.history.messages) == ["task", "[redacted]", "ok"]

    ag.remove_history_patch(key)
    assert _texts(ag.history.messages) == ["task", "secret", "ok"]

    patch_history(ag, "user", _block("secret"), [])
    assert _texts(ag.history.messages) == ["task", "ok"]
    ag.clear_history_patches()
    assert _texts(ag.history.messages) == ["task", "secret", "ok"]


def test_patch_history_returns_the_original_blocks_content_hash(ag):
    from agency.llm.message_store import block_payload, content_hash

    key = patch_history(ag, "user", _block("secret"), [_block("x")])
    assert key == content_hash(block_payload("user", _block("secret")))
    assert key == MessageBlock("user", _block("secret")).hash
    ag.context.resolve_prev_dependencies()
    assert key in ag.context.message_store.message_patch_lookup


def test_patch_history_rejects_bad_arguments(ag):
    secret = MessageBlock("user", _block("secret"))
    with pytest.raises(TypeError):
        ag.patch_history(_block("secret"), [secret])  # a raw dict is not a block
    with pytest.raises(TypeError):
        ag.patch_history(secret, secret)  # not a list
    with pytest.raises(TypeError):
        ag.patch_history(secret, [_block("x")])  # raw dicts are not blocks


def test_a_patch_submitted_to_the_agent_does_not_touch_earlier_contexts(ag):
    before = ag.context
    before.resolve_prev_dependencies()
    patch_history(ag, "user", _block("secret"), [_block("[redacted]")])
    ag.context.resolve_prev_dependencies()
    assert not before.message_store.message_patch_lookup
    assert ag.context.message_store.message_patch_lookup


# -- checkpoints ----------------------------------------------------------------


def test_checkpoint_saves_the_tables_not_a_message_list_and_load_restores_them(ag, tmp_path):
    patch_history(ag, "user", _block("secret"), [_block("[redacted]")])
    ckpt = tmp_path / "agent.ckpt"
    ag.save(ckpt)
    state = _state(ckpt)

    assert "history" not in state
    assert len(state["message_hash_list"]) == 3
    assert set(state["message_hash_list"]) <= set(state["message_structure_lookup"])
    assert len(state["message_patch_lookup"]) == 1

    name = ag.agname
    del ag
    _agname._allocated.discard(name)
    loaded = agent.load(ckpt, agconfig=_cfg())
    try:
        assert _texts(loaded.history.messages) == ["task", "[redacted]", "ok"]
        # The originals are still there beneath the patch.
        loaded.clear_history_patches()
        assert _texts(loaded.history.messages) == ["task", "secret", "ok"]
    finally:
        _release(loaded)


def test_a_repeated_block_is_stored_once_in_a_checkpoint(tmp_path):
    instance = agent(agconfig=_cfg())
    try:
        instance.history = agdata(messages=[_text("user", "same")] * 5)
        ckpt = tmp_path / "agent.ckpt"
        instance.save(ckpt)
        state = _state(ckpt)
        assert len(state["message_hash_list"]) == 5
        assert len(state["message_structure_lookup"]) == 1
        assert len(state["message_block_lookup"]) == 1
    finally:
        _release(instance)


def test_a_checkpoint_from_before_the_tables_still_loads(tmp_path):
    instance = agent(agconfig=_cfg())
    ckpt = tmp_path / "agent.ckpt"
    instance.save(ckpt)
    name = instance.agname
    del instance
    _agname._allocated.discard(name)

    # Rewrite it the way an older version stored history: a plain message list.
    with tarfile.open(ckpt, "r:gz") as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    state = json.loads(members["state.json"])
    for key in ("message_hash_list", "message_block_lookup", "message_patch_lookup"):
        state.pop(key)
    legacy = [_text("user", "old"), {"role": "assistant", "content": "plain"}]
    state["history"] = legacy
    members["state.json"] = json.dumps(state).encode()
    with tarfile.open(ckpt, "w:gz") as tar:
        for member, data in members.items():
            info = tarfile.TarInfo(name=member)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    loaded = agent.load(ckpt, agconfig=_cfg())
    try:
        assert loaded.history.messages == legacy
        assert not loaded.context.message_store.message_patch_lookup
    finally:
        _release(loaded)


# -- agcontext ------------------------------------------------------------------


def test_an_agcontext_copy_has_its_own_message_store_lookups():
    context = agcontext.from_messages([_text("user", "a")])
    patch(context.message_store, "user", _block("a"), [_block("b")])
    clone = context.copy()
    assert clone.message_store is not context.message_store
    # Rows added or removed in the copy never reach the original ...
    clone.message_store.clear_patches()
    clone.set_transcript([_text("user", "z")])
    assert _texts(context.get_resolved_transcript()) == ["b"]
    assert len(context.message_store.message_patch_lookup) == 1
    assert len(context.message_store.message_structure_lookup) == 1
    # ... and the copy starts with everything the original had.
    again = context.copy()
    assert _texts(again.get_resolved_transcript()) == ["b"]
    assert again.message_store.message_block_lookup == context.message_store.message_block_lookup


def test_a_replacement_block_is_a_row_of_message_block_lookup():
    context = agcontext.from_messages([_text("user", "old")])
    store = context.message_store
    key = patch(store, "user", _block("old"), [_block("one"), _block("two")])
    assert key in store.message_block_lookup  # the original was hashed when it was set
    assert set(store.message_patch_lookup[key]) <= set(store.message_block_lookup)
    (message,) = context.get_resolved_transcript()
    assert [b["text"] for b in message["blocks"]] == ["one", "two"]


def test_mutating_the_rebuilt_transcript_never_changes_the_context():
    context = agcontext.from_messages([_text("user", "a")])
    context.get_resolved_transcript()[0]["blocks"][0]["text"] = "mutated"
    assert _texts(context.get_resolved_transcript()) == ["a"]
