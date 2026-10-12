# The lossless message store and its patch overlay (agency/llm/message_store.py).

from __future__ import annotations

import copy
import json

import pytest

from agency.llm.message_store import (
    MessageBlock,
    MessageStore,
    block_of,
    block_payload,
    content_hash,
)


def text(role, value, index=0, **extra):
    return {"role": role, "blocks": [{"type": "text", "index": index, "text": value, **extra}]}


def tool_use(call_id="c1", arguments='{"cmd": "ls"}'):
    return {
        "role": "assistant",
        "blocks": [
            {"type": "tool_use", "index": 0, "id": call_id, "name": "bash", "arguments": arguments}
        ],
    }


def tool_result(call_id="c1", body="file.txt"):
    return {
        "role": "tool",
        "blocks": [{"type": "tool_result", "index": 0, "tool_call_id": call_id, "text": body}],
    }


# -- losslessness -------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: [text("user", "hi")],
        lambda: [text("user", "a"), text("user", "b")],  # two messages ...
        lambda: [
            {
                "role": "user",
                "blocks": [
                    {"type": "text", "index": 0, "text": "a"},
                    {"type": "text", "index": 1, "text": "b"},
                ],
            }
        ],  # ... versus one with two blocks
        lambda: [tool_use(), tool_result()],
        lambda: [text("user", "hi", citations=[{"cited_text": "x"}])],
        lambda: [{"role": "user", "blocks": [], "name": "extra-message-field"}],
        lambda: [{"role": "user", "blocks": [{"type": "text", "role": "inner", "text": "x"}]}],
        lambda: [{"role": "user", "content": "not the agency shape"}],
        lambda: [{"role": "user"}],
        lambda: [{"role": "user", "blocks": None}],
        lambda: ["not even a dict"],
    ],
)
def test_materialize_returns_exactly_what_was_interned(build):
    messages = build()
    store = MessageStore()
    hashes = store.add_messages(messages)
    assert store.materialize(hashes) == messages


def test_two_messages_and_one_message_with_two_blocks_have_different_hash_lists():
    store = MessageStore()
    two = store.add_messages([text("user", "a"), text("user", "b", index=0)])
    one = store.add_messages(
        [
            {
                "role": "user",
                "blocks": [
                    {"type": "text", "index": 0, "text": "a"},
                    {"type": "text", "index": 0, "text": "b"},
                ],
            }
        ]
    )
    assert two != one and len(two) == 2 and len(one) == 1


def test_a_block_differing_in_any_field_is_a_different_entry():
    store = MessageStore()
    base = tool_use()
    variants = [
        tool_use(arguments='{"cmd": "rm -rf /"}'),  # edited tool input
        tool_use(call_id="c2"),
        tool_result(body="[Old tool result content cleared]"),
        text("user", "hi", citations=[1]),
    ]
    seen = {store.add_message(base)}
    for variant in variants:
        digest = store.add_message(variant)
        assert digest not in seen
        seen.add(digest)
    # ... and each is stored exactly as given.
    assert store.materialize([store.add_message(tool_use(arguments='{"x": 1}'))]) == [
        tool_use(arguments='{"x": 1}')
    ]


def test_identical_blocks_are_stored_once_and_key_order_is_irrelevant():
    store = MessageStore()
    a = {"role": "user", "blocks": [{"type": "text", "index": 0, "text": "hi"}]}
    b = {"blocks": [{"text": "hi", "index": 0, "type": "text"}], "role": "user"}
    assert store.add_message(a) == store.add_message(b)
    assert len(store.message_block_lookup) == 1 and len(store.message_structure_lookup) == 1


def test_stored_payloads_are_not_aliased_to_the_callers_dicts():
    store = MessageStore()
    message = text("user", "original")
    digest = store.add_message(message)
    message["blocks"][0]["text"] = "mutated after the fact"
    out = store.materialize([digest])
    assert out[0]["blocks"][0]["text"] == "original"
    out[0]["blocks"][0]["text"] = "mutated the copy"
    assert store.materialize([digest])[0]["blocks"][0]["text"] == "original"


def test_block_payload_round_trips_even_with_a_role_key_inside_the_block():
    block = {"type": "text", "role": "inner", "text": "x"}
    assert block_of(block_payload("user", block)) == block
    assert block_of(block_payload("user", {"type": "text", "text": "x"})) == {
        "type": "text",
        "text": "x",
    }


def test_content_hash_is_the_sha256_of_canonical_json():
    assert content_hash({"b": 1, "a": 2}) == content_hash({"a": 2, "b": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})


# -- tables -------------------------------------------------------------------


def test_the_lookups_are_plain_json_dicts_that_rebuild_the_messages():
    store = MessageStore()
    hashes = store.add_messages([text("user", "a"), text("assistant", "b")])
    patch(store, "user", block("a"), [block("a2")])
    restored = MessageStore(
        json.loads(json.dumps(store.message_block_lookup)),
        json.loads(json.dumps(store.message_structure_lookup)),
        json.loads(json.dumps(store.message_patch_lookup)),
    )
    assert restored.materialize(hashes) == [text("user", "a2"), text("assistant", "b")]


# -- patches ------------------------------------------------------------------


def block(value, index=0):
    return {"type": "text", "index": index, "text": value}


def patch(store, role, old, new_blocks):
    """Add one patch: dicts in the agency block format in, a hash-to-hashes link out."""
    with store:
        return store.add_patch(
            MessageBlock(role, old).hash,
            [store.add_block(MessageBlock(role, b)) for b in new_blocks],
        )


def unpatched(store):
    """The same tables with no patches: what the harness sent."""
    return MessageStore(store.message_block_lookup, store.message_structure_lookup)


def test_patch_is_a_hash_to_hash_link_between_rows_of_the_block_lookup():
    store = MessageStore()
    key = patch(store, "user", block("secret"), [block("one"), block("two")])
    assert key == MessageBlock("user", block("secret")).hash
    one, two = (store.add_block(MessageBlock("user", block(v))) for v in ("one", "two"))
    assert store.message_patch_lookup == {key: [one, two]}
    # The replacements are rows of message_block_lookup; the original is not
    # written by add_patch (it is a row once something sent it).
    assert {one, two} <= set(store.message_block_lookup)
    assert key not in store.message_block_lookup
    store.add_block(MessageBlock("user", block("secret")))
    assert store.message_block_lookup[key] == block_payload("user", block("secret"))


def test_patch_replaces_the_keyed_block_when_the_history_is_rebuilt():
    store = MessageStore()
    hashes = store.add_messages([text("user", "a"), text("user", "secret"), text("user", "c")])
    key = patch(store, "user", block("secret"), [block("[redacted]")])

    assert store.materialize(hashes) == [
        text("user", "a"),
        text("user", "[redacted]"),
        text("user", "c"),
    ]
    # The original rows are untouched, so the patch can simply be withdrawn.
    assert unpatched(store).materialize(hashes) == [
        text("user", "a"),
        text("user", "secret"),
        text("user", "c"),
    ]
    assert store.remove_patch(key) is True
    assert store.remove_patch(key) is False
    assert store.materialize(hashes) == unpatched(store).materialize(hashes)


def test_clear_patches_withdraws_every_patch():
    store = MessageStore()
    hashes = store.add_messages([text("user", "a")])
    patch(store, "user", block("a"), [block("b")])
    store.clear_patches()
    assert store.message_patch_lookup == {}
    assert store.materialize(hashes) == [text("user", "a")]


def test_patch_can_delete_a_block_and_drops_a_message_that_becomes_empty():
    store = MessageStore()
    hashes = store.add_messages([text("user", "a"), text("user", "gone")])
    patch(store, "user", block("gone"), [])
    assert store.materialize(hashes) == [text("user", "a")]


def test_patch_can_expand_one_block_into_several_and_renumbers_indices():
    store = MessageStore()
    message = {
        "role": "user",
        "blocks": [block("x", 0), block("y", 1), block("z", 2)],
    }
    hashes = store.add_messages([message])
    patch(store, "user", block("y", 1), [block("y1", 9), block("y2", 9)])
    (out,) = store.materialize(hashes)
    assert [(b["index"], b["text"]) for b in out["blocks"]] == [
        (0, "x"),
        (1, "y1"),
        (2, "y2"),
        (3, "z"),
    ]


def test_patch_is_keyed_by_role_and_full_content():
    store = MessageStore()
    hashes = store.add_messages([text("user", "same"), text("assistant", "same")])
    patch(store, "assistant", block("same"), [block("changed")])
    assert store.materialize(hashes) == [
        text("user", "same"),
        text("assistant", "changed"),
    ]

    # A tool_use with other arguments is a different block and is not touched.
    store = MessageStore()
    hashes = store.add_messages([tool_use(arguments='{"cmd": "ls"}')])
    patch(store, "assistant", tool_use(arguments='{"cmd": "pwd"}')["blocks"][0], [block("x")])
    assert store.materialize(hashes) == [tool_use(arguments='{"cmd": "ls"}')]


def test_patch_is_not_applied_recursively():
    store = MessageStore()
    hashes = store.add_messages([text("user", "a")])
    patch(store, "user", block("a"), [block("b")])
    patch(store, "user", block("b"), [block("c")])
    assert store.materialize(hashes) == [text("user", "b")]


def test_materialize_applies_only_the_patches_whose_original_block_occurs():
    store = MessageStore()
    hashes = store.add_messages([text("user", "a"), text("user", "secret")])
    patch(store, "user", block("other"), [block("x")])
    assert store.materialize(hashes) == [text("user", "a"), text("user", "secret")]

    patch(store, "user", block("secret"), [block("[redacted]")])
    patched = store.materialize(hashes)
    assert patched == [text("user", "a"), text("user", "[redacted]")]
    # The hashes of the patched messages describe them exactly.
    assert store.materialize(store.add_messages(patched)) == patched


def test_copy_has_independent_patches():
    store = MessageStore()
    key = patch(store, "user", block("a"), [block("b")])
    clone = store.copy()
    clone.remove_patch(key)
    patch(clone, "user", block("c"), [block("d")])
    assert list(store.message_patch_lookup) == [key]


def test_add_patch_takes_hashes_and_the_replacement_blocks_must_already_be_stored():
    store = MessageStore()
    old = MessageBlock("user", block("a"))
    with pytest.raises(TypeError):
        store.add_patch(old, [])  # a block, not its hash
    with pytest.raises(TypeError):
        store.add_patch(old.hash, [block("b")])  # a dict, not a hash
    with pytest.raises(KeyError):
        store.add_patch(old.hash, [MessageBlock("user", block("b")).hash])  # not added yet
    assert store.message_patch_lookup == {}
    new = store.add_block(MessageBlock("user", block("b")))
    assert store.add_patch(old.hash, [new]) == old.hash
    assert store.message_patch_lookup == {old.hash: [new]}


def test_add_block_stores_once_and_takes_only_a_message_block():
    store = MessageStore()
    a = MessageBlock("user", block("a"))
    assert store.add_block(a) == store.add_block(MessageBlock("user", block("a"))) == a.hash
    assert len(store.message_block_lookup) == 1
    assert store.message_block_lookup[a.hash] == a.payload
    with pytest.raises(TypeError):
        store.add_block(block("a"))


def test_a_message_block_is_immutable_and_copies_what_it_is_given():
    fields = {"type": "text", "index": 0, "text": "a"}
    b = MessageBlock("user", fields)
    fields["text"] = "changed"
    assert b.to_dict()["text"] == "a"
    b.to_dict()["text"] = "changed"
    b.payload["text"] = "changed"
    assert b.to_dict()["text"] == "a"
    with pytest.raises(AttributeError):
        b.role = "assistant"
    assert (b.role, b.type) == ("user", "text")


def test_a_message_block_is_equal_by_hash_and_the_role_is_part_of_it():
    a = MessageBlock("user", block("a"))
    assert a == MessageBlock("user", block("a")) and hash(a) == hash(
        MessageBlock("user", block("a"))
    )
    assert a != MessageBlock("assistant", block("a"))
    assert a != MessageBlock("user", block("b"))
    assert a.hash == content_hash(block_payload("user", block("a")))
    assert MessageBlock.from_payload(a.payload) == a
    nested = MessageBlock("user", {"type": "x", "role": "inner"})  # its own role key survives
    assert MessageBlock.from_payload(nested.payload).to_dict() == {"type": "x", "role": "inner"}


def test_a_message_block_rejects_bad_arguments():
    for role, fields in [
        ("", block("a")),
        (None, block("a")),
        ("user", "a"),
        ("user", [block("a")]),
    ]:
        with pytest.raises(TypeError):
            MessageBlock(role, fields)
    with pytest.raises(TypeError):
        MessageBlock("user", MessageBlock("user", block("a")))


def test_adding_messages_is_thread_safe():
    import threading

    store = MessageStore()
    messages = [text("user", f"m{i}") for i in range(50)]
    results = []

    def work():
        results.append(store.add_messages(copy.deepcopy(messages)))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == results[0] for r in results)
    assert len(store.message_structure_lookup) == 50


def test_access_is_exclusive_so_two_patches_are_never_seen_half_applied():
    import threading

    store = MessageStore()
    digests = store.add_messages([text("user", "a"), text("user", "b")])
    stop = threading.Event()
    errors: list[BaseException] = []

    def flip():
        while not stop.is_set():
            with store:  # both patches or neither, as one step
                patch(store, "user", block("a"), [block("a2")])
                patch(store, "user", block("b"), [block("b2")])
            with store:
                store.clear_patches()
            store.copy()

    def request():
        try:
            for _ in range(2000):
                texts = [m["blocks"][0]["text"] for m in store.materialize(digests)]
                assert texts in (["a", "b"], ["a2", "b2"]), texts
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            errors.append(exc)

    writer = threading.Thread(target=flip)
    writer.start()
    try:
        request()
    finally:
        stop.set()
        writer.join()
    assert errors == []


def test_the_lock_is_reentrant_and_with_store_holds_it():
    store = MessageStore()
    with store:
        digest = store.add_message({"role": "user", "blocks": [block("x")]})
        assert store._message(digest)["blocks"][0]["text"] == "x"
        assert len(store) == 1
