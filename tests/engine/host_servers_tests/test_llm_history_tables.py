# The history hash list and tables as produced by LlmHandlerServer: hashed once
# when the request arrives, message_patch_lookup applied before the request is sent,
# the SQLite collector recording the very same hashes, and the skill run's final
# hash list set on the agcontext.

from __future__ import annotations

import copy
import json
import sqlite3

from agency.agcontext import agcontext
from agency.configs.agconfig import agconfig, dataloggerconfig, llmconfig
from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
from agency.llm.message_store import MessageBlock, MessageStore
from agency.llm.usage_tracker import LlmUsageTracker
from agency.observability.agdatalogger import (
    agDataLogger,
    reconstruct_llm_exchanges,
    reconstruct_llm_messages,
    reconstruct_llm_patches,
)


class _Logger:
    def __init__(self):
        self.events = []
        self.exchanges = []

    def record_event(self, type, payload, call_label=None, **_kw):
        self.events.append((type, payload, call_label))

    def record_stream_delta(self, *a, **kw):
        pass

    def record_llm_exchange(self, call_label, **kw):
        self.exchanges.append((call_label, kw))


class _Backend:
    model = "test-model"

    def __init__(self):
        self.requests = []

    def dispatch(self, request):
        self.requests.append(copy.deepcopy(request))
        n = len(self.requests)
        return {
            "message": {
                "role": "assistant",
                "blocks": [{"type": "text", "index": 0, "text": f"reply {n}"}],
            },
            "usage": {"prompt_tokens": 10 * n, "completion_tokens": 1},
            "stop_reason": "stop",
        }

    def dispatch_stream(self, request, on_client=None):
        self.requests.append(copy.deepcopy(request))
        yield {"type": "delta", "index": 0, "block_type": "text", "text": "streamed"}
        yield {
            "type": "usage",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            "stop_reason": "stop",
        }


def _text(role, value):
    return {"role": role, "blocks": [{"type": "text", "index": 0, "text": value}]}


def patch(store, role, old, new_blocks):
    """Add one patch: dicts in the agency block format in, a hash-to-hashes link out."""
    with store:
        return store.add_patch(
            MessageBlock(role, old).hash,
            [store.add_block(MessageBlock(role, b)) for b in new_blocks],
        )


def _block(value):
    return {"type": "text", "index": 0, "text": value}


def _texts(messages):
    return [m["blocks"][0]["text"] for m in messages if m["blocks"]]


def _server(context=None, logger=None):
    backend = _Backend()
    server = LlmHandlerServer(
        agconfig(llmconfig(model="test-model", base_url="http://x/v1")),
        logger if logger is not None else _Logger(),
        LlmUsageTracker(),
        context=context,
    )
    server._backend = backend
    return server, backend


def _call(server):
    """The hash lists kept for the one call this server has handled."""
    (call,) = server._call_messages.values()
    return call


# -- hashing once, and what each list means --------------------------------------


def test_each_call_keeps_the_hashes_of_what_the_harness_sent_and_what_the_backend_was_sent():
    context = agcontext()
    patch(context.message_store, "user", _block("secret"), [_block("[redacted]")])
    server, backend = _server(context)

    server.dispatch({"messages": [_text("user", "task"), _text("user", "secret")]})

    call = _call(server)
    message_store = server._message_store
    # Rebuilding always applies message_patch_lookup, so read what the harness sent
    # through the same rows without it.
    without_patches = MessageStore(
        message_store.message_block_lookup, message_store.message_structure_lookup
    )
    assert _texts(without_patches.materialize(call["harness_message_hashes"])) == [
        "task",
        "secret",
    ]
    assert _texts(message_store.materialize(call["patched_message_hashes"])) == [
        "task",
        "[redacted]",
    ]
    # The patched messages are what the backend received.
    assert (
        message_store.materialize(call["patched_message_hashes"]) == backend.requests[0]["messages"]
    )
    assert _texts(backend.requests[0]["messages"]) == ["task", "[redacted]"]


def test_without_patches_the_two_lists_are_one():
    server, backend = _server(agcontext())
    server.dispatch({"messages": [_text("user", "task")]})
    call = _call(server)
    assert call["harness_message_hashes"] == call["patched_message_hashes"]
    assert backend.requests[0]["messages"] == [_text("user", "task")]


def test_a_server_without_a_context_still_hashes_and_never_patches():
    server, backend = _server(None)
    server.dispatch({"messages": [_text("user", "secret")]})
    assert _texts(backend.requests[0]["messages"]) == ["secret"]
    assert len(_call(server)["harness_message_hashes"]) == 1


# -- the persistent message_patch_lookup ----------------------------------------------------


def test_patch_is_applied_to_the_request_on_both_the_streaming_and_plain_paths():
    context = agcontext()
    patch(context.message_store, "user", _block("secret"), [_block("[redacted]")])
    server, backend = _server(context)

    server.dispatch({"messages": [_text("user", "secret")]})
    handle = server.start_stream({"messages": [_text("user", "secret")]})
    items = [handle.first()]
    while items[-1]["type"] not in ("done", "error"):
        items.append(handle.first())
    handle.cancel_and_join()

    assert [_texts(r["messages"]) for r in backend.requests] == [["[redacted]"], ["[redacted]"]]


def test_a_patch_added_between_two_calls_lands_at_the_second_call():
    # This is what a live patch from agent.patch_history() does to a skill run.
    context = agcontext()
    server, backend = _server(context)
    server.dispatch({"messages": [_text("user", "secret")]})
    patch(context.message_store, "user", _block("secret"), [_block("[redacted]")])
    server.dispatch({"messages": [_text("user", "secret")]})
    context.message_store.clear_patches()
    server.dispatch({"messages": [_text("user", "secret")]})
    assert [_texts(r["messages"]) for r in backend.requests] == [
        ["secret"],
        ["[redacted]"],
        ["secret"],
    ]


def test_the_logger_is_handed_the_message_store_and_the_hashes_of_each_exchange():
    context = agcontext()
    patch(context.message_store, "user", _block("secret"), [_block("[redacted]")])
    logger = _Logger()
    server, _ = _server(context, logger=logger)
    server.dispatch({"messages": [_text("user", "secret")]})
    ((_label, exchange),) = logger.exchanges
    message_store = exchange["message_store"]
    assert message_store is context.message_store
    assert _texts(message_store.materialize(exchange["patched_message_hashes"])) == ["[redacted]"]
    assert exchange["harness_message_hashes"] != exchange["patched_message_hashes"]
    assert _texts(message_store.materialize(exchange["response_message_hashes"])) == ["reply 1"]


def test_a_patch_lasts_across_runs_although_the_harness_resends_the_original():
    context = agcontext()
    patch(context.message_store, "user", _block("secret"), [_block("[redacted]")])

    # Run 1: the harness sends the original; the model gets the patch.
    server1, backend1 = _server(context)
    server1.dispatch({"messages": [_text("user", "task"), _text("user", "secret")]})
    context.message_hash_list = server1.get_main_history("task")
    assert _texts(backend1.requests[0]["messages"]) == ["task", "[redacted]"]

    # Run 2: a new server, a restored session that resends the unedited history.
    server2, backend2 = _server(context)
    server2.dispatch(
        {
            "messages": [
                _text("user", "task"),
                _text("user", "secret"),
                _text("assistant", "reply 1"),
                _text("user", "again"),
            ]
        }
    )
    assert _texts(backend2.requests[0]["messages"]) == ["task", "[redacted]", "reply 1", "again"]


def test_the_history_shows_the_patch_and_a_later_patch_reaches_back_into_it():
    context = agcontext()
    server, _ = _server(context)
    server.dispatch({"messages": [_text("user", "task"), _text("user", "secret")]})
    context.message_hash_list = server.get_main_history("task")
    assert _texts(context.get_resolved_transcript()) == ["task", "secret", "reply 1"]

    # Registered after the fact: message_block_lookup still holds "secret", and the
    # rebuilt history now shows the patch in its place.
    patch(context.message_store, "user", _block("secret"), [_block("[redacted]")])
    assert _texts(context.get_resolved_transcript()) == ["task", "[redacted]", "reply 1"]
    context.message_store.clear_patches()
    assert _texts(context.get_resolved_transcript()) == ["task", "secret", "reply 1"]


def test_the_usage_tracker_still_matches_on_what_the_harness_sent():
    context = agcontext()
    patch(context.message_store, "user", _block("secret"), [_block("[redacted]")])
    server, _ = _server(context)
    tracked = []
    original = server._tracked_messages

    def spy(label, request):
        messages = original(label, request)
        tracked.append(_texts(messages))
        return messages

    server._tracked_messages = spy
    server.dispatch({"messages": [_text("user", "secret")]})
    assert tracked == [["secret"]]


# -- the committed history ----------------------------------------------------------


def test_the_final_history_is_the_main_calls_messages_plus_its_reply():
    server, _ = _server(agcontext())
    first = server.dispatch({"messages": [_text("user", "task")]})
    server.dispatch({"messages": [_text("user", "task"), first["message"], _text("user", "next")]})
    history = server.get_main_history("task")
    assert _texts(server._message_store.materialize(history)) == [
        "task",
        "reply 1",
        "next",
        "reply 2",
    ]


def test_nothing_is_pruned_and_the_server_hashes_straight_into_the_agcontext():
    context = agcontext()
    server, _ = _server(context)
    assert server._message_store is context.message_store
    server.dispatch({"messages": [_text("user", "a short side call")]})
    server.dispatch(
        {"messages": [_text("user", "task"), _text("user", "more"), _text("user", "x")]}
    )
    history = server.get_main_history("task")
    context.message_hash_list = history
    # Every message the skill run hashed is still a row of message_structure_lookup,
    # including the side call that is not part of the final message_hash_list.
    assert set(history) < set(context.message_store.message_structure_lookup)
    assert len(context.message_store.message_structure_lookup) > len(history)


def test_a_run_with_no_llm_calls_commits_an_empty_history():
    server, _ = _server(agcontext())
    history = server.get_main_history()
    assert history == []


# -- the SQLite collector ------------------------------------------------------------


def _tool_exchange(arguments):
    return [
        _text("user", "run it"),
        {
            "role": "assistant",
            "blocks": [
                {
                    "type": "tool_use",
                    "index": 0,
                    "id": "call_1",
                    "name": "bash",
                    "arguments": arguments,
                    "citations": [{"cited_text": "x"}],
                }
            ],
        },
        {
            "role": "tool",
            "blocks": [{"type": "tool_result", "index": 0, "tool_call_id": "call_1", "text": "ok"}],
        },
    ]


def test_the_collector_keeps_every_field_and_can_rebuild_each_request_exactly(tmp_path):
    db_path = tmp_path / "agent.db"
    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(db_path))))
    logger.start()
    context = agcontext()
    patch(context.message_store, "user", _block("run it"), [_block("run it (patched)")])
    server, backend = _server(context, logger=logger)

    server.dispatch({"messages": _tool_exchange('{"cmd": "ls"}')})
    # The same tool call with different input is a different block, not a collision.
    server.dispatch({"messages": _tool_exchange('{"cmd": "rm -rf /"}')})
    logger.flush()

    connection = sqlite3.connect(db_path)
    labels = [row[0] for row in connection.execute("SELECT call_label FROM exchanges ORDER BY id")]
    assert len(labels) == 2
    for label, request in zip(labels, backend.requests):
        # Rebuilt from the message tables: exactly what the backend was sent --
        # patched, with tool arguments and every extra field intact.
        assert reconstruct_llm_messages(connection, label, "prompt") == request["messages"]
    tool_uses = [
        json.loads(p)
        for (p,) in connection.execute(
            "SELECT payload FROM message_block_lookup WHERE payload LIKE '%tool_use%'"
        )
    ]
    assert sorted(b["arguments"] for b in tool_uses) == ['{"cmd": "ls"}', '{"cmd": "rm -rf /"}']
    assert all(b["citations"] == [{"cited_text": "x"}] for b in tool_uses)

    # The reply is recorded the same way.
    assert reconstruct_llm_messages(connection, labels[0], "response") == [
        {"role": "assistant", "blocks": [{"type": "text", "index": 0, "text": "reply 1"}]}
    ]
    connection.close()
    logger.stop()


def test_the_collector_and_the_history_use_the_same_hashes(tmp_path):
    db_path = tmp_path / "agent.db"
    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(db_path))))
    logger.start()
    server, _ = _server(agcontext(), logger=logger)
    server.dispatch({"messages": [_text("user", "task")]})
    history = server.get_main_history("task")
    message_store = server._message_store
    logger.flush()

    connection = sqlite3.connect(db_path)
    stored_messages = {
        row[0] for row in connection.execute("SELECT hash FROM message_structure_lookup")
    }
    stored_blocks = {row[0] for row in connection.execute("SELECT hash FROM message_block_lookup")}
    connection.close()
    logger.stop()

    assert set(history) <= stored_messages
    assert {
        b for d in history for b in message_store.message_structure_lookup[d]["blocks"]
    } <= stored_blocks


def test_flat_exchange_readers_still_see_the_block_chains(tmp_path):
    db_path = tmp_path / "agent.db"
    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(db_path))))
    logger.start()
    server, _ = _server(agcontext(), logger=logger)
    server.dispatch({"messages": [_text("user", "task")]})
    logger.flush()
    connection = sqlite3.connect(db_path)
    (exchange,) = reconstruct_llm_exchanges(connection, "llm_block")
    connection.close()
    logger.stop()
    assert [b["text"] for b in exchange["prompt"]] == ["task"]
    assert exchange["prompt"][0]["role"] == "user"
    assert exchange["response"][0]["text"] == "reply 1"


def test_the_collector_keeps_the_patch_table_as_hash_to_hash_links(tmp_path):
    db_path = tmp_path / "agent.db"
    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(db_path))))
    logger.start()
    context = agcontext()
    patches = context.message_store
    key_split = patch(patches, "user", _block("secret"), [_block("part one"), _block("part two")])
    key_gone = patch(patches, "user", _block("noise"), [])
    patch(patches, "user", _block("never sent"), [_block("unused")])
    server, backend = _server(context, logger=logger)

    server.dispatch(
        {"messages": [_text("user", "task"), _text("user", "secret"), _text("user", "noise")]}
    )
    server.dispatch({"messages": [_text("user", "task")]})  # no patch matches
    logger.flush()

    connection = sqlite3.connect(db_path)
    first, second = [
        row[0] for row in connection.execute("SELECT call_label FROM exchanges ORDER BY id")
    ]
    links = connection.execute(
        "SELECT original_hash, seq, replacement_hash FROM message_patch_lookup "
        "WHERE call_label = ? ORDER BY original_hash, seq",
        (first,),
    ).fetchall()
    # One original maps to several ordered replacements; a removed block is one NULL link.
    by_original: dict = {}
    for original, seq, replacement in links:
        by_original.setdefault(original, []).append((seq, replacement))
    assert set(by_original) == {key_split, key_gone}
    assert [seq for seq, _ in by_original[key_split]] == [0, 1]
    assert by_original[key_gone] == [(0, None)]
    # Both ends of every link are rows of the sqlite message_block_lookup.
    stored = {row[0] for row in connection.execute("SELECT hash FROM message_block_lookup")}
    assert {key_split, key_gone} <= stored
    assert {r for _, r in by_original[key_split]} <= stored

    rebuilt = {p["original_hash"]: p for p in reconstruct_llm_patches(connection, first)}
    assert rebuilt == {
        key_split: {
            "original_hash": key_split,
            "original": _block("secret"),
            "replacement": [_block("part one"), _block("part two")],
        },
        key_gone: {"original_hash": key_gone, "original": _block("noise"), "replacement": []},
    }
    assert reconstruct_llm_patches(connection, second) == []
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM message_patch_lookup WHERE call_label = ?", (second,)
        ).fetchone()[0]
        == 0
    )
    connection.close()
    logger.stop()
