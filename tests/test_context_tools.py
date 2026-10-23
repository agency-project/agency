# The context tools: the model listing, reading and editing its own LLM context
# through message_patch_lookup (agency/tools/context_tools.py, docs/Context_Tools.md).

from __future__ import annotations

import asyncio
import copy
import json
import sqlite3
import uuid
from types import SimpleNamespace

import httpx2
import pytest

from agency.agcontext import agcontext
from agency.agdata import agdata
from agency.agpolicy import agpolicy
from agency.agskill import agskill
from agency.configs.agconfig import agconfig, dataloggerconfig, hostserverconfig, llmconfig
from agency.tools.context_tools import REMOVED_TEXT, ContextView, context_tools
from agency.engine.host_servers.host_mcp_server import HostMcpServer
from agency.engine.host_servers.host_server_manager import HostServerManager
from agency.engine.host_servers.llm_handler_server import LlmHandlerServer
from agency.harness.protocol import ATTEMPT_TOKEN_HEADER
from agency.llm.message_store import MessageBlock, with_tool_schema
from agency.llm.usage_tracker import LlmUsageTracker
from agency.observability.agdatalogger import agDataLogger, reconstruct_llm_patches


class _Logger:
    def record_event(self, *a, **kw):
        pass

    def record_stream_delta(self, *a, **kw):
        pass

    def record_llm_exchange(self, *a, **kw):
        pass


class _Backend:
    model = "test-model"

    def __init__(self):
        self.requests = []

    def dispatch(self, request):
        self.requests.append(copy.deepcopy(request))
        return {
            "message": {
                "role": "assistant",
                "blocks": [{"type": "text", "index": 0, "text": "ok"}],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "stop_reason": "stop",
        }


def _text(role, *texts):
    return {
        "role": role,
        "blocks": [{"type": "text", "index": i, "text": t} for i, t in enumerate(texts)],
    }


def _tool_use(call_id, name, arguments="{}"):
    return {
        "role": "assistant",
        "blocks": [
            {"type": "tool_use", "index": 0, "id": call_id, "name": name, "arguments": arguments}
        ],
    }


def _tool_result(call_id, body):
    return {
        "role": "tool",
        "blocks": [{"type": "tool_result", "index": 0, "tool_call_id": call_id, "text": body}],
    }


FACTS = "Alice has a blue car, Bob has a red car."
BIG_OUTPUT = "line\n" * 2000


def _history():
    """A conversation as a harness would send it."""
    return [
        _text("system", "You are an agent."),
        _text("user", FACTS),
        _text("assistant", "Noted."),
        _tool_use("c1", "bash", '{"cmd": "cat log.txt"}'),
        _tool_result("c1", BIG_OUTPUT),
        _text("user", "keep", "drop me"),
    ]


@pytest.fixture
def run():
    """A skill run's LLM handler, its ContextView and the context tools by name."""
    context = agcontext()
    server = LlmHandlerServer(
        agconfig(llmconfig(model="test-model", base_url="http://x/v1")),
        _Logger(),
        LlmUsageTracker(),
        context=context,
    )
    backend = _Backend()
    server._backend = backend
    view = ContextView(server)
    return SimpleNamespace(
        context=context, server=server, backend=backend, view=view, tools=_tools(view)
    )


def _tools(view, **options):
    """``context_tools(**options)`` as plain functions of keyword arguments,
    called the way HostMcpServer calls them."""

    def bind(tool):
        return lambda **arguments: tool(agdata(**arguments), context_view=view).to_dict()

    return SimpleNamespace(**{tool.name: bind(tool) for tool in context_tools(**options)})


def _sent(backend, n=-1):
    """The texts of every block of request *n* the backend received."""
    return [
        [b.get("text") for b in m["blocks"]]
        for m in backend.requests[n]["messages"]
        if m["role"] != "tool_schema"
    ]


def _id_of(listing: str, needle: str) -> str:
    (line,) = [line for line in listing.splitlines() if needle in line]
    return line[1 : line.index("]")]


# -- listing and reading ---------------------------------------------------------------


def test_before_any_model_call_the_context_is_empty(run):
    assert "no model call" in run.tools.list_context()["context"]


def test_list_shows_every_block_of_the_latest_request_but_not_the_tool_schema(run):
    tools = [{"type": "function", "function": {"name": "bash"}}]
    run.server.dispatch({"messages": with_tool_schema(_history(), tools)})
    listing = run.tools.list_context()["context"]
    lines = listing.splitlines()
    assert len(lines) == 7  # 6 messages, the last with two blocks; no tool_schema line
    assert "user/text" in lines[1] and FACTS in lines[1]
    assert "(protected: the system prompt)" in lines[0]
    assert "tool_use bash" in lines[3] and "protected: a tool call" in lines[3]
    assert "10,000 chars" in lines[4]


def test_read_returns_a_block_in_pages(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], "tool/tool_result")
    page = run.tools.read_context_block(id=block_id, max_chars=100)
    assert page["text"] == BIG_OUTPUT[:100]
    assert page["total_chars"] == len(BIG_OUTPUT)
    assert page["next_offset"] == 100
    last = run.tools.read_context_block(id=block_id, offset=len(BIG_OUTPUT) - 5)
    assert last["text"] == BIG_OUTPUT[-5:] and "next_offset" not in last


def test_ids_are_stable_from_call_to_call(run):
    run.server.dispatch({"messages": _history()})
    first = _id_of(run.tools.list_context()["context"], FACTS)
    run.server.dispatch({"messages": [*_history(), _text("user", "more")]})
    assert _id_of(run.tools.list_context()["context"], FACTS) == first


def test_an_unknown_or_missing_id_is_an_error(run):
    run.server.dispatch({"messages": _history()})
    assert "no block" in run.tools.read_context_block(id="zzzzzzzz")["error"]
    assert "give the id" in run.tools.read_context_block(id="")["error"]


def test_a_too_short_id_that_names_several_blocks_is_an_error(run):
    run.server.dispatch({"messages": _history()})
    hashes = [e.block_hash for e in run.view.entries()]
    # Some one-character prefix is shared by two of the seven blocks, or none is.
    shared = {h[0] for h in hashes if sum(o[0] == h[0] for o in hashes) > 1}
    if not shared:
        pytest.skip("no two blocks share a first character in this history")
    assert "more than one" in run.tools.read_context_block(id=shared.pop())["error"]


# -- editing -------------------------------------------------------------------------


def test_an_edit_lands_at_the_next_model_call_and_the_harness_s_messages_are_unchanged(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], FACTS)

    result = run.tools.edit_context(id=block_id, text="Alice has a red car, Bob has a blue car.")
    assert result["edited"] == block_id and "next model call" in result["note"]

    run.server.dispatch({"messages": _history()})  # the harness resends its own history
    assert _sent(run.backend)[1] == ["Alice has a red car, Bob has a blue car."]
    # What the harness sent is still recorded as it was.
    harness = run.server.latest_request()["harness_message_hashes"]
    record = run.server.message_store.message_structure_lookup[harness[1]]
    (payload,) = [run.server.message_store.message_block_lookup[h] for h in record["blocks"]]
    assert payload["text"] == FACTS
    # The edit is the model's.
    (key,) = run.server.message_store.message_patch_lookup
    assert run.server.message_store.patch_origin(key) == "model"
    assert "(edited)" in run.tools.list_context()["context"]


def test_an_edit_still_lands_when_the_harness_puts_a_block_in_front(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], FACTS)
    run.tools.edit_context(id=block_id, text="corrected")

    history = _history()
    history[1] = _text("user", "<system-reminder>r</system-reminder>", FACTS)
    run.server.dispatch({"messages": history})
    assert _sent(run.backend)[1] == ["<system-reminder>r</system-reminder>", "corrected"]


def test_deleting_a_block_removes_it_but_never_empties_a_message(run):
    run.server.dispatch({"messages": _history()})
    listing = run.tools.list_context()["context"]
    run.tools.edit_context(id=_id_of(listing, "drop me"), delete=True)
    run.tools.edit_context(id=_id_of(listing, "Noted."), delete=True)

    run.server.dispatch({"messages": _history()})
    sent = _sent(run.backend)
    assert sent[-1] == ["keep"]
    assert sent[2] == [REMOVED_TEXT]  # the only block of its message


def test_deleting_a_repeated_block_never_empties_a_message_it_is_alone_in(run):
    reminder = "<system-reminder>r</system-reminder>"
    # The same block twice: beside another block, and alone in a later message.
    history = [*_history()[:2], _text("user", reminder, "task"), _text("user", reminder)]
    run.server.dispatch({"messages": history})
    listing = run.tools.list_context()["context"]
    (block_id,) = {line[1 : line.index("]")] for line in listing.splitlines() if reminder in line}
    run.tools.edit_context(id=block_id, delete=True)

    run.server.dispatch({"messages": history})
    sent = _sent(run.backend)
    assert sent[-2] == [REMOVED_TEXT, "task"] and sent[-1] == [REMOVED_TEXT]


def test_a_tool_result_keeps_its_pairing_when_edited_or_deleted(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], "tool/tool_result")
    run.tools.edit_context(id=block_id, text="(log: 2000 lines, nothing relevant)")
    run.server.dispatch({"messages": _history()})
    (result,) = [
        b
        for m in run.backend.requests[-1]["messages"]
        for b in m["blocks"]
        if b["type"] == "tool_result"
    ]
    assert result == {
        "type": "tool_result",
        "index": 0,
        "tool_call_id": "c1",
        "text": "(log: 2000 lines, nothing relevant)",
    }

    run.tools.edit_context(id=block_id, delete=True)
    run.server.dispatch({"messages": _history()})
    (result,) = [
        b
        for m in run.backend.requests[-1]["messages"]
        for b in m["blocks"]
        if b["type"] == "tool_result"
    ]
    assert result["text"] == REMOVED_TEXT and result["tool_call_id"] == "c1"


def test_edit_needs_exactly_one_of_text_and_delete(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], FACTS)
    assert "either" in run.tools.edit_context(id=block_id)["error"]
    assert "either" in run.tools.edit_context(id=block_id, text="x", delete=True)["error"]


def test_revert_brings_the_original_back(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], FACTS)
    run.tools.edit_context(id=block_id, text="changed")
    assert run.tools.revert_context_edit(id=block_id)["reverted"] == block_id
    run.server.dispatch({"messages": _history()})
    assert _sent(run.backend)[1] == [FACTS]
    assert "no edit of yours" in run.tools.revert_context_edit(id=block_id)["error"]


def test_editing_a_block_again_replaces_the_earlier_edit(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], FACTS)
    run.tools.edit_context(id=block_id, text="first")
    run.tools.edit_context(id=block_id, text="second")
    run.server.dispatch({"messages": _history()})
    assert _sent(run.backend)[1] == ["second"]
    assert len(run.server.message_store.message_patch_lookup) == 1


# -- what the model may not do ---------------------------------------------------------


@pytest.mark.parametrize(
    "needle, reason",
    [
        ("system/text", "the system prompt"),
        ("tool_use bash", "a tool call"),
    ],
)
def test_protected_blocks_cannot_be_edited(run, needle, reason):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], needle)
    error = run.tools.edit_context(id=block_id, text="x")["error"]
    assert reason in error
    assert run.server.message_store.message_patch_lookup == {}


def test_signed_reasoning_cannot_be_edited(run):
    thinking = {
        "role": "assistant",
        "blocks": [{"type": "thinking", "index": 0, "text": "hmm", "signature": "sig"}],
    }
    run.server.dispatch({"messages": [_text("user", "q"), thinking]})
    block_id = _id_of(run.tools.list_context()["context"], "assistant/thinking")
    assert "signed reasoning" in run.tools.edit_context(id=block_id, delete=True)["error"]


def test_a_block_the_operator_patched_cannot_be_edited_or_reverted(run):
    store = run.server.message_store
    old = MessageBlock("user", {"type": "text", "index": 0, "text": FACTS})
    with store:
        store.add_block(old)
        store.add_patch(
            old.hash, [store.add_block(MessageBlock("user", {"type": "text", "text": "op"}))]
        )
    run.server.dispatch({"messages": _history()})
    listing = run.tools.list_context()["context"]
    line = [line for line in listing.splitlines() if "user/text" in line and "op" in line][0]
    assert "edited by the operator" in line
    block_id = line[1 : line.index("]")]
    assert "edited by the operator" in run.tools.edit_context(id=block_id, text="x")["error"]
    assert "no edit of yours" in run.tools.revert_context_edit(id=block_id)["error"]


@pytest.mark.parametrize(
    "tool_name",
    # How harnesses name a tool of the "agency" MCP server (agharness.mcp_config_for).
    ["edit_context", "mcp__agency__edit_context", "agency.edit_context", "agency/edit_context"],
)
def test_the_context_tools_own_results_stay_unless_self_edit_is_allowed(run, tool_name):
    history = [
        *_history(),
        _tool_use("c2", tool_name),
        _tool_result("c2", '{"edited": "abc"}'),
    ]
    run.server.dispatch({"messages": history})
    block_id = _id_of(run.tools.list_context()["context"], '{"edited"')
    assert "own result" in run.tools.edit_context(id=block_id, delete=True)["error"]

    permissive = _tools(run.view, allow_self_edit=True)
    assert permissive.edit_context(id=block_id, delete=True)["action"] == "deleted"


def test_the_edit_limit_counts_edits_and_reverts(run):
    limited = _tools(run.view, max_edits=2)
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(limited.list_context()["context"], FACTS)
    limited.edit_context(id=block_id, text="a")
    limited.revert_context_edit(id=block_id)
    assert "limit of 2" in limited.edit_context(id=block_id, text="b")["error"]


# -- stale edits and the end of the skill run -----------------------------------------


def test_an_edit_the_harness_stopped_sending_is_reported_back(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], FACTS)
    run.tools.edit_context(id=block_id, text="corrected")
    run.server.dispatch({"messages": _history()})
    assert "warning" not in run.tools.list_context()  # the edit was used

    # The harness compacted its history: the edited block is gone.
    run.server.dispatch({"messages": [_text("user", "summary of earlier work")]})
    listing = run.tools.list_context()
    assert block_id in listing["warning"] and "no effect" in listing["warning"]


def test_an_edit_made_after_the_latest_request_is_not_stale_yet(run):
    run.server.dispatch({"messages": _history()})
    block_id = _id_of(run.tools.list_context()["context"], FACTS)
    run.tools.edit_context(id=block_id, text="corrected")
    assert "warning" not in run.tools.list_context()


def test_closing_withdraws_edits_unless_they_persist(run):
    run.server.dispatch({"messages": _history()})
    listing = run.tools.list_context()["context"]
    run.tools.edit_context(id=_id_of(listing, FACTS), text="for this run only")
    persistent = _tools(run.view, persist=True)
    persistent.edit_context(id=_id_of(listing, "Noted."), text="kept")

    run.view.close()
    run.view.close()  # idempotent

    store = run.context.message_store
    (key,) = store.message_patch_lookup
    assert store.patch_origin(key) == "model"
    assert [
        b["text"]
        for b in store.materialize([store.add_message(_text("assistant", "Noted."))])[0]["blocks"]
    ] == ["kept"]


def test_a_tool_without_a_context_view_is_an_error():
    (list_context, *_rest) = context_tools()
    assert "error" in list_context(agdata()).to_dict()


# -- through the host MCP server -------------------------------------------------------


def test_the_host_mcp_server_hands_the_tools_its_context_view(run):
    skill = agskill(name="s", prompt="p", add_host_mcp_tools=context_tools())
    mcp = HostMcpServer(
        SimpleNamespace(), skill, SimpleNamespace(), _Logger(), context_view=run.view
    )
    mcp.build_app()
    run.server.dispatch({"messages": _history()})

    result = asyncio.run(mcp._mcp_server.call_tool("list_context", {}))
    assert not result.is_error
    listing = json.loads(result.content[0].text)["context"]
    assert FACTS in listing


# -- the whole host path ---------------------------------------------------------------


@pytest.mark.parametrize("persist", [False, True])
def test_a_model_edit_through_the_real_host_server_reaches_the_backend_and_the_record(
    tmp_path, persist
):
    """HostServerManager over the host UDS: request 1, the model calls edit_context on
    the host MCP server, request 2 (the harness resending its own history) reaches the
    backend edited, the SQLite record says the model made the patch, and when the
    skill run's services stop, the edit is withdrawn unless it persists."""
    uds_path = f"/tmp/ctx_tools_{uuid.uuid4().hex[:8]}.sock"
    db_path = str(tmp_path / "agent.db")
    cfg = agconfig(
        llmconfig(model="test-model", base_url="http://x/v1"),
        hostserverconfig(uds_path=uds_path),
        dataloggerconfig(db_path=db_path),
    )
    logger = agDataLogger(cfg)
    logger.start()
    fake_agent = SimpleNamespace(
        agname="agent-1",
        agconfig=cfg,
        harness="claude_code",
        data_logger=logger,
        llm_usage_tracker=LlmUsageTracker(),
    )
    sandbox = SimpleNamespace(
        ensure_gpu_acquired=lambda _agname, *, is_cancelled: None,
        current_gpu_ids=lambda: None,
    )
    context = agcontext()
    skill = agskill(
        name="s", prompt="p", policy=agpolicy(), add_host_mcp_tools=context_tools(persist=persist)
    )
    manager = HostServerManager(fake_agent, sandbox, skill, SimpleNamespace(), context=context)
    backend = _Backend()
    manager.llm_handler_server._backend = backend
    history = [_text("user", "task"), _text("user", FACTS)]
    try:
        manager.start()
        manager.bind_attempt_token("tok")
        with httpx2.Client(
            transport=httpx2.HTTPTransport(uds=uds_path), base_url="http://localhost"
        ) as client:

            def dispatch(messages):
                response = client.post(
                    "/llm/dispatch",
                    json={"messages": messages},
                    headers={ATTEMPT_TOKEN_HEADER: "tok"},
                )
                assert response.status_code == 200

            dispatch(history)
            mcp = manager.host_mcp_server._mcp_server
            listed = asyncio.run(mcp.call_tool("list_context", {}))
            block_id = _id_of(json.loads(listed.content[0].text)["context"], FACTS)
            edited = asyncio.run(
                mcp.call_tool("edit_context", {"id": block_id, "text": "Alice has a red car."})
            )
            assert not edited.is_error
            dispatch(history)
        context.message_hash_list = manager.llm_handler_server.get_main_history("task")
    finally:
        manager.clear_attempt_token("tok")
        manager.stop()
        logger.flush()

    assert _sent(backend, 0)[1] == [FACTS]
    assert _sent(backend, 1)[1] == ["Alice has a red car."]

    connection = sqlite3.connect(db_path)
    first, second = [
        row[0] for row in connection.execute("SELECT call_label FROM exchanges ORDER BY id")
    ]
    assert reconstruct_llm_patches(connection, first) == []
    (patch,) = reconstruct_llm_patches(connection, second)
    assert patch["origin"] == "model"
    assert patch["original"]["text"] == FACTS
    assert patch["replacement"][0]["text"] == "Alice has a red car."

    # After the run: the agent's history shows the edit only if it persists.
    texts = [m["blocks"][0]["text"] for m in context.get_resolved_transcript()]
    assert texts[1] == ("Alice has a red car." if persist else FACTS)
