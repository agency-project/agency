"""Host tools that let the model edit its own LLM context.

``context_tools()`` returns host MCP tools a skill opts into with
``agskill(..., add_host_mcp_tools=context_tools())``. They drive the same
``message_patch_lookup`` as ``agent.patch_history`` (see docs/Message_Patches.md),
from inside the skill run, so the model -- not the operator -- decides what its next
LLM call sees. Nothing in the harness changes: the harness keeps sending its own
history, and ``LlmHandlerServer`` swaps the edited blocks in on the way out.

The model addresses blocks by a short id, the first characters of the block's hash
in the latest request the harness sent. The ids are stable: the same block keeps
its id from one call to the next. See docs/Context_Tools.md.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from .agdata import agdata, agerror
from .agtool import agtool
from .llm.message_store import (
    MODEL_ORIGIN,
    TOOL_SCHEMA_ROLE,
    MessageBlock,
    block_of,
)

if TYPE_CHECKING:
    from .engine.host_servers.llm_handler_server import LlmHandlerServer

LIST_CONTEXT = "list_context"
READ_CONTEXT_BLOCK = "read_context_block"
EDIT_CONTEXT = "edit_context"
REVERT_CONTEXT_EDIT = "revert_context_edit"
CONTEXT_TOOL_NAMES = frozenset(
    {LIST_CONTEXT, READ_CONTEXT_BLOCK, EDIT_CONTEXT, REVERT_CONTEXT_EDIT}
)

ID_LENGTH = 8
PREVIEW_CHARS = 100
REMOVED_TEXT = "[removed by context edit]"

# Block types the model can never edit: signed reasoning (an edit breaks the
# provider's signature check) and tool calls (a changed or missing tool_use
# orphans its tool_result).
_PROTECTED_TYPES = {
    "thinking": "signed reasoning",
    "redacted_thinking": "signed reasoning",
    "tool_use": "a tool call",
}


def _base_tool_name(name: str) -> str:
    """``list_context`` from ``mcp__agency-host-mcp-server__list_context`` and the
    other ways harnesses prefix an MCP tool's name."""
    return name.rsplit("__", 1)[-1].rsplit(".", 1)[-1].rsplit("/", 1)[-1]


def _block_text(block: dict) -> str:
    """What a block says, as plain text, for previews and ``read_context_block``."""
    kind = block.get("type")
    if kind in ("text", "tool_result", "thinking") and isinstance(block.get("text"), str):
        return block["text"]
    if kind == "tool_use":
        return f"{block.get('name')}({block.get('arguments')})"
    return json.dumps(block, sort_keys=True, default=str)


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 1] + "…"


class _Entry:
    """One block of the latest request, as the context tools see it."""

    __slots__ = (
        "id",
        "block_hash",
        "role",
        "original",
        "shown",
        "patch_key",
        "status",
        "protected",
        "only_block",
    )

    def __init__(self, **fields) -> None:
        for name in self.__slots__:
            setattr(self, name, fields[name])


class ContextView:
    """A skill run's own LLM context, for the context tools: the blocks of the
    latest request the harness made, and the model's edits to them.

    Made by ``HostServerManager`` for every skill run. Edits go straight into the
    run's ``message_store`` (``origin="model"``), so they land at the next LLM
    call. ``close()`` withdraws the ones not made to persist, when the skill run
    ends."""

    def __init__(self, llm_handler_server: "LlmHandlerServer") -> None:
        self._server = llm_handler_server
        self._lock = threading.Lock()
        # patch key -> {"seq": seq of the latest request when made, "persist": bool}
        self._edits: "dict[str, dict]" = {}
        self._calls = 0

    # -- reading -------------------------------------------------------------------

    def entries(self, *, allow_self_edit: bool = False) -> "list[_Entry]":
        """The blocks of the latest request, with edits shown, in order. Blocks of
        the ``tool_schema`` message are left out."""
        latest = self._server.latest_request()
        if latest is None:
            return []
        store = self._server.message_store
        with store:
            rows: "list[tuple[str, str, str, dict]]" = []  # (message hash, role, hash, block)
            for message_hash in latest["harness_message_hashes"]:
                record = store.message_structure_lookup[message_hash]
                role = record.get("role")
                if "opaque" in record or role == TOOL_SCHEMA_ROLE:
                    continue
                for block_hash, payload in store.payloads(message_hash):
                    rows.append((message_hash, role, block_hash, block_of(payload)))
            patch_keys = store.patch_keys_for(row[2] for row in rows)
            context_call_ids = {
                block.get("id")
                for _m, _r, _h, block in rows
                if block.get("type") == "tool_use"
                and _base_tool_name(str(block.get("name") or "")) in CONTEXT_TOOL_NAMES
            }
            message_sizes: "dict[str, int]" = {}
            for message_hash, *_ in rows:
                message_sizes[message_hash] = message_sizes.get(message_hash, 0) + 1
            entries = []
            for message_hash, role, block_hash, block in rows:
                key = patch_keys.get(block_hash)
                origin = store.patch_origin(key) if key is not None else None
                if key is None:
                    shown, status = [block], ""
                else:
                    shown = [
                        block_of(store.message_block_lookup[h])
                        for h in store.message_patch_lookup[key]
                    ]
                    status = "deleted" if not shown else "edited"
                    if origin != MODEL_ORIGIN:
                        status += " by the operator"
                protected = self._protection(
                    role,
                    block,
                    origin,
                    context_call_ids,
                    allow_self_edit=allow_self_edit,
                )
                entries.append(
                    _Entry(
                        id=block_hash[:ID_LENGTH],
                        block_hash=block_hash,
                        role=role,
                        original=block,
                        shown=shown,
                        patch_key=key,
                        status=status,
                        protected=protected,
                        only_block=message_sizes[message_hash] == 1,
                    )
                )
        return entries

    @staticmethod
    def _protection(
        role, block: dict, origin, context_call_ids: set, *, allow_self_edit: bool
    ) -> "str | None":
        """Why the model may not edit *block*, or ``None`` if it may."""
        kind = block.get("type")
        if role == "system":
            return "the system prompt"
        if kind in _PROTECTED_TYPES:
            return _PROTECTED_TYPES[kind]
        if kind == "tool_result":
            if block.get("tool_call_id") in context_call_ids and not allow_self_edit:
                return "a context tool's own result"
            if not isinstance(block.get("text"), str):
                return "a tool result that is not plain text"
        if origin is not None and origin != MODEL_ORIGIN:
            return "edited by the operator"
        return None

    def find(self, block_id: str, *, allow_self_edit: bool = False) -> "_Entry":
        block_id = (block_id or "").strip().strip("[]")
        if not block_id:
            raise LookupError("give the id of a block, as list_context shows it")
        matches = [
            e
            for e in self.entries(allow_self_edit=allow_self_edit)
            if e.block_hash.startswith(block_id)
        ]
        if not matches:
            raise LookupError(f"no block {block_id!r} in your context; call list_context")
        if len({e.block_hash for e in matches}) > 1:
            raise LookupError(f"{block_id!r} names more than one block; give more of its id")
        return matches[0]

    def stale_edits(self) -> "list[str]":
        """Ids of the model's edits that the latest request did not use although it
        was made after them: the harness no longer sends that block (it compacted
        or changed it), so the edit has no effect."""
        latest = self._server.latest_request()
        if latest is None:
            return []
        store = self._server.message_store
        with self._lock:
            return [
                key[:ID_LENGTH]
                for key, edit in self._edits.items()
                if edit["seq"] < latest["seq"]
                and key in store.message_patch_lookup
                and key not in latest["applied_patch_keys"]
            ]

    # -- editing -------------------------------------------------------------------

    def edit(
        self,
        block_id: str,
        text: "str | None",
        *,
        persist: bool = False,
        allow_self_edit: bool = False,
        max_edits: "int | None" = None,
    ) -> "_Entry":
        """Replace a block's text with *text*, or remove the block if *text* is
        ``None``. Returns the entry edited."""
        entry = self.find(block_id, allow_self_edit=allow_self_edit)
        if entry.protected is not None:
            raise PermissionError(f"block {entry.id} is {entry.protected} and cannot be edited")
        replacement = self._replacement(entry, text)
        store = self._server.message_store
        with self._lock:
            self._count_call(max_edits)
            latest = self._server.latest_request()
            key = entry.patch_key if entry.patch_key is not None else entry.block_hash
            with store:
                store.add_block(MessageBlock(entry.role, entry.original))
                store.add_patch(
                    key,
                    [store.add_block(MessageBlock(entry.role, b)) for b in replacement],
                    origin=MODEL_ORIGIN,
                )
            self._edits[key] = {"seq": latest["seq"] if latest else 0, "persist": persist}
        return entry

    @staticmethod
    def _replacement(entry: "_Entry", text: "str | None") -> "list[dict]":
        block = entry.original
        kind = block.get("type")
        if kind == "tool_result":
            # A tool_result must stay, paired with its tool_use; only its text changes.
            return [{**block, "text": REMOVED_TEXT if text is None else text}]
        if text is None:
            # A message left with no blocks would be dropped, which can break the
            # provider's turn order; leave a marker instead.
            if entry.only_block:
                return [{"type": "text", "index": block.get("index", 0), "text": REMOVED_TEXT}]
            return []
        if kind == "text":
            return [{**block, "text": text}]
        return [{"type": "text", "index": block.get("index", 0), "text": text}]

    def revert(self, block_id: str, *, max_edits: "int | None" = None) -> "_Entry":
        """Withdraw the model's edit of a block."""
        entry = self.find(block_id, allow_self_edit=True)
        store = self._server.message_store
        if entry.patch_key is None or store.patch_origin(entry.patch_key) != MODEL_ORIGIN:
            raise LookupError(f"block {entry.id} has no edit of yours to revert")
        with self._lock:
            self._count_call(max_edits)
            store.remove_patch(entry.patch_key)
            self._edits.pop(entry.patch_key, None)
        return entry

    def _count_call(self, max_edits: "int | None") -> None:
        if max_edits is not None and self._calls >= max_edits:
            raise PermissionError(
                f"the limit of {max_edits} context edits for this task is used up"
            )
        self._calls += 1

    def close(self) -> None:
        """Withdraw this run's edits that were not made to persist. Idempotent."""
        store = self._server.message_store
        with self._lock:
            for key, edit in list(self._edits.items()):
                if not edit["persist"] and store.patch_origin(key) == MODEL_ORIGIN:
                    store.remove_patch(key)
                del self._edits[key]


# -- the tools ---------------------------------------------------------------------


def _render(entry: "_Entry") -> str:
    shown = "\n".join(_block_text(b) for b in entry.shown)
    kind = entry.original.get("type")
    if kind == "tool_use":
        kind = f"tool_use {entry.original.get('name')}"
    notes = [
        n for n in (entry.status, f"protected: {entry.protected}" if entry.protected else "") if n
    ]
    suffix = f" ({'; '.join(notes)})" if notes else ""
    if entry.status.startswith("deleted"):
        return f"[{entry.id}] {entry.role}/{kind}{suffix}"
    return f"[{entry.id}] {entry.role}/{kind}, {len(shown):,} chars{suffix}: {_preview(shown)}"


def _stale_note(view: ContextView) -> "str | None":
    stale = view.stale_edits()
    if not stale:
        return None
    return (
        "these edits no longer match anything sent to the model (the harness "
        f"compacted or changed those blocks), so they have no effect: {', '.join(stale)}"
    )


def context_tools(
    *, max_edits: "int | None" = 50, allow_self_edit: bool = False, persist: bool = False
) -> "list[agtool]":
    """Host MCP tools that let the model list, read and edit its own LLM context.

    * *max_edits* caps the ``edit_context``/``revert_context_edit`` calls of one
      skill run (each edit makes the provider re-read the prompt from that block
      on); ``None`` is no cap.
    * *allow_self_edit* lets the model edit the results of these tools themselves
      (by default they stay, as a record of its edits; the calls always stay).
    * *persist* keeps the model's edits after the skill run ends (in the agent's
      ``agcontext``, so later runs see them too); by default they are withdrawn.

    The model can never edit the system prompt, tool definitions, signed reasoning,
    tool calls, or a block the operator patched with ``agent.patch_history``.
    """

    def missing_view() -> agdata:
        return agerror("context tools need a skill run's host servers")

    def list_context(arg: agdata, context_view: "ContextView | None" = None) -> agdata:
        if context_view is None:
            return missing_view()
        entries = context_view.entries(allow_self_edit=allow_self_edit)
        if not entries:
            return agdata(context="(no model call has been made yet)")
        result = {"context": "\n".join(_render(e) for e in entries)}
        note = _stale_note(context_view)
        if note:
            result["warning"] = note
        return agdata(**result)

    def read_context_block(arg: agdata, context_view: "ContextView | None" = None) -> agdata:
        if context_view is None:
            return missing_view()
        try:
            entry = context_view.find(arg._data.get("id"), allow_self_edit=allow_self_edit)
        except LookupError as e:
            return agerror(str(e))
        text = "\n".join(_block_text(b) for b in entry.shown)
        offset = max(int(arg._data.get("offset") or 0), 0)
        max_chars = max(int(arg._data.get("max_chars") or 4000), 1)
        chunk = text[offset : offset + max_chars]
        result = {"id": entry.id, "text": chunk, "total_chars": len(text)}
        if offset + max_chars < len(text):
            result["next_offset"] = offset + max_chars
        return agdata(**result)

    def edit_context(arg: agdata, context_view: "ContextView | None" = None) -> agdata:
        if context_view is None:
            return missing_view()
        text = arg._data.get("text")
        delete = bool(arg._data.get("delete"))
        if delete == (text is not None):
            return agerror("give either text (to replace the block) or delete=true")
        try:
            entry = context_view.edit(
                arg._data.get("id"),
                None if delete else str(text),
                persist=persist,
                allow_self_edit=allow_self_edit,
                max_edits=max_edits,
            )
        except (LookupError, PermissionError) as e:
            return agerror(str(e))
        result = {
            "edited": entry.id,
            "action": "deleted" if delete else "replaced",
            "note": "takes effect from your next model call",
        }
        note = _stale_note(context_view)
        if note:
            result["warning"] = note
        return agdata(**result)

    def revert_context_edit(arg: agdata, context_view: "ContextView | None" = None) -> agdata:
        if context_view is None:
            return missing_view()
        try:
            entry = context_view.revert(arg._data.get("id"), max_edits=max_edits)
        except (LookupError, PermissionError) as e:
            return agerror(str(e))
        return agdata(
            reverted=entry.id, note="the original block is back from your next model call"
        )

    block_id = {"type": "string", "description": "the block's id, as list_context shows it"}
    return [
        agtool(
            name=LIST_CONTEXT,
            description=(
                "List the blocks of your own context -- everything sent to the model on "
                "each call -- one line each: id, role/type, size, and a preview. Use the "
                "ids with read_context_block, edit_context and revert_context_edit."
            ),
            fn=list_context,
        ),
        agtool(
            name=READ_CONTEXT_BLOCK,
            description="Read one block of your context in full (in pages, for long ones).",
            fn=read_context_block,
            params={
                "type": "object",
                "properties": {
                    "id": block_id,
                    "offset": {"type": "integer", "description": "first character to return"},
                    "max_chars": {"type": "integer", "description": "characters to return (4000)"},
                },
                "required": ["id"],
            },
        ),
        agtool(
            name=EDIT_CONTEXT,
            description=(
                "Edit your own context: replace one block's text, or delete the block. "
                "From your next model call on, the model sees the edited version instead "
                "of the original. Use it to drop tool output you no longer need, replace "
                "a long block with a short summary, or correct something outdated."
            ),
            fn=edit_context,
            params={
                "type": "object",
                "properties": {
                    "id": block_id,
                    "text": {"type": "string", "description": "the block's new text"},
                    "delete": {"type": "boolean", "description": "remove the block instead"},
                },
                "required": ["id"],
            },
        ),
        agtool(
            name=REVERT_CONTEXT_EDIT,
            description="Undo your edit of one block, so the model sees the original again.",
            fn=revert_context_edit,
            params={"type": "object", "properties": {"id": block_id}, "required": ["id"]},
        ),
    ]


__all__ = ["ContextView", "context_tools", "CONTEXT_TOOL_NAMES"]
