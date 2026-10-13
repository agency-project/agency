"""Lossless content-addressed storage of LLM messages, plus a patch overlay.

One scheme serves both the SQLite data collector and ``agcontext``:

* a **block** is one ``{"role": role, **block}`` payload, keyed by the SHA-256
  of its *complete* canonical JSON. Nothing is left out of the key (tool-call
  arguments, tool-result content, ``citations``, ``index``, ...), so two
  blocks that differ in any field are two entries, and a stored payload is
  always exactly what was seen;
* a **message** is a record ``{"role", "blocks": [block hashes], "extra"?}``
  keyed the same way, which keeps message boundaries and any message-level
  fields (a flat chain of blocks cannot tell two messages from one);
* a **history** is just an ordered list of message hashes.

The patch lookup sits on top: it maps the hash of an original block to the
ordered hashes of the blocks that stand in for it (none = the block is
removed). Both sides are ordinary rows of ``message_block_lookup``. Patches are
applied whenever a message list is *rebuilt* (``MessageStore.materialize``), which
is also how a request is built before it is sent, never by rewriting the stored
originals, so the original tables stay a faithful record
of what the harness sent and a patch can be dropped again.

A patch matches a block by its hash, and failing that by its hash with
``index`` left out (``position_free_hash``): a harness that inserts a block in
front of a patched one (Claude Code's ``<system-reminder>`` blocks, say) shifts
its ``index`` without changing the block, and the patch still applies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import functools
import threading
from typing import Any, Iterable, Mapping

# A block that itself has a top-level "role" key cannot be flattened next to the
# message role without one overwriting the other, so it is nested instead.
_NESTED = "agency_nested_block"

# The role of the one message that carries a request's tool definitions, its
# tool_choice and its other non-message fields (see with_tool_schema).
TOOL_SCHEMA_ROLE = "tool_schema"


def content_hash(payload) -> str:
    """SHA-256 over the complete canonical JSON of *payload*."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def block_payload(role, block: dict) -> dict:
    """The stored form of one block: the message role plus the whole block."""
    if "role" in block:
        return {"role": role, _NESTED: block}
    return {"role": role, **block}


def block_of(payload: dict) -> dict:
    """Inverse of ``block_payload``: the block, without its role."""
    if _NESTED in payload:
        return payload[_NESTED]
    return {key: value for key, value in payload.items() if key != "role"}


def position_free_hash(payload: dict) -> str:
    """``content_hash`` of a stored block payload with its ``index`` left out, so
    the same block at another position in its message has the same key."""
    if _NESTED in payload:
        nested = {k: v for k, v in payload[_NESTED].items() if k != "index"}
        return content_hash({**payload, _NESTED: nested})
    return content_hash({k: v for k, v in payload.items() if k != "index"})


# Who made a patch (``MessageStore.message_patch_origin``). A patch with no entry
# was made by the operator, through ``agent.patch_history``.
OPERATOR_ORIGIN = "operator"
MODEL_ORIGIN = "model"


def with_tool_schema(
    messages: "list[dict]",
    tools: "list | None" = None,
    tool_choice: Any = None,
    extra: "dict | None" = None,
) -> "list[dict]":
    """*messages* with the ``tool_schema`` message in front: the harness's *tools*,
    its *tool_choice* and its *extra* fields. Nothing to carry adds no message.

    A message is any data exchanged between a harness and an LLM backend, so a
    request's tools, ``tool_choice`` and other fields are messages too. A harness
    adapter writes them into the request's ``messages`` with this function and an
    LLM backend reads them back with ``split_tool_schema`` just before building its
    own wire format; hashing, patching and recording see only the message list.
    The message has ``role: "tool_schema"`` and one block per item, holding every
    key it was given:

    * ``{"type": "tool_definition", "tool": <the whole tool dict>}`` per tool;
    * ``{"type": "tool_choice", "value": <tool_choice>}``;
    * ``{"type": "request_extra", "value": {<every other field>}}``."""
    blocks: "list[dict]" = [{"type": "tool_definition", "tool": tool} for tool in tools or ()]
    if tool_choice is not None:
        blocks.append({"type": "tool_choice", "value": tool_choice})
    if extra:
        blocks.append({"type": "request_extra", "value": dict(extra)})
    if not blocks:
        return list(messages)
    for index, block in enumerate(blocks):
        block["index"] = index
    return [{"role": TOOL_SCHEMA_ROLE, "blocks": blocks}, *messages]


def split_tool_schema(messages: "list[dict]") -> "tuple[list[dict], list | None, Any, dict]":
    """Inverse of ``with_tool_schema``: ``(chat messages, tools, tool_choice, extra)``.
    ``tools`` is ``None`` and ``extra`` is ``{}`` when the request carries none."""
    chat: "list[dict]" = []
    tools: list = []
    tool_choice = None
    extra: dict = {}
    for message in messages:
        if message.get("role") != TOOL_SCHEMA_ROLE:
            chat.append(message)
            continue
        for block in message.get("blocks") or ():
            kind = block.get("type")
            if kind == "tool_definition":
                tools.append(block["tool"])
            elif kind == "tool_choice":
                tool_choice = block["value"]
            elif kind == "request_extra":
                extra.update(block["value"])
    return chat, tools or None, tool_choice, extra


class MessageBlock:
    """One content block of a message in the agency block format, together with
    the role of the message that carries it. Immutable.

    ``hash`` is the key of the block's row in ``message_block_lookup``: the SHA-256
    of the whole payload ``{"role", **fields}``, so blocks that differ in any field,
    or in role, are different blocks. ``MessageBlock("user", {...})`` copies its
    fields, so later changes to the dict passed in do not reach the block."""

    __slots__ = ("_role", "_fields", "_hash")

    def __init__(self, role: str, fields: "Mapping[str, Any]") -> None:
        if not isinstance(role, str) or not role:
            raise TypeError("a MessageBlock's role must be a non-empty string")
        if isinstance(fields, MessageBlock) or not isinstance(fields, Mapping):
            raise TypeError(
                f"a MessageBlock's fields must be a mapping, not {type(fields).__name__}"
            )
        object.__setattr__(self, "_role", role)
        object.__setattr__(self, "_fields", copy.deepcopy(dict(fields)))
        object.__setattr__(self, "_hash", content_hash(block_payload(role, self._fields)))

    @classmethod
    def from_payload(cls, payload: "Mapping[str, Any]") -> "MessageBlock":
        """Inverse of ``payload``: a block from a row of ``message_block_lookup``."""
        return cls(payload["role"], block_of(dict(payload)))

    def __setattr__(self, name, value) -> None:
        raise AttributeError("a MessageBlock is immutable")

    @property
    def role(self) -> str:
        return self._role

    @property
    def type(self) -> "str | None":
        return self._fields.get("type")

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def payload(self) -> dict:
        """The row stored in ``message_block_lookup`` (a copy)."""
        return copy.deepcopy(block_payload(self._role, self._fields))

    def to_dict(self) -> dict:
        """The block in the agency block format, without its role (a copy)."""
        return copy.deepcopy(self._fields)

    def __eq__(self, other) -> bool:
        return isinstance(other, MessageBlock) and self._hash == other._hash

    def __hash__(self) -> int:
        return hash(self._hash)

    def __repr__(self) -> str:
        return f"MessageBlock({self._role!r}, {self._fields!r})"


def _renumber(blocks: "list[dict]") -> None:
    """Keep ``index`` consistent with position after blocks were swapped."""
    for position, block in enumerate(blocks):
        if "index" in block:
            block["index"] = position


def _exclusive(method):
    """Run a ``MessageStore`` method holding the store's lock."""

    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


class MessageStore:
    """The three lookup tables, all keyed by content hash:

    * ``message_block_lookup``: block hash -> block payload (the content);
    * ``message_structure_lookup``: message hash -> message record, which names
      the blocks of one message in order (the structure);
    * ``message_patch_lookup``: original block hash -> ordered list of the
      hashes of the blocks that replace it (an empty list removes the block).

    ``message_patch_origin`` says who made a patch (patch key -> ``"model"``);
    a key it does not name was made by the operator.

    The dicts are shared, not copied, when passed in. Stored payloads are never
    mutated; every read hands out a copy.

    Access is exclusive: every method holds one reentrant lock, so a patch
    added, removed or cleared from another thread is never seen halfway through
    a request or a rebuild. Code that reads the lookups directly, or needs several
    calls to see one state, does so inside ``with store:``."""

    def __init__(
        self,
        message_block_lookup: "dict | None" = None,
        message_structure_lookup: "dict | None" = None,
        message_patch_lookup: "dict | None" = None,
        message_patch_origin: "dict | None" = None,
    ) -> None:
        self.message_block_lookup: "dict[str, dict]" = (
            message_block_lookup if message_block_lookup is not None else {}
        )
        self.message_structure_lookup: "dict[str, dict]" = (
            message_structure_lookup if message_structure_lookup is not None else {}
        )
        self.message_patch_lookup: "dict[str, list[str]]" = (
            message_patch_lookup if message_patch_lookup is not None else {}
        )
        self.message_patch_origin: "dict[str, str]" = (
            message_patch_origin if message_patch_origin is not None else {}
        )
        self._lock = threading.RLock()

    def __enter__(self) -> "MessageStore":
        self._lock.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self._lock.release()

    @_exclusive
    def __len__(self) -> int:
        return len(self.message_structure_lookup)

    # -- writing -----------------------------------------------------------

    @_exclusive
    def add_block(self, block: MessageBlock) -> str:
        """Store *block* in ``message_block_lookup`` (once, whatever the number of
        calls) and return its hash."""
        if not isinstance(block, MessageBlock):
            raise TypeError(f"add_block takes a MessageBlock, not {type(block).__name__}")
        if block.hash not in self.message_block_lookup:
            self.message_block_lookup[block.hash] = block.payload
        return block.hash

    @_exclusive
    def add_message(self, message) -> str:
        """Store a message and its blocks and return the message's hash. A message
        that is not in the agency shape is kept whole."""
        blocks = message.get("blocks") if isinstance(message, dict) else None
        if not isinstance(blocks, list) or not all(isinstance(b, Mapping) for b in blocks):
            # Not the agency shape: keep it whole rather than lose anything.
            record: dict = {"opaque": message}
        else:
            role = message.get("role")
            record = {
                "role": role,
                "blocks": [self.add_block(MessageBlock(role, b)) for b in blocks],
            }
            extra = {k: v for k, v in message.items() if k not in ("role", "blocks")}
            if extra:
                record["extra"] = extra
        digest = content_hash(record)
        with self._lock:
            if digest not in self.message_structure_lookup:
                self.message_structure_lookup[digest] = copy.deepcopy(record)
        return digest

    @_exclusive
    def add_messages(self, messages: "Iterable") -> "list[str]":
        return [self.add_message(message) for message in messages]

    # -- patches -----------------------------------------------------------

    @_exclusive
    def add_patch(
        self, old_hash: str, new_hashes: "list[str]", *, origin: str = OPERATOR_ORIGIN
    ) -> str:
        """Replace every block whose hash is *old_hash* with the blocks *new_hashes*,
        in order (none removes it). Both sides are rows of ``message_block_lookup``;
        *old_hash* need not be there yet, but every hash in *new_hashes* must be
        (``add_block`` stores a new one). Only an *old_hash* that is a row also
        matches the same block at another ``index``. *origin* records who made the
        patch. Returns *old_hash*, the key."""
        if not isinstance(old_hash, str) or not all(isinstance(h, str) for h in new_hashes):
            raise TypeError("a patch links a block hash to a list of block hashes")
        missing = [h for h in new_hashes if h not in self.message_block_lookup]
        if missing:
            raise KeyError(f"not in message_block_lookup (add_block them first): {missing}")
        self.message_patch_lookup[old_hash] = list(new_hashes)
        if origin == OPERATOR_ORIGIN:
            self.message_patch_origin.pop(old_hash, None)
        else:
            self.message_patch_origin[old_hash] = origin
        return old_hash

    @_exclusive
    def remove_patch(self, key: str) -> bool:
        self.message_patch_origin.pop(key, None)
        return self.message_patch_lookup.pop(key, None) is not None

    @_exclusive
    def clear_patches(self) -> None:
        self.message_patch_lookup.clear()
        self.message_patch_origin.clear()

    @_exclusive
    def patch_origin(self, key: str) -> str:
        return self.message_patch_origin.get(key, OPERATOR_ORIGIN)

    @_exclusive
    def _position_free_patch_keys(self) -> "dict[str, str]":
        """Position-free hash -> patch key, for every patch whose original is a row."""
        return {
            position_free_hash(self.message_block_lookup[key]): key
            for key in self.message_patch_lookup
            if key in self.message_block_lookup
        }

    @_exclusive
    def _patch_key(self, block_hash: str, position_free: "dict[str, str]") -> "str | None":
        if block_hash in self.message_patch_lookup:
            return block_hash
        if not position_free:
            return None
        return position_free.get(position_free_hash(self.message_block_lookup[block_hash]))

    @_exclusive
    def patch_keys_for(self, block_hashes: "Iterable[str]") -> "dict[str, str]":
        """block hash -> the key of the patch that replaces it, for each of
        *block_hashes* that some patch matches (by hash, or position-free)."""
        position_free = self._position_free_patch_keys()
        out = {}
        for block_hash in block_hashes:
            key = self._patch_key(block_hash, position_free)
            if key is not None:
                out[block_hash] = key
        return out

    # -- reading -----------------------------------------------------------

    @_exclusive
    def _block_hashes(self, digest: str) -> "list[str]":
        return list(self.message_structure_lookup[digest].get("blocks") or [])

    @_exclusive
    def role(self, digest: str):
        record = self.message_structure_lookup[digest]
        if "opaque" in record:
            opaque = record["opaque"]
            return opaque.get("role") if isinstance(opaque, dict) else None
        return record.get("role")

    @_exclusive
    def payloads(self, digest: str) -> "list[tuple[str, dict]]":
        """``(block hash, stored payload)`` for each block of a message."""
        return [(h, self.message_block_lookup[h]) for h in self._block_hashes(digest)]

    @_exclusive
    def _message(
        self,
        digest: str,
        position_free: "dict[str, str] | None" = None,
        applied: "set[str] | None" = None,
    ) -> "dict | None":
        """The message as a dict, with the patch lookup applied. ``None`` if the
        patches removed every block it had."""
        if position_free is None:
            position_free = self._position_free_patch_keys()
        record = self.message_structure_lookup[digest]
        if "opaque" in record:
            return copy.deepcopy(record["opaque"])
        blocks: "list[dict]" = []
        patched = False
        for block_hash in record["blocks"]:
            key = self._patch_key(block_hash, position_free)
            if key is None:
                blocks.append(block_of(self.message_block_lookup[block_hash]))
            else:
                patched = True
                if applied is not None:
                    applied.add(key)
                blocks.extend(
                    block_of(self.message_block_lookup[h]) for h in self.message_patch_lookup[key]
                )
        if patched:
            if not blocks and record["blocks"]:
                return None
            _renumber(blocks)
        message = {"role": record["role"], **record.get("extra", {}), "blocks": blocks}
        return copy.deepcopy(message)

    @_exclusive
    def materialize(self, digests: "Iterable[str]", applied: "set[str] | None" = None) -> list:
        """Rebuild the message list, patched blocks superseding the originals. The
        key of every patch that replaced a block is added to *applied*, if given."""
        position_free = self._position_free_patch_keys()
        out = []
        for digest in digests:
            message = self._message(digest, position_free, applied)
            if message is not None:
                out.append(message)
        return out

    @_exclusive
    def copy(self) -> "MessageStore":
        # Stored payloads are immutable by convention, so sharing them is safe.
        return MessageStore(
            dict(self.message_block_lookup),
            dict(self.message_structure_lookup),
            {key: list(value) for key, value in self.message_patch_lookup.items()},
            dict(self.message_patch_origin),
        )
