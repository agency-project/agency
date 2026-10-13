from __future__ import annotations
import copy
from typing import TYPE_CHECKING

from .llm.message_store import MessageStore

if TYPE_CHECKING:
    from concurrent.futures import Future


class agcontext:
    """Persistent conversation state owned by an agent and passed through each skill run.

    Fields
    ------
    message_hash_list   : ordered message hashes of the most recent skill run
                           (replaced each run, not accumulated)
    message_store       : a MessageStore (see llm/message_store.py) holding the
                           three lookups that ``message_hash_list`` resolves
                           against, all keyed by content hash:
                           ``message_structure_lookup`` (message hash -> record
                           of its role and ordered block hashes),
                           ``message_block_lookup`` (block hash -> block
                           payload) and ``message_patch_lookup`` (original
                           block hash -> ordered hashes of the blocks that
                           replace it), plus ``message_patch_origin`` (who
                           made each patch); the lookups are kept across skill runs
    harness_sessions    : per-harness session continuity, e.g.
                           {"claude_code": {"session_id": ..., "blob_b64": ...}}
    retained_messages   : ordered user/system messages retained independently
                           of a harness transcript or restorable session
    harness_message_cursors
                        : last retained-message sequence incorporated into
                           each stateful harness session
    """

    def __init__(
        self,
        harness_sessions: "dict[str, dict] | None" = None,
        retained_messages: "list[dict] | None" = None,
        harness_message_cursors: "dict[str, int] | None" = None,
        _future: "Future[agcontext] | None" = None,
        *,
        message_hash_list: "list[str] | None" = None,
        message_store: "MessageStore | None" = None,
    ) -> None:
        self.message_hash_list: "list[str]" = (
            list(message_hash_list) if message_hash_list is not None else []
        )
        self.message_store: MessageStore = (
            message_store if message_store is not None else MessageStore()
        )
        self.harness_sessions = harness_sessions if harness_sessions is not None else {}
        self.retained_messages = retained_messages if retained_messages is not None else []
        self.harness_message_cursors = (
            harness_message_cursors if harness_message_cursors is not None else {}
        )
        self._future = _future

    # ------------------------------------------------------------------
    # Future / lazy-resolution support
    # ------------------------------------------------------------------

    def is_pending(self) -> bool:
        return self._future is not None

    def resolve_prev_dependencies(self) -> None:
        """Block until the pending future resolves and merge its state into self."""
        future = self._future
        if future is None:
            return
        prev_ctx = future.result()
        self.message_hash_list = prev_ctx.message_hash_list
        self.message_store = prev_ctx.message_store
        self.harness_sessions = prev_ctx.harness_sessions
        self.retained_messages = prev_ctx.retained_messages
        self.harness_message_cursors = prev_ctx.harness_message_cursors
        self._future = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_resolved_transcript(self) -> "list[dict]":
        """Block until pending, then rebuild the message list from
        ``message_hash_list``, with ``message_patch_lookup`` superseding the
        original blocks."""
        self.resolve_prev_dependencies()
        return self.message_store.materialize(self.message_hash_list)

    @classmethod
    def from_messages(cls, messages: "list[dict]", **fields) -> "agcontext":
        """An agcontext whose ``message_hash_list`` is *messages*, interned into a
        fresh ``message_store``."""
        context = cls(**fields)
        context.set_transcript(messages)
        return context

    def set_transcript(self, messages: "list[dict]") -> None:
        """Replace ``message_hash_list`` with the hashes of *messages*, interned
        into ``message_store`` (whose lookups only ever gain rows)."""
        self.message_hash_list = self.message_store.add_messages(messages)

    def message_state(self) -> dict:
        """JSON-serializable form of ``message_hash_list`` and the three lookups,
        as stored in a checkpoint's ``state.json``."""
        store = self.message_store
        with store:
            return {
                "message_hash_list": list(self.message_hash_list),
                "message_block_lookup": dict(store.message_block_lookup),
                "message_structure_lookup": dict(store.message_structure_lookup),
                "message_patch_lookup": {k: list(v) for k, v in store.message_patch_lookup.items()},
                "message_patch_origin": dict(store.message_patch_origin),
            }

    @classmethod
    def message_state_fields(cls, state: dict) -> dict:
        """Constructor keywords for a ``message_state`` -- or for a checkpoint
        from before the lookups, which kept a plain message list as "history"."""
        if "message_hash_list" in state:
            return {
                "message_hash_list": list(state["message_hash_list"]),
                "message_store": MessageStore(
                    dict(state.get("message_block_lookup") or {}),
                    dict(state.get("message_structure_lookup") or {}),
                    {k: list(v) for k, v in (state.get("message_patch_lookup") or {}).items()},
                    dict(state.get("message_patch_origin") or {}),
                ),
            }
        store = MessageStore()
        return {
            "message_hash_list": store.add_messages(state.get("history", [])),
            "message_store": store,
        }

    def append_retained_message(self, message: dict) -> int:
        """Append one validated, JSON-serializable retained message."""
        entry = copy.deepcopy(message)
        sequence = entry.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise ValueError("retained message sequence must be a positive integer")
        if entry.get("type") != "message":
            raise ValueError("retained message type must be 'message'")
        if entry.get("role") not in {"user", "system"}:
            raise ValueError("retained message role must be 'user' or 'system'")
        if not isinstance(entry.get("content"), str):
            raise ValueError("retained message content must be a string")
        if self.retained_messages and sequence <= int(self.retained_messages[-1]["sequence"]):
            raise ValueError("retained message sequences must be strictly increasing")
        self.retained_messages.append(entry)
        return sequence

    def pending_retained_messages(self, harness: str) -> "list[dict]":
        """Return retained entries not captured by *harness*'s session."""
        cursor = int(self.harness_message_cursors.get(str(harness), 0))
        return copy.deepcopy(
            [entry for entry in self.retained_messages if int(entry["sequence"]) > cursor]
        )

    def advance_retained_cursor(self, harness: str, sequence: int) -> None:
        """Monotonically mark retained messages through *sequence* as session-backed."""
        harness = str(harness)
        sequence = int(sequence)
        current = int(self.harness_message_cursors.get(harness, 0))
        if sequence < current:
            raise ValueError("retained message cursor cannot move backwards")
        self.harness_message_cursors[harness] = sequence

    def copy(self) -> "agcontext":
        """Return a copy of the resolved agcontext (blocks if pending). The copy's
        ``message_store`` has its own three lookups, so adding or removing rows
        in either agcontext never touches the other; the stored payloads
        themselves are shared, and are never mutated."""
        self.resolve_prev_dependencies()
        result = agcontext(
            message_hash_list=self.message_hash_list,
            message_store=self.message_store.copy(),
            harness_sessions=copy.deepcopy(self.harness_sessions),
            retained_messages=copy.deepcopy(self.retained_messages),
            harness_message_cursors=copy.deepcopy(self.harness_message_cursors),
        )
        # Execution evidence is host-local metadata, never conversation data
        # or checkpoint state. Keeping it through copies records dependencies
        # on a completed context without retaining scheduler requests.
        for key in ("_execution_id", "_execution_agent_id"):
            if key in self.__dict__:
                setattr(result, key, self.__dict__[key])
        return result

    def __repr__(self) -> str:
        pending = " (pending)" if self._future is not None else ""
        return (
            f"agcontext(message_hash_list={len(self.message_hash_list)}"
            f"  harnesses={sorted(self.harness_sessions)}"
            f"  retained={len(self.retained_messages)}"
            f"){pending}"
        )
