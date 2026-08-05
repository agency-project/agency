from __future__ import annotations
import queue
import threading
from ..agtool import agtool
from ..agdata import agdata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASK_ID_HEX_LENGTH = 12  # Number of hex characters taken from a UUID4 to form a unique ask_human request ID sent to the web UI emitter.

_TIMEOUT_REPLY = "[no human available — timed out]"
DEFAULT_TIMEOUT_S = 300  # 5 minutes


def ask_human_and_wait(agname: str, question: str, timeout_s: "float | None", ag=None) -> str:
    """Ask a human operator *question* and block for their reply -- the
    actual implementation, shared by the host-side `ask_human` agtool
    below (which looks up its own `ag` by agname, an older convention: it
    was only ever given the name at construction time) and
    `agmcp_server.py`'s `ask_human` MCP tool (which already has a live
    `ag` reference from its own token registry, so it skips the lookup).
    Pass ``timeout_s=None`` to wait indefinitely. Pass ``ag=None`` to skip
    UI-state save/restore when no live agent object is available."""
    prev_state, prev_skill, prev_tool = ag._state.snapshot() if ag else (None, None, None)
    if ag:
        ag._set_ui_state("human", skill=prev_skill)

    from .. import agwebui as _agwebui

    if _agwebui._active is not None:
        import uuid as _uuid

        ask_id = _uuid.uuid4().hex[:ASK_ID_HEX_LENGTH]
        reply = _agwebui._active.emitter.ask_human(agname, ask_id, question, timeout_s=timeout_s)
    else:
        print(f"\n[{agname}] asks: {question}")
        q: queue.SimpleQueue[str] = queue.SimpleQueue()

        def _read() -> None:
            try:
                q.put(input("> "))
            except EOFError:
                q.put(_TIMEOUT_REPLY)

        threading.Thread(target=_read, daemon=True).start()
        try:
            reply = q.get(timeout=timeout_s)
        except queue.Empty:
            print(f"[{agname}] ask_human timed out after {timeout_s:.0f}s")
            reply = _TIMEOUT_REPLY
        # timeout_s=None blocks forever — Empty is never raised

    if ag:
        ag._set_ui_state(prev_state or "skill", skill=prev_skill, tool=prev_tool)
    return reply


def make_ask_human(agname: str, timeout_s: float | None = DEFAULT_TIMEOUT_S) -> agtool:
    """Return an ask_human tool bound to the given agent's agname.

    Pass ``timeout_s=None`` to wait indefinitely for a human reply.
    """

    def fn(arg: agdata) -> agdata:
        question = str(arg._data.get("question", ""))

        def _find_agent():
            from ..agent import agent as Agent

            for a in Agent.all():
                if a.agname == agname:
                    return a
            return None

        reply = ask_human_and_wait(agname, question, timeout_s, ag=_find_agent())
        return agdata(reply=reply)

    return agtool(
        name="ask_human",
        description=(
            "Ask the human operator a question and wait for their reply. "
            "Use when you need information, a decision, or clarification "
            "that only a human can provide. Prefer autonomous action; "
            "only ask when genuinely blocked."
        ),
        fn=fn,
        params={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the human operator.",
                }
            },
            "required": ["question"],
        },
        run_in_subprocess=False,
    )
