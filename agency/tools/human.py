from __future__ import annotations
from ..agtool import agtool
from ..agdata import agdata


def make_ask_human(agname: str) -> agtool:
    """Return an ask_human tool bound to the given agent's agname."""

    def fn(arg: agdata) -> agdata:
        question = str(arg._data.get("question", ""))

        def _find_agent():
            from ..agent import agent as Agent
            for a in Agent.all():
                if a.agname == agname:
                    return a
            return None

        a = _find_agent()
        prev_state = dict(a._ui_state) if a else {}
        if a:
            a._set_ui_state("human", skill=prev_state.get("skill"))

        from .. import agui
        if agui._active is not None:
            reply = agui._active.ask_human(agname, question)
        else:
            print(f"\n[{agname}] asks: {question}")
            reply = input("> ")

        if a:
            a._set_ui_state(
                prev_state.get("state", "skill"),
                skill=prev_state.get("skill"),
                tool=prev_state.get("tool"),
            )
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
    )
