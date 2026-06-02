from __future__ import annotations
import json
from typing import TYPE_CHECKING, Callable
import openai
from .agdata import agdata
from .agtool import agtool

if TYPE_CHECKING:
    from .agterm import agterm

_TYPE_MAP: dict[str, type] = {
    "str": str, "int": int, "float": float,
    "bool": bool, "list": list, "dict": dict,
}


class agskill:
    """A named skill with its own system prompt and a self-contained ReAct loop.

    input_schema / output_schema are agdata objects whose keys define required
    fields and whose values are either a recognised type name ("str", "int",
    "float", "bool", "list", "dict") or a plain description string.  Both are
    serialised and appended to the system prompt so the LLM knows the contract.

    Input is validated before the loop runs.  Output is validated after each
    final (non-tool-call) LLM response; on failure a correction message is
    injected and the loop retries up to max_retries times.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[agtool] | None = None,
        input_schema: agdata | None = None,
        output_schema: agdata | None = None,
        output_validator: "Callable[[agdata], list[str]] | None" = None,
        max_retries: int = 3,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools          # None → inherit from agent
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.output_validator = output_validator   # extra check beyond type schema
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        parts = [self.system_prompt]
        if self.input_schema is not None:
            parts.append(f"\nInput JSON format:\n{self.input_schema.to_json()}")
        if self.output_schema is not None:
            parts.append(
                f"\nOutput JSON format (respond ONLY with this JSON):\n"
                f"{self.output_schema.to_json()}"
            )
        return "\n".join(parts)

    def _check_schema(self, data: agdata, schema: agdata) -> list[str]:
        """Return a list of error strings; empty list means the data is valid."""
        errors: list[str] = []
        for key, hint in schema._data.items():
            if key not in data._data:
                errors.append(f"missing required field '{key}'")
                continue
            type_name = hint if isinstance(hint, str) else None
            if type_name and type_name.lower() in _TYPE_MAP:
                expected = _TYPE_MAP[type_name.lower()]
                actual = data._data[key]
                if not isinstance(actual, expected):
                    errors.append(
                        f"field '{key}': expected {type_name}, "
                        f"got {type(actual).__name__}"
                    )
        return errors

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def run(
        self,
        llm_config: dict,
        input: agdata,
        history: agdata,
        agent_tools: list[agtool],
        max_steps: int = 10,
        term: "agterm | None" = None,
        _is_continuation: bool = False,
    ) -> tuple[agdata, agdata, list[dict]]:
        """Run the ReAct loop.

        Returns (result_agdata, updated_history_agdata, history_delta).
        history_delta is the list of new messages added during this skill's
        execution (user input → tool calls / results → final answer).
        The system_prompt (+ schemas) is prepended to every call but is NOT
        persisted in history.

        When *_is_continuation* is True (outer monitoring loop re-entry), input
        schema validation is skipped so process-status ping messages can flow
        through without matching the skill's declared input schema.
        """
        # --- Input validation ------------------------------------------------
        if self.input_schema is not None and not _is_continuation:
            errors = self._check_schema(input, self.input_schema)
            if errors:
                sys_msg = {"role": "system", "content": self._build_system_prompt()}
                return agdata(error=f"input schema error: {errors}"), history, [sys_msg]

        active_tools: list[agtool] = self.tools if self.tools is not None else agent_tools
        tool_map = {t.name: t for t in active_tools}
        openai_tools = [t.to_openai_tool() for t in active_tools] or None

        client = openai.OpenAI(
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", None),
        )

        history_msgs: list[dict] = list(history._data.get("messages", []))
        n_before = len(history_msgs)
        messages: list[dict] = (
            [{"role": "system", "content": self._build_system_prompt()}]
            + history_msgs
            + [{"role": "user", "content": input.to_json()}]
        )

        retries_left = self.max_retries

        for _ in range(max_steps):
            kwargs: dict = dict(
                model=llm_config.get("model", "gpt-4o"),
                messages=messages,
            )
            if openai_tools:
                kwargs["tools"] = openai_tools

            if term:
                term.log("LLM      ", f"model={llm_config.get('model','?')}  messages={len(messages)}")
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            msg_dict: dict = {"role": "assistant"}
            if msg.content:
                msg_dict["content"] = msg.content
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(msg_dict)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    t = tool_map.get(tc.function.name)
                    if t is None:
                        if term:
                            term.log("TOOL     ", f"{tc.function.name}  → unknown tool")
                        result_content = json.dumps({"error": f"unknown tool: {tc.function.name}"})
                    else:
                        if term:
                            term.log("TOOL     ", f"{tc.function.name}({tc.function.arguments[:80]})")
                        try:
                            result_content = t(agdata.from_json(tc.function.arguments)).to_json()
                        except Exception as e:
                            result_content = json.dumps({"error": str(e)})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_content})

            else:
                # --- Parse final answer --------------------------------------
                content = msg.content or "{}"
                # Strip markdown code fences that some models add despite instructions
                stripped = content.strip()
                if stripped.startswith("```"):
                    stripped = stripped[stripped.find("\n") + 1:] if "\n" in stripped else stripped[3:]
                    if stripped.endswith("```"):
                        stripped = stripped[:-3]
                    content = stripped.strip()
                try:
                    result = agdata.from_json(content)
                except (json.JSONDecodeError, TypeError):
                    result = agdata(result=content)

                # --- Output validation + retry --------------------------------
                if self.output_schema is not None:
                    errors = self._check_schema(result, self.output_schema)
                    if not errors and self.output_validator is not None:
                        errors = self.output_validator(result)
                    if errors:
                        if retries_left > 0:
                            retries_left -= 1
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Output schema errors: {errors}. "
                                    f"Respond ONLY with valid JSON matching exactly: "
                                    f"{self.output_schema.to_json()}"
                                ),
                            })
                            continue  # retry in the same loop
                        updated_history = agdata(messages=messages[1:])
                        return (
                            agdata(error=f"output schema error after retries: {errors}"),
                            updated_history,
                            [messages[0]] + messages[1:][n_before:],
                        )

                updated_history = agdata(messages=messages[1:])
                return result, updated_history, [messages[0]] + messages[1:][n_before:]

        updated_history = agdata(messages=messages[1:])
        return agdata(error="max_steps exceeded"), updated_history, [messages[0]] + messages[1:][n_before:]

    def __repr__(self) -> str:
        return f"agskill(name={self.name!r})"
