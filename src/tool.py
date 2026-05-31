from __future__ import annotations
from typing import Callable
from .agdata import agdata


class tool:
    """A named callable tool that an LLM can invoke via function calling.

    Provides the OpenAI tool schema and executes when called.
    """

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[[agdata], agdata],
        params: dict | None = None,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.params = params or {"type": "object", "properties": {}}

    def __call__(self, arg: agdata) -> agdata:
        return self.fn(arg)

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params,
            },
        }

    def __repr__(self) -> str:
        return f"tool(name={self.name!r})"
