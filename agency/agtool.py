from __future__ import annotations
import time
from typing import TYPE_CHECKING, Callable
from .agdata import agdata

if TYPE_CHECKING:
    from .aglog import aglog
    from .agterm import agterm


class agtool:
    """A named callable tool that an LLM can invoke via function calling.

    Provides the OpenAI tool schema and executes when called.

    Logging
    -------
    Call ``attach_logger(term, log)`` after construction (done automatically by
    ``agent.__init__``) to wire up terminal and file logging.  Every invocation
    then calls ``self.log(arg, result, elapsed_ms)``, which can be overridden
    per-tool by passing a custom *log_fn* to the constructor.
    """

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[[agdata], agdata],
        params: dict | None = None,
        log_fn: "Callable[[agtool, agdata, agdata, int], None] | None" = None,
    ):
        self.name        = name
        self.description = description
        self.fn          = fn
        self.params      = params or {"type": "object", "properties": {}}
        self._log_fn     = log_fn
        self._term:  "agterm | None" = None
        self._aglog: "aglog  | None" = None

    def attach_logger(self, term: "agterm", aglog: "aglog") -> None:
        """Wire up terminal and structured file logging for this tool."""
        self._term  = term
        self._aglog = aglog

    # ------------------------------------------------------------------
    # Logging — override by supplying log_fn to __init__
    # ------------------------------------------------------------------

    def log(self, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        """Called after every invocation.  Default: log input/output key names."""
        if self._log_fn is not None:
            self._log_fn(self, arg, result, elapsed_ms)
            return
        if self._term is not None:
            in_keys  = list(arg._data.keys())
            out_keys = list(result._data.keys())
            self._term.log(
                "TOOL ✓   ",
                f"{self.name}  in={in_keys}  out={out_keys}  ({elapsed_ms}ms)",
            )
        if self._aglog is not None:
            self._aglog._tool_call(
                self.name, arg.to_dict(), result.to_dict(), elapsed_ms
            )

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def __call__(self, arg: agdata) -> agdata:
        t0     = time.monotonic()
        result = self.fn(arg)
        self.log(arg, result, int((time.monotonic() - t0) * 1000))
        return result

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.params,
            },
        }

    def __repr__(self) -> str:
        return f"agtool(name={self.name!r})"
