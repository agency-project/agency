from __future__ import annotations
import inspect
import time
from typing import Callable
from .agdata import agdata, agerror
from .agutil import format_exception
from .agconfig import DynamicConfigParam, _AgConfigViewBase


# Exists only to register agtool's config fields (via __set_name__ at import
# time). Reads use a throwaway instance -- _AgToolFields(agconfig) -- since
# __init__ does nothing but (optionally) store an agconfig; there's no
# persistent agtool instance to hang descriptors on for reading.
class _AgToolFields:
    timeout_s = DynamicConfigParam(
        "agtool", default=1800
    )  # Historical ceiling on tool execution time -- no longer enforced now
    # that tool calls run directly in the caller's own process/thread (see
    # agtool.__call__), kept only as a config field other code may still read.
    output_offload_chars = DynamicConfigParam(
        "agtool", default=40_000
    )  # minimum floor for tool-output offloading
    offload_id_prefix_len = DynamicConfigParam(
        "agtool", default=12
    )  # Chars of the tool_call_id kept when naming an offloaded-output file.

    def __init__(self, agconfig=None) -> None:
        self._agconfig = agconfig


class agToolConfig(_AgConfigViewBase):
    """View over an agConfig for pre-setting agtool tunables in one call::

        cfg = agConfig(agToolConfig(timeout_s=60))

    See `_AgConfigViewBase` in agconfig.py for the shared mechanics.
    """

    _OWNER = "agtool"


class agtool:
    """A named callable tool that an LLM can invoke via function calling.

    Provides the OpenAI tool schema and executes when called, directly in
    the calling thread/process -- whichever process actually holds the real
    `fn` closure and whatever host state it references (a sandbox object, a
    live resource pool, etc.), never shipped across a process boundary to
    run elsewhere.

    Logging
    -------
    Every invocation calls ``self.log(arg, result, elapsed_ms)``, which can
    be overridden per-tool by passing a custom *log_fn* to the constructor.
    """

    def __init__(
        self,
        name: str,
        description: str,
        fn: "Callable[..., agdata]",
        params: dict | None = None,
        log_fn: "Callable[[agtool, agdata, agdata, int], None] | None" = None,
        run_in_subprocess: bool = True,
        persistent_vars: "dict[str, Callable[[], object]] | None" = None,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.params = params or {"type": "object", "properties": {}}
        self._log_fn = log_fn
        self.run_in_subprocess = run_in_subprocess
        self.persistent_vars = persistent_vars or {}

    # ------------------------------------------------------------------
    # Pickle support — a custom log_fn may hold state that doesn't survive
    # serialization (needed if a caller ever ships an agtool elsewhere,
    # e.g. cloudpickle-ing `fn` into a container process).
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "fn": self.fn,
            "params": self.params,
            "_log_fn": self._log_fn,
            "run_in_subprocess": self.run_in_subprocess,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.run_in_subprocess = state.get("run_in_subprocess", True)

    # ------------------------------------------------------------------
    # Logging — override by supplying log_fn to __init__
    # ------------------------------------------------------------------

    def log(self, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        """Called after every invocation. Only does something if a custom
        log_fn was supplied to __init__."""
        if self._log_fn is not None:
            self._log_fn(self, arg, result, elapsed_ms)

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def __call__(self, arg: agdata, timeout: int | None = None, **context: object) -> agdata:
        # Always runs directly in the calling thread/process -- no
        # subprocess isolation, no pickling `fn` across a process boundary
        # (a host-authored closure often captures host-only state, like a
        # sandbox object or a live resource pool, that has no meaning
        # anywhere else). `timeout` is accepted for call-site compatibility
        # but not enforced -- there is no separate process/thread to bound
        # without reintroducing the isolation this deliberately avoids;
        # the caller controls blocking behavior instead (e.g. ask_human).
        t0 = time.monotonic()
        try:
            wanted = self._wanted_context(context)
            result = self.fn(arg, **wanted)
        except Exception as e:
            result = agerror(format_exception(e))
        self.log(arg, result, int((time.monotonic() - t0) * 1000))
        return result

    def _wanted_context(self, context: dict) -> dict:
        if not context:
            return {}
        params = list(inspect.signature(self.fn).parameters.values())[1:]
        names = {p.name for p in params}
        return {k: v for k, v in context.items() if k in names}

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
        return f"agtool(name={self.name!r})"


__all__ = ["agtool", "agToolConfig", "_AgToolFields"]
