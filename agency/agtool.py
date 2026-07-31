from __future__ import annotations
import atexit
import threading
import time
import multiprocessing as _mp
from concurrent.futures import (
    ProcessPoolExecutor,
    TimeoutError as _FutureTimeoutError,
    BrokenExecutor,
)
import json
from typing import TYPE_CHECKING, Callable
from .agdata import agdata, agerror
from .profiler import agprof
from .agutil import format_exception
from .agtype import type_hint_to_string_type, get_return_tool_description_prompt
from .agconfig import GlobalConfigParam, DynamicConfigParam, _AgConfigViewBase

if TYPE_CHECKING:
    from .aglog import aglog
    from .agterm import agterm
    from .agsandbox import agSandbox
    from .agconfig import agConfig

# ---------------------------------------------------------------------------
# Process pool — workers are created lazily on first tool call and scale up
# to match concurrent demand (one worker per in-flight tool call, up to 256).
# Uses "spawn" start method to avoid fork-in-multithreaded-process deadlocks.
# ---------------------------------------------------------------------------
_pool: ProcessPoolExecutor | None = None
_pool_lock: threading.Lock = threading.Lock()


def _ignore_sigint_in_worker() -> None:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


# Exists only to register agtool's config fields (via __set_name__ at import
# time). Other code in this file needing the same hardcoded value -- e.g.
# agtool.__call__'s own timeout fallback -- reads the descriptor's frozen
# default directly (_AgToolFields.timeout_s.default) instead of going through
# a separate plain constant, so there's a single source of truth.
# Reads use a throwaway instance -- _AgToolFields(agconfig) -- since __init__
# does nothing but (optionally) store an agconfig; there's no persistent
# agtool instance to hang descriptors on for reading.
class _AgToolFields:
    pool_max_workers = GlobalConfigParam(
        "agtool", default=256
    )  # Max worker processes in the tool executor pool; one per in-flight tool call.
    timeout_s = DynamicConfigParam(
        "agtool", default=1800
    )  # Default ceiling on tool execution time (seconds). Prevents a crashed or
    # hung worker process from blocking an agent thread forever via future.result();
    # agents can pass "timeout": <seconds> in tool arguments to override per-call.
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


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                max_workers = _AgToolFields().pool_max_workers
                _pool = ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=_mp.get_context("spawn"),
                    initializer=_ignore_sigint_in_worker,
                )
    return _pool


def shutdown_tool_pool(*, wait: bool = False, cancel_futures: bool = True) -> None:
    global _pool
    with _pool_lock:
        pool = _pool
        _pool = None
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=cancel_futures)


atexit.register(shutdown_tool_pool)


def _process_worker(fn_bytes: bytes, arg_bytes: bytes) -> bytes:
    """Worker entry-point: unpickle the tool fn and call it."""
    import cloudpickle
    import pickle

    fn: Callable[[agdata], agdata] = cloudpickle.loads(fn_bytes)
    arg: agdata = pickle.loads(arg_bytes)
    return pickle.dumps(fn(arg))


class agtool:
    """A named callable tool that an LLM can invoke via function calling.

    Provides the OpenAI tool schema and executes when called.  Each call is
    offloaded to a dedicated worker process so CPU-bound tools cannot block
    the agent thread pool and the GIL cannot starve other agents.

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
        run_in_subprocess: bool = True,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.params = params or {"type": "object", "properties": {}}
        self._log_fn = log_fn
        self.run_in_subprocess = run_in_subprocess
        self._term: "agterm | None" = None
        self._aglog: "aglog  | None" = None

    # ------------------------------------------------------------------
    # Pickle support — exclude loggers; they hold locks / file handles and
    # are not needed in the worker process.
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
        self._term = None
        self._aglog = None

    # ------------------------------------------------------------------
    # Logger attachment
    # ------------------------------------------------------------------

    def attach_logger(self, term: "agterm", aglog: "aglog") -> None:
        """Wire up terminal and structured file logging for this tool."""
        self._term = term
        self._aglog = aglog

    # ------------------------------------------------------------------
    # Logging — override by supplying log_fn to __init__
    # ------------------------------------------------------------------

    def log_start(self, arg: agdata) -> None:
        """Called immediately before invocation."""
        if self._term is not None:
            in_keys = list(arg._data.keys())
            self._term.log("TOOL ▶   ", f"{self.name}  in={in_keys}")

    def log(self, arg: agdata, result: agdata, elapsed_ms: int) -> None:
        """Called after every invocation.  Default: log input/output key names."""
        if self._log_fn is not None:
            self._log_fn(self, arg, result, elapsed_ms)
            return
        if self._term is not None:
            in_keys = list(arg._data.keys())
            out_keys = list(result._data.keys())
            self._term.log(
                "TOOL ✓   ",
                f"{self.name}  in={in_keys}  out={out_keys}  ({elapsed_ms}ms)",
            )
        if self._aglog is not None:
            self._aglog._tool_call(self.name, arg.to_dict(), result.to_dict(), elapsed_ms)

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def __call__(self, arg: agdata, timeout: int | None = None) -> agdata:
        self.log_start(arg)
        t0 = time.monotonic()

        if not self.run_in_subprocess:
            # Run directly in the calling thread — no subprocess isolation or
            # timeout needed (caller controls blocking behaviour, e.g. ask_human).
            try:
                result = self.fn(arg)
            except Exception as e:
                result = agerror(format_exception(e))
            self.log(arg, result, int((time.monotonic() - t0) * 1000))
            return result

        import cloudpickle
        import pickle

        fn_bytes = cloudpickle.dumps(self.fn)
        arg_bytes = pickle.dumps(arg)
        effective_timeout = timeout if timeout is not None else _AgToolFields.timeout_s.default
        try:
            result_bytes = (
                _get_pool()
                .submit(_process_worker, fn_bytes, arg_bytes)
                .result(timeout=effective_timeout)
            )
        except _FutureTimeoutError:
            elapsed = int((time.monotonic() - t0) * 1000)
            result = agerror(f"tool timed out after {effective_timeout}s")
            self.log(arg, result, elapsed)
            return result
        except BrokenExecutor:
            # Worker process died (OOM kill, crash). Reset the pool so future
            # calls get fresh workers, then report the error.
            global _pool
            with _pool_lock:
                _pool = None
            elapsed = int((time.monotonic() - t0) * 1000)
            result = agerror("tool worker process died unexpectedly")
            self.log(arg, result, elapsed)
            return result
        result = pickle.loads(result_bytes)
        self.log(arg, result, int((time.monotonic() - t0) * 1000))
        return result

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


# ---------------------------------------------------------------------------
# Return-output tool builders
# ---------------------------------------------------------------------------


def make_return_output_tools(schema) -> list[dict]:
    """Build one typed tool per output field from the schema.

    Each tool is named ``return_<field>`` and has a single parameter named
    after the field itself with the correct JSON Schema type.
    schema is an agdata instance; accessed via duck typing.
    """
    tools = []
    for field, hint in schema._data.items():
        json_type = type_hint_to_string_type(hint)
        tool_desc, value_desc = get_return_tool_description_prompt(field, hint)
        value_schema: dict = {"type": json_type, "description": value_desc}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"return_{field}",
                    "description": tool_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {field: value_schema},
                        "required": [field],
                    },
                },
            }
        )
    return tools


def _make_return_output_tool(schema) -> list[dict]:
    """Alias kept for test compatibility — returns the full per-field tool list."""
    return make_return_output_tools(schema)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def dispatch_tools(
    tool_calls: list[dict],
    toolkit: dict,
    messages: list[dict],
    sandbox: "agSandbox",
    skill_name: str,
    state_fn: "Callable | None",
    live_messages_fn: "Callable | None",
    full_history_fn: "Callable | None",
    term: "agterm | None",
    tool_offload_chars: "int | None" = None,
    agconfig: "agConfig | None" = None,
) -> None:
    """Execute all tool calls from one LLM response, appending results to messages.

    Mutates toolkit in place if the read tool is lazily injected due to a large
    tool output being offloaded — the caller derives wire-format schemas from the
    toolkit each iteration, so the injection is automatically visible to the LLM.
    """
    _state_fn = state_fn
    _live_fn = live_messages_fn
    _hist_fn = full_history_fn
    _term = term
    _fields = _AgToolFields(agconfig)
    if tool_offload_chars is None:
        tool_offload_chars = _fields.output_offload_chars
    _default_tool_timeout = _fields.timeout_s

    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        fn_args = tc["function"]["arguments"]
        tc_id = tc["id"]
        # Ensure arguments is valid JSON before it goes back into history.
        # A malformed string (truncated generation, Python repr, etc.) causes
        # vLLM to crash on the next request when it re-parses the history.
        try:
            json.loads(fn_args)
        except (json.JSONDecodeError, TypeError):
            fn_args = "{}"
            tc["function"]["arguments"] = fn_args

        t = toolkit.get(fn_name)
        if t is None:
            with agprof.span(f"tool:{fn_name}"):
                agprof.annotate(outcome="failure", error_type="unknown_tool")
                if _term:
                    _term.log("TOOL ✗   ", f"{fn_name}  → unknown tool")
                result_content = json.dumps({"error": f"unknown tool: {fn_name}"})
        else:
            try:
                if _state_fn:
                    _state_fn("tool", skill=skill_name, tool=fn_name)
                # Let the agent specify a custom timeout (seconds) via a
                # "timeout" key in the tool arguments.
                _tool_timeout: int | None = None
                try:
                    _parsed = json.loads(fn_args)
                    if isinstance(_parsed.get("timeout"), int):
                        _tool_timeout = _parsed["timeout"]
                except (json.JSONDecodeError, TypeError, AttributeError):
                    # Malformed/non-dict arguments -- fall back to the default timeout below.
                    pass
                if _tool_timeout is None:
                    _tool_timeout = _default_tool_timeout
                with agprof.span(f"tool:{fn_name}"):
                    _tool_result = t(agdata.from_json(fn_args), timeout=_tool_timeout)
                    _tool_error = _tool_result._data.get("error")
                    agprof.annotate(
                        outcome="failure" if _tool_error else "success",
                        error_type="tool_error" if _tool_error else None,
                    )
                    result_content = _tool_result.to_json()
                if _state_fn:
                    _state_fn("skill", skill=skill_name)
                # Offload large tool outputs regardless of whether the tool
                # itself uses the sandbox — fetch_paper and other add_tools have
                # run_in_subprocess=False but can still produce huge outputs that
                # bloat the context.
                if len(result_content) > tool_offload_chars:
                    safe_id = tc_id.replace("-", "")[: _fields.offload_id_prefix_len]
                    offload_path = f"/workspace/long_tool_call_outputs/{fn_name}_{safe_id}.txt"
                    try:
                        try:
                            file_body = json.loads(result_content).get("content", result_content)
                        except (json.JSONDecodeError, AttributeError):
                            file_body = result_content
                        sandbox.write_file(offload_path, file_body)
                        result_content = json.dumps(
                            {
                                "note": f"Output was too large and has been saved to {offload_path}. Use the read tool to access it."
                            }
                        )
                        # Inject read into the toolkit so the LLM can access the file.
                        # The caller derives wire-format schemas from the toolkit each
                        # iteration, so this is automatically visible on the next step.
                        if "read" not in toolkit:
                            from .tools import make_read as _make_read

                            toolkit["read"] = _make_read(sandbox)
                    except Exception as _e:
                        print(
                            f"[agtool] WARNING: failed to offload large tool output to {offload_path}: {_e}"
                        )
                # stop() (hibernate) runs after every tool call regardless
                # of run_in_subprocess -- releasing the sandbox's runtime
                # slot between calls isn't specific to subprocess-isolated
                # tools. Rollback no longer happens at this granularity: a
                # tool call's own success or failure no longer decides
                # whether the sandbox gets checkpointed or discarded --
                # only the skill as a whole does, at its own teardown (see
                # agskill.py). The one exception here: if this tool call
                # left background work still running inside the sandbox
                # (e.g. a bash `cmd &`), stop() must NOT run yet -- it kills
                # every process inside, which would end that work before the
                # agent ever gets a chance to check on it in a later tool
                # call. Skipping here just defers the hibernate; the next
                # tool call that finds nothing pending will catch it up.
                if not sandbox._has_pending_background_work():
                    sandbox.stop()
            except Exception as e:
                if _state_fn:
                    _state_fn("skill", skill=skill_name)
                result_content = json.dumps({"error": format_exception(e)})
                # An exception escaping tool dispatch is handled the same
                # way as an ordinary result at this granularity -- hibernate,
                # not discard. Same pending-work deferral as above.
                if not sandbox._has_pending_background_work():
                    sandbox.stop()
        tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result_content}
        messages.append(tool_msg)
        if _live_fn:
            _live_fn(messages[1:])
        if _hist_fn:
            _hist_fn(tool_msg)
