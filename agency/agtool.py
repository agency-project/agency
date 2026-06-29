from __future__ import annotations
import threading
import time
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as _FutureTimeoutError, BrokenExecutor
import json
from typing import TYPE_CHECKING, Callable
from .agdata import agdata, agerror
from .agutil import format_exception
from .agtype import _hint_to_json_type, _return_tool_descriptions

if TYPE_CHECKING:
    from .aglog import aglog
    from .agterm import agterm
    from .agsandbox import agSandbox

# Default ceiling on tool execution time. Prevents a crashed or hung worker
# process from blocking an agent thread forever via future.result().
# Set to 1800s (30 min) to accommodate long-running bash commands; agents
# can pass "timeout": <seconds> in tool arguments to override per-call.
TOOL_TIMEOUT_S: int = 1800

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOOL_POOL_MAX_WORKERS = 256  # Maximum number of worker processes in the tool executor pool; one worker per in-flight tool call.

# ---------------------------------------------------------------------------
# Process pool — workers are created lazily on first tool call and scale up
# to match concurrent demand (one worker per in-flight tool call, up to 256).
# Uses "spawn" start method to avoid fork-in-multithreaded-process deadlocks.
# ---------------------------------------------------------------------------
_pool:      ProcessPoolExecutor | None = None
_pool_lock: threading.Lock             = threading.Lock()


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=TOOL_POOL_MAX_WORKERS, mp_context=_mp.get_context("spawn"))
    return _pool


def _process_worker(fn_bytes: bytes, arg_bytes: bytes) -> bytes:
    """Worker entry-point: unpickle the tool fn and call it."""
    import cloudpickle
    import pickle
    fn:  Callable[[agdata], agdata] = cloudpickle.loads(fn_bytes)
    arg: agdata                     = pickle.loads(arg_bytes)
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
        need_sandbox: bool = True,
    ):
        self.name         = name
        self.description  = description
        self.fn           = fn
        self.params       = params or {"type": "object", "properties": {}}
        self._log_fn      = log_fn
        self.need_sandbox = need_sandbox
        self._term:  "agterm | None" = None
        self._aglog: "aglog  | None" = None

    # ------------------------------------------------------------------
    # Pickle support — exclude loggers; they hold locks / file handles and
    # are not needed in the worker process.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        return {
            "name":         self.name,
            "description":  self.description,
            "fn":           self.fn,
            "params":       self.params,
            "_log_fn":      self._log_fn,
            "need_sandbox": self.need_sandbox,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.need_sandbox = state.get("need_sandbox", True)
        self._term  = None
        self._aglog = None

    # ------------------------------------------------------------------
    # Logger attachment
    # ------------------------------------------------------------------

    def attach_logger(self, term: "agterm", aglog: "aglog") -> None:
        """Wire up terminal and structured file logging for this tool."""
        self._term  = term
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

    def __call__(self, arg: agdata, timeout: int | None = None) -> agdata:
        self.log_start(arg)
        t0 = time.monotonic()

        if not self.need_sandbox:
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
        fn_bytes         = cloudpickle.dumps(self.fn)
        arg_bytes        = pickle.dumps(arg)
        effective_timeout = timeout if timeout is not None else TOOL_TIMEOUT_S
        try:
            result_bytes = _get_pool().submit(_process_worker, fn_bytes, arg_bytes).result(timeout=effective_timeout)
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
                "name":        self.name,
                "description": self.description,
                "parameters":  self.params,
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
        json_type = _hint_to_json_type(hint)
        tool_desc, value_desc = _return_tool_descriptions(field, hint)
        value_schema: dict = {"type": json_type, "description": value_desc}
        tools.append({
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
        })
    return tools


def _make_return_output_tool(schema) -> list[dict]:
    """Alias kept for test compatibility — returns the full per-field tool list."""
    return make_return_output_tools(schema)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

TOOL_OUTPUT_OFFLOAD_CHARS: int = 40_000  # minimum floor for tool-output offloading


def dispatch_tools(
    tool_calls: list[dict],
    tool_map: dict,
    messages: list[dict],
    sandbox: "agSandbox",
    skill_name: str,
    _state_fn: "Callable | None",
    _live_messages_fn: "Callable | None",
    _full_history_fn: "Callable | None",
    term: "agterm | None",
    _intercept: "dict[str, Callable[[dict], str]] | None" = None,
    tool_offload_chars: int = TOOL_OUTPUT_OFFLOAD_CHARS,
) -> bool:
    """Execute all tool calls from one LLM response, appending results to messages.

    Returns True if the read tool was lazily injected into tool_map during this
    dispatch (because a large output was offloaded and read was not already present).
    The caller should then add the read tool's schema to openai_tools so the LLM
    can use it on the next step.
    """
    _injected_read = False
    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        fn_args = tc["function"]["arguments"]
        tc_id   = tc["id"]
        # Ensure arguments is valid JSON before it goes back into history.
        # A malformed string (truncated generation, Python repr, etc.) causes
        # vLLM to crash on the next request when it re-parses the history.
        try:
            json.loads(fn_args)
        except (json.JSONDecodeError, TypeError):
            fn_args = "{}"
            tc["function"]["arguments"] = fn_args

        # Framework-internal tools (e.g. return_output) are handled in the
        # calling thread before normal tool dispatch.
        if _intercept and fn_name in _intercept:
            try:
                args = json.loads(fn_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
            result_content = _intercept[fn_name](args)
            if term:
                try:
                    _ret_result = json.loads(result_content)
                except (json.JSONDecodeError, TypeError):
                    _ret_result = {}
                if "error" in _ret_result:
                    term.log("TOOL ✗   ", f"{fn_name}({fn_args})  → {_ret_result['error']}")
                else:
                    term.log("TOOL ✓   ", f"{fn_name}({fn_args})")
            tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result_content}
            messages.append(tool_msg)
            if _live_messages_fn:
                _live_messages_fn(messages[1:])
            if _full_history_fn:
                _full_history_fn(tool_msg)
            continue

        t = tool_map.get(fn_name)
        if t is None:
            if term:
                term.log("TOOL ✗   ", f"{fn_name}  → unknown tool")
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
                    pass
                result_content = t(agdata.from_json(fn_args), timeout=_tool_timeout).to_json()
                if _state_fn:
                    _state_fn("skill", skill=skill_name)
                # Offload large tool outputs regardless of whether the tool
                # itself uses the sandbox — fetch_paper and other add_tools have
                # need_sandbox=False but can still produce huge outputs that
                # bloat the context.
                if len(result_content) > tool_offload_chars:
                    safe_id = tc_id.replace("-", "")[:12]
                    offload_path = f"/workspace/long_tool_call_outputs/{fn_name}_{safe_id}.txt"
                    try:
                        try:
                            file_body = json.loads(result_content).get("content", result_content)
                        except (json.JSONDecodeError, AttributeError):
                            file_body = result_content
                        sandbox.write_file(offload_path, file_body)
                        result_content = json.dumps({
                            "note": f"Output was too large and has been saved to {offload_path}. Use the read tool to access it."
                        })
                        if "read" not in tool_map:
                            from .tools import make_read as _make_read
                            _read_tool = _make_read(sandbox)
                            tool_map["read"] = _read_tool
                            _injected_read = True
                    except Exception as _e:
                        print(f"[agtool] WARNING: failed to offload large tool output to {offload_path}: {_e}")
                if t.need_sandbox:
                    # A tool may signal failure via agdata(error=...) without raising —
                    # treat that the same as an exception: discard dirty state.
                    _result_errored = False
                    try:
                        if "error" in json.loads(result_content):
                            _result_errored = True
                    except (json.JSONDecodeError, TypeError):
                        pass
                    if _result_errored:
                        sandbox.stop(commit=False)
                        try:
                            _result_obj = json.loads(result_content)
                            _result_obj["workspace_reverted"] = (
                                "The workspace has been reverted to the state "
                                "before this tool call."
                            )
                            result_content = json.dumps(_result_obj)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    else:
                        sandbox.stop(commit=True)
            except Exception as e:
                if _state_fn:
                    _state_fn("skill", skill=skill_name)
                result_content = json.dumps({"error": format_exception(e)})
                # On failure: remove without committing to discard dirty state.
                # The next tool call restores from the last successful checkpoint.
                if t.need_sandbox:
                    sandbox.stop(commit=False)
                    try:
                        _result_obj = json.loads(result_content)
                        _result_obj["workspace_reverted"] = (
                            "The workspace has been reverted to the state "
                            "before this tool call."
                        )
                        result_content = json.dumps(_result_obj)
                    except (json.JSONDecodeError, TypeError):
                        pass
        tool_msg = {"role": "tool", "tool_call_id": tc_id, "content": result_content}
        messages.append(tool_msg)
        if _live_messages_fn:
            _live_messages_fn(messages[1:])
        if _full_history_fn:
            _full_history_fn(tool_msg)
    return _injected_read
