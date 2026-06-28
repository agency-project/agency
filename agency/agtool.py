from __future__ import annotations
import threading
import time
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as _FutureTimeoutError, BrokenExecutor
from typing import TYPE_CHECKING, Callable
from .agdata import agdata, agerror

if TYPE_CHECKING:
    from .aglog import aglog
    from .agterm import agterm

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
                from .agdata import _fmt_exc
                result = agerror(_fmt_exc(e))
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
