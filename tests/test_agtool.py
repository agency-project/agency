"""Tests for the agtool class."""

import importlib
import os
from unittest.mock import MagicMock, patch
from agency.agdata import agdata, agerror
from agency.agtool import agtool, dispatch_tools
from agency.profiler import agprof

# `agency/__init__.py` does `from .agtool import agtool`, which overwrites the
# `agtool` attribute on the `agency` package with the class — so a plain
# `import agency.agtool as _agtool_mod` would resolve to the class, not the
# module. Go through importlib to get the actual module object.
_agtool_mod = importlib.import_module("agency.agtool")


def _echo(arg: agdata) -> agdata:
    return agdata(echoed=arg.to_dict())


def _identity(arg: agdata) -> agdata:
    return arg


def _get_pid(arg: agdata) -> agdata:
    return agdata(worker_pid=os.getpid())


def make_tool() -> agtool:
    return agtool(
        name="echo",
        description="Echoes the input.",
        fn=_echo,
        params={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Text to echo"},
            },
            "required": ["message"],
        },
    )


def test_name():
    t = make_tool()
    assert t.name == "echo"


def test_call_returns_agdata():
    t = make_tool()
    result = t(agdata(message="hello"))
    assert isinstance(result, agdata)
    assert result.echoed == {"message": "hello"}


def test_to_openai_tool_shape():
    t = make_tool()
    schema = t.to_openai_tool()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "echo"
    assert fn["description"] == "Echoes the input."
    assert "properties" in fn["parameters"]
    assert "message" in fn["parameters"]["properties"]


def test_repr():
    t = make_tool()
    assert "echo" in repr(t)


def test_dispatch_failure_is_recorded_in_profiler_metadata(monkeypatch):
    class FakeRecordFunction:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(agprof, "_records", [])
    monkeypatch.setattr(agprof, "_open_spans", {})
    monkeypatch.setattr(agprof, "_session", agprof._TorchSession(FakeRecordFunction))

    failing = agtool(
        name="failing",
        description="",
        fn=lambda arg: agerror("boom"),
        run_in_subprocess=False,
    )
    sandbox = MagicMock()
    sandbox._has_pending_background_work.return_value = False
    messages = []
    dispatch_tools(
        [
            {
                "id": "call-1",
                "function": {"name": "failing", "arguments": "{}"},
            }
        ],
        {"failing": failing},
        messages,
        sandbox,
        "test",
        None,
        None,
        None,
        None,
    )

    tool_record = next(record for record in agprof._records if record[1] == "tool:failing")
    assert tool_record[6]["outcome"] == "failure"
    assert tool_record[6]["error_type"] == "tool_error"


def test_default_params():
    t = agtool(name="noop", description="", fn=_identity)
    assert t.to_openai_tool()["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# Pickle / serialisation — loggers must be excluded
# ---------------------------------------------------------------------------


def test_getstate_excludes_loggers():
    t = make_tool()
    mock_term = MagicMock()
    mock_log = MagicMock()
    t.attach_logger(mock_term, mock_log)

    state = t.__getstate__()
    assert "_term" not in state
    assert "_aglog" not in state
    assert state["name"] == "echo"
    assert state["fn"] is _echo


def test_setstate_restores_none_loggers():
    t = make_tool()
    t2 = agtool.__new__(agtool)
    t2.__setstate__(t.__getstate__())
    assert t2._term is None
    assert t2._aglog is None
    assert t2.name == "echo"


def test_pickle_round_trip():
    """Tool and its fn survive a pickle serialisation cycle."""
    import pickle

    t = make_tool()
    restored = pickle.loads(pickle.dumps(t))
    result = restored.fn(agdata(message="ping"))
    assert result.echoed == {"message": "ping"}


# ---------------------------------------------------------------------------
# Process pool — run_in_subprocess=True runs in a separate worker process
# ---------------------------------------------------------------------------


def test_process_pool_runs_in_different_pid():
    """run_in_subprocess=True (default) offloads fn to a worker process (different PID)."""
    t = agtool(name="pid_check", description="", fn=_get_pid)
    result = t(agdata())
    assert result.worker_pid != os.getpid()


# ---------------------------------------------------------------------------
# run_in_subprocess=False — runs in-process, in the calling thread
# ---------------------------------------------------------------------------


def test_no_sandbox_runs_in_same_pid():
    """run_in_subprocess=False runs fn directly in the calling thread (same PID)."""
    t = agtool(name="pid_inproc", description="", fn=_get_pid, run_in_subprocess=False)
    result = t(agdata())
    assert result.worker_pid == os.getpid()


def test_no_sandbox_sees_host_state():
    """run_in_subprocess=False fn can read module-level state set in the main process.

    This is the key property that sandboxed tools cannot provide: a subprocess
    worker would see the module-level sentinel as None (freshly imported module),
    while the in-process path sees the value set by the test.
    """
    import agency.agtool as _agtool_mod

    _agtool_mod._TEST_SENTINEL = "host-value"

    def _read_sentinel(arg: agdata) -> agdata:
        import agency.agtool as _m

        return agdata(value=getattr(_m, "_TEST_SENTINEL", None))

    try:
        t = agtool(name="sentinel", description="", fn=_read_sentinel, run_in_subprocess=False)
        result = t(agdata())
        assert result.value == "host-value"
    finally:
        del _agtool_mod._TEST_SENTINEL


def test_no_sandbox_exception_returns_error_agdata():
    """run_in_subprocess=False catches exceptions and returns agdata(error=...) like sandboxed tools."""

    def _boom(arg: agdata) -> agdata:
        raise ValueError("intentional failure")

    t = agtool(name="boom", description="", fn=_boom, run_in_subprocess=False)
    result = t(agdata())
    assert result.error is not None
    assert "intentional failure" in result.error


def test_no_sandbox_timeout_not_enforced():
    """run_in_subprocess=False ignores the timeout parameter — it runs in the calling thread."""
    import time

    def _slow(arg: agdata) -> agdata:
        time.sleep(0.05)
        return agdata(done=True)

    # Pass a very short timeout; with run_in_subprocess=True this would race, but
    # run_in_subprocess=False bypasses the pool entirely so timeout has no effect.
    t = agtool(name="slow_inproc", description="", fn=_slow, run_in_subprocess=False)
    result = t(agdata(), timeout=1)
    assert result.done is True


# ---------------------------------------------------------------------------
# Process pool lifecycle — SIGINT-ignoring workers, explicit shutdown
# ---------------------------------------------------------------------------


def test_ignore_sigint_in_worker_sets_sig_ign():
    """The pool initializer makes workers ignore SIGINT so Ctrl+C doesn't kill
    a tool call mid-flight; call it directly rather than actually changing
    this process's signal disposition."""
    import signal

    with patch("signal.signal") as mock_signal:
        _agtool_mod._ignore_sigint_in_worker()
    mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)


def test_get_pool_uses_sigint_ignoring_initializer():
    pool = _agtool_mod._get_pool()
    assert pool._initializer is _agtool_mod._ignore_sigint_in_worker


def test_shutdown_tool_pool_resets_pool_and_calls_shutdown():
    mock_pool = MagicMock()
    original = _agtool_mod._pool
    _agtool_mod._pool = mock_pool
    try:
        _agtool_mod.shutdown_tool_pool()
        assert _agtool_mod._pool is None
        mock_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    finally:
        _agtool_mod._pool = original


def test_shutdown_tool_pool_forwards_custom_kwargs():
    mock_pool = MagicMock()
    original = _agtool_mod._pool
    _agtool_mod._pool = mock_pool
    try:
        _agtool_mod.shutdown_tool_pool(wait=True, cancel_futures=False)
        mock_pool.shutdown.assert_called_once_with(wait=True, cancel_futures=False)
    finally:
        _agtool_mod._pool = original


def test_shutdown_tool_pool_noop_when_no_pool():
    original = _agtool_mod._pool
    _agtool_mod._pool = None
    try:
        _agtool_mod.shutdown_tool_pool()  # must not raise
        assert _agtool_mod._pool is None
    finally:
        _agtool_mod._pool = original
