"""Tests for the agtool class."""

import importlib
import json
import os
from unittest.mock import MagicMock, patch
from agency.agdata import agdata
from agency.agtool import agtool

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


# ---------------------------------------------------------------------------
# dispatch_tools() -- optional agpolicy mediation (Phase 6 retrofit)
# ---------------------------------------------------------------------------


def _build_tool_call(name, args, call_id="c1"):
    import json

    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_dispatch_tools_no_policy_behaves_unchanged():
    """policy=None (the default) must not change dispatch_tools' behavior
    at all -- this is the existing native path, untouched."""
    from agency.agtool import dispatch_tools

    t = agtool(
        name="calc",
        description="",
        fn=lambda arg: agdata(val=arg.x * 10),
        params={"type": "object", "properties": {"x": {"type": "integer"}}},
        run_in_subprocess=False,
    )
    toolkit = {"calc": t}
    messages = [{"role": "system", "content": "sys"}]
    tool_calls = [_build_tool_call("calc", {"x": 7})]

    dispatch_tools(tool_calls, toolkit, messages, MagicMock(), "skill", None, None, None, None)

    tool_msg = next(m for m in messages if m.get("role") == "tool")
    assert json.loads(tool_msg["content"]) == {"val": 70}


def test_dispatch_tools_policy_allow_runs_the_tool():
    from agency.agtool import dispatch_tools
    from agency.agpolicy import agpolicy, agdecision

    class AllowPolicy(agpolicy):
        def check(self, ag, event):
            return agdecision.allow()

    t = agtool(
        name="calc",
        description="",
        fn=lambda arg: agdata(val=arg.x * 10),
        params={"type": "object", "properties": {"x": {"type": "integer"}}},
        run_in_subprocess=False,
    )
    toolkit = {"calc": t}
    messages = [{"role": "system", "content": "sys"}]
    tool_calls = [_build_tool_call("calc", {"x": 7})]

    dispatch_tools(
        tool_calls, toolkit, messages, MagicMock(), "skill", None, None, None, None,
        policy=AllowPolicy(), ag=None,
    )

    tool_msg = next(m for m in messages if m.get("role") == "tool")
    assert json.loads(tool_msg["content"]) == {"val": 70}


def test_dispatch_tools_policy_deny_skips_the_tool_and_reports_reason():
    from agency.agtool import dispatch_tools
    from agency.agpolicy import agpolicy, agdecision

    calls = []

    class DenyPolicy(agpolicy):
        def check(self, ag, event):
            calls.append(event)
            return agdecision.deny("blocked for testing")

    ran = []

    def fn(arg):
        ran.append(arg.x)
        return agdata(val=arg.x * 10)

    t = agtool(
        name="calc",
        description="",
        fn=fn,
        params={"type": "object", "properties": {"x": {"type": "integer"}}},
        run_in_subprocess=False,
    )
    toolkit = {"calc": t}
    messages = [{"role": "system", "content": "sys"}]
    tool_calls = [_build_tool_call("calc", {"x": 7})]

    dispatch_tools(
        tool_calls, toolkit, messages, MagicMock(), "skill", None, None, None, None,
        policy=DenyPolicy(), ag=None,
    )

    assert ran == []  # the tool itself never executed
    tool_msg = next(m for m in messages if m.get("role") == "tool")
    assert json.loads(tool_msg["content"]) == {"error": "blocked for testing"}
    assert len(calls) == 1
    event = calls[0]
    assert event.syscall == "tool_call"
    assert event.tool_name == "calc"
    assert event.tool_args == {"x": 7}


def test_dispatch_tools_policy_not_consulted_for_unknown_tool():
    from agency.agtool import dispatch_tools
    from agency.agpolicy import agpolicy, agdecision

    checked = []

    class RecordingPolicy(agpolicy):
        def check(self, ag, event):
            checked.append(event)
            return agdecision.allow()

    toolkit = {}
    messages = [{"role": "system", "content": "sys"}]
    tool_calls = [_build_tool_call("nonexistent", {})]

    dispatch_tools(
        tool_calls, toolkit, messages, MagicMock(), "skill", None, None, None, None,
        policy=RecordingPolicy(), ag=None,
    )

    assert checked == []  # no policy check for a tool that doesn't exist
    tool_msg = next(m for m in messages if m.get("role") == "tool")
    assert "unknown tool" in json.loads(tool_msg["content"])["error"]


def test_dispatch_tools_policy_receives_ag():
    from agency.agtool import dispatch_tools
    from agency.agpolicy import agpolicy, agdecision

    seen_agents = []

    class RecordingPolicy(agpolicy):
        def check(self, ag, event):
            seen_agents.append(ag)
            return agdecision.allow()

    t = agtool(
        name="noop", description="", fn=lambda arg: agdata(ok=True),
        params={"type": "object", "properties": {}}, run_in_subprocess=False,
    )
    sentinel_agent = object()
    dispatch_tools(
        [_build_tool_call("noop", {})], {"noop": t},
        [{"role": "system", "content": "sys"}], MagicMock(), "skill",
        None, None, None, None,
        policy=RecordingPolicy(), ag=sentinel_agent,
    )
    assert seen_agents == [sentinel_agent]
