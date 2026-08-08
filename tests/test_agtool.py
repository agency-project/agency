"""Tests for the agtool class."""

import json
import os
from unittest.mock import MagicMock
from agency.agdata import agdata
from agency.agtool import agtool


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


# test_process_pool_runs_in_different_pid was retired here: agtool.__call__
# no longer has a subprocess-pool path at all -- every call always runs in
# the calling thread/process (see agtool.py's own module docstring).

# ---------------------------------------------------------------------------
# Invocation always runs in the calling thread/process -- no subprocess
# isolation, ever (run_in_subprocess is still accepted as a constructor
# kwarg for call-site compatibility with existing tool factories, but no
# longer changes behavior; see agtool.py's __call__).
# ---------------------------------------------------------------------------


def test_call_runs_in_same_pid():
    t = agtool(name="pid_inproc", description="", fn=_get_pid)
    result = t(agdata())
    assert result.worker_pid == os.getpid()


def test_call_sees_host_state():
    """A tool's fn can read module-level state set in the main process --
    the property that made run_in_subprocess=False necessary for any tool
    closing over live host objects (a sandbox, a resource pool, ...), now
    true unconditionally."""
    import agency.agtool as _agtool_mod

    _agtool_mod._TEST_SENTINEL = "host-value"

    def _read_sentinel(arg: agdata) -> agdata:
        import agency.agtool as _m

        return agdata(value=getattr(_m, "_TEST_SENTINEL", None))

    try:
        t = agtool(name="sentinel", description="", fn=_read_sentinel)
        result = t(agdata())
        assert result.value == "host-value"
    finally:
        del _agtool_mod._TEST_SENTINEL


def test_call_exception_returns_error_agdata():
    def _boom(arg: agdata) -> agdata:
        raise ValueError("intentional failure")

    t = agtool(name="boom", description="", fn=_boom)
    result = t(agdata())
    assert result.error is not None
    assert "intentional failure" in result.error


def test_call_timeout_not_enforced():
    """`timeout` is accepted for call-site compatibility but not enforced --
    there's no separate process/thread left to bound (see agtool.py's
    __call__ docstring)."""
    import time

    def _slow(arg: agdata) -> agdata:
        time.sleep(0.05)
        return agdata(done=True)

    t = agtool(name="slow_inproc", description="", fn=_slow)
    result = t(agdata(), timeout=1)
    assert result.done is True
