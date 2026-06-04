"""Tests for the agtool class."""
import os
import sys
import json
import pytest
from unittest.mock import MagicMock
from agency.agdata import agdata
from agency.agtool import agtool
import agency.agtool   # ensure module in sys.modules
_agtool_module = sys.modules["agency.agtool"]


def _echo(arg: agdata) -> agdata:
    return agdata(echoed=arg.to_dict())


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
    t = agtool(name="noop", description="", fn=lambda a: a)
    assert t.to_openai_tool()["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# Pickle / serialisation — loggers must be excluded
# ---------------------------------------------------------------------------

def test_getstate_excludes_loggers():
    t = make_tool()
    mock_term = MagicMock()
    mock_log  = MagicMock()
    t.attach_logger(mock_term, mock_log)

    state = t.__getstate__()
    assert "_term"  not in state
    assert "_aglog" not in state
    assert state["name"] == "echo"
    assert state["fn"] is _echo


def test_setstate_restores_none_loggers():
    t = make_tool()
    t2 = agtool.__new__(agtool)
    t2.__setstate__(t.__getstate__())
    assert t2._term  is None
    assert t2._aglog is None
    assert t2.name   == "echo"


def test_cloudpickle_round_trip():
    """Tool and its fn survive a cloudpickle serialisation cycle."""
    import cloudpickle
    t = make_tool()
    restored = cloudpickle.loads(cloudpickle.dumps(t))
    result = restored.fn(agdata(message="ping"))
    assert result.echoed == {"message": "ping"}


# ---------------------------------------------------------------------------
# _use_process_pool bypass (conftest sets this to False for all tests)
# ---------------------------------------------------------------------------

def test_bypass_calls_fn_directly(monkeypatch):
    """When _use_process_pool is False, __call__ invokes fn in the same process."""
    monkeypatch.setattr(_agtool_module, "_use_process_pool", False)
    seen_pids = []

    def fn(arg: agdata) -> agdata:
        seen_pids.append(os.getpid())
        return agdata(ok=True)

    t = agtool(name="pid_check", description="", fn=fn)
    t(agdata())
    assert seen_pids == [os.getpid()]


def test_process_pool_runs_in_different_pid(monkeypatch):
    """When _use_process_pool is True, fn runs in a worker process (different PID)."""
    monkeypatch.setattr(_agtool_module, "_use_process_pool", True)

    def fn(arg: agdata) -> agdata:
        return agdata(worker_pid=os.getpid())

    t = agtool(name="pid_check", description="", fn=fn)
    result = t(agdata())
    assert result.worker_pid != os.getpid()
