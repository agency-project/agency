"""Tests for the agtool class."""
import os
import pytest
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


def test_pickle_round_trip():
    """Tool and its fn survive a pickle serialisation cycle."""
    import pickle
    t = make_tool()
    restored = pickle.loads(pickle.dumps(t))
    result = restored.fn(agdata(message="ping"))
    assert result.echoed == {"message": "ping"}


# ---------------------------------------------------------------------------
# Process pool — fn always runs in a separate worker process
# ---------------------------------------------------------------------------

def test_process_pool_runs_in_different_pid():
    """__call__ always offloads fn to a worker process (different PID)."""
    t = agtool(name="pid_check", description="", fn=_get_pid)
    result = t(agdata())
    assert result.worker_pid != os.getpid()
