"""Tests for the agtool class."""
import json
from agency.agdata import agdata
from agency.agtool import agtool


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
