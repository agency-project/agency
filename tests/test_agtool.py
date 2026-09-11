"""Tests for the agtool class."""

import os
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


def test_default_params():
    t = agtool(name="noop", description="", fn=_identity)
    assert t.to_openai_tool()["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# Pickle / serialisation
# ---------------------------------------------------------------------------


def test_pickle_round_trip():
    """Tool and its fn survive a pickle serialisation cycle."""
    import pickle

    t = make_tool()
    restored = pickle.loads(pickle.dumps(t))
    result = restored.fn(agdata(message="ping"))
    assert result.echoed == {"message": "ping"}


def test_cloudpickle_preserves_persistent_factories_and_tool_definition():
    import cloudpickle

    seed = 7
    tool = make_tool()
    tool.persistent_vars = {"state": lambda: {"count": seed}}
    restored = cloudpickle.loads(cloudpickle.dumps(tool))

    assert restored is not tool
    assert restored.to_openai_tool() == tool.to_openai_tool()
    assert restored.persistent_vars["state"]() == {"count": 7}
    assert restored(agdata(message="hello")).echoed == {"message": "hello"}


# ---------------------------------------------------------------------------
# Invocation always runs in the calling thread/process -- no subprocess
# isolation. Sandbox MCP transport owns any cross-process serialization.
# ---------------------------------------------------------------------------


def test_call_runs_in_same_pid():
    t = agtool(name="pid_inproc", description="", fn=_get_pid)
    result = t(agdata())
    assert result.worker_pid == os.getpid()


def test_call_sees_host_state():
    """A tool's fn can read module-level state set in the main process --
    including closures over live host objects such as sandboxes and pools."""
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


# ---------------------------------------------------------------------------
# __call__ context injection -- a fn declares only the extra names (past its
# first, agdata, parameter) it wants; __call__ forwards only those, sourced
# from whatever **context a caller happens to pass in.
# ---------------------------------------------------------------------------


def test_plain_single_arg_fn_ignores_unrelated_context():
    def _plain(arg: agdata) -> agdata:
        return agdata(x=arg._data["x"])

    t = agtool(name="plain", description="", fn=_plain)
    result = t(agdata(x=1), sandbox="S", resource_pool="P")
    assert result.x == 1


def test_fn_declaring_extra_param_receives_only_the_matching_context():
    seen = {}

    def _needs_sandbox(arg: agdata, sandbox) -> agdata:
        seen["sandbox"] = sandbox
        return agdata(ok=True)

    t = agtool(name="needs_sandbox", description="", fn=_needs_sandbox)
    result = t(agdata(), sandbox="S", resource_pool="P")
    assert result.ok is True
    assert seen == {"sandbox": "S"}


def test_fn_declaring_a_context_param_not_supplied_fails_gracefully():
    def _needs_sandbox(arg: agdata, sandbox) -> agdata:
        return agdata(sandbox=sandbox)

    t = agtool(name="needs_sandbox", description="", fn=_needs_sandbox)
    result = t(agdata())  # no context supplied at all
    assert result.error is not None
    assert "sandbox" in result.error
