"""Tests for agskill as a self-contained ReAct skill."""

import json
import shutil
from unittest.mock import MagicMock
import pytest
from agency.agdata import agdata
from agency.configs.agconfig import agconfig, llmconfig
from agency.agschema import agschema
from agency.agskill import agskill
from agency.agtool import agtool

LLM_MAX_RETRIES = agconfig().llm.max_retries
LLM_IDLE_TIMEOUT = agconfig().llm.idle_timeout
LLM_STREAM_TIMEOUT = agconfig().llm.stream_timeout


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    from agency.sandbox.container import _runtime_works

    return _runtime_works("docker")


docker = pytest.mark.skipif(not _docker_available(), reason="Docker daemon not reachable")


def make_mock_agent(sandbox=None, ping_interval_s=300, poll_interval_s=5):
    _ping = ping_interval_s
    _poll = poll_interval_s

    class _MockAgent:
        agresource_pool = MagicMock()
        ping_interval_s = _ping
        poll_interval_s = _poll
        agconfig = None

    ag = _MockAgent()
    if sandbox is not None:
        ag.sandbox = sandbox
    else:
        ag.sandbox = MagicMock()
        # A bare MagicMock()'s _has_pending_background_work() would
        # otherwise auto-mock to a truthy value, making agtool.py's
        # dispatch_tools() defer stop() forever -- default to "nothing
        # pending" so tests get the common case without configuring it.
        # Same reasoning for `.persistent` -- a bare MagicMock auto-mocks
        # it truthy too (see agtool.py's `not sandbox.persistent and ...`).
        ag.sandbox._has_pending_background_work.return_value = False
        ag.sandbox.persistent = False
    ag.data_logger = MagicMock()
    ag.agname = "test"
    return ag


def _noop(arg: agdata) -> agdata:
    return agdata()


# ---------------------------------------------------------------------------
# Streaming mock helpers
# agskill uses stream=True; the mock must return a list of chunk objects.
# Using a list (not iter()) lets the same return_value be re-iterated across
# multiple calls (e.g. retry tests).
# ---------------------------------------------------------------------------


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {}
        self.reasoning_content = None


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Usage:
    prompt_tokens = 5


class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.usage = usage
        self.choices = (
            [_Choice(_Delta(content, tool_calls))] if (content is not None or tool_calls) else []
        )


class _TCDelta:
    def __init__(self, name, args_json, call_id):
        self.id = call_id
        self.index = 0
        self.function = _TCFnDelta(name, args_json)


class _TCFnDelta:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


def _direct(content: str) -> list:
    """Streaming response list for a plain-text or JSON reply."""
    return [_Chunk(content=content), _Chunk(usage=_Usage())]


def _tool_call(name: str, args: dict, call_id: str = "c1") -> list:
    """Streaming response list for a tool-call reply."""
    tc = _TCDelta(name, json.dumps(args), call_id)
    return [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]


def make_skill(name="summarise", add_host_mcp_tools=None) -> agskill:
    return agskill(
        name=name,
        prompt="You are a summarisation assistant.",
        add_host_mcp_tools=add_host_mcp_tools,
    )


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_name_and_repr():
    s = make_skill()
    assert s.name == "summarise"
    assert "summarise" in repr(s)


# test_run_returns_agdata_and_history / test_run_no_schema_returns_raw_content /
# test_run_plain_text_fallback were retired here along with execute_react()
# itself -- basic "the loop returns the model's content correctly" coverage
# now lives in tests/harness/adapters/test_native_loop_fast.py
# (test_bash_tool_round_trip, test_final_text_preserves_raw_content_verbatim),
# exercising the native loop that replaces execute_react() for every engine.


# ---------------------------------------------------------------------------
# agskill.check_schema — Python type object hints
# ---------------------------------------------------------------------------


def test_check_schema_accepts_python_type_objects():
    assert agschema(agdata(x=int, name=str)).check(agdata(x=5, name="hi")) == []


def test_check_schema_type_mismatch_with_type_object():
    errors = agschema(agdata(x=int)).check(agdata(x="bad"))
    assert len(errors) == 1
    assert "x" in errors[0]
    assert "int" in errors[0]


def test_prompt_type_names_shown_correctly():
    from agency.agtype import agfile

    sk = agskill(
        "t",
        "",
        input_schema=agdata(n=int, s=str, doc=agfile),
        output_schema=agdata(result=float),
    )
    prompt = sk._build_prompt()
    assert '"n": "int"' in prompt  # input schema still uses to_json()
    assert '"s": "str"' in prompt
    assert '"doc": "file"' in prompt
    output_instruction = sk._build_output_instruction()
    assert "result" in output_instruction  # output field listed by name
    assert "float" in output_instruction  # output field type shown as "float"


# ---------------------------------------------------------------------------
# Tool call path
# ---------------------------------------------------------------------------


# test_tool_call_executes_and_continues / test_replace_tools_overrides_defaults /
# test_replace_tools_empty_list_gives_no_tools / test_add_tools_extends_sandbox_defaults
# were retired here along with execute_react() itself: add_tools/
# replace_tools are host-authored Python tool closures with no execution
# path today for ANY engine -- execute_react() was the only one that ever
# ran them, and native.py's `NativeAdapter.execute()` explicitly rejects
# them (a real, currently-open gap -- see that module's "Known gaps"
# docstring section; building real container-side support, e.g. shipping a
# picklable closure into the container plus a minimal pure-agdata shim
# there, is deliberately scoped as separate follow-up work, not done here).


# ---------------------------------------------------------------------------
# replace_tools / add_tools
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# input_schema and output_schema
# ---------------------------------------------------------------------------


def test_input_schema_missing_field_returns_error():
    # Input schema validation is engine-agnostic, so it needs no live harness.
    s = agskill(
        name="s",
        prompt="",
        input_schema=agdata(question=str, context=str),
    )
    error = s.input_schema.validate_input(agdata(question="hi"))
    assert error is not None
    assert "context" in error


def test_input_schema_type_error_returns_error():
    s = agskill(
        name="s",
        prompt="",
        input_schema=agdata(count=int),
    )
    error = s.input_schema.validate_input(agdata(count="not-an-int"))
    assert error is not None
    assert "count" in error


def test_input_schema_valid_proceeds():
    s = agskill(
        name="s",
        prompt="",
        input_schema=agdata(text=str),
    )
    assert s.input_schema.validate_input(agdata(text="hello")) is None


def test_input_schema_description_value_only_checks_presence():
    """Non-type-name values (descriptions) only trigger a missing-key error."""
    s = agskill(
        name="s",
        prompt="",
        input_schema=agdata(query="the search query"),
    )
    assert s.input_schema.validate_input(agdata(query=42)) is None  # 42 is not type-checked


# test_output_schema_missing_field_triggers_retry,
# test_output_schema_retry_exhausted_returns_error,
# test_output_schema_type_mismatch_triggers_retry, and
# test_correction_message_appended_on_retry were retired here: they tested
# execute_react()'s own inline retry loop around the (also retired)
# per-field `return_<field>` tool mechanism. Native's structured output
# uses a different mechanism entirely -- a single `submit_output` MCP tool
# validated per-call (fast coverage:
# tests/harness/adapters/test_native_loop_fast.py's
# test_submit_output_all_fields_collected /
# test_submit_output_type_error_returns_immediate_feedback) plus a bounded
# reprompt-across-turns loop one level up in native.py's
# `NativeAdapter.execute()` (Docker-only coverage today, see
# test_native.py's TestNativeBackendRealEndToEnd).


def test_schemas_appended_to_prompt():
    s = agskill(
        name="s",
        prompt="Be helpful.",
        input_schema=agdata(text=str),
        output_schema=agdata(summary=str),
    )
    prompt = s._build_prompt()
    assert "Be helpful." in prompt
    assert "Input JSON format" in prompt
    assert '"text"' in prompt
    assert "submit_output" not in prompt  # moved to _build_output_instruction()

    output_instruction = s._build_output_instruction()
    assert "submit_output" in output_instruction
    assert "return_summary" not in output_instruction
    assert "`field`" in output_instruction and "`value`" in output_instruction
    assert "Do not answer" in output_instruction
    assert "summary" in output_instruction
    assert "string" in output_instruction  # per-field description for str output


def test_no_schemas_prompt_unchanged():
    s = agskill(name="s", prompt="Be helpful.")
    assert s._build_prompt() == "Be helpful."


# test_return_output_* / test_return_tool_* (all fields correct, type
# error feedback, unknown field, list/dict shapes, agrawstring passthrough,
# tool schema shape/ordering, parameter naming, any/wrong key extraction,
# term logging) were retired here: they tested the per-field `return_<field>`
# tool mechanism (agschema.make_return_output_agtool/agtool.
# make_return_output_tools), only ever called from execute_react()'s
# _build_toolkit(). Native's structured output uses a single `submit_output`
# MCP tool instead -- fast coverage for the all-fields-correct and
# type-error-immediate-feedback cases now lives in
# tests/harness/adapters/test_native_loop_fast.py
# (test_submit_output_all_fields_collected /
# test_submit_output_type_error_returns_immediate_feedback).
# test_semaphore_* / test_timeout_* / test_ssl_error_* / test_oserror_*
# (concurrency semaphore, exponential-backoff retry on idle-timeout/SSL/OS
# errors) were retired here: they all exercised agllm.py's own `call()`
# method's retry-with-backoff and call-concurrency semaphore, only ever
# invoked from execute_react(). Native's own retry lives in a different
# place with different scope (_native_in_container_entrypoint.py's
# _dispatch_via_terminus, retrying only a terminus 503/connection-failure,
# not SSL/idle-timeout errors from a real openai SDK client -- see that
# function's own docstring) -- already covered by test_native.py's real-
# Docker test_dispatch_retries_transient_terminus_error_and_recovers.

# _make_sandbox() helper and test_short_tool_output_not_offloaded /
# test_long_tool_output_offloaded_to_file / _to_sandbox /
# test_long_output_injects_read_tool_into_openai_tools /
# _read_tool_persists_for_skill_run / _no_duplicate_read_when_already_present
# were retired here: they tested agtool.py's dispatch_tools() host-side
# tool-output-offload-to-sandbox-file mechanism (only ever called from
# execute_react()) -- native has its own, simpler offload (a plain local
# file write, no sandbox bridge, `read` always available so no lazy
# tool-injection step exists), already covered fast by
# tests/harness/adapters/test_native_loop_fast.py's
# test_oversized_tool_output_is_offloaded_to_a_file.

# ---------------------------------------------------------------------------
# build_llm_kwargs
# ---------------------------------------------------------------------------

from agency.llm.agllm import agllm as _agllm_mod


def build_llm_kwargs(cfg, messages, openai_tools=None):
    return _agllm_mod.for_config(cfg).build_kwargs(messages, openai_tools)


def _llm_cfg(**fields) -> agconfig:
    """Test helper: build an agconfig from LLM fields."""
    fields.setdefault("base_url", "http://x/v1")
    return agconfig(llmconfig(**fields))


def test_build_llm_kwargs_model_and_messages():
    msgs = [{"role": "user", "blocks": [{"type": "text", "index": 0, "text": "hi"}]}]
    kw = build_llm_kwargs(_llm_cfg(model=""), msgs, None)
    assert kw["model"] == ""
    assert kw["messages"] == [{"role": "user", "content": "hi"}]


def test_build_llm_kwargs_strips_private_keys():
    msgs = [
        {
            "role": "assistant",
            "blocks": [{"type": "text", "index": 0, "text": "ok"}],
            "_thinking": "secret",
        }
    ]
    kw = build_llm_kwargs(_llm_cfg(model="m"), msgs, None)
    assert "_thinking" not in kw["messages"][0]
    assert kw["messages"][0]["content"] == "ok"


def test_build_llm_kwargs_openai_gen_params():
    kw = build_llm_kwargs(_llm_cfg(model="m", temperature=0.7, max_completion_tokens=100), [], None)
    assert kw["temperature"] == 0.7
    assert kw["max_completion_tokens"] == 100


def test_build_llm_kwargs_extra_body_vllm_params():
    kw = build_llm_kwargs(_llm_cfg(model="m", top_k=50, repetition_penalty=1.1), [], None)
    assert kw["extra_body"]["top_k"] == 50
    assert kw["extra_body"]["repetition_penalty"] == 1.1


def test_build_llm_kwargs_tools_included_when_provided():
    tools = [{"type": "function", "function": {"name": "f"}}]
    kw = build_llm_kwargs(_llm_cfg(model="m"), [], tools)
    assert kw["tools"] == tools


def test_build_llm_kwargs_no_tools_key_when_none():
    kw = build_llm_kwargs(_llm_cfg(model="m"), [], None)
    assert "tools" not in kw


# ---------------------------------------------------------------------------
# wait_for_processes
# ---------------------------------------------------------------------------

from agency.sandbox.agsandbox import agSandbox


def _make_real_sandbox(watched_pids=None):
    """Minimal sandbox stub with real _watched_pids dict for process monitoring tests."""

    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = dict(watched_pids or {})

        def _has_pending_background_work(self):
            return bool(self._watched_pids)

        def get_live_pids(self):
            return set(self._watched_pids.keys())

        def pid_status_summary(self):
            return ", ".join(f"PID {p}" for p in self._watched_pids)

    return _FakeSandbox()


def test_wait_for_processes_clean_sandbox_returns_none():
    sb = _make_real_sandbox()
    assert agSandbox.wait_for_processes(sb, "skill", "", 300, 5) is None


def test_wait_for_processes_no_watched_pids_attr_returns_none():
    class NoPids:
        def _has_pending_background_work(self):
            return False

    assert agSandbox.wait_for_processes(NoPids(), "skill", "", 300, 5) is None


def test_wait_for_processes_mock_sandbox_returns_none():
    from unittest.mock import MagicMock

    sb = MagicMock()
    sb._has_pending_background_work.return_value = False
    assert agSandbox.wait_for_processes(sb, "skill", "", 300, 5) is None


def test_wait_for_processes_completes_quickly_returns_completed_msg():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1234: 0.0}
            self._call_count = 0

        def _has_pending_background_work(self):
            # wait_for_processes() polls THIS method in its loop, not
            # get_live_pids() -- the state transition has to happen here,
            # not there, or the loop would just spin until ping_interval_s.
            self._call_count += 1
            if self._call_count >= 3:  # gate call + a couple of poll iterations
                self._watched_pids.clear()
            return bool(self._watched_pids)

        def get_live_pids(self):
            return set(self._watched_pids.keys())

        def pid_status_summary(self):
            return "PID 1234"

    sb = _FakeSandbox()
    result = agSandbox.wait_for_processes(sb, "skill", "", ping_interval_s=30, poll_interval_s=0.01)
    assert result is not None
    assert "completed" in result.lower() or "Background processes have completed" in result


def test_wait_for_processes_still_running_returns_update_msg():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1234: 0.0}

        def _has_pending_background_work(self):
            return bool(self._watched_pids)

        def get_live_pids(self):
            return {1234}

        def pid_status_summary(self):
            return "PID 1234"

    sb = _FakeSandbox()
    result = agSandbox.wait_for_processes(
        sb, "skill", "", ping_interval_s=0.02, poll_interval_s=0.01
    )
    assert result is not None
    assert "still running" in result.lower() or "Background processes are still running" in result


def test_wait_for_processes_calls_state_fn():
    class _FakeSandbox:
        def __init__(self):
            self._watched_pids = {1: 0.0}
            self._call_count = 0

        def _has_pending_background_work(self):
            # Same reasoning as test_wait_for_processes_completes_quickly_returns_completed_msg:
            # the loop polls this method, so the transition must live here.
            self._call_count += 1
            if self._call_count >= 3:
                self._watched_pids.clear()
            return bool(self._watched_pids)

        def get_live_pids(self):
            return set(self._watched_pids.keys())

        def pid_status_summary(self):
            return "PID 1"

    states = []
    agSandbox.wait_for_processes(
        _FakeSandbox(),
        "myskill",
        "",
        30,
        0.01,
        state_fn=lambda state, **kw: states.append(state),
    )
    assert "proc_wait" in states


# ---------------------------------------------------------------------------
# agskill.validate_input
# ---------------------------------------------------------------------------

from agency.agtype import agrawstring


def test_validate_input_no_schema_returns_none():
    assert agschema(agdata(x=int)).validate_input(agdata(x=1)) is None


def test_validate_input_schema_mismatch_returns_error():
    error = agschema(agdata(x=agrawstring)).validate_input(agdata())
    assert error is not None
    assert "x" in error


# ---------------------------------------------------------------------------
# Randomised nested-schema fuzz: type_hint_to_string_type + JSON round-trip + validation
# ---------------------------------------------------------------------------


def test_random_nested_schema_roundtrip():
    """100 randomly generated nested schemas exercising every container/leaf combination.

    For each trial:
    - Python → serialized-str: type_hint_to_string_type must return the correct JSON Schema
      type, and json.dumps must succeed.
    - Serialized-str → Python: json.loads must round-trip cleanly, and
      validate_output_field_against_schema must accept the recovered value.
    - Wrong-container rejection: a value with the opposite container type (list vs
      dict) must be rejected by validate_output_field_against_schema for bare / generic hints that
      the framework validates at the top level.
    """
    import random
    from typing import get_origin, get_args
    from agency.agtype import type_hint_to_string_type, validate_output_field_against_schema
    from agency.agtype import agrawstring, agtype, agfile, agbinary, agimage

    _validate_output_field_against_schema = validate_output_field_against_schema

    rng = random.Random(20240624)

    LEAF_TYPES = [str, int, float, bool, agrawstring, agfile, agbinary, agimage]

    def rand_hint(depth: int):
        if depth >= 4 or (depth > 0 and rng.random() < 0.30 * depth):
            return rng.choice(LEAF_TYPES)
        kind = rng.choice(("list", "dict", "tuple"))
        n = rng.randint(1, 4)
        if kind == "list":
            return list[rand_hint(depth + 1)]
        if kind == "dict":
            return dict[str, rand_hint(depth + 1)]
        # tuple: 1-4 heterogeneous element types
        inners = tuple(rand_hint(depth + 1) for _ in range(n))
        return tuple[inners] if len(inners) > 1 else tuple[inners[0]]

    def rand_value(hint):
        if hint is bool:
            return rng.choice([True, False])
        if hint is int:
            return rng.randint(-9, 9)
        if hint is float:
            return round(rng.uniform(-9.0, 9.0), 1)
        if hint is str or (isinstance(hint, type) and issubclass(hint, agtype)):
            return rng.choice(["a", "bb", "ccc"])
        origin = get_origin(hint)
        args = get_args(hint)
        if origin is list:
            return [rand_value(args[0]) for _ in range(rng.randint(1, 4))]
        if origin is dict:
            return {f"k{i}": rand_value(args[1]) for i in range(rng.randint(1, 4))}
        if origin is tuple:
            # Serialise as list — JSON has no tuple type
            return [rand_value(t) for t in args]
        # bare container types
        if hint is list:
            return [rng.randint(0, 5) for _ in range(rng.randint(1, 4))]
        if hint is dict:
            return {f"k{i}": rng.randint(0, 5) for i in range(rng.randint(1, 4))}
        if hint is tuple:
            return [rng.randint(0, 5) for _ in range(rng.randint(1, 4))]
        return "?"

    def ground_truth_json_type(hint) -> str:
        if isinstance(hint, type):
            if issubclass(hint, bool):
                return "boolean"
            if issubclass(hint, int):
                return "integer"
            if issubclass(hint, float):
                return "number"
            if issubclass(hint, (list, tuple)):
                return "array"
            if issubclass(hint, dict):
                return "object"
            return "string"  # str and agtype subclasses
        origin = get_origin(hint)
        if origin in (list, tuple):
            return "array"
        if origin is dict:
            return "object"
        return "string"

    failures = []
    for trial in range(100):
        hint = rand_hint(0)
        value = rand_value(hint)
        exp = ground_truth_json_type(hint)

        # -- Python → JSON Schema type --
        got = type_hint_to_string_type(hint)
        if got != exp:
            failures.append(f"[{trial}] type_hint_to_string_type({hint!r}) = {got!r}, want {exp!r}")
            continue

        # -- Python value → JSON string --
        try:
            json_str = json.dumps(value)
        except (TypeError, ValueError) as exc:
            failures.append(f"[{trial}] json.dumps raised {exc} for hint={hint!r} value={value!r}")
            continue

        # -- JSON string → Python value --
        try:
            recovered = json.loads(json_str)
        except (ValueError, TypeError) as exc:
            failures.append(f"[{trial}] json.loads raised {exc}")
            continue

        # round-trip structural equality (tuples serialise as lists, both sides agree)
        if json.dumps(recovered) != json_str:
            failures.append(
                f"[{trial}] round-trip mismatch: {value!r} → {json_str!r} → {recovered!r}"
            )
            continue

        # -- Valid value must pass validate_output_field_against_schema --
        schema = agdata(v=hint)
        err = _validate_output_field_against_schema("v", recovered, schema)
        if err is not None:
            failures.append(
                f"[{trial}] valid value rejected — hint={hint!r} value={recovered!r} err={err!r}"
            )
            continue

        # -- Wrong container type must be rejected for bare/generic hints --
        # Parameterised generics with no top-level validation (e.g. dict[str, int])
        # intentionally skip this check — only bare container types and list[T] validate.
        wrong = {"__wrong__": 1} if exp == "array" else [1, 2] if exp == "object" else None
        if wrong is not None:
            validates_top_level = (
                isinstance(hint, type)  # bare list / dict / tuple
                or get_origin(hint) in (list, tuple, dict)  # generic list[T] / dict[K,V] / tuple[T]
            )
            if validates_top_level:
                err2 = _validate_output_field_against_schema("v", wrong, schema)
                if err2 is None:
                    failures.append(
                        f"[{trial}] wrong value not rejected — hint={hint!r} wrong={wrong!r}"
                    )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])


def test_random_schema_prompt_examples_parseable():
    """100 randomly generated schemas: the example in every auto-generated tool
    description must be valid JSON AND must pass validate_output_field_against_schema.

    Also checks that error messages (wrong container type) include a parseable
    example that itself validates correctly.
    """
    import random
    from agency.agtype import (
        get_json_example_for_type_hint,
        type_hint_to_string_type,
        get_return_tool_description_prompt,
        validate_output_field_against_schema,
    )
    from agency.agtype import agrawstring, agtype, agfile, agbinary, agimage

    _validate_output_field_against_schema = validate_output_field_against_schema

    rng = random.Random(20240625)

    LEAF_TYPES = [str, int, float, bool, agrawstring, agfile, agbinary, agimage]

    def rand_hint(depth: int):
        if depth >= 4 or (depth > 0 and rng.random() < 0.30 * depth):
            return rng.choice(LEAF_TYPES)
        kind = rng.choice(("list", "dict", "tuple", "list_of_dicts"))
        n = rng.randint(1, 4)
        if kind == "list":
            return list[rand_hint(depth + 1)]
        if kind == "dict":
            return dict[str, rand_hint(depth + 1)]
        if kind == "tuple":
            inners = tuple(rand_hint(depth + 1) for _ in range(n))
            return tuple[inners] if len(inners) > 1 else tuple[inners[0]]
        # literal list-of-dicts: [{key: type, ...}]
        keys = [f"f{i}" for i in range(rng.randint(1, 3))]
        return [{k: rng.choice([str, int, float, bool]) for k in keys}]

    failures = []
    for trial in range(100):
        hint = rand_hint(0)

        # -- get_json_example_for_type_hint must produce valid JSON --
        ex_str = get_json_example_for_type_hint(hint)
        try:
            ex_val = json.loads(ex_str)
        except (ValueError, TypeError) as exc:
            failures.append(
                f"[{trial}] get_json_example_for_type_hint({hint!r}) = {ex_str!r} is not valid JSON: {exc}"
            )
            continue

        # -- that example must pass validate_output_field_against_schema --
        schema = agdata(v=hint)
        err = _validate_output_field_against_schema("v", ex_val, schema)
        if err is not None:
            failures.append(
                f"[{trial}] example from hint {hint!r} = {ex_val!r} failed validation: {err}"
            )
            continue

        # -- example must appear in the generated value description --
        _, vd = get_return_tool_description_prompt("v", hint)
        if not isinstance(hint, type) or not issubclass(hint, agtype):
            # agtype delegates to its own classmethods; skip appearance check there
            if ex_str not in vd:
                failures.append(f"[{trial}] example {ex_str!r} not found in value_desc {vd!r}")
                continue

        # -- the tool JSON Schema type must match the example's top-level type --
        json_type = type_hint_to_string_type(hint)
        type_ok = (
            (json_type == "array" and isinstance(ex_val, list))
            or (json_type == "object" and isinstance(ex_val, dict))
            or (json_type == "string" and isinstance(ex_val, str))
            or (json_type == "integer" and isinstance(ex_val, int) and not isinstance(ex_val, bool))
            or (json_type == "number" and isinstance(ex_val, float))
            or (json_type == "boolean" and isinstance(ex_val, bool))
        )
        if not type_ok:
            failures.append(
                f"[{trial}] example type mismatch — hint={hint!r} json_type={json_type!r} "
                f"example={ex_val!r} (type {type(ex_val).__name__})"
            )

    assert not failures, f"{len(failures)}/100 trials failed:\n" + "\n".join(failures[:20])


# ---------------------------------------------------------------------------
# host_mcp_tools / sandbox_mcp_tools / policy
# ---------------------------------------------------------------------------


def test_host_mcp_tools_defaults_to_the_default_set():
    from agency.agskill import _DEFAULT_HOST_MCP_TOOLS

    s = agskill(name="s", prompt="")
    assert [t.name for t in s.host_mcp_tools] == [t.name for t in _DEFAULT_HOST_MCP_TOOLS]


def test_add_host_mcp_tools_extends_the_defaults():
    from agency.agskill import _DEFAULT_HOST_MCP_TOOLS

    extra = agtool(name="extra", description="d", fn=_noop)
    s = agskill(name="s", prompt="", add_host_mcp_tools=[extra])
    assert [t.name for t in s.host_mcp_tools] == [t.name for t in _DEFAULT_HOST_MCP_TOOLS] + [
        "extra"
    ]


def test_sandbox_mcp_tools_defaults_to_empty():
    s = agskill(name="s", prompt="")
    assert s.sandbox_mcp_tools == []


def test_add_sandbox_mcp_tools_populates_sandbox_mcp_tools():
    sbx_tool = agtool(name="sbx", description="d", fn=_noop)
    s = agskill(name="s", prompt="", add_sandbox_mcp_tools=[sbx_tool])
    assert [t.name for t in s.sandbox_mcp_tools] == ["sbx"]


def test_policy_defaults_to_a_fresh_agpolicy():
    from agency.agpolicy import agpolicy

    s = agskill(name="s", prompt="")
    assert isinstance(s.policy, agpolicy)
    assert s.policy.tool_hooks is None
    assert s.policy.default_to_deny is False


def test_policy_stored_verbatim_when_supplied():
    from agency.agpolicy import agpolicy

    policy = agpolicy(default_to_deny=True)
    s = agskill(name="s", prompt="", policy=policy)
    assert s.policy is policy
