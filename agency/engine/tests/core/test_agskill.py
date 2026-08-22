"""Tests for agskill as a self-contained ReAct skill."""

import json
from unittest.mock import MagicMock
from agency.agdata import agdata, agerror
from agency.agcontext import agcontext
from agency.agconfig import agConfig
from agency.agschema import agschema
from agency.agskill import agskill
from agency.llm.agllm import _AgLLMFields, agllm
from agency.agtool import agtool
from agency.agent import agent as _agent_cls, agent_state as _agent_state_cls

LLM_MAX_RETRIES = _AgLLMFields.max_retries.default
LLM_IDLE_TIMEOUT = _AgLLMFields.idle_timeout.default
LLM_STREAM_TIMEOUT = _AgLLMFields.stream_timeout.default

LLM_CONFIG = {"api_key": "test", "model": ""}
LLM = agllm(agConfig({"agllm_backend": LLM_CONFIG}), context_limit=128_000)


def make_mock_agent(llm=None, sandbox=None, ping_interval_s=300, poll_interval_s=5):
    _ping = ping_interval_s
    _poll = poll_interval_s

    class _MockAgent:
        agresource_pool = MagicMock()
        ping_interval_s = _ping
        poll_interval_s = _poll
        agconfig = None
        _drain_inbox = _agent_cls._drain_inbox

    ag = _MockAgent()
    ag.llm = llm or LLM
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
    ag.terminal = MagicMock()
    ag._state = _agent_state_cls("test")
    ag.log = MagicMock()
    ag.log.token_usage = {}
    ag.agname = "test"
    ag._set_ui_state = MagicMock()
    ag._push_live_messages = MagicMock()
    ag._append_full_history = MagicMock()
    ag._next_inbox_msg = MagicMock(return_value=None)
    ag.push_token_count_update_to_ui = MagicMock()
    return ag


def _noop(arg: agdata) -> agdata:
    return agdata()


def _noop_r1(arg: agdata) -> agdata:
    return agdata(r=1)


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
        system_prompt="You are a summarisation assistant.",
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
# now lives in tests/harness/agharness_backends/test_native_loop_fast.py
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


def test_system_prompt_type_names_shown_correctly():
    from agency.agtype import agfile

    sk = agskill(
        "t",
        "",
        input_schema=agdata(n=int, s=str, doc=agfile),
        output_schema=agdata(result=float),
    )
    prompt = sk._build_system_prompt()
    assert '"n": "int"' in prompt  # input schema still uses to_json()
    assert '"s": "str"' in prompt
    assert '"doc": "file"' in prompt
    assert "result" in prompt  # output field listed by name
    assert "float" in prompt  # output field type shown as "float"


# ---------------------------------------------------------------------------
# System prompt is sent but NOT stored in history
# ---------------------------------------------------------------------------


def test_system_prompt_prepended_to_llm_call():
    # _build_initial_messages() is the shared, engine-agnostic method both
    # execute_react() and every agharness_backend's execute() build their
    # first turn from -- calling it directly tests the same contract
    # without needing a real (or execute_react-only) loop around it.
    s = make_skill()
    messages, _n_before = s._build_initial_messages(agdata(x=1), agcontext(), None, None, None)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a summarisation assistant."


def test_system_prompt_not_in_returned_history():
    s = make_skill()
    messages, _n_before = s._build_initial_messages(agdata(x=1), agcontext(), None, None, None)
    # The delta a caller appends back to agcontext.messages is messages[1:]
    # (dropping the system prompt) -- see _build_initial_messages()'s own
    # docstring on n_before/messages[n_before+1:].
    roles = [m["role"] for m in messages[1:]]
    assert "system" not in roles


def test_existing_history_included_in_call():
    s = make_skill()
    prior = agcontext(
        messages=[{"role": "user", "content": "prior"}, {"role": "assistant", "content": "ok"}]
    )
    messages, _n_before = s._build_initial_messages(agdata(x=1), prior, None, None, None)
    # system at [0], prior messages at [1] and [2], new user at [-1]
    assert messages[1]["content"] == "prior"
    assert messages[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Tool call path
# ---------------------------------------------------------------------------


# test_tool_call_executes_and_continues / test_replace_tools_overrides_defaults /
# test_replace_tools_empty_list_gives_no_tools / test_add_tools_extends_sandbox_defaults
# were retired here along with execute_react() itself: add_tools/
# replace_tools are host-authored Python tool closures with no execution
# path today for ANY engine -- execute_react() was the only one that ever
# ran them, and native.py's `_NativeBackend.execute()` explicitly rejects
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
    # input_schema validation is shared, engine-agnostic code
    # (self.input_schema.validate_input(), called directly by both
    # execute_react() and execute_harness() before any engine/backend is
    # touched) -- testing it directly here needs no LLM/loop at all.
    s = agskill(
        name="s",
        system_prompt="",
        input_schema=agdata(question=str, context=str),
    )
    error = s.input_schema.validate_input(agdata(question="hi"))
    assert error is not None
    assert "context" in error


def test_input_schema_type_error_returns_error():
    s = agskill(
        name="s",
        system_prompt="",
        input_schema=agdata(count=int),
    )
    error = s.input_schema.validate_input(agdata(count="not-an-int"))
    assert error is not None
    assert "count" in error


def test_input_schema_valid_proceeds():
    s = agskill(
        name="s",
        system_prompt="",
        input_schema=agdata(text=str),
    )
    assert s.input_schema.validate_input(agdata(text="hello")) is None


def test_input_schema_description_value_only_checks_presence():
    """Non-type-name values (descriptions) only trigger a missing-key error."""
    s = agskill(
        name="s",
        system_prompt="",
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
# tests/harness/agharness_backends/test_native_loop_fast.py's
# test_submit_output_all_fields_collected /
# test_submit_output_type_error_returns_immediate_feedback) plus a bounded
# reprompt-across-turns loop one level up in native.py's
# `_NativeBackend.execute()` (Docker-only coverage today, see
# test_native.py's TestNativeBackendRealEndToEnd).


def test_schemas_appended_to_system_prompt():
    s = agskill(
        name="s",
        system_prompt="Be helpful.",
        input_schema=agdata(text=str),
        output_schema=agdata(summary=str),
    )
    prompt = s._build_system_prompt()
    assert "Be helpful." in prompt
    assert "Input JSON format" in prompt
    assert '"text"' in prompt
    assert "return_summary" in prompt
    assert "summary" in prompt
    assert "string" in prompt  # per-field description for str output


def test_no_schemas_system_prompt_unchanged():
    s = agskill(name="s", system_prompt="Be helpful.")
    assert s._build_system_prompt() == "Be helpful."


# test_return_output_* / test_return_tool_* (all fields correct, type
# error feedback, unknown field, list/dict shapes, agrawstring passthrough,
# tool schema shape/ordering, parameter naming, any/wrong key extraction,
# term logging) were retired here: they tested the per-field `return_<field>`
# tool mechanism (agschema.make_return_output_agtool/agtool.
# make_return_output_tools), only ever called from execute_react()'s
# _build_toolkit(). Native's structured output uses a single `submit_output`
# MCP tool instead -- fast coverage for the all-fields-correct and
# type-error-immediate-feedback cases now lives in
# tests/harness/agharness_backends/test_native_loop_fast.py
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
# tests/harness/agharness_backends/test_native_loop_fast.py's
# test_oversized_tool_output_is_offloaded_to_a_file.

# ---------------------------------------------------------------------------
# Skill-exit sandbox teardown (agskill.py's own run()/_task() finally block)
#
# Per-tool commit/rollback is gone (agtool.py's dispatch_tools() now just
# calls sandbox.stop() unconditionally after each call -- a separate,
# already-updated concern, not tested here). The remaining rollback boundary
# lives one level up, in agskill.py's _task(): on a *skill's own* result
# being an error, it calls ag.sandbox.rm_container() (discarding everything
# since the last successful skill's commit()) and pushes a plain string onto
# ag.inbox describing the revert -- surfaced at the START of the next skill
# call via ag._drain_inbox() (see execute_react()'s loop), since the failed
# skill's own result is already final by the time _task() reaches teardown.
# On success, it calls ag.sandbox.commit() (no args -- squashing is now
# fully automatic, the old force_squash parameter is gone) and then
# ag.sandbox.stop() to hibernate again (releasing the session keyring);
# execute_react()'s output path may have re-woken the container after the
# last tool-call hibernate, and commit() itself does not stop it.
#
# These tests exercise that finally block directly by running skills through
# the real agent.run()/_task() path (not execute_react() in isolation, which
# never reaches this teardown) against a real agent and a mocked sandbox,
# with execute_react() itself replaced by a fake so the scenario -- success,
# an agerror result, or an uncaught exception -- is fully controlled. Any
# per-tool run_in_subprocess distinction is irrelevant at this layer (kept
# only in a few names/docstrings for traceability from the pre-refactor
# suite these evolved from).
# ---------------------------------------------------------------------------


def _make_sandbox_with_tracking():
    """Return a sandbox mock that records stop()/commit()/rm_container() calls."""
    sandbox = MagicMock()
    sandbox._name = "testbox"
    sandbox.stop.return_value = None
    sandbox._has_pending_background_work.return_value = False
    # A bare MagicMock's auto-attribute for `.persistent` is a truthy Mock,
    # not the real agSandbox default (False) -- since dispatch_tools() now
    # short-circuits on `not sandbox.persistent` before ever consulting
    # `_has_pending_background_work()` (see agtool.py), leaving this unset
    # would silently skip that check (and any side_effect list queued on
    # it) in every test using this helper. Match the real default here.
    sandbox.persistent = False
    return sandbox


def _run_skill_via_agent(s, sandbox, skill_input=None):
    """Run *s* to completion through a real agent.run() -- the only code
    path that reaches agskill.py's _task() finally block -- against a real
    agent and *sandbox* (typically a MagicMock so commit()/rm_container()/
    inbox.put() calls can be asserted on). Returns (ag, resolved pending
    agdata)."""
    cfg = agConfig({"agllm_backend": LLM_CONFIG})
    ag = _agent_cls(agconfig=cfg, llm=LLM, sandbox=sandbox)
    ag.inbox = MagicMock()
    pending = ag.run(s, skill_input if skill_input is not None else agdata(x=1))
    pending.wait()
    return ag, pending


def test_tool_success_commits_and_stops():
    """A skill run that completes successfully must call sandbox.commit()
    (no args) then sandbox.stop() in agskill.py's _task() finally block,
    and must not rm_container() or push anything onto the inbox."""
    sandbox = _make_sandbox_with_tracking()
    s = make_skill()
    s.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agdata(result="ok"),
        prev_ctx,
        [],
    )

    ag, _ = _run_skill_via_agent(s, sandbox)

    sandbox.commit.assert_called_once_with()
    sandbox.stop.assert_called_once_with()
    method_names = [c[0] for c in sandbox.method_calls]
    assert method_names.index("commit") < method_names.index("stop")
    sandbox.rm_container.assert_not_called()
    ag.inbox.put.assert_not_called()


def test_tool_success_defers_hibernate_when_background_work_pending():
    """Same deferral as per-tool stop(): do not hibernate over live
    background work at skill teardown."""
    sandbox = _make_sandbox_with_tracking()
    sandbox._has_pending_background_work.return_value = True
    s = make_skill()
    s.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agdata(result="ok"),
        prev_ctx,
        [],
    )

    _run_skill_via_agent(s, sandbox)

    sandbox.commit.assert_called_once_with()
    sandbox.stop.assert_not_called()


def test_tool_failure_triggers_stop_without_commit():
    """When the skill's own result is an error, the finally block must call
    sandbox.rm_container() (discard since the last successful commit)
    instead of sandbox.commit()."""
    sandbox = _make_sandbox_with_tracking()
    s = make_skill()
    s.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agerror("boom"),
        prev_ctx,
        [],
    )

    ag, pending = _run_skill_via_agent(s, sandbox)

    sandbox.rm_container.assert_called_once_with()
    sandbox.commit.assert_not_called()
    assert pending._data.get("error") == "boom"


def test_tool_failure_adds_workspace_reverted_note():
    """The old "workspace_reverted key injected into the tool result JSON"
    behavior is gone entirely -- the revert notice now goes on ag.inbox as a
    plain string (queue.Queue[str]), not in the failed skill's own result,
    since that result is already final by the time _task() reaches
    teardown. The note is meant to surface at the START of the next skill
    call via ag._drain_inbox()."""
    sandbox = _make_sandbox_with_tracking()
    s = make_skill()
    s.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agerror("disk full"),
        prev_ctx,
        [],
    )

    ag, pending = _run_skill_via_agent(s, sandbox)

    ag.inbox.put.assert_called_once()
    (note,), _kwargs = ag.inbox.put.call_args
    assert isinstance(note, str)
    assert "revert" in note.lower() or "discard" in note.lower()
    # The failed skill's own result carries only its own error -- no
    # revert-related key was added to it at this layer.
    assert pending._data == {"error": "disk full"}


def test_tool_failure_reverts_even_without_subprocess():
    """The revert-and-notify teardown is triggered purely by the skill's own
    result being an error -- it doesn't matter whether any tool ran at all,
    let alone in a subprocess; _task() only ever inspects outer_result."""
    sandbox = MagicMock()
    s = make_skill()
    s.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agerror("nope"),
        prev_ctx,
        [],
    )

    ag, pending = _run_skill_via_agent(s, sandbox)

    sandbox.rm_container.assert_called_once_with()
    ag.inbox.put.assert_called_once()
    assert pending._data.get("error") == "nope"


def test_run_in_subprocess_false_still_stops():
    """The commit()/rm_container() decision is made fresh for every skill
    call on the same agent -- a later call's failure must still trigger
    rm_container() (and a fresh inbox note) even though an earlier call
    already committed successfully."""
    sandbox = _make_sandbox_with_tracking()
    ok_skill = make_skill(name="ok")
    ok_skill.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agdata(result="ok"),
        prev_ctx,
        [],
    )
    bad_skill = make_skill(name="bad")
    bad_skill.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agerror("second call failed"),
        prev_ctx,
        [],
    )

    cfg = agConfig({"agllm_backend": LLM_CONFIG})
    ag = _agent_cls(agconfig=cfg, llm=LLM, sandbox=sandbox)
    ag.inbox = MagicMock()
    ag.run(ok_skill, agdata(x=1)).wait()
    ag.run(bad_skill, agdata(x=1)).wait()

    assert sandbox.commit.call_count == 1
    assert sandbox.stop.call_count == 1  # success path hibernates after commit
    assert sandbox.rm_container.call_count == 1
    ag.inbox.put.assert_called_once()


def test_tool_exception_triggers_stop_without_commit():
    """An uncaught exception raised out of execute_react() is caught by
    _task()'s own outer try/except and turned into an agerror -- which must
    then trigger the same rm_container()-without-commit teardown as an
    ordinary agerror result."""
    sandbox = _make_sandbox_with_tracking()
    s = make_skill()

    def _raise(ag, prev_ctx, skill_input, max_steps=None):
        raise RuntimeError("exploded")

    s.execute_harness = _raise

    ag, pending = _run_skill_via_agent(s, sandbox)

    sandbox.rm_container.assert_called_once_with()
    sandbox.commit.assert_not_called()
    assert "exploded" in pending._data.get("error", "")


def test_run_in_subprocess_false_success_still_commits():
    """A skill call's own commit()/rm_container() decision doesn't carry
    over from an earlier call on the same agent -- a successful call must
    still commit() even immediately after a prior call's failure already
    triggered a revert."""
    sandbox = _make_sandbox_with_tracking()
    bad_skill = make_skill(name="bad")
    bad_skill.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agerror("first call failed"),
        prev_ctx,
        [],
    )
    ok_skill = make_skill(name="ok")
    ok_skill.execute_harness = lambda ag, prev_ctx, skill_input, max_steps=None: (
        agdata(result="ok"),
        prev_ctx,
        [],
    )

    cfg = agConfig({"agllm_backend": LLM_CONFIG})
    ag = _agent_cls(agconfig=cfg, llm=LLM, sandbox=sandbox)
    ag.inbox = MagicMock()
    ag.run(bad_skill, agdata(x=1)).wait()
    ag.run(ok_skill, agdata(x=1)).wait()

    assert sandbox.rm_container.call_count == 1
    assert sandbox.commit.call_count == 1
    assert sandbox.stop.call_count == 1  # success path hibernates after commit


# test_pending_background_work_defers_stop_entirely was retired here: it
# tested agtool.py's dispatch_tools() deferring sandbox.stop() while
# `_has_pending_background_work()` is true -- the per-tool-call hibernate
# model itself was already retired in Phase 1 (persistent containers), so
# this check (and the "wait_for_processes() right after" ordering the
# comment describes) no longer exists in the new execute_harness() path.

# test_pending_background_work_omits_workspace_reverted_note_on_error was
# retired here: same retired mechanism, errored-tool-call path.


def test_tool_exception_with_run_in_subprocess_false_still_stops():
    """Same exception-triggers-revert teardown as
    test_tool_exception_triggers_stop_without_commit, confirmed here via a
    plain ValueError (rather than RuntimeError) raised very early -- before
    execute_react() ever reaches a tool call -- to show the finally block's
    rm_container()+inbox path doesn't depend on how far execute_react() got
    before failing."""
    sandbox = _make_sandbox_with_tracking()
    s = make_skill()

    def _raise_early(ag, prev_ctx, skill_input, max_steps=None):
        raise ValueError("early failure")

    s.execute_harness = _raise_early

    ag, pending = _run_skill_via_agent(s, sandbox)

    sandbox.rm_container.assert_called_once_with()
    sandbox.commit.assert_not_called()
    assert "early failure" in pending._data.get("error", "")
    ag.inbox.put.assert_called_once()
    (note,), _kwargs = ag.inbox.put.call_args
    assert "revert" in note.lower() or "discard" in note.lower()


# test_tool_exception_with_pending_background_work_defers_stop was
# retired here: same retired dispatch_tools() pending-background-work
# stop-deferral mechanism, exception-handler path.


# test_dispatch_tools_accepts_camel_case_llm_arguments /
# test_tool_timeout_uses_agent_provided_value / test_tool_timeout_ignored_if_not_int
# were retired here: all three tested agtool.py's dispatch_tools()-specific
# mechanics (camelCase argument-key coercion via agdata.from_json(), and
# per-call `timeout` override from LLM tool-call args) -- only ever
# exercised via execute_react(). Native's built-in tools have fixed
# timeouts and no camelCase-coercion step of their own (they parse JSON
# arguments directly, see _native_in_container_entrypoint.py's
# _parse_tool_args), so neither mechanism carries over.


# ---------------------------------------------------------------------------
# build_llm_kwargs
# ---------------------------------------------------------------------------

from agency.llm.agllm import agllm as _agllm_mod

build_llm_kwargs = _agllm_mod.build_llm_kwargs


def _llm_cfg(**fields) -> agConfig:
    """Test helper: wrap agllm_backend fields in an agConfig."""
    return agConfig({"agllm_backend": fields})


def test_build_llm_kwargs_model_and_messages():
    msgs = [{"role": "user", "content": "hi"}]
    kw = build_llm_kwargs(_llm_cfg(model=""), msgs, None)
    assert kw["model"] == ""
    assert kw["messages"] == msgs


def test_build_llm_kwargs_strips_private_keys():
    msgs = [{"role": "assistant", "content": "ok", "_thinking": "secret"}]
    kw = build_llm_kwargs(_llm_cfg(model="m"), msgs, None)
    assert "_thinking" not in kw["messages"][0]
    assert "content" in kw["messages"][0]


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
# build_assistant_msg
# ---------------------------------------------------------------------------

build_assistant_msg = _agllm_mod.build_assistant_msg


def test_build_assistant_msg_plain_content():
    msg = build_assistant_msg(["hello", " world"], [], {})
    assert msg["role"] == "assistant"
    assert msg["content"] == "hello world"


def test_build_assistant_msg_reasoning_parts():
    msg = build_assistant_msg(["answer"], ["think ", "harder"], {})
    assert msg["_thinking"] == "think harder"
    assert msg["content"] == "answer"


def test_build_assistant_msg_think_tag_stripped():
    msg = build_assistant_msg(["<think>reasoning</think>answer"], [], {})
    assert msg.get("_thinking") == "reasoning"
    assert msg["content"] == "answer"


def test_build_assistant_msg_tool_calls_included():
    tc = {0: {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}}
    msg = build_assistant_msg([], [], tc)
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "f"


def test_build_assistant_msg_tool_calls_sorted_by_index():
    tc = {
        1: {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        0: {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
    }
    msg = build_assistant_msg([], [], tc)
    assert msg["tool_calls"][0]["function"]["name"] == "a"
    assert msg["tool_calls"][1]["function"]["name"] == "b"


# ---------------------------------------------------------------------------
# _drain_inbox
# ---------------------------------------------------------------------------


def test_drain_inbox_empty_queue_returns_false():
    ag = make_mock_agent()
    ag._next_inbox_msg = MagicMock(return_value=None)
    messages = []
    assert ag._drain_inbox(messages) is False
    assert messages == []


def test_drain_inbox_single_message_appended():
    ag = make_mock_agent()
    ag._next_inbox_msg = MagicMock(side_effect=[{"type": "message", "content": "hello"}, None])
    messages = [{"role": "system", "content": "sys"}]
    had = ag._drain_inbox(messages)
    assert had is True
    assert messages[-1] == {"type": "message", "content": "hello"}


def test_drain_inbox_multiple_messages_all_appended():
    ag = make_mock_agent()
    ag._next_inbox_msg = MagicMock(
        side_effect=[{"type": "message", "content": "msg1"}, {"type": "pause"}, None]
    )
    messages = []
    ag._drain_inbox(messages)
    assert len(messages) == 2
    assert messages[0] == {"type": "message", "content": "msg1"}
    assert messages[1] == {"type": "pause"}


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
    assert agSandbox.wait_for_processes(sb, "skill", None, None, "", 300, 5) is None


def test_wait_for_processes_no_watched_pids_attr_returns_none():
    class NoPids:
        def _has_pending_background_work(self):
            return False

    assert agSandbox.wait_for_processes(NoPids(), "skill", None, None, "", 300, 5) is None


def test_wait_for_processes_mock_sandbox_returns_none():
    from unittest.mock import MagicMock

    sb = MagicMock()
    sb._has_pending_background_work.return_value = False
    assert agSandbox.wait_for_processes(sb, "skill", None, None, "", 300, 5) is None


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
    result = agSandbox.wait_for_processes(
        sb, "skill", None, None, "", ping_interval_s=30, poll_interval_s=0.01
    )
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
        sb, "skill", None, None, "", ping_interval_s=0.02, poll_interval_s=0.01
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
        None,
        None,
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
# agskill._build_initial_messages
# ---------------------------------------------------------------------------


def test_build_initial_messages_structure():
    s = make_skill()
    history = agcontext(messages=[{"role": "user", "content": "prior"}])
    msgs, n_before = s._build_initial_messages(agdata(q="hi"), history, None, None, None)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "prior"
    assert msgs[-1]["role"] == "user"
    assert n_before == 1


def test_build_initial_messages_fires_live_fn():
    s = make_skill()
    live_calls = []
    s._build_initial_messages(agdata(), agcontext(), None, lambda m: live_calls.append(m), None)
    assert len(live_calls) == 1


def test_build_initial_messages_fires_full_history_fn():
    s = make_skill()
    history_items = []
    s._build_initial_messages(
        agdata(q="test"), agcontext(), None, None, lambda m: history_items.append(m["role"])
    )
    assert "system" in history_items
    assert "user" in history_items


# ---------------------------------------------------------------------------
# run() — sandbox process monitoring
# ---------------------------------------------------------------------------


# test_run_continues_loop_when_sandbox_has_live_pids /
# test_run_injects_process_completed_message / test_run_clean_sandbox_returns_immediately
# were retired here: they tested execute_react()'s specific "loop back and
# reprompt the model" behavior when agSandbox.wait_for_processes()/
# get_live_pids() finds pending background work after a final answer --
# retired along with the per-tool-call hibernate model itself (Phase 1).
# execute_harness() now calls wait_for_processes() once, non-looping, for
# native only (see that method's own comment) -- there is no equivalent
# "reprompt and continue in the same call" behavior to test for any engine
# today.

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# run() — thinking extraction from stream
# ---------------------------------------------------------------------------


# test_run_extracts_thinking_from_think_tag was retired here, not ported:
# it tested agllm.py's own <think>-tag stripping (build_assistant_msg's
# `_thinking` extraction) around execute_react()'s streaming reassembly.
# Native's own reassembly (_native_in_container_entrypoint.py's
# _dispatch_via_terminus) does no such stripping today -- a genuine
# behavior gap, not a like-for-like port; noted in native.py's own
# "Known gaps" docstring section rather than silently dropped.

# test_run_returns_token_counts was retired here: covered fast, for native,
# by tests/harness/agharness_backends/test_native_loop_fast.py's
# test_token_usage_is_tracked (proving _run_react_loop()'s response usage
# is real, accumulated per-dispatch data, not an execute_react()-only
# concern anymore).


# ---------------------------------------------------------------------------
# run() defensively copies skill_input before mutating it
# ---------------------------------------------------------------------------
#
# prepare_inputs_in_sandbox() (called from execute_react(), itself called from
# run()'s _task()) mutates its skill_input argument in place -- offloaded
# agtype/oversized fields get overwritten with a sandbox path reference. If a
# caller hands the *same* agdata object to more than one concurrently-running
# agent.run() call (a real pattern: fanning one shared input out to several
# agents, e.g. autoresearch's ClassificationTeam.run()), those calls race on
# that shared mutation -- whichever run finishes its offload last clobbers the
# field with its own path, leaving every other run trying to read a file that
# only exists in that one run's own sandbox. _task() must give each run its
# own private copy from the moment it starts, regardless of what the caller
# does with the object it passed in.


def test_run_does_not_mutate_callers_shared_input_object():
    """Regression test: run() must not mutate the skill_input object the
    caller passed in -- prepare_inputs_in_sandbox()'s offload rewrite must
    land on a private copy, not the caller's own object."""
    from agency.agschema import agSchemaConfig
    from agency.sandbox import agSandboxBackendConfig

    # Force the docker sandbox backend: sandbox' "auto" selection
    # prefers podman over docker when both are usable, but CI's
    # images/build.sh only builds/tags agency-sandbox:latest for docker, so
    # podman has no local image and would try (and fail) to pull one.
    cfg = agConfig(
        agSchemaConfig(input_offload_chars=10),
        agSandboxBackendConfig(backend="docker"),
        {"agllm_backend": LLM_CONFIG},
    )
    s = agskill(name="offload_test", system_prompt="", input_schema=agdata(text=str))

    def fake_execute_react(ag, prev_ctx, skill_input, max_steps=None, **_):
        s.input_schema.prepare_inputs_in_sandbox(
            skill_input,
            ag.sandbox,
            s.name,
            context_limit=ag.llm.context_limit,
            agconfig=ag.agconfig,
        )
        return agdata(answer=skill_input.text), prev_ctx, []

    s.execute_harness = fake_execute_react

    shared_input = agdata(text="x" * 100)
    ag = _agent_cls(agconfig=cfg)
    try:
        result = ag.run(s, shared_input)
        assert "saved to" in result.answer  # this run's own copy WAS offloaded
        assert shared_input.text == "x" * 100  # the caller's object was not
    finally:
        if ag.sandbox is not None:
            ag.sandbox.destroy()


def test_run_gives_concurrent_runs_sharing_one_input_independent_copies():
    """Two agents' run() calls sharing one input agdata (the exact
    ClassificationTeam.run() pattern) must each read back their own
    offloaded file, not race on the shared object's mutation."""
    from agency.agschema import agSchemaConfig
    from agency.sandbox import agSandboxBackendConfig

    # Force the docker sandbox backend: sandbox' "auto" selection
    # prefers podman over docker when both are usable, but CI's
    # images/build.sh only builds/tags agency-sandbox:latest for docker, so
    # podman has no local image and would try (and fail) to pull one.
    cfg = agConfig(
        agSchemaConfig(input_offload_chars=10),
        agSandboxBackendConfig(backend="docker"),
        {"agllm_backend": LLM_CONFIG},
    )
    s = agskill(name="offload_test", system_prompt="", input_schema=agdata(text=str))

    def fake_execute_react(ag, prev_ctx, skill_input, max_steps=None, **_):
        s.input_schema.prepare_inputs_in_sandbox(
            skill_input,
            ag.sandbox,
            s.name,
            context_limit=ag.llm.context_limit,
            agconfig=ag.agconfig,
        )
        from agency.tools.read import make_read

        path = skill_input.text.split("saved to ")[1].split(" —")[0]
        r = make_read(ag.sandbox).fn(agdata(file_path=path))
        return agdata(answer=r.content), prev_ctx, []

    s.execute_harness = fake_execute_react

    shared_input = agdata(text="x" * 100)
    agents = [_agent_cls(agconfig=cfg) for _ in range(2)]
    try:
        pending = [a.run(s, shared_input) for a in agents]
        for p in pending:
            assert "x" * 100 in p.answer
        assert shared_input.text == "x" * 100
    finally:
        for a in agents:
            if a.sandbox is not None:
                a.sandbox.destroy()


# plan_mode / replace_tools were removed from agskill entirely in this
# refactor (see agency/agskill.py) -- add_tools was renamed to
# add_host_mcp_tools and now only extends the default host_mcp_tools set,
# never replaces it, so there is no "replace"/"plan_mode suppresses tools"
# concept left to test. See test_add_host_mcp_tools_extends_the_defaults
# and test_host_mcp_tools_defaults_to_the_default_set below for the current
# equivalent coverage.


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

    s = agskill(name="s", system_prompt="")
    assert [t.name for t in s.host_mcp_tools] == [t.name for t in _DEFAULT_HOST_MCP_TOOLS]


def test_add_host_mcp_tools_extends_the_defaults():
    from agency.agskill import _DEFAULT_HOST_MCP_TOOLS

    extra = agtool(name="extra", description="d", fn=_noop)
    s = agskill(name="s", system_prompt="", add_host_mcp_tools=[extra])
    assert [t.name for t in s.host_mcp_tools] == [t.name for t in _DEFAULT_HOST_MCP_TOOLS] + [
        "extra"
    ]


def test_sandbox_mcp_tools_defaults_to_empty():
    s = agskill(name="s", system_prompt="")
    assert s.sandbox_mcp_tools == []


def test_add_sandbox_mcp_tools_populates_sandbox_mcp_tools():
    sbx_tool = agtool(name="sbx", description="d", fn=_noop)
    s = agskill(name="s", system_prompt="", add_sandbox_mcp_tools=[sbx_tool])
    assert [t.name for t in s.sandbox_mcp_tools] == ["sbx"]


def test_policy_defaults_to_a_fresh_agpolicy():
    from agency.agpolicy import agpolicy

    s = agskill(name="s", system_prompt="")
    assert isinstance(s.policy, agpolicy)
    assert s.policy.tool_hooks is None
    assert s.policy.default_to_deny is False


def test_policy_stored_verbatim_when_supplied():
    from agency.agpolicy import agpolicy

    policy = agpolicy(default_to_deny=True)
    s = agskill(name="s", system_prompt="", policy=policy)
    assert s.policy is policy
