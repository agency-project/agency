"""Tests for agharness.py -- the thin, engine-agnostic glue shared by every
agharness_backends/* concrete backend."""

from __future__ import annotations

from unittest.mock import MagicMock

from agency import agharness
from agency.agcontext import agcontext
from agency.agdata import agdata
from agency.agskill import agskill


def _make_agent(agname="test-agent"):
    ag = MagicMock()
    ag.agname = agname
    return ag


def test_materialize_config_home_creates_isolated_directory():
    ag = _make_agent()
    d1 = agharness.materialize_config_home(ag, token="t1", base_url="http://x")
    d2 = agharness.materialize_config_home(ag, token="t2", base_url="http://x")
    assert d1.is_dir()
    assert d2.is_dir()
    assert d1 != d2  # each launch gets its own directory
    agharness.cleanup_config_home(d1)
    agharness.cleanup_config_home(d2)
    assert not d1.exists()
    assert not d2.exists()


def test_cleanup_config_home_is_idempotent(tmp_path):
    d = tmp_path / "nonexistent"
    agharness.cleanup_config_home(d)  # must not raise


def test_build_harness_messages_keeps_every_canonical_input_part():
    skill = MagicMock()
    skill._build_system_prompt.return_value = "system policy"
    skill._build_user_content.return_value = [
        {"type": "text", "text": "current task"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    skill.output_schema = None
    previous = agcontext(messages=[{"role": "assistant", "content": "prior answer"}])

    messages = agharness.build_harness_messages(
        skill, previous, agdata(task="go"), file_notice="saved at /workspace/input.txt"
    )

    assert messages.system_instructions == "system policy"
    assert messages.previous_context == ({"role": "assistant", "content": "prior answer"},)
    assert messages.current_user_input == "current task"
    assert messages.file_notices == ("saved at /workspace/input.txt",)
    assert messages.attachments[0]["type"] == "image_url"

    rendered = agharness.render_harness_messages(messages)
    assert "[SYSTEM INSTRUCTIONS]" in rendered
    assert "prior answer" in rendered
    assert "/workspace/input.txt" in rendered
    assert "data:image/png;base64,abc" in rendered

    resumed = agharness.render_harness_messages(messages, include_previous_context=False)
    assert "[PREVIOUS CONTEXT]" not in resumed
    assert "prior answer" not in resumed
    assert "[CURRENT USER INPUT]\ncurrent task" in resumed


def test_finalize_harness_result_wraps_raw_text():
    skill = agskill(name="s", system_prompt="do the thing")

    result = agharness.finalize_harness_result(
        agharness.HarnessResult(final_text="done"), skill, MagicMock()
    )

    assert result == agdata(result="done")


def test_finalize_harness_result_validates_json_output():
    skill = agskill(
        name="s", system_prompt="do the thing", output_schema=agdata(answer=str, count=int)
    )

    result = agharness.finalize_harness_result(
        agharness.HarnessResult(final_text='{"answer": "yes", "count": 2}'),
        skill,
        MagicMock(),
    )

    assert result == agdata(answer="yes", count=2)


def test_finalize_harness_result_recovers_submitted_fields_once():
    schema = MagicMock()
    schema.check.return_value = []
    skill = MagicMock(output_schema=schema)
    sandbox = MagicMock()

    result = agharness.finalize_harness_result(
        agharness.HarnessResult(submitted_fields={"answer": "yes"}), skill, sandbox
    )

    assert result == agdata(answer="yes")
    schema.recover_outputs.assert_called_once_with(result, sandbox)


def test_run_harness_cli_uses_sandbox_stdin_workspace_and_pid_wiring():
    ag = _make_agent()
    ag.agconfig = MagicMock()
    ag.sandbox = MagicMock()
    handle = MagicMock()
    handle.wait.return_value = ("out", "", 0)

    from unittest.mock import patch

    with (
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as ptrace_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox") as wire,
    ):
        ptrace_cls.return_value.launch.return_value = handle
        result = agharness.run_harness_cli(
            ag, ["/bin/harness"], {"PATH": "/bin"}, stdin="complete task"
        )

    assert result == ("out", "", 0)
    assert ptrace_cls.return_value.launch.call_args.kwargs["cwd"] == "/workspace"
    assert ptrace_cls.return_value.launch.call_args.kwargs["sandbox"] is ag.sandbox
    assert ptrace_cls.return_value.launch.call_args.kwargs["stdin"] == "complete task"
    wire.assert_called_once_with(handle, ag.sandbox)


def test_run_harness_cli_kills_and_reaps_after_timeout():
    ag = _make_agent()
    ag.agconfig = MagicMock()
    ag.sandbox = None
    handle = MagicMock()
    handle.wait.side_effect = [("partial", "", -1), ("final", "killed", -9)]

    from unittest.mock import patch

    with patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as ptrace_cls:
        ptrace_cls.return_value.launch.return_value = handle
        stdout, stderr, rc = agharness.run_harness_cli(
            ag, ["/bin/harness"], {}, stdin="task", timeout_s=0.01
        )

    assert (stdout, stderr, rc) == ("final", "killed", -1)
    handle.kill.assert_called_once_with()
    assert handle.wait.call_count == 2


def test_execute_harness_builds_canonical_input_before_adapter_dispatch():
    skill = agskill(name="s", system_prompt="system policy")
    previous = agcontext(messages=[{"role": "assistant", "content": "prior"}])
    ag = MagicMock()
    ag.engine = "opencode"
    ag.agconfig = MagicMock()
    ag.sandbox = MagicMock()
    ag.llm.context_limit = 1000
    backend = MagicMock()
    backend.execute.return_value = (agdata(result="ok"), previous, [])

    from unittest.mock import patch

    with patch(
        "agency.agharness_internal.agharness_backends.base.agharness_backend.for_config",
        return_value=backend,
    ):
        result, _, _ = skill.execute_harness(ag, previous, agdata(task="current"))

    assert result.result == "ok"
    canonical = backend.execute.call_args.kwargs["canonical_input"]
    assert canonical.system_instructions.startswith("system policy")
    assert canonical.previous_context[0]["content"] == "prior"
    assert "current" in canonical.current_user_input


def test_build_output_format_instruction_none_when_no_schema():
    skill = agskill(name="s", system_prompt="do the thing")
    assert agharness.build_output_format_instruction(skill) is None


def test_build_output_format_instruction_none_for_raw_string_schema():
    from agency.agtype import agrawstring

    skill = agskill(name="s", system_prompt="do the thing", output_schema=agdata(text=agrawstring))
    assert agharness.build_output_format_instruction(skill) is None


def test_build_output_format_instruction_present_for_structured_schema():
    skill = agskill(
        name="s", system_prompt="do the thing", output_schema=agdata(answer=str, count=int)
    )
    instruction = agharness.build_output_format_instruction(skill)
    assert instruction is not None
    assert "JSON object" in instruction
    assert "answer" in instruction
    assert "count" in instruction


def test_build_mcp_output_format_instruction_none_when_no_schema():
    skill = agskill(name="s", system_prompt="do the thing")
    assert agharness.build_mcp_output_format_instruction(skill) is None


def test_build_mcp_output_format_instruction_none_for_raw_string_schema():
    from agency.agtype import agrawstring

    skill = agskill(name="s", system_prompt="do the thing", output_schema=agdata(text=agrawstring))
    assert agharness.build_mcp_output_format_instruction(skill) is None


def test_build_mcp_output_format_instruction_present_for_structured_schema():
    skill = agskill(
        name="s", system_prompt="do the thing", output_schema=agdata(answer=str, count=int)
    )
    instruction = agharness.build_mcp_output_format_instruction(skill)
    assert instruction is not None
    assert "submit_output" in instruction
    assert "answer" in instruction
    assert "count" in instruction


def test_default_policy_check_returns_allow():

    ag = _make_agent()
    ag.log = MagicMock()
    policy = agharness.default_policy(ag)
    event = MagicMock(syscall="execve", argv=["/bin/echo"], path=None)
    decision = policy.check(ag, event)
    assert decision.kind == "allow"
    ag.log._tool_call.assert_called_once()


def test_default_policy_logging_failure_does_not_raise():
    ag = _make_agent()
    ag.log._tool_call.side_effect = RuntimeError("log write failed")
    policy = agharness.default_policy(ag)
    event = MagicMock(syscall="execve", argv=["/bin/echo"], path=None)
    decision = policy.check(ag, event)  # must not raise
    assert decision.kind == "allow"
