# Tests for engine.py -- AgentEngine, the per-launch host-side orchestrator.

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from agency.agdata import agdata
from agency.agcontext import agcontext
from agency.agschema import agschema
from agency.agdata import agerror
from agency.engine import engine as mod
from agency.engine.engine import AgentEngine
from agency.harness.protocol import HarnessAttemptRequest, HarnessAttemptResult, PromptPayload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TrackingLock:
    def __init__(self, events):
        self.events = events
        self.held = False

    def acquire(self):
        assert not self.held
        self.held = True
        self.events.append("acquire")

    def release(self):
        assert self.held
        self.events.append("release")
        self.held = False


class _FakeSandbox:
    def __init__(self, marker="sandbox"):
        self.marker = marker
        self.events = []
        self._lock = _TrackingLock(self.events)
        self.background_work_pending = False
        self.commit_error = None
        self.discard_error = None

    def commit(self):
        assert self._lock.held
        self.events.append("commit")
        if self.commit_error is not None:
            raise self.commit_error

    def rm_container(self):
        assert self._lock.held
        self.events.append("discard")
        if self.discard_error is not None:
            raise self.discard_error

    def _has_pending_background_work(self):
        assert self._lock.held
        self.events.append("check_background_work")
        return self.background_work_pending

    def stop(self):
        assert self._lock.held
        self.events.append("stop")


class _FakeAgent:
    output_dir = None

    def __init__(self):
        self.agconfig = SimpleNamespace(marker="agconfig")
        self.sandbox = _FakeSandbox()
        self.harness = "claude_code"
        self.agname = "test-agent"
        self.change_config_calls = []
        self.inbox = queue.Queue()

    def change_config(self, agconfig):
        self.change_config_calls.append(agconfig)

    def _drain_inbox(self, messages):
        while not self.inbox.empty():
            messages.append(self.inbox.get_nowait())


def _install_fake_host_server_manager(monkeypatch, results, collected_sequence=None):
    """Install a fake host manager and request/response daemon client."""
    holder: dict = {"requests": []}
    remaining_results = list(results)
    collected_iter = iter(collected_sequence) if collected_sequence is not None else None

    class _FakeHostServerManager:
        def __init__(self, agent, sandbox, skill, resource_pool):
            self.agent = agent
            self.sandbox = sandbox
            self.skill = skill
            self.resource_pool = resource_pool
            self.started = False
            self.stopped = False
            self.set_config_calls = []

            self.host_mcp_server = SimpleNamespace(
                collected_output=lambda: next(collected_iter) if collected_iter is not None else {}
            )
            holder["manager"] = self

        def start(self) -> str:
            self.started = True
            return "/tmp/host.sock"

        def stop(self) -> None:
            self.stopped = True

        def set_config(self, agconfig) -> None:
            self.set_config_calls.append(agconfig)

    monkeypatch.setattr(mod, "HostServerManager", _FakeHostServerManager)

    class _FakeClient:
        def __init__(self):
            self.closed = False

        def run_harness_attempt(self, request):
            holder["requests"].append(request)
            return remaining_results.pop(0)

        def close(self):
            self.closed = True

    client = _FakeClient()
    holder["client"] = client

    def fake_ensure_harness_daemon(sandbox, host_uds_path, engine_name, harness, agconfig):
        holder["daemon_sandbox"] = sandbox
        return SimpleNamespace(client=lambda: client)

    monkeypatch.setattr(mod, "ensure_harness_daemon", fake_ensure_harness_daemon)
    return holder


# ---------------------------------------------------------------------------
# __init__ / host_server_manager property
# ---------------------------------------------------------------------------


def test_init_stores_agent_and_starts_with_no_host_server_manager():
    agent = _FakeAgent()
    engine = AgentEngine(agent)
    assert engine._agent is agent
    assert engine._host_server_manager is None
    assert engine._sandbox_interaction_client is None


def test_host_server_manager_property_raises_before_run():
    engine = AgentEngine(_FakeAgent())
    with pytest.raises(RuntimeError):
        engine.host_server_manager


# ---------------------------------------------------------------------------
# set_config
# ---------------------------------------------------------------------------


def test_set_config_does_not_propagate_back_to_agent():
    agent = _FakeAgent()
    engine = AgentEngine(agent)
    new_cfg = SimpleNamespace(marker="new")
    engine.set_config(new_cfg)
    assert agent.change_config_calls == []


def test_set_config_does_not_touch_host_server_manager_when_not_yet_built():
    engine = AgentEngine(_FakeAgent())
    engine.set_config(SimpleNamespace())  # should not raise


def test_set_config_forwards_to_host_server_manager_when_built(monkeypatch):
    holder = _install_fake_host_server_manager(monkeypatch, results=[])
    agent = _FakeAgent()
    engine = AgentEngine(agent)
    engine._host_server_manager = mod.HostServerManager(agent, agent.sandbox, None, None)
    new_cfg = SimpleNamespace(marker="new")
    engine.set_config(new_cfg)
    assert holder["manager"].set_config_calls == [new_cfg]


# ---------------------------------------------------------------------------
# _build_prompt_payload / _build_retry_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_payload_uses_agharness_helpers(monkeypatch):
    from agency.harness import agharness

    monkeypatch.setattr(
        agharness, "build_user_turn_prompt", lambda skill, skill_input: "the-prompt"
    )
    monkeypatch.setattr(agharness, "build_output_format_instruction", lambda skill: "the-format")
    engine = AgentEngine(_FakeAgent())
    skill = SimpleNamespace(_build_system_prompt=lambda: "the-system")
    payload = engine._build_prompt_payload(skill, SimpleNamespace())
    assert payload == PromptPayload(
        system_instruction="the-system",
        user_content="the-prompt",
        output_instruction="the-format",
    )


def test_build_retry_prompt_mentions_missing_fields():
    engine = AgentEngine(_FakeAgent())
    payload = engine._build_retry_prompt(["a", "b"], system_instruction="the-system")
    assert payload.system_instruction == "the-system"
    assert "a" in payload.user_content and "b" in payload.user_content
    assert "submit_output" in payload.user_content
    assert payload.output_instruction is None


# ---------------------------------------------------------------------------
# _missing_output_fields
# ---------------------------------------------------------------------------


def test_missing_output_fields_empty_when_skill_has_no_output_schema():
    engine = AgentEngine(_FakeAgent())
    skill = SimpleNamespace(output_schema=None)
    assert engine._missing_output_fields(skill) == []


def test_missing_output_fields_diffs_against_collected_output():
    engine = AgentEngine(_FakeAgent())
    engine._host_server_manager = SimpleNamespace(
        host_mcp_server=SimpleNamespace(collected_output=lambda: {"summary": "hi"})
    )
    skill = SimpleNamespace(output_schema=agdata(summary=str, count=int))
    assert engine._missing_output_fields(skill) == ["count"]


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


def test_execute_uses_only_the_explicit_sandbox(monkeypatch):
    agent = _FakeAgent()
    agent_sandbox = agent.sandbox
    explicit_sandbox = _FakeSandbox(marker="explicit")
    engine = AgentEngine(agent)
    execution = agdata(result="done")

    def fake_execute_harness(*args, **_kwargs):
        assert args[4] is explicit_sandbox
        explicit_sandbox.events.append("execute")
        return execution

    monkeypatch.setattr(engine, "_execute_harness", fake_execute_harness)

    result = engine.execute(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        explicit_sandbox,
    )

    assert result is execution
    assert agent.sandbox is agent_sandbox
    assert agent_sandbox.events == []
    assert explicit_sandbox.events == [
        "acquire",
        "execute",
        "commit",
        "check_background_work",
        "stop",
        "release",
    ]


def test_execute_holds_lock_through_successful_commit_and_stop(monkeypatch):
    agent = _FakeAgent()
    engine = AgentEngine(agent)
    execution = agdata(done=True)

    def fake_execute_harness(*_args, **_kwargs):
        assert agent.sandbox._lock.held
        agent.sandbox.events.append("execute")
        return execution

    monkeypatch.setattr(engine, "_execute_harness", fake_execute_harness)

    result = engine.execute(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        agent.sandbox,
    )

    assert result is execution
    assert agent.sandbox.events == [
        "acquire",
        "execute",
        "commit",
        "check_background_work",
        "stop",
        "release",
    ]


def test_execute_discards_failed_result_before_releasing_lock(monkeypatch):
    agent = _FakeAgent()
    engine = AgentEngine(agent)
    execution = agerror("boom")
    monkeypatch.setattr(
        engine,
        "_execute_harness",
        lambda *_args, **_kwargs: agent.sandbox.events.append("execute") or execution,
    )

    result = engine.execute(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        agent.sandbox,
    )

    assert result is execution
    assert agent.sandbox.events == ["acquire", "execute", "discard", "release"]
    assert "previous skill call failed" in agent.inbox.get_nowait()


def test_execute_discards_raised_exception_and_releases_lock(monkeypatch):
    agent = _FakeAgent()
    engine = AgentEngine(agent)

    def fail(*_args, **_kwargs):
        agent.sandbox.events.append("execute")
        raise RuntimeError("harness failed")

    monkeypatch.setattr(engine, "_execute_harness", fail)

    with pytest.raises(RuntimeError, match="harness failed"):
        engine.execute(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            agent.sandbox,
        )

    assert agent.sandbox.events == ["acquire", "execute", "discard", "release"]
    assert agent.sandbox._lock.held is False


def test_execute_discards_commit_failure_and_releases_lock(monkeypatch):
    agent = _FakeAgent()
    agent.sandbox.commit_error = RuntimeError("commit failed")
    engine = AgentEngine(agent)
    execution = agdata(done=True)
    monkeypatch.setattr(engine, "_execute_harness", lambda *_args, **_kwargs: execution)

    with pytest.raises(RuntimeError, match="commit failed"):
        engine.execute(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            agent.sandbox,
        )

    assert agent.sandbox.events == [
        "acquire",
        "commit",
        "check_background_work",
        "stop",
        "discard",
        "release",
    ]
    assert agent.sandbox._lock.held is False


def test_execute_releases_lock_when_discard_fails(monkeypatch):
    agent = _FakeAgent()
    agent.sandbox.discard_error = RuntimeError("discard failed")
    engine = AgentEngine(agent)
    execution = agerror("boom")
    monkeypatch.setattr(engine, "_execute_harness", lambda *_args, **_kwargs: execution)

    with pytest.raises(RuntimeError, match="discard failed"):
        engine.execute(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            agent.sandbox,
        )

    assert agent.sandbox.events == ["acquire", "discard", "release"]
    assert agent.sandbox._lock.held is False


def test_execute_defers_stop_while_background_work_is_pending(monkeypatch):
    agent = _FakeAgent()
    agent.sandbox.background_work_pending = True
    engine = AgentEngine(agent)
    execution = agdata(done=True)
    monkeypatch.setattr(engine, "_execute_harness", lambda *_args, **_kwargs: execution)

    engine.execute(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        agent.sandbox,
    )

    assert agent.sandbox.events == [
        "acquire",
        "commit",
        "check_background_work",
        "release",
    ]


def test_execute_stops_the_host_server_manager_when_daemon_launch_fails(monkeypatch):
    holder = _install_fake_host_server_manager(monkeypatch, results=[])
    monkeypatch.setattr(
        mod,
        "ensure_harness_daemon",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    engine = AgentEngine(_FakeAgent())
    skill = SimpleNamespace(output_schema=None, max_output_schema_retries=3)
    with pytest.raises(RuntimeError, match="launch failed"):
        engine.execute(
            SimpleNamespace(), skill, SimpleNamespace(), SimpleNamespace(), engine._agent.sandbox
        )
    assert holder["manager"].started is True
    assert holder["manager"].stopped is True


def test_execute_calls_run_prompt_once_and_returns_execution_result_on_first_success(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch, results=[HarnessAttemptResult(ok=True, final_text="done")]
    )
    engine = AgentEngine(_FakeAgent())
    execution = agdata(result="done")
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: "p0")
    monkeypatch.setattr(engine, "_build_execution_result", lambda *_args: execution)
    skill = SimpleNamespace(output_schema=None, max_output_schema_retries=3)

    result = engine.execute(
        agcontext(), skill, SimpleNamespace(), SimpleNamespace(), engine._agent.sandbox
    )

    manager = holder["manager"]
    assert [request.prompt for request in holder["requests"]] == ["p0"]
    assert holder["requests"][0].harness == "claude_code"
    assert manager.sandbox is engine._agent.sandbox
    assert holder["daemon_sandbox"] is engine._agent.sandbox
    assert manager.started is True
    assert manager.stopped is True
    assert result is execution


def test_execute_stops_immediately_on_a_failed_attempt_without_retrying(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch, results=[HarnessAttemptResult(ok=False, error_message="boom")]
    )
    engine = AgentEngine(_FakeAgent())
    execution = agerror("boom")
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: "p0")
    monkeypatch.setattr(engine, "_build_execution_result", lambda *_args: execution)
    skill = SimpleNamespace(output_schema=agdata(summary=str), max_output_schema_retries=3)

    result = engine.execute(
        agcontext(), skill, SimpleNamespace(), SimpleNamespace(), engine._agent.sandbox
    )

    assert [request.prompt for request in holder["requests"]] == ["p0"]
    assert result is execution


def test_execute_retries_on_missing_output_fields_then_succeeds(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch,
        results=[
            HarnessAttemptResult(ok=True, final_text="first"),
            HarnessAttemptResult(ok=True, final_text="second"),
        ],
        collected_sequence=[{}, {"summary": "x"}],
    )
    engine = AgentEngine(_FakeAgent())
    execution = agdata(summary="x")
    p0 = PromptPayload("the-system", "p0", None)
    p1 = PromptPayload("the-system", "p1", None)
    prompts = iter([p0, p1])
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: next(prompts))
    monkeypatch.setattr(
        engine,
        "_build_retry_prompt",
        lambda missing, *, system_instruction: next(prompts),
    )
    monkeypatch.setattr(engine, "_build_execution_result", lambda *_args: execution)
    skill = SimpleNamespace(output_schema=agdata(summary=str), max_output_schema_retries=3)

    result = engine.execute(
        agcontext(), skill, SimpleNamespace(), SimpleNamespace(), engine._agent.sandbox
    )

    assert [request.prompt for request in holder["requests"]] == [p0, p1]
    assert result is execution


def test_execute_transports_and_captures_session_blobs(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch,
        results=[
            HarnessAttemptResult(
                ok=True,
                final_text="first",
                session_id="session-2",
                session_blob_b64="dXBkYXRlZA==",
            ),
            HarnessAttemptResult(
                ok=True,
                final_text="second",
                session_id="session-2",
                session_blob_b64="ZmluYWw=",
            ),
        ],
        collected_sequence=[{}, {"summary": "x"}],
    )
    agent = _FakeAgent()
    context = agcontext(
        harness_sessions={"claude_code": {"session_id": "session-1", "blob_b64": "cHJpb3I="}}
    )
    engine = AgentEngine(agent)
    execution = agdata(result="second")
    prompt = PromptPayload("system", "prompt")
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda *_args: prompt)
    monkeypatch.setattr(engine, "_build_retry_prompt", lambda *_args, **_kwargs: prompt)
    monkeypatch.setattr(engine, "_build_execution_result", lambda *_args: execution)
    skill = SimpleNamespace(output_schema=agdata(summary=str), max_output_schema_retries=1)

    result = engine.execute(context, skill, SimpleNamespace(), SimpleNamespace(), agent.sandbox)

    first, second = holder["requests"]
    assert (first.resume_session_id, first.prior_session_blob_b64) == (
        "session-1",
        "cHJpb3I=",
    )
    assert (second.resume_session_id, second.prior_session_blob_b64) == (
        "session-2",
        "dXBkYXRlZA==",
    )
    assert context.harness_sessions["claude_code"] == {
        "session_id": "session-2",
        "blob_b64": "ZmluYWw=",
    }
    assert result is execution


def test_execute_stops_retrying_once_retries_are_exhausted(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch,
        results=[HarnessAttemptResult(ok=True, final_text=f"attempt-{i}") for i in range(3)],
        collected_sequence=[{}, {}, {}],
    )
    engine = AgentEngine(_FakeAgent())
    execution = agdata(result="attempt-2")
    p0 = PromptPayload("the-system", "p0", None)
    retry = PromptPayload("the-system", "retry", None)
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: p0)
    monkeypatch.setattr(
        engine,
        "_build_retry_prompt",
        lambda missing, *, system_instruction: retry,
    )
    monkeypatch.setattr(engine, "_build_execution_result", lambda *_args: execution)
    skill = SimpleNamespace(output_schema=agdata(summary=str), max_output_schema_retries=2)

    result = engine.execute(
        agcontext(), skill, SimpleNamespace(), SimpleNamespace(), engine._agent.sandbox
    )

    # 1 initial attempt + 2 retries = 3 calls total, then gives up
    assert [request.prompt for request in holder["requests"]] == [p0, retry, retry]
    assert result is execution


def test_execute_builds_execution_result_from_final_attempt(monkeypatch):
    attempt = HarnessAttemptResult(ok=True, final_text="done")
    holder = _install_fake_host_server_manager(monkeypatch, results=[attempt])
    engine = AgentEngine(_FakeAgent())
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: "p0")
    seen = {}
    expected = agdata(result="done")

    def fake_build_result(context, skill, got_attempt, sandbox):
        seen["context"], seen["skill"], seen["attempt"] = context, skill, got_attempt
        seen["sandbox"] = sandbox
        return expected

    monkeypatch.setattr(engine, "_build_execution_result", fake_build_result)
    skill = SimpleNamespace(output_schema=None, max_output_schema_retries=3)
    context = SimpleNamespace(marker="ctx", harness_sessions={})

    result = engine.execute(
        context, skill, SimpleNamespace(), SimpleNamespace(), engine._agent.sandbox
    )

    assert result is expected
    assert seen == {
        "context": context,
        "skill": skill,
        "attempt": attempt,
        "sandbox": engine._agent.sandbox,
    }
    assert holder["manager"].stopped is True


def test_run_attempt_returns_original_rpc_response_without_host_callback():
    expected = HarnessAttemptResult(ok=True, final_text="same response")
    seen = []
    engine = AgentEngine(_FakeAgent())
    engine._sandbox_interaction_client = SimpleNamespace(
        run_harness_attempt=lambda request: seen.append(request) or expected
    )
    prompt = PromptPayload("system", "user")

    result = engine._run_attempt(prompt, max_steps=7)

    assert result is expected
    assert seen == [HarnessAttemptRequest(prompt=prompt, harness="claude_code", max_steps=7)]


# ---------------------------------------------------------------------------
# _build_execution_result
# ---------------------------------------------------------------------------


def _fake_host_server_manager_with_transcript(transcript, *, collected_output=None):
    seen: dict = {}

    def get_main_transcript(needle):
        seen["needle"] = needle
        return transcript

    return (
        SimpleNamespace(
            llm_handler_server=SimpleNamespace(get_main_transcript=get_main_transcript),
            host_mcp_server=SimpleNamespace(
                collected_output=lambda: collected_output if collected_output is not None else {}
            ),
        ),
        seen,
    )


def test_build_execution_result_populates_recent_transcript_from_llm_handler_server():
    engine = AgentEngine(_FakeAgent())
    engine._execution_prompt = PromptPayload("system", "do the work")
    transcript = [
        {"role": "user", "content": "do the work"},
        {"role": "assistant", "content": "done"},
    ]
    engine._host_server_manager, seen = _fake_host_server_manager_with_transcript(transcript)
    context = agcontext()
    skill = SimpleNamespace(output_schema=None, _build_system_prompt=lambda: "system")

    result = engine._build_execution_result(
        context,
        skill,
        HarnessAttemptResult(ok=True, final_text="done", input_tokens=4, output_tokens=2),
        engine._agent.sandbox,
    )

    assert result == agdata(result="done")
    assert context.recent_transcript == transcript
    assert seen["needle"] == "do the work"


def test_build_execution_result_uses_collected_structured_output():
    engine = AgentEngine(_FakeAgent())
    engine._host_server_manager, _ = _fake_host_server_manager_with_transcript(
        [], collected_output={"summary": "finished", "count": 2}
    )
    skill = SimpleNamespace(
        output_schema=agschema(agdata(summary=str, count=int)),
        _build_system_prompt=lambda: "system",
    )

    result = engine._build_execution_result(
        agcontext(),
        skill,
        HarnessAttemptResult(ok=True, final_text="ignored"),
        engine._agent.sandbox,
    )

    assert result == agdata(summary="finished", count=2)


def test_build_execution_result_accepts_valid_structured_json_without_mcp_calls():
    engine = AgentEngine(_FakeAgent())
    engine._host_server_manager, _ = _fake_host_server_manager_with_transcript([])
    skill = SimpleNamespace(
        output_schema=agschema(agdata(summary=str, count=int)),
        _build_system_prompt=lambda: "system",
    )

    result = engine._build_execution_result(
        agcontext(),
        skill,
        HarnessAttemptResult(ok=True, final_text='{"summary": "finished", "count": 2}'),
        engine._agent.sandbox,
    )

    assert result == agdata(summary="finished", count=2)


def test_build_execution_result_reports_incomplete_structured_output():
    engine = AgentEngine(_FakeAgent())
    engine._host_server_manager, _ = _fake_host_server_manager_with_transcript(
        [], collected_output={"summary": "finished"}
    )
    skill = SimpleNamespace(
        output_schema=agschema(agdata(summary=str, count=int)),
        _build_system_prompt=lambda: "system",
    )

    result = engine._build_execution_result(
        agcontext(),
        skill,
        HarnessAttemptResult(ok=True, final_text="done"),
        engine._agent.sandbox,
    )

    assert isinstance(result, agerror)
    assert "count" in result.error


def test_build_execution_result_converts_failed_or_missing_attempt_to_error():
    engine = AgentEngine(_FakeAgent())
    engine._host_server_manager, _ = _fake_host_server_manager_with_transcript([])
    skill = SimpleNamespace(output_schema=None, _build_system_prompt=lambda: "system")

    failed = engine._build_execution_result(
        agcontext(),
        skill,
        HarnessAttemptResult(ok=False, error_message="daemon failed"),
        engine._agent.sandbox,
    )
    missing = engine._build_execution_result(agcontext(), skill, None, engine._agent.sandbox)

    assert isinstance(failed, agerror)
    assert failed.error == "daemon failed"
    assert isinstance(missing, agerror)
    assert missing.error == "no attempt was made"
