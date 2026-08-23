# Tests for engine.py -- agentEngine, the per-launch host-side orchestrator.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agency.agdata import agdata
from agency.engine import engine as mod
from agency.engine.engine import agentEngine
from agency.engine.types import HarnessAttemptResult, PromptPayload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self):
        self.agconfig = SimpleNamespace(marker="agconfig")
        self.sandbox = SimpleNamespace(marker="sandbox")
        self.change_config_calls = []

    def change_config(self, agconfig):
        self.change_config_calls.append(agconfig)


def _install_fake_host_server_manager(monkeypatch, results, collected_sequence=None):
    """Patches engine.py's HostServerManager import with a fake whose
    harness_interaction_server.run_prompt() pops from `results` in order,
    and whose host_mcp_server.collected_output() steps through
    `collected_sequence`. Returns a dict that will hold the constructed
    instance under "manager" once run() builds one."""
    holder: dict = {}
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

            def run_prompt(prompt, timeout=None):
                run_prompt_calls.append(prompt)
                return remaining_results.pop(0)

            run_prompt_calls: list = []
            self.harness_interaction_server = SimpleNamespace(
                run_prompt=run_prompt, run_prompt_calls=run_prompt_calls
            )
            self.host_mcp_server = SimpleNamespace(
                collected_output=lambda: next(collected_iter) if collected_iter is not None else {}
            )
            holder["manager"] = self

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def set_config(self, agconfig) -> None:
            self.set_config_calls.append(agconfig)

    monkeypatch.setattr(mod, "HostServerManager", _FakeHostServerManager)
    return holder


# ---------------------------------------------------------------------------
# __init__ / host_server_manager property
# ---------------------------------------------------------------------------


def test_init_stores_agent_and_starts_with_no_host_server_manager():
    agent = _FakeAgent()
    engine = agentEngine(agent)
    assert engine._agent is agent
    assert engine._host_server_manager is None


def test_host_server_manager_property_raises_before_run():
    engine = agentEngine(_FakeAgent())
    with pytest.raises(RuntimeError):
        engine.host_server_manager


# ---------------------------------------------------------------------------
# set_config
# ---------------------------------------------------------------------------


def test_set_config_forwards_to_agent_change_config():
    agent = _FakeAgent()
    engine = agentEngine(agent)
    new_cfg = SimpleNamespace(marker="new")
    engine.set_config(new_cfg)
    assert agent.change_config_calls == [new_cfg]


def test_set_config_does_not_touch_host_server_manager_when_not_yet_built():
    engine = agentEngine(_FakeAgent())
    engine.set_config(SimpleNamespace())  # should not raise


def test_set_config_forwards_to_host_server_manager_when_built(monkeypatch):
    holder = _install_fake_host_server_manager(monkeypatch, results=[])
    agent = _FakeAgent()
    engine = agentEngine(agent)
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
    engine = agentEngine(_FakeAgent())
    payload = engine._build_prompt_payload(SimpleNamespace(), SimpleNamespace())
    assert payload == PromptPayload(
        prompt="the-prompt", output_format_instruction="the-format", extra_system=None
    )


def test_build_retry_prompt_mentions_missing_fields():
    engine = agentEngine(_FakeAgent())
    payload = engine._build_retry_prompt(["a", "b"])
    assert "a" in payload.prompt and "b" in payload.prompt
    assert "submit_output" in payload.prompt
    assert payload.output_format_instruction is None


# ---------------------------------------------------------------------------
# _missing_output_fields
# ---------------------------------------------------------------------------


def test_missing_output_fields_empty_when_skill_has_no_output_schema():
    engine = agentEngine(_FakeAgent())
    skill = SimpleNamespace(output_schema=None)
    assert engine._missing_output_fields(skill) == []


def test_missing_output_fields_diffs_against_collected_output():
    engine = agentEngine(_FakeAgent())
    engine._host_server_manager = SimpleNamespace(
        host_mcp_server=SimpleNamespace(collected_output=lambda: {"summary": "hi"})
    )
    skill = SimpleNamespace(output_schema=agdata(summary=str, count=int))
    assert engine._missing_output_fields(skill) == ["count"]


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_stops_the_host_server_manager_even_when_bootup_is_not_implemented(monkeypatch):
    holder = _install_fake_host_server_manager(monkeypatch, results=[])
    engine = agentEngine(_FakeAgent())
    skill = SimpleNamespace(output_schema=None, max_output_schema_retries=3)
    with pytest.raises(NotImplementedError):
        engine.run(SimpleNamespace(), skill, SimpleNamespace(), SimpleNamespace())
    assert holder["manager"].started is True
    assert holder["manager"].stopped is True


def test_run_calls_run_prompt_once_and_returns_execution_result_on_first_success(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch, results=[HarnessAttemptResult(ok=True, final_text="done")]
    )
    engine = agentEngine(_FakeAgent())
    monkeypatch.setattr(engine, "_ensure_harness_manager_launched", lambda: None)
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: "p0")
    monkeypatch.setattr(
        engine, "_build_execution_result", lambda context, skill, attempt: ("built", attempt)
    )
    skill = SimpleNamespace(output_schema=None, max_output_schema_retries=3)

    result = engine.run(SimpleNamespace(), skill, SimpleNamespace(), SimpleNamespace())

    manager = holder["manager"]
    assert manager.harness_interaction_server.run_prompt_calls == ["p0"]
    assert manager.started is True
    assert manager.stopped is True
    assert result == ("built", HarnessAttemptResult(ok=True, final_text="done"))


def test_run_stops_immediately_on_a_failed_attempt_without_retrying(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch, results=[HarnessAttemptResult(ok=False, error_message="boom")]
    )
    engine = agentEngine(_FakeAgent())
    monkeypatch.setattr(engine, "_ensure_harness_manager_launched", lambda: None)
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: "p0")
    monkeypatch.setattr(engine, "_build_execution_result", lambda context, skill, attempt: attempt)
    skill = SimpleNamespace(output_schema=agdata(summary=str), max_output_schema_retries=3)

    result = engine.run(SimpleNamespace(), skill, SimpleNamespace(), SimpleNamespace())

    assert holder["manager"].harness_interaction_server.run_prompt_calls == ["p0"]
    assert result == HarnessAttemptResult(ok=False, error_message="boom")


def test_run_retries_on_missing_output_fields_then_succeeds(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch,
        results=[
            HarnessAttemptResult(ok=True, final_text="first"),
            HarnessAttemptResult(ok=True, final_text="second"),
        ],
        collected_sequence=[{}, {"summary": "x"}],
    )
    engine = agentEngine(_FakeAgent())
    monkeypatch.setattr(engine, "_ensure_harness_manager_launched", lambda: None)
    prompts = iter(["p0", "p1"])
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: next(prompts))
    monkeypatch.setattr(engine, "_build_retry_prompt", lambda missing: next(prompts))
    monkeypatch.setattr(engine, "_build_execution_result", lambda context, skill, attempt: attempt)
    skill = SimpleNamespace(output_schema=agdata(summary=str), max_output_schema_retries=3)

    result = engine.run(SimpleNamespace(), skill, SimpleNamespace(), SimpleNamespace())

    assert holder["manager"].harness_interaction_server.run_prompt_calls == ["p0", "p1"]
    assert result == HarnessAttemptResult(ok=True, final_text="second")


def test_run_stops_retrying_once_retries_are_exhausted(monkeypatch):
    holder = _install_fake_host_server_manager(
        monkeypatch,
        results=[HarnessAttemptResult(ok=True, final_text=f"attempt-{i}") for i in range(3)],
        collected_sequence=[{}, {}, {}],
    )
    engine = agentEngine(_FakeAgent())
    monkeypatch.setattr(engine, "_ensure_harness_manager_launched", lambda: None)
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: "p0")
    monkeypatch.setattr(engine, "_build_retry_prompt", lambda missing: "retry")
    monkeypatch.setattr(engine, "_build_execution_result", lambda context, skill, attempt: attempt)
    skill = SimpleNamespace(output_schema=agdata(summary=str), max_output_schema_retries=2)

    result = engine.run(SimpleNamespace(), skill, SimpleNamespace(), SimpleNamespace())

    # 1 initial attempt + 2 retries = 3 calls total, then gives up
    assert holder["manager"].harness_interaction_server.run_prompt_calls == ["p0", "retry", "retry"]
    assert result == HarnessAttemptResult(ok=True, final_text="attempt-2")


def test_run_builds_execution_result_from_final_attempt(monkeypatch):
    attempt = HarnessAttemptResult(ok=True, final_text="done")
    holder = _install_fake_host_server_manager(monkeypatch, results=[attempt])
    engine = agentEngine(_FakeAgent())
    monkeypatch.setattr(engine, "_ensure_harness_manager_launched", lambda: None)
    monkeypatch.setattr(engine, "_build_prompt_payload", lambda skill, skill_input: "p0")
    seen = {}

    def fake_build_result(context, skill, got_attempt):
        seen["context"], seen["skill"], seen["attempt"] = context, skill, got_attempt
        return "the-result"

    monkeypatch.setattr(engine, "_build_execution_result", fake_build_result)
    skill = SimpleNamespace(output_schema=None, max_output_schema_retries=3)
    context = SimpleNamespace(marker="ctx")

    result = engine.run(context, skill, SimpleNamespace(), SimpleNamespace())

    assert result == "the-result"
    assert seen == {"context": context, "skill": skill, "attempt": attempt}
    assert holder["manager"].stopped is True
