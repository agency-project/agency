from __future__ import annotations

import inspect

import pytest

from agency.agdata import agdata
from agency.engine.engine import agentEngine
from agency.engine.types import CompletedResult


class _Builder:
    def __init__(self, result: CompletedResult) -> None:
        self.result = result
        self.calls: list[dict] = []
        self.error: BaseException | None = None

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _completed() -> CompletedResult:
    return CompletedResult(
        output=agdata(result="ok"),
        context="updated-context",
        delta=[{"role": "assistant", "content": "ok"}],
    )


def test_execute_is_the_only_public_execution_method_and_delegates_once():
    agent = object()
    context = object()
    skill = object()
    skill_input = object()
    resource_pool = object()
    result = _completed()
    builder = _Builder(result)
    engine = agentEngine(
        agent=agent,
        context=context,
        skill=skill,
        skill_input=skill_input,
        resource_pool=resource_pool,
        execution_builder=builder,
    )

    assert engine.execute() is result
    assert builder.calls == [
        {
            "agent": agent,
            "context": context,
            "skill": skill,
            "skill_input": skill_input,
            "resource_pool": resource_pool,
        }
    ]

    public_methods = {
        name
        for name, member in inspect.getmembers(agentEngine, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public_methods == {"execute"}
    assert not hasattr(engine, "start")
    assert not hasattr(engine, "run")
    assert not hasattr(engine, "stop")


def test_execute_preserves_the_builder_exception():
    builder = _Builder(_completed())
    builder.error = RuntimeError("builder exploded")
    engine = agentEngine(
        agent=object(),
        context=object(),
        skill=object(),
        skill_input=object(),
        resource_pool=object(),
        execution_builder=builder,
    )

    with pytest.raises(RuntimeError, match="builder exploded"):
        engine.execute()


def test_default_composition_passes_the_provisioner_to_the_builder():
    provisioner = object()
    engine = agentEngine(
        agent=object(),
        context=object(),
        skill=object(),
        skill_input=object(),
        resource_pool=object(),
        sandbox_provisioner=provisioner,
    )

    assert engine._execution_builder._sandbox_provisioner is provisioner
