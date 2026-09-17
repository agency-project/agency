"""Ordered host-only message requests on the global orchestrator."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agency import agdata, agent, agskill
from agency.configs.agconfig import agconfig, agentconfig, llmconfig, orchestratorconfig
from agency.agname import agname as _agname
from agency.engine import AgentEngine
from agency.engine import engine as engine_module
from agency.harness.protocol import HarnessAttemptResult
from agency.orchestrator import get_orchestrator


def _request_for(handle: agdata):
    """Look up the orchestrator's internal request behind a bare handle."""
    orchestrator = get_orchestrator()
    future = object.__getattribute__(handle, "_future")
    with orchestrator._event_cond:
        request_id = orchestrator._future_producers[future]
        return orchestrator._requests[request_id]


def _message_request(ag: agent, orchestrator=None):
    """Find the (only) pending queued-message request for *ag*.

    queue_message() returns None -- this is the only way a test can locate
    the request it produced.
    """
    orchestrator = orchestrator if orchestrator is not None else get_orchestrator()
    with orchestrator._event_cond:
        return next(
            r
            for r in orchestrator._requests.values()
            if r.agent is ag and r.kind == "context_message"
        )


def _config(tmp_path, *, max_engines: int | None = None) -> agconfig:
    return agconfig(
        orchestratorconfig(max_concurrent_engines=max_engines),
        agentconfig(log_dir=str(tmp_path)),
        llmconfig(api_key="test", model="m"),
    )


def _agent(tmp_path, *, max_engines: int | None = None) -> agent:
    sandbox = MagicMock()
    sandbox._lock = threading.RLock()
    sandbox._checkpoint_image = None
    return agent(sandbox=sandbox, agconfig=_config(tmp_path, max_engines=max_engines))


@pytest.mark.parametrize("invalid", [None, 1, object()])
def test_queue_message_rejects_non_strings(tmp_path, invalid):
    ag = _agent(tmp_path)
    with pytest.raises(TypeError, match="message must be a string"):
        ag.queue_message(invalid)


@pytest.mark.parametrize("invalid", ["", "   ", "\n\t"])
def test_queue_message_rejects_empty_strings(tmp_path, invalid):
    ag = _agent(tmp_path)
    with pytest.raises(ValueError, match="message must be a non-empty string"):
        ag.queue_message(invalid)


def test_message_uses_exact_context_position_and_no_engine_infrastructure(monkeypatch, tmp_path):
    constructed: list[AgentEngine] = []
    observed_contexts: list[tuple[str, list[dict]]] = []
    original_init = AgentEngine.__init__

    def tracked_init(self, owner):
        constructed.append(self)
        original_init(self, owner)

    def execute(self, *, context, skill_input, **_kwargs):
        observed_contexts.append((skill_input.label, list(context.retained_messages)))
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "__init__", tracked_init)
    monkeypatch.setattr(AgentEngine, "execute", execute)
    ag = _agent(tmp_path)
    skill = agskill("ordered", "")
    gate_dependency: Future[agdata] = Future()

    blocked = ag.run(
        skill,
        agdata(label="blocked", dependency=agdata(_future=gate_dependency)),
    )
    blocked_request = _request_for(blocked)
    ag.queue_message("Remember the exact order")
    orchestrator = get_orchestrator()
    message_request = _message_request(ag, orchestrator)
    later = ag.run(skill, agdata(label="later"))
    later_request = _request_for(later)
    callback_context_done: list[bool] = []
    callback_finished = threading.Event()

    def observe_success(_future) -> None:
        callback_context_done.append(message_request.context_future.done())
        callback_finished.set()

    message_request.result_future.add_done_callback(observe_success)

    assert message_request.context_dependency._future is blocked_request.context_future
    assert later_request.context_dependency._future is message_request.context_future
    assert object.__getattribute__(ag.context, "_future") is later_request.context_future
    assert not message_request.result_future.done()
    assert constructed == []
    assert ag.engine is None

    with orchestrator._event_cond:
        assert message_request.kind == "context_message"
        assert message_request.skill is None
        assert message_request.skill_input is None

    gate_dependency.set_result(agdata(open=True))
    assert blocked.wait(timeout=2).label == "blocked"
    assert message_request.result_future.result(timeout=2).to_dict() == {}
    assert callback_finished.wait(timeout=2)
    assert callback_context_done == [True]
    assert later.wait(timeout=2).label == "later"

    entry = {
        "sequence": 1,
        "type": "message",
        "role": "user",
        "content": "Remember the exact order",
        "source": "queue_message",
    }
    assert message_request.context_future.result().retained_messages == [entry]
    assert observed_contexts == [("blocked", []), ("later", [entry])]
    assert len(constructed) == 2


def test_real_engine_replays_queued_message_into_a_stateless_run(monkeypatch, tmp_path):
    requests = []
    managers = []

    class FakeHostServerManager:
        def __init__(
            self,
            agent,
            sandbox,
            skill,
            resource_pool,
            *,
            is_cancelled=None,
            request_id=None,
            recent_transcript=None,
        ):
            self.agent = agent
            self.sandbox = sandbox
            self.skill = skill
            self.resource_pool = resource_pool
            self.is_cancelled = is_cancelled
            self.request_id = request_id
            self.recent_transcript = recent_transcript
            self.bound_tokens: list[str] = []
            self.cleared_tokens: list[str] = []
            self.active_token = None
            self.host_mcp_server = SimpleNamespace(collected_output=lambda: {})
            self.llm_handler_server = SimpleNamespace(get_main_transcript=lambda _needle: [])
            self.interaction_server = SimpleNamespace(last_bootstrap_ping_ts=None)
            managers.append(self)

        def start(self) -> str:
            return "/tmp/stateless-retained-host.sock"

        def stop(self) -> None:
            return None

        def bind_attempt_token(self, token: str) -> None:
            assert self.active_token is None
            self.active_token = token
            self.bound_tokens.append(token)

        def clear_attempt_token(self, token: str) -> bool:
            assert token == self.active_token
            self.active_token = None
            self.cleared_tokens.append(token)
            return True

    class FakeSandboxClient:
        def run_harness_attempt(self, request):
            requests.append(request)
            return HarnessAttemptResult(ok=True, final_text="retained context observed")

        def close(self) -> None:
            return None

    client = FakeSandboxClient()

    def ensure_daemon(
        _sandbox, _host_path, _engine_name, _harness, *, agconfig, progress_source=None
    ):
        del agconfig, progress_source
        return SimpleNamespace(client=lambda timeout_s=None: client)

    monkeypatch.setattr(engine_module, "HostServerManager", FakeHostServerManager)
    monkeypatch.setattr(engine_module, "ensure_harness_daemon", ensure_daemon)
    ag = _agent(tmp_path)
    ag.sandbox._has_pending_background_work.return_value = False

    ag.queue_message("Remember this stateless fact")
    invocation = ag.run(
        agskill("read-retained", "Use the retained message."),
        agdata(question="What should be remembered?"),
    )

    invocation.wait(timeout=2)
    assert invocation.result == "retained context observed"
    assert len(requests) == 1
    prompt = requests[0].prompt
    assert "[AGENCY RETAINED CONTEXT]" in prompt.user_content
    assert "[USER]\nRemember this stateless fact" in prompt.user_content
    assert requests[0].resume_session_id is None
    assert requests[0].prior_session_blob_b64 is None
    assert len(managers) == 1
    assert managers[0].bound_tokens == managers[0].cleared_tokens
    assert len(managers[0].bound_tokens) == 1
    assert managers[0].active_token is None
    assert ag.context.copy().harness_message_cursors == {}
    assert [entry["content"] for entry in ag.context.retained_messages] == [
        "Remember this stateless fact"
    ]


def test_full_capacity_does_not_block_host_only_messages(monkeypatch, tmp_path):
    holder_started = threading.Event()
    release_holder = threading.Event()
    constructed_for: list[agent] = []
    original_init = AgentEngine.__init__

    def tracked_init(self, owner):
        constructed_for.append(owner)
        original_init(self, owner)

    def execute(self, *, skill_input, **_kwargs):
        if skill_input.label == "holder":
            holder_started.set()
            assert release_holder.wait(timeout=2)
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "__init__", tracked_init)
    monkeypatch.setattr(AgentEngine, "execute", execute)
    holder_agent = _agent(tmp_path, max_engines=1)
    message_agent = _agent(tmp_path, max_engines=1)

    holder = holder_agent.run(agskill("holder", ""), agdata(label="holder"))
    assert holder_started.wait(timeout=2)

    # Nothing blocks this message's own predecessor context, so it may
    # complete near-instantly -- resolve the agent's own context chain
    # (rather than reaching for the internal, possibly-already-settled
    # tracking node) to observe the committed result race-free.
    message_agent.queue_message("host-only at full capacity")
    message_agent.context.resolve_prev_dependencies()

    assert message_agent.engine is None
    assert constructed_for == [holder_agent]
    assert get_orchestrator().snapshot().running_count == 1
    assert message_agent.context.copy().retained_messages[0]["content"] == (
        "host-only at full capacity"
    )

    release_holder.set()
    assert holder.wait(timeout=2).label == "holder"


def test_concurrent_messages_follow_atomic_publication_order(monkeypatch, tmp_path):
    constructed: list[AgentEngine] = []
    original_init = AgentEngine.__init__

    def tracked_init(self, owner):
        constructed.append(self)
        original_init(self, owner)

    monkeypatch.setattr(AgentEngine, "__init__", tracked_init)
    monkeypatch.setattr(AgentEngine, "execute", lambda self, **_kwargs: agdata(done=True))
    ag = _agent(tmp_path)
    gate_dependency: Future[agdata] = Future()
    gate = ag.run(
        agskill("gate", ""),
        agdata(dependency=agdata(_future=gate_dependency)),
    )
    barrier = threading.Barrier(9)
    failures: list[BaseException] = []

    def submit(index: int) -> None:
        try:
            barrier.wait(timeout=2)
            ag.queue_message(f"message-{index}")
        except BaseException as exc:
            failures.append(exc)

    callers = [threading.Thread(target=submit, args=(index,)) for index in range(8)]
    for caller in callers:
        caller.start()
    barrier.wait(timeout=2)
    for caller in callers:
        caller.join(timeout=2)
        assert not caller.is_alive()

    assert failures == []
    orchestrator = get_orchestrator()
    gate_request = _request_for(gate)
    with orchestrator._event_cond:
        submissions = [
            r for r in orchestrator._requests.values() if r.agent is ag and r is not gate_request
        ]
    assert len(submissions) == 8
    ordered = sorted(submissions, key=lambda r: r.sequence)
    predecessor_future = gate_request.context_future
    for request in ordered:
        assert request.context_dependency._future is predecessor_future
        assert not request.result_future.done()
        predecessor_future = request.context_future
    assert object.__getattribute__(ag.context, "_future") is ordered[-1].context_future
    assert constructed == []

    gate_dependency.set_result(agdata(open=True))
    assert gate.wait(timeout=2).done is True
    for request in ordered:
        assert request.result_future.result(timeout=2).to_dict() == {}

    final_context = ag.context.copy()
    assert [entry["content"] for entry in final_context.retained_messages] == [
        request.message for request in ordered
    ]
    assert [entry["sequence"] for entry in final_context.retained_messages] == list(range(1, 9))
    assert len(constructed) == 1


def test_fork_and_checkpoint_preserve_messages_cursors_and_sequence(tmp_path):
    ag = _agent(tmp_path)
    ag.queue_message("persist me")
    ag.context.resolve_prev_dependencies()
    expected_retained_messages = ag.context.retained_messages
    ag.context.harness_message_cursors["claude_code"] = 1

    forked = agent.fork(ag, name="message-fork")
    assert forked.context.retained_messages == ag.context.retained_messages
    assert forked.context.harness_message_cursors == {"claude_code": 1}
    assert forked.context.retained_messages is not ag.context.retained_messages
    assert forked.context.harness_message_cursors is not ag.context.harness_message_cursors

    checkpoint = tmp_path / "messages.ckpt"
    ag.save(checkpoint)
    saved_name = str(ag.agname)
    _agname._allocated.discard(saved_name)

    loaded = agent.load(checkpoint, agconfig=_config(tmp_path))
    assert loaded.context.retained_messages == expected_retained_messages
    assert loaded.context.harness_message_cursors == {"claude_code": 1}

    loaded.queue_message("after load")
    loaded.context.resolve_prev_dependencies()
    assert [entry["sequence"] for entry in loaded.context.copy().retained_messages] == [1, 2]
    assert [entry["content"] for entry in loaded.context.retained_messages] == [
        "persist me",
        "after load",
    ]
