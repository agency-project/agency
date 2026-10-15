"""Fenced profiler interventions; all execution is deterministic or mocked."""

from concurrent.futures import Future
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agency import agent, agdata, agskill
from agency.configs.agconfig import agconfig, agentconfig, llmconfig
from agency.engine import AgentEngine
from agency.observability.agwebui import server
from agency.observability.agwebui.relay import ProfilerRelay
from agency.observability.agwebui.trajectory import Trajectory


class FakeAgent(SimpleNamespace):
    __hash__ = object.__hash__


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    calls = []
    agents = []
    for name in ("alpha", "beta"):
        orchestrator = SimpleNamespace(
            _event_cond=threading.RLock(), _active_by_agent={}, _requests={}
        )
        target = FakeAgent(
            agname=name,
            _orchestrator=orchestrator,
            harness="native",
            is_paused=lambda: False,
            pause=lambda name=name: calls.append((name, "pause")),
            resume=lambda name=name: calls.append((name, "resume")),
            stop=lambda *, force, name=name: calls.append((name, "kill" if force else "stop")),
        )
        agents.append(target)
    monkeypatch.setattr(agent, "all", classmethod(lambda cls: agents))
    relay = ProfilerRelay(tmp_path)
    monkeypatch.setattr(relay, "_event", lambda *_args: None)
    relay.publish()
    (tmp_path / ".heartbeat").touch()
    model = Trajectory("live", "test")
    for name in ("alpha", "beta"):
        model.apply(
            {
                "id": f"registered:{name}",
                "type": "agent_registered",
                "actor": name,
                "ts": 100,
                "payload": {},
            }
        )
    source = SimpleNamespace(model=model, lock=threading.RLock())
    monkeypatch.setattr(server, "_command_dir", tmp_path)
    monkeypatch.setattr(server, "_command_ids", {})
    return SimpleNamespace(
        relay=relay, agents=agents, calls=calls, source=source, directory=tmp_path
    )


def request(bridge, command="pause", **kwargs):
    return {
        "id": "command-one",
        "command": command,
        "agname": "alpha",
        "relay_id": bridge.relay.identifier,
        "execution": server._execution_controls(bridge.source)["execution"],
        **kwargs,
    }


def queued_file(bridge):
    return next(bridge.directory.glob("*.json"))


def test_file_enqueue_is_not_application_and_duplicate_id_is_not_resubmitted(bridge):
    first = server._trajectory_execution_command(bridge.source, request(bridge))
    assert first["queued"] and first["state"] == "queued" and bridge.calls == []
    second = server._trajectory_execution_command(bridge.source, request(bridge))
    assert second["duplicate"] and len(list(bridge.directory.glob("*.json"))) == 1
    result = bridge.relay.dispatch(json.loads(queued_file(bridge).read_text()))
    assert result["state"] == "accepted" and result["targets"][0]["scope"] == "agent_api_accepted"
    bridge.relay.dispatch(json.loads(queued_file(bridge).read_text()))
    assert bridge.calls == [("alpha", "pause")]
    changed = server._trajectory_execution_command(bridge.source, request(bridge, "resume"))
    assert "already used" in changed["error"]


@pytest.mark.parametrize("operation", ["pause", "resume"])
def test_real_agent_receipt_shape_does_not_collide_or_override_target_state(bridge, operation):
    shape = {
        "scope": "agent_pause_intent",
        "requested": operation,
        "rpc_acknowledged": False,
        "detail": "No daemon was ready; intent remains recorded.",
        "state": "applied",
        "agname": "different-agent",
    }
    setattr(
        bridge.agents[0],
        operation,
        lambda: setattr(bridge.agents[0], "_last_control_receipt", dict(shape)),
    )
    result = bridge.relay.dispatch(
        {
            "id": operation,
            "type": operation,
            "targets": ["alpha"],
            "relay_id": bridge.relay.identifier,
        }
    )
    target = result["targets"][0]
    assert target["state"] == "accepted" and target["agname"] == "alpha"
    assert target["scope"] == "recorded_agent_intent"
    assert target["requested"] == operation and target["rpc_acknowledged"] is False
    assert target["detail"] == "No daemon was ready; intent remains recorded."


def test_real_agent_rpc_error_receipt_remains_failed_after_intent_was_recorded(bridge):
    shape = {
        "scope": "agent_pause_intent",
        "requested": "pause",
        "rpc_acknowledged": False,
        "error": "daemon connection lost",
    }
    bridge.agents[0].pause = lambda: setattr(bridge.agents[0], "_last_control_receipt", dict(shape))
    receipt = bridge.relay.dispatch(
        {
            "id": "failed-pause",
            "type": "pause",
            "targets": ["alpha"],
            "relay_id": bridge.relay.identifier,
        }
    )
    assert receipt["state"] == "failed"
    assert receipt["targets"][0]["detail"] == "daemon connection lost"
    assert receipt["targets"][0]["requested"] == "pause"


def test_failed_event_recording_does_not_prevent_authorized_dispatch(bridge):
    def failed_recording(*_args):
        raise OSError("recording database unavailable")

    bridge.relay._event = failed_recording
    receipt = bridge.relay.dispatch(
        {
            "id": "pause-without-recording",
            "type": "pause",
            "targets": ["alpha"],
            "relay_id": bridge.relay.identifier,
        }
    )
    assert receipt["state"] == "accepted"
    assert bridge.calls == [("alpha", "pause")]
    assert receipt["recording_error"] == "recording database unavailable"


def test_public_queue_override_without_receipt_cannot_claim_stale_delivery(bridge):
    target = bridge.agents[0]
    messages = []
    target.queue_message = messages.append
    target._invoke_queue_message_receipt = agent._invoke_queue_message_receipt.__get__(target)
    target._last_queue_message_receipt = {"request_id": "stale-request", "future": Future()}
    receipt = bridge.relay.dispatch(
        {
            "id": "custom-queue",
            "type": "queue_message",
            "targets": ["alpha"],
            "relay_id": bridge.relay.identifier,
            "message": "preserve override",
        }
    )
    result = receipt["targets"][0]
    assert messages == ["preserve override"]
    assert result["state"] == "accepted"
    assert result["scope"] == "sdk_dispatch_without_queue_receipt"
    assert "request_id" not in result and "context_state" not in result


def test_bulk_reports_partial_failure_and_snapshots_targets(bridge):
    def failed():
        raise ConnectionError("harness disconnected")

    bridge.agents[0].pause = failed
    accepted = server._trajectory_execution_command(bridge.source, request(bridge, "pause_all"))
    assert accepted["targets"] == ["alpha", "beta"]
    bridge.agents.append(SimpleNamespace(agname="later-agent"))
    receipt = bridge.relay.dispatch(json.loads(queued_file(bridge).read_text()))
    assert receipt["state"] == "partial"
    assert [target["state"] for target in receipt["targets"]] == ["failed", "accepted"]
    assert bridge.calls == [("beta", "pause")]


def test_server_and_relay_reject_replacement_execution(bridge):
    stale = request(bridge)
    assert (
        "Execution changed"
        in server._trajectory_execution_command(bridge.source, {**stale, "relay_id": "previous"})[
            "error"
        ]
    )
    assert (
        "Execution changed"
        in server._trajectory_execution_command(
            bridge.source, {**stale, "execution": {"id": "live", "time_origin": 0}}
        )["error"]
    )
    server._trajectory_execution_command(bridge.source, stale)
    replacement = ProfilerRelay(bridge.directory)
    receipt = replacement.dispatch(json.loads(queued_file(bridge).read_text()))
    assert receipt["state"] == "failed" and bridge.calls == []


def test_disconnect_after_enqueue_does_not_erase_receipt(bridge):
    server._trajectory_execution_command(bridge.source, request(bridge, "stop"))
    bridge.relay.dispatch(json.loads(queued_file(bridge).read_text()))
    receipts = server._execution_controls(bridge.source)["receipts"]
    assert receipts[0]["targets"][0]["scope"] == "stop_requested_not_process_exit"
    assert bridge.calls == [("alpha", "stop")]


@pytest.mark.parametrize(
    "command, fields",
    [
        ("queue_message", {"message": " \n "}),
        ("queue_message", {"message": "x" * 32001}),
        ("redirect", {"message": "hello", "invocation": "ended-run"}),
        ("run", {"task": "unregistered", "input": {}}),
        ("run", {"task": ["invalid"], "input": {}}),
    ],
)
def test_invalid_or_unsupported_interactions_do_not_queue(bridge, command, fields):
    assert "error" in server._trajectory_execution_command(
        bridge.source, request(bridge, command, **fields)
    )
    assert not list(bridge.directory.glob("*.json"))


def test_invalid_target_scope_and_non_string_identifiers_are_rejected(bridge):
    for fields in ({"targets": ["alpha"]}, {"command": {}}, {"agname": []}):
        assert "error" in server._trajectory_execution_command(
            bridge.source, {**request(bridge, "pause_all"), **fields}
        )
    assert not list(bridge.directory.glob("*.json"))


def test_registered_task_uses_existing_schema_validation(bridge):
    bridge.relay.tasks = {"answer": agskill("Answer", "", input_schema=agdata(question=str))}
    bridge.relay.publish()
    dispatched = []
    handle = agdata()
    object.__setattr__(handle, "_execution_id", "run-explicit")
    bridge.agents[0].run = lambda skill, data: (
        dispatched.append((skill.name, data.to_dict())) or handle
    )
    for identifier, data, expected in (
        ("bad", {"question": 7}, "failed"),
        ("private", {"question": "why?", "_future": {}}, "failed"),
        ("good", {"question": "why?"}, "submitted"),
    ):
        message = request(bridge, "run", id=identifier, task="answer", input=data)
        assert server._trajectory_execution_command(bridge.source, message)["queued"]
        path = sorted(bridge.directory.glob("*.json"))[-1]
        receipt = bridge.relay.dispatch(json.loads(path.read_text()))
        assert receipt["targets"][0]["state"] == expected
    assert dispatched == [("Answer", {"question": "why?"})]


def test_old_relay_shutdown_does_not_remove_replacement_session(bridge):
    replacement = ProfilerRelay(bridge.directory)
    replacement.publish()
    bridge.relay.close()
    assert (
        json.loads((bridge.directory / ".session").read_text())["relay_id"]
        == replacement.identifier
    )
    assert (bridge.directory / ".heartbeat").exists()


def test_task_registry_validates_without_starting_work(tmp_path):
    with pytest.raises(ValueError, match="ui_tasks"):
        ProfilerRelay(tmp_path, {"arbitrary": lambda: None})


@pytest.mark.parametrize("fail_recording", [False, True])
def test_queue_receipt_reports_context_commit_without_model_delivery(
    tmp_path, monkeypatch, fail_recording
):
    contexts = []

    def execute(_self, *, context, skill_input, **_kwargs):
        contexts.append(
            (skill_input.label, [entry["content"] for entry in context.retained_messages])
        )
        return agdata(label=skill_input.label)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    target = agent(
        sandbox=MagicMock(),
        agconfig=agconfig(
            agentconfig(log_dir=str(tmp_path)), llmconfig(api_key="test", model="mock")
        ),
    )
    dependency = Future()
    first = target.run(
        agskill("test", ""), agdata(label="A", dependency=agdata(_future=dependency))
    )
    second = target.run(agskill("test", ""), agdata(label="B"))
    relay = ProfilerRelay(tmp_path / "commands")
    events = []

    def record_event(command, name, state, result=None):
        events.append(state)
        if fail_recording and state == "context_committed":
            raise OSError("recording database unavailable")

    monkeypatch.setattr(relay, "_event", record_event)
    receipt = relay.dispatch(
        {
            "id": "message",
            "relay_id": relay.identifier,
            "type": "queue_message",
            "targets": [str(target.agname)],
            "message": "line one\nline two",
        }
    )
    assert receipt["targets"][0]["state"] == "queued"
    assert receipt["targets"][0].get("context_state") is None
    third = target.run(agskill("test", ""), agdata(label="C"))
    dependency.set_result(agdata())
    first.wait(timeout=3)
    second.wait(timeout=3)
    third.wait(timeout=3)
    assert contexts == [("A", []), ("B", []), ("C", ["line one\nline two"])]
    assert receipt["targets"][0]["context_state"] == "committed"
    assert "context_committed" in events and "delivered" not in events
    if fail_recording:
        assert receipt["recording_error"] == "recording database unavailable"


def test_redirect_outcome_is_explicit_and_not_a_successor_invocation(bridge):
    target = bridge.agents[0]
    target.harness = "codex"
    future = Future()
    target._orchestrator._active_by_agent[target] = "run-a"
    target._orchestrator._requests["run-a"] = SimpleNamespace(
        agent=target, kind="skill", result_future=future, state="running"
    )
    target._redirect_receipt = lambda handle, message: {
        "state": "delivered",
        "invocation": object.__getattribute__(handle, "_execution_id"),
        "scope": "native_prompt_acknowledgement",
    }
    receipt = bridge.relay.dispatch(
        {
            "id": "redirect",
            "relay_id": bridge.relay.identifier,
            "type": "redirect",
            "targets": ["alpha"],
            "invocation": "run-a",
            "message": "change approach",
        }
    )
    assert receipt["targets"][0]["state"] == "delivered"
    assert receipt["targets"][0]["invocation"] == "run-a"
    target._orchestrator._requests.clear()
    late = bridge.relay.dispatch(
        {
            "id": "late",
            "relay_id": bridge.relay.identifier,
            "type": "redirect",
            "targets": ["alpha"],
            "invocation": "run-a",
            "message": "late",
        }
    )
    assert late["state"] == "failed"


def test_interventions_project_requested_and_known_outcome_without_agent_failure():
    model = Trajectory("live", "test")
    common = {
        "command_id": "stop-one",
        "relay_id": "relay",
        "requested_action": "stop",
        "target_agent": "alpha",
    }
    model.apply(
        {
            "id": "request",
            "type": "ui_intervention",
            "actor": "alpha",
            "ts": 100,
            "payload": {**common, "state": "requested"},
        }
    )
    model.apply(
        {
            "id": "response",
            "type": "ui_intervention",
            "actor": "alpha",
            "ts": 101,
            "payload": {
                **common,
                "state": "accepted",
                "result": {"scope": "stop_requested_not_process_exit"},
            },
        }
    )
    assert len(model.run["actions"]) == 1
    action = model.run["actions"][0]
    assert action["kind"] == "control" and action["outcome"] == "accepted"
    assert action["requested_action"] == "stop" and action["target_agent"] == "alpha"
    assert model.agents["alpha"]["status"] == "unknown"


@pytest.mark.parametrize("count", [10, 32])
def test_many_agent_recorded_catalog_review_replay_and_full_evidence(count, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_run_dir", tmp_path)
    identifier = f"trajectory-agents-{count}"
    catalog = asyncio.run(server.investigator_runs())
    assert (
        next(entry for entry in catalog["runs"] if entry["id"] == identifier)["source"]
        == "synthetic"
    )
    run = asyncio.run(server.investigator_run(identifier))
    assert len(run["agents"]) == count and run["mode"] == "review" and run["source"] == "synthetic"
    replay_run, events = asyncio.run(server._trajectory_replay_data(identifier))
    assert replay_run["id"] == identifier
    long_output = next(
        event for event in events if len(str(event["payload"].get("result", ""))) > 8000
    )
    evidence = asyncio.run(server.trajectory_replay_event(identifier, long_output["id"]))
    assert len(evidence["payload"]["result"]) > 8000 and "<script>" in evidence["payload"]["result"]


def test_public_control_override_cannot_reuse_stale_observation(bridge):
    bridge.agents[0]._last_control_receipt = {
        "requested": "pause",
        "rpc_acknowledged": True,
        "observations": [{"id": "stale"}],
    }
    receipt = bridge.relay.dispatch(
        {"id": "stale", "type": "pause", "targets": ["alpha"], "relay_id": bridge.relay.identifier}
    )
    result = receipt["targets"][0]
    assert result["scope"] == "agent_api_accepted"
    assert result["application"]["state"] == "unavailable"
    assert not bridge.relay._observers


class ImmediatePool:
    def __init__(self):
        self.calls = []

    def submit(self, fn, observations):
        self.calls.append(observations)
        future = Future()
        future.set_result(fn(observations))
        return future

    def shutdown(self, **kwargs):
        pass


@pytest.mark.parametrize("outcome", ["confirmed", "partial", "unknown"])
def test_application_is_separate_recorded_evidence_not_api_acceptance(bridge, outcome):
    target = bridge.agents[0]
    observations = [
        {
            "id": "a" * 32,
            "request_id": "A",
            "attempt_token": "attempt-A",
            "action": "pause",
            "uds_path": "/unused",
        }
    ]
    target.pause = lambda: setattr(
        target,
        "_last_control_receipt",
        {"requested": "pause", "rpc_acknowledged": True, "observations": observations},
    )
    relay = bridge.relay
    relay._observation_pool.shutdown()
    relay._observation_pool = ImmediatePool()
    recorded = []
    relay._event = lambda command, name, state, result=None: recorded.append(state)
    relay._read_observations = lambda _: {
        "state": outcome,
        "scope": "local_harness_tasks",
        "observations": [{"state": outcome, "request_id": "A", "observed_at": 100}],
    }
    receipt = relay.dispatch(
        {"id": "observed", "type": "pause", "targets": ["alpha"], "relay_id": relay.identifier}
    )
    assert (
        receipt["state"] == "accepted"
        and receipt["targets"][0]["application"]["state"] == "pending"
    )
    relay.publish()  # Schedule read-only probe.
    relay.publish()  # Harvest evidence.
    assert receipt["targets"][0]["application"]["state"] == outcome
    assert receipt["state"] == "accepted" and not relay._observers
    assert recorded[-1] == ("observed" if outcome == "confirmed" else "unknown")
    assert len(relay._observation_pool.calls) == 1
    relay.close()


def test_observation_timeout_does_not_retry_mutation_or_claim_success(bridge):
    relay = bridge.relay
    receipt = {"id": "timeout", "targets": []}
    result = {"agname": "alpha", "state": "accepted"}
    receipt["targets"].append(result)
    relay._watch_control(
        receipt, {"id": "timeout", "type": "pause"}, result, {"observations": [{"id": "a" * 32}]}
    )
    next(iter(relay._observers.values()))["deadline"] = 0
    relay.publish()
    assert result["application"]["state"] == "unknown"
    assert "no mutation retry" in result["application"]["detail"]
    assert bridge.calls == [] and not relay._observers
    relay.close()


@pytest.mark.parametrize(
    "change", ["none", "attempt_token", "request_id", "action", "disconnect", "partial"]
)
def test_read_observations_fences_identity_and_reports_partial_evidence(monkeypatch, change):
    from agency.engine.clients import HarnessInteractionClient

    expected = {
        "id": "a" * 32,
        "uds_path": "/unused",
        "request_id": "A",
        "attempt_token": "attempt-A",
        "action": "pause",
    }
    actual = {key: value for key, value in expected.items() if key != "uds_path"}
    actual.update(state="confirmed", local_state="suspended")
    if change in actual:
        actual[change] = "replacement"

    def read(self, identifier):
        if change == "disconnect":
            raise OSError("socket closed after acceptance")
        return {
            **actual,
            "id": identifier,
            "state": "unknown" if identifier == "b" * 32 else actual["state"],
        }

    monkeypatch.setattr(HarnessInteractionClient, "_control_observation", read)
    observations = [expected]
    if change == "partial":
        observations.append({**expected, "id": "b" * 32})
    result = ProfilerRelay._read_observations(observations)
    assert result["state"] == (
        "confirmed" if change == "none" else "partial" if change == "partial" else "unknown"
    )
    assert all("uds_path" not in item for item in result["observations"])


def test_dispatch_time_confirmed_evidence_survives_later_socket_loss(monkeypatch):
    from agency.engine.clients import HarnessInteractionClient

    monkeypatch.setattr(
        HarnessInteractionClient,
        "_control_observation",
        lambda *_: pytest.fail("A confirmed fact must not be probed again"),
    )
    result = ProfilerRelay._read_observations(
        [
            {
                "id": "a" * 32,
                "state": "confirmed",
                "local_state": "suspended",
                "observed_at": 100,
                "uds_path": "/gone",
                "request_id": "A",
                "attempt_token": "a",
                "action": "pause",
            }
        ]
    )
    assert result["state"] == "confirmed"
    assert result["observations"][0]["observed_at"] == 100
    assert "uds_path" not in result["observations"][0]


def test_completed_observation_is_not_erased_by_delayed_receipt_harvesting(bridge):
    relay = bridge.relay
    receipt = {"id": "delayed", "targets": []}
    result = {"agname": "alpha", "state": "accepted"}
    receipt["targets"].append(result)
    relay._watch_control(
        receipt, {"id": "delayed", "type": "pause"}, result, {"observations": [{"id": "a" * 32}]}
    )
    row = next(iter(relay._observers.values()))
    row["deadline"] = 0
    row["future"] = Future()
    row["future"].set_result(
        {
            "state": "confirmed",
            "scope": "local_harness_tasks",
            "observations": [{"id": "a" * 32, "state": "confirmed", "observed_at": 100}],
        }
    )
    relay.publish()
    assert result["application"]["state"] == "confirmed"
    relay.close()


def test_observers_are_bounded_and_missing_targets_never_claim_bulk_success(bridge):
    relay = bridge.relay
    for index in range(129):
        receipt = {"id": str(index), "targets": []}
        result = {"agname": "alpha", "state": "accepted"}
        relay._watch_control(
            receipt,
            {"id": str(index), "type": "pause"},
            result,
            {"observations": [{"id": "a" * 32}]},
        )
    assert len(relay._observers) == 128
    assert result["application"]["state"] == "unavailable"
    result = {"agname": "beta", "state": "accepted"}
    relay._watch_control(
        {"id": "partial"},
        {"type": "stop"},
        result,
        {"observations": [{"id": "a" * 32}], "observation_expected": 2},
    )
    assert result["application"]["state"] == "unavailable"
    relay.close()


def test_relay_close_retains_dispatch_time_confirmed_outcome(bridge):
    relay = bridge.relay
    receipt = {"id": "closed", "targets": []}
    result = {"agname": "alpha", "state": "accepted"}
    receipt["targets"].append(result)
    relay._watch_control(
        receipt,
        {"id": "closed", "type": "pause"},
        result,
        {
            "observations": [
                {
                    "id": "a" * 32,
                    "state": "confirmed",
                    "local_state": "suspended",
                    "observed_at": 100,
                }
            ]
        },
    )
    relay.close()
    assert result["application"]["state"] == "confirmed"
    assert result["application"]["observations"][0]["observed_at"] == 100
