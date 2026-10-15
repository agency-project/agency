import copy
import json

import pytest

from agency.native_harness.annotations import ARMS, augment_schemas, extract, instruction
from agency.native_harness.react_loop import run_react_loop
from agency.native_harness import tools


class FakeLLM:
    def __init__(self, calls):
        self.calls = calls
        self.requests = []

    def dispatch(self, model, messages, schemas):
        self.requests.append(copy.deepcopy((messages, schemas)))
        if len(self.requests) == 1:
            return {
                "message": {"role": "assistant", "content": None, "tool_calls": self.calls},
                "usage": None,
            }
        return {
            "message": {"role": "assistant", "content": "finished"},
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }


class FakeMCP:
    def __init__(self):
        self.arguments = []

    def discover(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "custom",
                    "description": "custom",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                    },
                },
            }
        ]

    def call(self, name, arguments):
        self.arguments.append(json.loads(arguments))
        return '{"result": 3}'


class FakeBridge:
    def __init__(self, denied=False):
        self.denied = denied
        self.admissions = []
        self.completions = []
        self.annotations = []

    def check_tool_policy(self, name, arguments, *, annotation=None):
        self.admissions.append((name, arguments))
        self.annotations.append(annotation)
        return {"call_id": "agency-call-1", "decision": "deny" if self.denied else "allow"}

    def complete_tool_policy(self, call_id, result, **kwargs):
        self.completions.append(call_id)


def call(metadata=None):
    args = {"x": 3}
    if metadata is not None:
        args["_agency"] = metadata
    return {
        "id": "model-call-1",
        "type": "function",
        "function": {"name": "custom", "arguments": json.dumps(args)},
    }


@pytest.mark.parametrize("arm", ARMS)
def test_schemas_are_copies_and_annotations_are_optional(arm):
    schemas = list(tools.BUILTIN_TOOL_SCHEMAS.values())
    original = copy.deepcopy(schemas)
    augmented = augment_schemas(schemas, arm)
    assert schemas == original
    if arm == "baseline":
        assert augmented is schemas
        assert instruction(arm) == ""
    else:
        for schema in augmented:
            params = schema["function"]["parameters"]
            assert "_agency" not in params.get("required", [])
            metadata = params["properties"]["_agency"]
            assert "required" not in metadata
            assert ("workstream_ids" in metadata["properties"]) == (arm == "purpose_workstreams")
        augmented[0]["function"]["parameters"]["properties"]["_agency"]["type"] = "string"
        assert schemas == original


def test_reserved_argument_collision_is_rejected_only_in_active_arms():
    schemas = FakeMCP().discover()
    schemas[0]["function"]["parameters"]["properties"]["_agency"] = {"type": "string"}
    assert augment_schemas(schemas, "baseline") == schemas
    with pytest.raises(ValueError, match="collision"):
        augment_schemas(schemas, "purpose")


@pytest.mark.parametrize(
    "metadata,status",
    [
        (None, "missing"),
        ("bad", "malformed"),
        ({}, "malformed"),
        ({"purpose": "Check x"}, "valid"),
        ({"purpose": ""}, "malformed"),
        ({"purpose": "x", "extra": 2}, "malformed"),
    ],
)
def test_missing_or_malformed_annotations_do_not_cause_retries(metadata, status, tmp_path):
    tc = call(metadata)
    llm, mcp, bridge = FakeLLM([tc]), FakeMCP(), FakeBridge()
    events = []
    result = run_react_loop(
        [],
        "fake",
        llm,
        mcp=mcp,
        bridge=bridge,
        annotation_arm="purpose",
        observer=lambda kind, payload: events.append({"kind": kind, **payload}),
        offload_dir=str(tmp_path),
    )
    assert result.status == "done"
    assert len(llm.requests) == 2
    assert mcp.arguments == [{"x": 3}]
    assert bridge.admissions == [("custom", {"x": 3})]
    assert bridge.annotations[0]["model_tool_call_id"] == tc["id"]
    assert bridge.annotations[0]["status"] == status
    if status == "valid":
        assert bridge.annotations[0]["raw"] == metadata
    assert bridge.completions == ["agency-call-1"]
    assert (
        next(e for e in events if e["kind"] == "tool_annotation")["annotation"]["status"] == status
    )
    assert llm.requests[1][0][1]["tool_calls"] == [tc]
    admission = next(e for e in events if e["kind"] == "tool_admission")
    completion = next(e for e in events if e["kind"] == "tool_result")
    assert admission["event_id"] == completion["event_id"]
    assert admission["call_id"] == completion["call_id"] == "agency-call-1"


def test_denied_call_annotation_is_saved_before_admission(tmp_path):
    events = []
    mcp, bridge = FakeMCP(), FakeBridge(denied=True)
    run_react_loop(
        [],
        "fake",
        FakeLLM([call({"purpose": "Check x"})]),
        mcp=mcp,
        bridge=bridge,
        annotation_arm="purpose",
        observer=lambda kind, payload: events.append({"kind": kind, **payload}),
        offload_dir=str(tmp_path),
    )
    kinds = [e["kind"] for e in events]
    assert (
        kinds.index("tool_annotation") < kinds.index("tool_admission") < kinds.index("tool_result")
    )
    assert not mcp.arguments
    assert len(bridge.admissions) == 1


def test_baseline_prompt_and_schemas_are_unchanged(tmp_path):
    llm = FakeLLM([call()])
    messages = [{"role": "system", "content": "original"}]
    run_react_loop(messages, "fake", llm, mcp=FakeMCP(), offload_dir=str(tmp_path))
    assert llm.requests[0][0] == messages
    assert llm.requests[0][1] == list(tools.BUILTIN_TOOL_SCHEMAS.values()) + FakeMCP().discover()


def test_schema_only_may_spontaneously_annotate():
    arguments, metadata = extract(
        json.dumps({"x": 1, "_agency": {"purpose": "Check"}}), "schema_only"
    )
    assert json.loads(arguments) == {"x": 1}
    assert metadata["status"] == "valid"
    assert instruction("schema_only") == ""


def test_multiple_workstream_memberships_and_reuse_are_preserved():
    for ids in [["left", "shared"], ["right", "shared"], ["left"]]:
        arguments, metadata = extract(
            json.dumps({"_agency": {"purpose": "Work", "workstream_ids": ids}, "x": 1}),
            "purpose_workstreams",
        )
        assert metadata["status"] == "valid"
        assert metadata["raw"]["workstream_ids"] == ids
        assert json.loads(arguments) == {"x": 1}


@pytest.mark.parametrize("arguments", ["{", "[]", "null"])
def test_malformed_executable_json_is_preserved(arguments):
    executable, metadata = extract(arguments, "purpose")
    assert executable == arguments
    assert metadata["status"] == "malformed_arguments"


def test_native_daemon_config_transports_annotation_and_trace_settings_without_credentials():
    from agency.configs.agconfig import agconfig, agentconfig, llmconfig
    from agency.engine.harness_daemon_launcher import _daemon_config

    cfg = agconfig(
        agentconfig(
            annotation_arm="purpose",
            native_trace_file="/tmp/events.jsonl",
            experiment_run_id="run",
            experiment_agent_id="worker",
        ),
        llmconfig(api_key="private-secret-key"),
    )
    payload = _daemon_config(cfg)
    assert payload["agent"] == {
        "annotation_arm": "purpose",
        "native_trace_file": "/tmp/events.jsonl",
        "experiment_run_id": "run",
        "experiment_agent_id": "worker",
    }
    assert "private-secret-key" not in json.dumps(payload)
    assert "agent" not in _daemon_config(agconfig())


def test_all_batch_annotations_are_captured_even_if_first_call_is_interrupted(tmp_path):
    class InterruptedMCP(FakeMCP):
        def call(self, name, arguments):
            raise KeyboardInterrupt()

    calls = [call({"purpose": "first"}), call({"purpose": "second"})]
    calls[1]["id"] = "model-call-2"
    events = []
    with pytest.raises(KeyboardInterrupt):
        run_react_loop(
            [],
            "fake",
            FakeLLM(calls),
            mcp=InterruptedMCP(),
            annotation_arm="purpose",
            observer=lambda kind, payload: events.append({"kind": kind, **payload}),
            offload_dir=str(tmp_path),
        )
    assert len([event for event in events if event["kind"] == "tool_annotation"]) == 2


def test_trace_writer_redacts_echoed_credentials(tmp_path, monkeypatch):
    from agency.native_harness.annotations import TraceWriter

    monkeypatch.setenv("TEST_API_KEY", "private-test-key")
    trace = TraceWriter(
        str(tmp_path / "events.jsonl"), agent_id="a", run_id="r", secrets=("bridge-secret",)
    )
    trace("model_exchange", {"response": {"error": "private-test-key bridge-secret"}})
    text = (tmp_path / "events.jsonl").read_text()
    assert "private-test-key" not in text and "bridge-secret" not in text
    bridge = FakeBridge()
    bridge.token = "bridge-secret"
    run_react_loop(
        [],
        "fake",
        FakeLLM([call({"purpose": "Check private-test-key bridge-secret"})]),
        mcp=FakeMCP(),
        bridge=bridge,
        annotation_arm="purpose",
        offload_dir=str(tmp_path),
    )
    forwarded = json.dumps(bridge.annotations)
    assert "private-test-key" not in forwarded and "bridge-secret" not in forwarded
    assert "[REDACTED]" in forwarded


def test_builtin_arguments_are_stripped_and_nonzero_shell_exit_is_an_error(tmp_path):
    arguments = {"command": "false", "_agency": {"purpose": "Check failure"}}
    tool_call = {
        "id": "builtin-call",
        "function": {"name": "bash", "arguments": json.dumps(arguments)},
    }
    received, events = [], []

    def fake_shell(arguments):
        received.append(json.loads(arguments))
        return '{"returncode": 1, "output": ""}'

    run_react_loop(
        [],
        "fake",
        FakeLLM([tool_call]),
        annotation_arm="purpose",
        toolset=([tools.BUILTIN_TOOL_SCHEMAS["bash"]], {"bash": fake_shell}),
        observer=lambda kind, payload: events.append({"kind": kind, **payload}),
        offload_dir=str(tmp_path),
    )
    assert received == [{"command": "false"}]
    result = next(event for event in events if event["kind"] == "tool_result")
    assert result["category"] == "error"
    assert result["duration_ns"] is not None
