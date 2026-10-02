import copy
import json

import pytest

from benchmarks.tool_annotation_effect.codex import CodexGateway, result_category
from benchmarks.tool_annotation_effect.common import read_events
from benchmarks.tool_annotation_effect.planning import validate


class Client:
    def __init__(self, arguments):
        self.arguments = arguments
        self.requests = []

    def dispatch(self, model, messages, tools):
        self.requests.append(copy.deepcopy((messages, tools)))
        if len(self.requests) == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-real",
                            "type": "function",
                            "function": {
                                "name": "mcp__apply_patch__custom",
                                "arguments": json.dumps(self.arguments),
                            },
                        }
                    ],
                },
                "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
            }
        return {"message": {"role": "assistant", "content": "done"}, "usage": None}

    def close(self):
        pass


def gateway(tmp_path, monkeypatch, arm, arguments):
    monkeypatch.setenv("TEST_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("TEST_KEY", "fake-key-for-tests")
    model = {
        "id": "test",
        "provider": "openai",
        "model": "fake",
        "base_url_env": "TEST_URL",
        "api_key_env": "TEST_KEY",
    }
    config = {
        "budgets": {"max_steps": 2, "timeout_s": 30},
        "context_limit": 196000,
        "codex_version": "0.147.0",
    }
    client = Client(arguments)
    return CodexGateway(
        model, config, {"arm": arm, "trial_id": "test"}, tmp_path, client=client
    ), client


@pytest.mark.parametrize("arm", ["baseline", "schema_only", "purpose", "purpose_workstreams"])
def test_codex_custom_tool_executes_original_input_and_retains_annotation_history(
    tmp_path, monkeypatch, arm
):
    patch = "*** Begin Patch\n*** Add File: x\n+hello\n*** End Patch"
    args = {"input": patch}
    if arm != "baseline":
        args["_agency"] = {"purpose": "Create x."}
        if arm == "purpose_workstreams":
            args["_agency"]["workstream_ids"] = ["x"]
    gw, client = gateway(tmp_path, monkeypatch, arm, args)
    body = {
        "instructions": "Codex prompt",
        "input": "task",
        "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "text"}}],
    }
    original = copy.deepcopy(body)
    result = gw.dispatch(body)
    assert body == original
    output = gw.adapter._format_context_agency_to_harness(
        result,
        "fake",
        tool_routes=__import__(
            "agency.harness.adapters.codex", fromlist=["_responses_tool_routes"]
        )._responses_tool_routes(body),
    )
    tool = output["output"][0]
    assert tool["type"] == "custom_tool_call"
    assert tool["input"] == patch
    assert tool["call_id"] == "call-real"
    params = client.requests[0][1][0]["function"]["parameters"]
    assert ("_agency" in params["properties"]) == (arm != "baseline")
    assert "_agency" not in params.get("required", [])
    text = client.requests[0][0][0]["content"]
    assert ("immediate purpose" in text) == (arm in ("purpose", "purpose_workstreams"))
    body["input"] = [
        tool,
        {
            "type": "custom_tool_call_output",
            "call_id": "call-real",
            "output": "Success. Updated files: x",
        },
    ]
    gw.dispatch(body)
    assert json.loads(client.requests[1][0][1]["tool_calls"][0]["function"]["arguments"]) == args
    events = read_events(tmp_path / "events.jsonl")
    labels = [e for e in events if e["kind"] == "tool_annotation"]
    assert len(labels) == 1
    assert labels[0]["annotation"]["status"] == ("not_requested" if arm == "baseline" else "valid")
    assert labels[0]["arguments"] == {"input": patch}
    results = [e for e in events if e["kind"] == "tool_result"]
    assert results[0]["event_id"] == labels[0]["event_id"]
    with pytest.raises(RuntimeError, match="max_steps"):
        gw.dispatch(body)
    assert len(client.requests) == 2
    assert (
        len([e for e in read_events(tmp_path / "events.jsonl") if e["kind"] == "tool_result"]) == 1
    )


def test_codex_missing_annotations_do_not_retry(tmp_path, monkeypatch):
    gw, client = gateway(tmp_path, monkeypatch, "purpose", {"input": "patch"})
    gw.dispatch({"input": "task", "tools": [{"type": "custom", "name": "apply_patch"}]})
    assert len(client.requests) == 1
    event = next(
        e for e in read_events(tmp_path / "events.jsonl") if e["kind"] == "tool_annotation"
    )
    assert event["annotation"]["status"] == "missing"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Process exited with code 1", "error"),
        ("Process exited with code 0", "ok"),
        ("Error: missing file", "error"),
    ],
)
def test_codex_exit_status_categories(text, expected):
    assert result_category(text) == expected


def test_codex_requires_its_actual_compaction_policy():
    with pytest.raises(ValueError, match="codex-default"):
        validate(
            {
                "harness": "codex",
                "profile": "custom",
                "models": [
                    {
                        "id": "x",
                        "model": "x",
                        "provider": "openai",
                        "base_url_env": "URL",
                        "api_key_env": "KEY",
                    }
                ],
                "budgets": {"max_steps": 50, "timeout_s": 600},
                "context_limit": 196000,
                "cache_policy": "unknown",
                "suites": {"swebench": {}},
                "compaction_policy": "native-v1",
            }
        )


def test_codex_gateway_accepts_real_http_body_and_authenticates_before_dispatch(
    tmp_path, monkeypatch
):
    import httpx

    gw, client = gateway(
        tmp_path, monkeypatch, "purpose", {"input": "patch", "_agency": {"purpose": "Apply patch."}}
    )
    gw.config["codex_gateway_host"] = "127.0.0.1"
    body = {"input": "task", "tools": [{"type": "custom", "name": "apply_patch"}]}
    with gw.serve() as url:
        rejected = httpx.post(url + "/v1/responses", json=body)
        assert rejected.status_code == 401
        assert not client.requests
        accepted = httpx.post(
            url + "/v1/responses", headers={"Authorization": "Bearer " + gw.token}, json=body
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["output"][0]["input"] == "patch"
        assert len(client.requests) == 1


def test_codex_binary_must_match_frozen_checksum(tmp_path):
    import hashlib
    from benchmarks.tool_annotation_effect.codex import verified_binary

    binary = tmp_path / "codex"
    binary.write_bytes(b"original")
    config = {
        "codex_binary_path": str(binary),
        "codex_binary_sha256": hashlib.sha256(b"original").hexdigest(),
    }
    assert verified_binary(config) == binary
    binary.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        verified_binary(config)


def test_harbor_codex_budget_without_final_file_still_reaches_verifier(tmp_path, monkeypatch):
    import asyncio
    import importlib
    import sys
    from contextlib import contextmanager
    from types import ModuleType, SimpleNamespace
    from benchmarks.tool_annotation_effect import codex

    base = ModuleType("harbor.agents.base")

    class Base:
        def __init__(self, logs_dir, model_name, **kwargs):
            self.logs_dir, self.model_name = logs_dir, model_name

    base.BaseAgent = Base
    installed = ModuleType("harbor.agents.installed.base")
    installed.NonZeroAgentExitCodeError = type("NonZeroAgentExitCodeError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "harbor.agents.base", base)
    monkeypatch.setitem(sys.modules, "harbor.agents.installed.base", installed)
    monkeypatch.delitem(
        sys.modules, "benchmarks.tool_annotation_effect.harbor_agent", raising=False
    )
    module = importlib.import_module("benchmarks.tool_annotation_effect.harbor_agent")

    class Gateway:
        error = "exceeded max_steps=1"

        def __init__(self, *args):
            pass

        @contextmanager
        def serve(self):
            yield "http://fake"

    class Environment:
        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=1, stdout=None)

    monkeypatch.setattr(codex, "CodexGateway", Gateway)
    monkeypatch.setattr(codex, "codex_files", lambda *args: ({}, "codex"))
    agent = module.AgencyCodexAgent(
        logs_dir=tmp_path,
        model_name="fake",
        experiment_config={"system_prompt": "solve", "budgets": {"timeout_s": 30}},
    )
    with pytest.raises(installed.NonZeroAgentExitCodeError, match="max_steps"):
        asyncio.run(agent.run("task", Environment(), SimpleNamespace()))
    assert (tmp_path / "final.txt").read_text() == ""
    assert (
        json.loads((tmp_path / "codex-result.json").read_text())["error"] == "exceeded max_steps=1"
    )
