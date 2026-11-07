"""Agency's OpenAI-compatible backend, instrumented, against a local mock endpoint.

These are integration tests of the real request path below the harness:
Agency request -> backend wire format -> OpenAI SDK -> HTTP -> mock server, with
the shared limiter and telemetry observing every HTTP attempt.
"""

import json

import pytest

from benchmarks.terminalbench_contention import api_instrument, mock_model
from benchmarks.terminalbench_contention.api_budget import (
    BudgetPolicy,
    DeadlineReached,
    SharedBudget,
    read_ledger,
    verify_ledger,
)
from benchmarks.terminalbench_contention.tokens import EstimatorConfig
from benchmarks.terminalbench_contention.util import read_jsonl

SECRET = "sk-test-secret-key-0123456789abcdef"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
]


def agency_request(text="Run the tests.", tools=True):
    request = {
        "messages": [
            {
                "role": "system",
                "blocks": [{"type": "text", "index": 0, "text": "You are an agent."}],
            },
            {"role": "user", "blocks": [{"type": "text", "index": 0, "text": text}]},
        ]
    }
    if tools:
        request["tools"] = TOOLS
    return request


class Harness:
    def __init__(self, tmp_path, mock_arguments=(), policy=None, deadline=None, **llm):
        from agency.configs.agconfig import agconfig, llmconfig
        from agency.llm.agllm import agllm
        from agency.llm.openai import _OpenAICompatibleBackend

        self.tmp_path = tmp_path
        self.server, self.base_url, self.mock = mock_model.start(
            ["--log", str(tmp_path / "mock.jsonl"), *mock_arguments]
        )
        self.policy = policy or BudgetPolicy(
            minimum_request_spacing_seconds=0,
            rate_limit_cooldown_seconds=0.2,
            estimator=EstimatorConfig(safety_factor=1.0, safety_tokens=0),
        )
        self.budget = SharedBudget.create(
            tmp_path / "ledger", self.policy, run_id="t", deadline=deadline
        )
        self.telemetry = api_instrument.ApiTelemetry(
            self.budget, tmp_path / "api-requests.jsonl", agent_id="A01"
        )
        self.backend_class = _OpenAICompatibleBackend
        api_instrument.install(self.backend_class, self.telemetry)
        settings = dict(
            provider="openai",
            model="mock-luna",
            api_key=SECRET,
            base_url=self.base_url,
            reasoning_effort="none",
            max_completion_tokens=256,
            stream_timeout=20,
            idle_timeout=20,
        )
        settings.update(llm)
        self.backend = agllm.for_config(agconfig(llmconfig(**settings)))

    def close(self):
        api_instrument.uninstall(self.backend_class)
        self.server.shutdown()

    def records(self, kind="request"):
        return [r for r in read_jsonl(self.tmp_path / "api-requests.jsonl") if r["kind"] == kind]

    def ledger(self):
        return read_ledger(self.tmp_path / "ledger")


@pytest.fixture
def harness(tmp_path):
    created = []

    def make(*mock_arguments, **kwargs):
        instance = Harness(tmp_path, mock_arguments, **kwargs)
        created.append(instance)
        return instance

    yield make
    for instance in created:
        instance.close()


def test_streamed_request_is_admitted_observed_and_reconciled(harness):
    h = harness("--prompt-tokens", "77", "--cached-tokens", "5")
    items = list(h.backend.dispatch_stream(agency_request()))
    assert any(i.get("block_type") == "tool_use" for i in items)
    assert h.mock.count == 1
    [record] = h.records()
    for field in (
        "agent_id",
        "request_id",
        "ready_ns",
        "admitted_ns",
        "headers_ns",
        "first_chunk_ns",
        "completed_ns",
        "estimated_prompt_tokens",
        "reserved_completion_tokens",
        "admission_charge",
        "reported_prompt_tokens",
        "reported_completion_tokens",
        "reconciliation",
        "api_wait_seconds",
        "response_latency_seconds",
        "stream_seconds",
        "retries",
        "http_429",
        "available_budget_at_admission",
        "waiters_at_admission",
    ):
        assert record[field] is not None, field
    assert record["outcome"] == "completed" and record["streaming"] is True
    assert record["ready_ns"] <= record["admitted_ns"] <= record["headers_ns"]
    assert record["headers_ns"] <= record["first_chunk_ns"] <= record["completed_ns"]
    assert record["reported_prompt_tokens"] == 77 and record["reported_cached_tokens"] == 5
    assert record["reserved_completion_tokens"] == 256  # the configured output cap
    assert record["admission_charge"] == record["estimated_prompt_tokens"] + 256
    assert record["retries"] == 0 and record["attempts"] == 1 and record["status_codes"] == [200]
    assert record["available_budget_at_admission"] == h.policy.tokens_per_window
    events = h.ledger()
    assert [e["kind"] for e in events] == ["admit", "reconcile"]
    assert events[0]["agent_id"] == "A01" and events[1]["reported_prompt_tokens"] == 77
    report = verify_ledger(h.tmp_path / "ledger")
    assert report["ok"] and not report["unsettled"]


def test_the_admitted_bytes_are_the_bytes_the_endpoint_received(harness):
    """The estimator sees the real LiteLLM-compatible request, tools included."""
    h = harness()
    list(h.backend.dispatch_stream(agency_request()))
    [admitted] = h.records("attempt_admitted")
    [received] = [r for r in read_jsonl(h.tmp_path / "mock.jsonl") if r["event"] == "request"]
    assert admitted["request_bytes"] == received["bytes"]
    assert admitted["estimate_method"] == "tokenizer"
    wire = h.backend._format_context_agency_to_backend(agency_request())
    plain = h.backend._format_context_agency_to_backend(agency_request(tools=False))
    estimator = h.budget.estimator
    assert estimator.estimate(wire).tool_schema_tokens > 0
    assert estimator.estimate(plain).tool_schema_tokens == 0
    assert admitted["estimated_prompt_tokens"] > estimator.estimate(plain).estimated_tokens


def test_non_streaming_request_is_accounted_the_same_way(harness):
    h = harness("--prompt-tokens", "40")
    result = h.backend.dispatch(agency_request())
    assert result["usage"]["prompt_tokens"] == 40
    [record] = h.records()
    assert record["outcome"] == "completed" and record["streaming"] is False
    assert record["reported_prompt_tokens"] == 40
    assert verify_ledger(h.tmp_path / "ledger")["ok"]


def test_http_429_is_recorded_cools_down_and_the_retry_is_charged_separately(harness):
    h = harness("--fail-429", "1", "--retry-after", "0")
    items = list(h.backend.dispatch_stream(agency_request()))
    assert items and h.mock.count == 2
    [record] = h.records()
    assert record["outcome"] == "completed"
    assert record["attempts"] == 2 and record["retries"] == 1 and record["http_429"] == 1
    assert record["status_codes"] == [429, 200]
    assert record["attempt_outcomes"] == ["http_429", "completed"]
    events = h.ledger()
    assert [e["kind"] for e in events] == ["admit", "rate_limited", "fail", "admit", "reconcile"]
    assert events[2]["status_code"] == 429 and events[2]["reason"] == "http_429"
    # The 429 put every client into cooldown; the retry waited for it.
    assert events[3]["blocked_by"] == ["rate_limit_cooldown"]
    assert events[3]["wait_seconds"] >= 0.15
    assert record["admission_charge_all_attempts"] == events[0]["charge"] + events[3]["charge"]
    report = verify_ledger(h.tmp_path / "ledger")
    assert report["ok"] and report["totals"]["rate_limited"] == 1


def test_persistent_server_errors_fail_the_request_and_every_attempt_is_settled(harness):
    import openai

    h = harness("--fail-500", "10")
    with pytest.raises(openai.APIStatusError):
        list(h.backend.dispatch_stream(agency_request()))
    [record] = h.records()
    assert record["outcome"] == "request_error" and record["error_type"]
    assert record["attempts"] == h.mock.count >= 2
    assert set(record["status_codes"]) == {500}
    assert record["reported_prompt_tokens"] is None
    report = verify_ledger(h.tmp_path / "ledger")
    assert report["ok"] and not report["unsettled"]
    assert report["totals"]["failed"] == record["attempts"]


def test_api_timeout_is_recorded_as_a_failed_request(harness):
    import openai

    h = harness("--hang-first", "10", "--hang-seconds", "30", stream_timeout=0.4)
    with pytest.raises(openai.APIError):
        list(h.backend.dispatch_stream(agency_request()))
    [record] = h.records()
    assert record["outcome"] == "request_error"
    assert record["attempts"] >= 2 and record["status_codes"] == [None] * record["attempts"]
    assert set(record["attempt_outcomes"]) <= {"transport_error", "request_error"}
    report = verify_ledger(h.tmp_path / "ledger")
    assert report["ok"] and not report["unsettled"] and report["totals"]["failed"] >= 2


def test_cancelling_a_stream_settles_the_reservation(harness):
    h = harness("--stream-seconds", "3")
    stream = h.backend.dispatch_stream(agency_request())
    assert next(stream) is not None
    stream.close()  # what Agency does when a request is cancelled mid-stream
    [record] = h.records()
    assert record["outcome"] == "cancelled" and record["reported_prompt_tokens"] is None
    events = h.ledger()
    assert [e["kind"] for e in events] == ["admit", "fail"] and events[1]["reason"] == "cancelled"
    assert verify_ledger(h.tmp_path / "ledger")["ok"]


def test_deadline_during_admission_never_reaches_the_endpoint(harness, tmp_path):
    import time

    policy = BudgetPolicy(
        tokens_per_window=400,
        minimum_request_spacing_seconds=0,
        poll_interval_seconds=0.02,
        estimator=EstimatorConfig(safety_factor=1.0, safety_tokens=0),
    )
    stop_at = time.monotonic() + 0.4
    h = harness(policy=policy, deadline=lambda: stop_at, max_completion_tokens=64)
    # Fill the window so the next request must wait past the deadline.
    filler = json.dumps(
        dict(max_completion_tokens=64, messages=[dict(role="user", content="alpha " * 250)])
    ).encode()
    h.budget.admit(filler, request_id="filler")
    with pytest.raises(Exception) as caught:
        list(h.backend.dispatch_stream(agency_request()))
    assert (
        isinstance(caught.value, DeadlineReached)
        or isinstance(caught.value.__cause__, DeadlineReached)
        or "deadline" in str(caught.value).lower()
        or caught.type.__name__ in ("APIConnectionError", "APITimeoutError")
    )
    assert h.mock.count == 0
    assert h.records("wait_censored")
    assert any(e["kind"] == "wait_censored" for e in h.ledger())
    assert verify_ledger(h.tmp_path / "ledger")["ok"]


def test_credentials_and_bodies_never_reach_the_artifacts(harness):
    h = harness()
    marker = "UNIQUE-PROMPT-MARKER-7f3a"
    list(h.backend.dispatch_stream(agency_request(text=marker)))
    written = (
        (h.tmp_path / "api-requests.jsonl").read_text()
        + (h.tmp_path / "ledger" / "ledger.jsonl").read_text()
        + (h.tmp_path / "ledger" / "state.json").read_text()
    )
    assert SECRET not in written and "Authorization" not in written
    assert marker not in written and h.base_url not in written


def test_model_listing_is_not_charged_as_a_model_request(harness):
    h = harness()
    assert [m.id for m in h.backend.list_models()] == ["mock-luna"]
    assert h.ledger() == [] and h.records() == []


def test_instrumentation_can_be_removed(harness):
    h = harness()
    originals = h.backend_class._tb_contention_originals
    api_instrument.uninstall(h.backend_class)
    assert h.backend_class.make_client is originals[0]
    assert not hasattr(h.backend_class, "_tb_contention_originals")
    list(h.backend.dispatch_stream(agency_request()))
    assert h.ledger() == []
    api_instrument.install(h.backend_class, h.telemetry)


def test_codex_responses_request_maps_to_the_wire_format_that_is_estimated(harness):
    """Codex -> Agency adapter -> backend: tools and instructions survive to the wire."""
    from agency.harness.adapters.codex import CodexAdapter

    h = harness()
    codex_body = {
        "model": "mock-luna",
        "stream": True,
        "instructions": "You are Codex.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "List files."}],
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            }
        ],
        "tool_choice": "auto",
    }
    context = CodexAdapter.__new__(CodexAdapter)._format_context_harness_to_agency(codex_body)
    wire = h.backend._format_context_agency_to_backend(context)
    assert [m["role"] for m in wire["messages"]] == ["system", "user"]
    assert wire["tools"][0]["function"]["name"] == "exec_command"
    assert wire["max_completion_tokens"] == 256 and wire["tool_choice"] == "auto"
    items = list(h.backend.dispatch_stream(context))
    assert any(i.get("block_type") == "tool_use" and i.get("name") == "exec_command" for i in items)
    [admitted] = h.records("attempt_admitted")
    estimate = h.budget.estimator.estimate(wire)
    assert (
        admitted[
            "tool_schema_tokens" if "tool_schema_tokens" in admitted else "estimated_prompt_tokens"
        ]
        > 0
    )
    assert estimate.tool_schema_tokens > 0 and estimate.by_role["system"] > 0
