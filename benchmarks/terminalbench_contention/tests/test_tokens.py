"""Token estimation: the prompt charge must track what the model is sent."""

import json
import math

import pytest

from benchmarks.terminalbench_contention import tokens
from benchmarks.terminalbench_contention.tokens import (
    EstimatorConfig,
    TokenEstimator,
    completion_cap,
    legacy_estimate,
    render_tools_typescript,
)

RAW = EstimatorConfig(safety_factor=1.0, safety_tokens=0)

TOOL = {
    "type": "function",
    "function": {
        "name": "exec_command",
        "description": "Run a shell command.\nReturns its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Command line"},
                "timeout_ms": {"type": "integer"},
                "mode": {"type": "string", "enum": ["fast", "safe"]},
                "env": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["cmd"],
        },
    },
}


@pytest.fixture(scope="module")
def encoding():
    try:
        return tokens.load_encoding("o200k_base")
    except tokens.TokenizerUnavailable:
        pytest.skip("o200k_base tokenizer cache is not available")


def test_message_count_matches_tokenizer_plus_documented_overhead(encoding):
    estimator = TokenEstimator(RAW)
    messages = [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "List the files in /app and summarise them."},
    ]
    expected = RAW.tokens_reply_priming
    for message in messages:
        expected += RAW.tokens_per_message
        expected += len(encoding.encode(message["role"])) + len(encoding.encode(message["content"]))
    estimate = estimator.estimate({"messages": messages})
    assert estimate.raw_tokens == expected == estimate.estimated_tokens
    assert estimate.method == "tokenizer" and estimate.messages == 2
    assert estimate.by_role.keys() == {"system", "user"}


def test_tool_calls_tool_results_and_names_are_counted(encoding):
    estimator = TokenEstimator(RAW)
    arguments = json.dumps({"cmd": "pytest -q tests/test_storage.py"})
    base = estimator.estimate({"messages": [{"role": "assistant", "content": None}]}).raw_tokens
    with_call = estimator.estimate(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "exec_command", "arguments": arguments},
                        }
                    ],
                }
            ]
        }
    ).raw_tokens
    assert with_call - base == (
        RAW.tokens_per_tool_call
        + len(encoding.encode("exec_command"))
        + len(encoding.encode(arguments))
    )
    result = estimator.estimate(
        {"messages": [{"role": "tool", "tool_call_id": "call_1", "content": "3 passed"}]}
    )
    assert result.raw_tokens > base
    named = estimator.estimate({"messages": [{"role": "user", "name": "ops", "content": "hi"}]})
    plain = estimator.estimate({"messages": [{"role": "user", "content": "hi"}]})
    assert named.raw_tokens == plain.raw_tokens + RAW.tokens_per_name + len(encoding.encode("ops"))


def test_tool_schemas_are_charged_as_rendered_not_as_json(encoding):
    rendered = render_tools_typescript([TOOL])
    assert "namespace functions {" in rendered and "type exec_command = (_: {" in rendered
    assert "// Command line" in rendered and "cmd: string," in rendered
    assert "timeout_ms?: number," in rendered and 'mode?: "fast" | "safe",' in rendered
    assert "env?: string[]," in rendered
    typescript = TokenEstimator(RAW).estimate({"messages": [], "tools": [TOOL]})
    as_json = TokenEstimator(
        EstimatorConfig(tool_schema_mode="json", safety_factor=1.0, safety_tokens=0)
    ).estimate({"messages": [], "tools": [TOOL]})
    assert typescript.tool_schema_tokens == (
        len(encoding.encode(rendered)) + RAW.tokens_tools_preamble + RAW.tokens_per_tool
    )
    assert 0 < typescript.tool_schema_tokens < as_json.tool_schema_tokens
    assert typescript.tools == 1


def test_estimate_is_far_below_the_legacy_bytes_over_two_rule(encoding):
    payload = {
        "model": "gpt-6-luna",
        "max_completion_tokens": 4096,
        "stream": True,
        "messages": [
            {"role": "system", "content": "Follow the repository conventions. " * 200},
            {"role": "user", "content": "def f(x):\n    return x + 1\n" * 150},
        ],
        "tools": [TOOL] * 8,
    }
    body = json.dumps(payload).encode()
    estimate = TokenEstimator().estimate_bytes(body)
    legacy_prompt = legacy_estimate(body, payload) - 4096
    assert legacy_prompt == math.ceil(len(body) / 2)
    assert estimate.estimated_tokens < 0.6 * legacy_prompt


def test_safety_margin_is_applied_and_recorded(encoding):
    payload = {"messages": [{"role": "user", "content": "hello " * 500}]}
    raw = TokenEstimator(RAW).estimate(payload)
    padded = TokenEstimator(EstimatorConfig(safety_factor=1.1, safety_tokens=50)).estimate(payload)
    assert padded.raw_tokens == raw.raw_tokens
    assert padded.estimated_tokens == math.ceil(raw.raw_tokens * 1.1) + 50


def test_special_token_text_and_unicode_are_plain_text(encoding):
    estimator = TokenEstimator(RAW)
    estimate = estimator.estimate(
        {
            "messages": [
                {"role": "user", "content": "<|endoftext|> <|im_start|> 日本語のテキスト 🚀"}
            ]
        }
    )
    assert estimate.raw_tokens > 10


def test_content_parts_and_unknown_shapes_are_counted(encoding):
    estimator = TokenEstimator(RAW)
    text = estimator.estimate({"messages": [{"role": "user", "content": "describe this"}]})
    parts = estimator.estimate(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "describe this"}]}]}
    )
    assert parts.raw_tokens == text.raw_tokens
    odd = estimator.estimate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:..."}}],
                }
            ]
        }
    )
    assert odd.raw_tokens > RAW.tokens_per_message


def test_fallback_is_conservative_and_labelled(monkeypatch):
    def unavailable(name):
        raise tokens.TokenizerUnavailable("no cache")

    monkeypatch.setattr(tokens, "load_encoding", unavailable)
    estimator = TokenEstimator(RAW)
    assert estimator.method == "fallback_bytes" and "no cache" in estimator.fallback_reason
    text = "The quick brown fox jumps over the lazy dog. " * 50
    estimate = estimator.estimate({"messages": [{"role": "user", "content": text}]})
    assert estimate.method == "fallback_bytes"
    assert estimate.raw_tokens >= math.ceil(len(text.encode()) / 3)


def test_fallback_overestimates_relative_to_the_tokenizer(encoding, monkeypatch):
    text = "for index in range(10):\n    print(index, 'value')\n" * 80
    payload = {"messages": [{"role": "user", "content": text}]}
    real = TokenEstimator(RAW).estimate(payload).raw_tokens
    monkeypatch.setattr(
        tokens,
        "load_encoding",
        lambda name: (_ for _ in ()).throw(tokens.TokenizerUnavailable("x")),
    )
    assert TokenEstimator(RAW).estimate(payload).raw_tokens > real


def test_undecodable_body_is_charged_by_size():
    estimate = TokenEstimator(RAW).estimate_bytes(b"\xff\xfe not json" * 30)
    assert estimate.method == "fallback_bytes" and estimate.estimated_tokens >= 100


def test_completion_cap_reads_the_declared_output_limit():
    assert completion_cap({"max_completion_tokens": 4096}, 99) == 4096
    assert completion_cap({"max_tokens": 512}, 99) == 512
    assert completion_cap({"max_completion_tokens": 100, "n": 3}, 99) == 300
    assert completion_cap({}, 777) == 777
    assert completion_cap({"max_completion_tokens": None}, 777) == 777
    assert completion_cap({"max_completion_tokens": True}, 777) == 777


@pytest.mark.parametrize(
    "bad",
    [
        dict(safety_factor=0.9),
        dict(safety_tokens=-1),
        dict(tool_schema_mode="xml"),
        dict(fallback_bytes_per_token=0),
    ],
)
def test_estimator_rejects_unsafe_configuration(bad):
    with pytest.raises(ValueError):
        EstimatorConfig(**bad).validate()
