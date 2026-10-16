"""Every harness compacts near 90% of the real context limit, whatever the
model and whichever backend reported the usage.

Two things decide when an external harness compacts, and both broke silently
before: the window Agency configures (each harness caps, ignores or shrinks it
in its own way) and the token counts Agency forwards (backends report
`prompt_tokens` in two conventions). The tests below go over every harness,
every backend's usage format, and every model in Agency's tables plus names no
harness knows yet, so a new model, backend or harness lands here first.

The `_threshold` functions restate each harness's own rule, for the versions
tests/host/fixtures/Dockerfile.harnesses pins. When a pin moves, re-check them
against the new release (docs/Context_Window.md says how).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import agency.llm.anthropic as anthropic_llm
import agency.llm.bedrock as bedrock_llm
import agency.llm.openai as openai_llm
import agency.llm.openai_responses as openai_responses_llm
from agency.configs.agconfig import agconfig
from agency.harness.adapters.base import AdapterRuntime, HarnessAdapter
from agency.harness.adapters.claude_code import (
    ClaudeDriver,
    _forward_usage_to_anthropic,
    claude_code_window_settings,
)
from agency.harness.adapters.codex import CODEX_UNKNOWN_MODEL_MAX_WINDOW, _responses_usage
from agency.harness.adapters.openai_chat_completions import _forward_usage
from agency.harness.adapters.pty.driver import driver_for
from agency.llm.agllm import agllm
from agency.native_harness import compaction
from agency.native_harness.usage import prompt_tokens

# One prompt of 190,000 tokens: 185,000 read from the cache, 2,000 written to
# it, 3,000 neither; 500 output tokens.
PROMPT, CACHE_READ, CACHE_WRITE, OUTPUT = 190_000, 185_000, 2_000, 500
UNCACHED = PROMPT - CACHE_READ - CACHE_WRITE


def _anthropic_usage():
    return anthropic_llm._serialize_anthropic_usage(
        SimpleNamespace(
            input_tokens=UNCACHED,
            output_tokens=OUTPUT,
            cache_read_input_tokens=CACHE_READ,
            cache_creation_input_tokens=CACHE_WRITE,
        )
    )


def _bedrock_converse_usage():
    backend = object.__new__(bedrock_llm._BedrockConverseBackend)
    result = backend._format_context_backend_to_agency(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "usage": {
                "inputTokens": UNCACHED,
                "outputTokens": OUTPUT,
                "cacheReadInputTokens": CACHE_READ,
                "cacheWriteInputTokens": CACHE_WRITE,
                "totalTokens": PROMPT + OUTPUT,
            },
            "stopReason": "end_turn",
        }
    )
    return result["usage"]


def _openai_usage():
    return openai_llm._serialize_openai_usage(
        SimpleNamespace(
            prompt_tokens=PROMPT,
            completion_tokens=OUTPUT,
            total_tokens=PROMPT + OUTPUT,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=CACHE_READ, cache_write_tokens=CACHE_WRITE
            ),
            completion_tokens_details=None,
        )
    )


def _openai_responses_usage():
    return openai_responses_llm._serialize_responses_usage(
        SimpleNamespace(
            input_tokens=PROMPT,
            output_tokens=OUTPUT,
            total_tokens=PROMPT + OUTPUT,
            input_tokens_details=SimpleNamespace(
                cached_tokens=CACHE_READ, cache_write_tokens=CACHE_WRITE
            ),
            output_tokens_details=None,
        )
    )


# Backend class -> the usage it reports for the prompt above. Anthropic-on-AWS
# and Claude-on-Bedrock inherit the Anthropic serializer; Bedrock's
# OpenAI-compatible (Mantle) route inherits OpenAI's.
BACKEND_USAGE = {
    "_AnthropicBackend": _anthropic_usage,
    "_AnthropicBedrockBackend": _anthropic_usage,
    "_AnthropicAWSBackend": _anthropic_usage,
    "_BedrockConverseBackend": _bedrock_converse_usage,
    "_OpenAICompatibleBackend": _openai_usage,
    "_OpenAICompatibleBedrockBackend": _openai_usage,
    "_OpenAIResponsesBackend": _openai_responses_usage,
}


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def test_every_llm_backend_has_a_usage_case_here():
    """A new backend must say how it reports a cached prompt."""
    backends = {cls.__name__ for cls in _all_subclasses(agllm)} - {"_MockBackend"}
    assert backends == set(BACKEND_USAGE)


@pytest.mark.parametrize("backend", sorted(BACKEND_USAGE))
def test_prompt_tokens_is_the_whole_prompt_for_every_backend(backend):
    usage = BACKEND_USAGE[backend]()
    assert prompt_tokens(usage) == (PROMPT, CACHE_READ, CACHE_WRITE)
    assert prompt_tokens(usage).uncached == UNCACHED


# -- What each harness counts, from what its adapter forwards ----------------
#
# Claude Code: input + cache_read + cache_creation (Anthropic's convention).
# Codex: total_tokens, with cached_tokens a part of input_tokens.
# Grok: total_tokens. OpenCode: prompt_tokens + completion_tokens (AI SDK's
# totalTokens). Kimi: prompt_tokens, top-level cached_tokens split out of it.
# Native: prompt_tokens() itself.


def _claude_code_view(usage):
    wire = _forward_usage_to_anthropic(usage)
    assert min(wire["input_tokens"], wire["cache_read_input_tokens"]) >= 0
    return (
        wire["input_tokens"] + wire["cache_read_input_tokens"] + wire["cache_creation_input_tokens"]
    )


def _codex_view(usage):
    wire = _responses_usage(usage)
    assert wire["input_tokens_details"]["cached_tokens"] <= wire["input_tokens"]
    return wire["total_tokens"] - wire["output_tokens"]


def _chat_view(usage):
    wire = _forward_usage(usage)
    assert wire["total_tokens"] == wire["prompt_tokens"] + wire["completion_tokens"]
    cached = wire.get("cached_tokens", 0)
    assert cached == wire.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    assert cached <= wire["prompt_tokens"]
    return wire["prompt_tokens"]


HARNESS_VIEW = {
    "claude_code": _claude_code_view,
    "codex": _codex_view,
    "grok": _chat_view,
    "opencode": _chat_view,
    "kimi": _chat_view,
    "native": lambda usage: prompt_tokens(usage).total,
}


def _harness_names() -> "set[str]":
    with pytest.raises(ValueError) as error:
        HarnessAdapter.for_config("no-such-harness", agconfig())
    message = str(error.value)
    return set(message[message.index("one of") :].split("'")[1::2])


def test_every_harness_has_a_usage_view_and_a_threshold_here():
    """A new harness must say how it counts tokens and when it compacts."""
    assert set(HARNESS_VIEW) == _harness_names()
    assert set(THRESHOLD) == _harness_names()


@pytest.mark.parametrize("harness", sorted(HARNESS_VIEW))
@pytest.mark.parametrize("backend", sorted(BACKEND_USAGE))
def test_every_harness_sees_the_whole_prompt_from_every_backend(harness, backend):
    assert HARNESS_VIEW[harness](BACKEND_USAGE[backend]()) == PROMPT


def test_claude_code_is_told_the_cached_reads():
    wire = _forward_usage_to_anthropic(_openai_usage())
    assert wire["cache_read_input_tokens"] == CACHE_READ
    assert wire["cache_creation_input_tokens"] == CACHE_WRITE
    assert wire["input_tokens"] == UNCACHED


# -- When each harness compacts, from the configuration its adapter writes ---


def _runtime(model):
    return AdapterRuntime(
        agconfig=agconfig(),
        model=model,
        engine_name="test",
        harness_base_url="http://127.0.0.1:8766",
        token="attempt-token",
        syscall_policy=object(),
        register_control_handle=Mock(),
        register_redirect=Mock(),
    )


def _write_config(harness, model, limit, tmp_path, monkeypatch):
    monkeypatch.setattr(
        f"agency.harness.adapters.{harness}.fetch_context_limit", lambda *a, **k: limit
    )
    runtime = _runtime(model)
    driver_for(
        HarnessAdapter.for_config(harness, runtime.agconfig), runtime, tmp_path, None, None, 4
    )


def _claude_code_threshold(model, limit, tmp_path, monkeypatch):
    # Claude Code 2.1.251; see claude_code.py's notes above
    # claude_code_window_settings.
    model, argv, env = claude_code_window_settings(model, limit)
    if model.endswith("[1m]"):
        model_window = 1_000_000
    elif not model.startswith("claude-") and "CLAUDE_CODE_MAX_CONTEXT_TOKENS" in env:
        model_window = int(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"])
    else:
        model_window = 200_000
    if "CLAUDE_CODE_AUTO_COMPACT_WINDOW" in env:
        setting = min(1_000_000, max(100_000, int(env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"])))
    elif "--autocompact" in argv:
        setting = int(argv[argv.index("--autocompact") + 1])
        assert 100_000 <= setting <= 1_000_000
    else:
        setting = None
    window = min(model_window, setting) if setting else model_window
    return window - 20_000 - 13_000


def _codex_threshold(model, limit, tmp_path, monkeypatch):
    # Codex 0.147.0: 90% of model_context_window, capped at the model's
    # max_context_window: 272K for one it doesn't know, unless the adapter's
    # model_catalog_json gives it one.
    _write_config("codex", model, limit, tmp_path, monkeypatch)
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    window = config["model_context_window"]
    cap = CODEX_UNKNOWN_MODEL_MAX_WINDOW
    if "model_catalog_json" in config:
        entry = json.loads(Path(config["model_catalog_json"]).read_text())["models"][0]
        assert entry["slug"] == model
        cap = entry["max_context_window"]
    return int(min(window, cap) * 0.9)


def _grok_threshold(model, limit, tmp_path, monkeypatch):
    # Grok 1.0.0: [session] auto_compact_threshold_percent, 85 by default.
    _write_config("grok", model, limit, tmp_path, monkeypatch)
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    percent = config.get("session", {}).get("auto_compact_threshold_percent", 85)
    return int(config["model"]["agency-proxy"]["context_window"] * percent / 100)


def _opencode_threshold(model, limit, tmp_path, monkeypatch):
    # OpenCode 1.18.15 (session/overflow.ts): context - min(output, 32_000).
    _write_config("opencode", model, limit, tmp_path, monkeypatch)
    config = json.loads((tmp_path / "opencode.json").read_text())
    entry = config["provider"]["agency-proxy"]["models"][model]["limit"]
    return entry["context"] - min(entry["output"], 32_000)


def _kimi_threshold(model, limit, tmp_path, monkeypatch):
    # Kimi Code 0.42.0: min(ratio * max, max - reserved), 0.85 and 50K by
    # default, the reserve only when smaller than max. It checks a step late
    # (kimi.py), so it lands near 90% at 0.85.
    _write_config("kimi", model, limit, tmp_path, monkeypatch)
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    size = config["models"][model]["max_context_size"]
    loop = config.get("loop_control", {})
    ratio = loop.get("compaction_trigger_ratio", 0.85)
    reserve = loop.get("reserved_context_size", 50_000)
    by_reserve = size - reserve if 0 < reserve < size else size
    return int(min(size * ratio, by_reserve))


def _native_threshold(model, limit, tmp_path, monkeypatch):
    return int(limit * compaction.COMPACT_THRESHOLD)


THRESHOLD = {
    "claude_code": _claude_code_threshold,
    "codex": _codex_threshold,
    "grok": _grok_threshold,
    "opencode": _opencode_threshold,
    "kimi": _kimi_threshold,
    "native": _native_threshold,
}

# Every model Agency knows a limit for, plus names no harness has heard of.
MODELS = sorted(
    {
        **openai_llm._OPENAI_MAX_INPUT_TOKENS,
        **{name: info.context_window for name, info in anthropic_llm._ANTHROPIC_MODELS.items()},
        "claude-future-9": 1_000_000,
        "claude-future-9-small": 200_000,
        "gpt-future-9": 2_000_000,
        "open-weights-model": 131_072,
        "small-open-weights-model": 32_768,
    }.items()
)


def _expected_range(harness, model, limit):
    """(lowest, highest) acceptable threshold; 80-92% unless a harness can't."""
    low, high = 0.80 * limit, 0.92 * limit
    if harness == "claude_code":
        if limit * 0.9 + 33_000 > 1_000_000:
            # --autocompact stops at 1M: Claude Code can't wait past 967K.
            return 1_000_000 - 33_000, 1_000_000 - 33_000
        if model.startswith("claude-") and limit * 0.9 + 33_000 < 100_000:
            pytest.skip("Claude Code's 100K minimum window binds; no Claude model is that small")
    return low, high


@pytest.mark.parametrize("model,limit", MODELS, ids=[m for m, _ in MODELS])
@pytest.mark.parametrize("harness", sorted(THRESHOLD))
def test_every_harness_compacts_near_90_percent_for_every_model(
    harness, model, limit, tmp_path, monkeypatch
):
    threshold = THRESHOLD[harness](model, limit, tmp_path, monkeypatch)
    low, high = _expected_range(harness, model, limit)
    assert low <= threshold <= high, (harness, model, limit, threshold)
    assert threshold < limit


@pytest.mark.parametrize("limit", [32_768, 131_072, 200_000, 922_000, 1_050_000])
@pytest.mark.parametrize("model", ["gpt-6-luna", "open-weights-model", "claude-haiku-5-5"])
def test_claude_driver_applies_the_window_settings(model, limit, tmp_path, monkeypatch):
    (tmp_path / "events").mkdir()
    monkeypatch.setattr(
        "agency.harness.adapters.claude_code.fetch_context_limit", lambda *a, **k: limit
    )
    adapter = SimpleNamespace(
        agconfig=agconfig(),
        prepare_pty=lambda *args, **kwargs: (["claude", "--model", model], {"PATH": "/bin"}),
    )
    runtime = AdapterRuntime(
        agconfig(), model, "engine", "http://127.0.0.1:8766", "token", SimpleNamespace()
    )
    driver = ClaudeDriver(adapter, runtime, tmp_path, None, None, None)
    expected_model, argv, env = claude_code_window_settings(model, limit)
    assert driver.argv[driver.argv.index("--model") + 1] == expected_model
    for flag in argv:
        assert flag in driver.argv
    assert env.items() <= driver.env.items()


# -- The native harness counts what the backend counted ----------------------


class _SummaryLLM:
    def __init__(self):
        self.calls = 0

    def dispatch(self, model, messages, *args, **kwargs):
        self.calls += 1
        return {"message": {"role": "assistant", "content": "summary"}}


def _conversation(turns):
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
    ]
    for i in range(turns):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{i}", "function": {"name": "bash", "arguments": "{}"}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 400})
    return messages


def test_native_compacts_on_the_backends_count_not_only_its_estimate():
    messages = _conversation(12)
    llm = _SummaryLLM()
    # ~1.3K tokens by the 4-chars-per-token estimate: far from 90% of 10K.
    kept, _ = compaction.maybe_compact(messages, 10_000, llm, "m", None)
    assert kept is messages and llm.calls == 0
    # The backend counted 9.5K for the same conversation (tool schemas, dense
    # text): over 90%, so it compacts.
    compacted, _ = compaction.maybe_compact(
        messages, 10_000, llm, "m", None, measured=(9_500, len(messages))
    )
    assert llm.calls == 1 and compacted is not messages


# -- Context-limit lookup ----------------------------------------------------


def test_a_listed_limit_never_exceeds_the_known_input_cap(monkeypatch):
    """LiteLLM lists gpt-6-luna at 1,050,000 (its whole window); its input cap
    is 922,000."""
    backend = object.__new__(openai_llm._OpenAICompatibleBackend)
    backend.agconfig = agconfig()
    backend.agconfig.llm.model = "gpt-6-luna"
    listing = [SimpleNamespace(id="gpt-6-luna", max_input_tokens=1_050_000, model_extra={})]
    monkeypatch.setattr(backend, "list_models", lambda: listing, raising=False)
    monkeypatch.setattr(backend, "retrieve_model", lambda model: None, raising=False)
    assert backend.fetch_context_limit() == 922_000
    backend.agconfig.llm.model = "unknown-open-model"
    listing[0].id = "unknown-open-model"
    assert backend.fetch_context_limit() == 1_050_000


# -- An overflow the backend rejects reads as one to every harness -----------


@pytest.mark.parametrize(
    "message",
    [
        "anthropic 400: prompt is too long: 250000 tokens > 200000 maximum",
        "litellm 400: This model's maximum context length is 128000 tokens. However, your "
        "messages resulted in 250000 tokens.",
        "BedrockException - Input is too long for requested model.",
        "Error code: 400 - {'error': {'code': 'context_length_exceeded'}}",
        "vllm 400: This model's maximum context length is 131072 tokens",
    ],
)
def test_overflow_errors_carry_the_words_and_code_harnesses_recover_on(message):
    from agency.harness.adapters.streaming import context_overflow_message, upstream_error_body
    from agency.harness.clients.host_services_client import HostDispatchError

    worded = context_overflow_message(message)
    assert worded is not None and "prompt is too long" in worded.lower() and message in worded
    body, status = upstream_error_body(HostDispatchError({"message": message, "status_code": 400}))
    assert status == 400
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["error"]["message"] == worded


@pytest.mark.parametrize(
    "message",
    ["Rate limit reached for tokens per minute", "internal server error", "invalid tool schema"],
)
def test_other_errors_pass_through_unchanged(message):
    from agency.harness.adapters.streaming import context_overflow_message, upstream_error_body
    from agency.harness.clients.host_services_client import HostDispatchError

    assert context_overflow_message(message) is None
    body, _ = upstream_error_body(HostDispatchError({"message": message, "status_code": 400}))
    assert body["error"]["message"] == message and "code" not in body["error"]


def test_native_counts_what_came_after_at_the_measured_rate():
    """Dense text (a tool output of random words) runs ~1 token per 1.5
    characters; the 4-chars-per-token guess would put the next step a whole
    file under the backend's count."""
    messages = _conversation(12)
    llm = _SummaryLLM()
    # The backend counted 4.8K tokens for ~5K characters: about 1 per char.
    measured = (4_800, len(messages))
    messages.append({"role": "tool", "tool_call_id": "late", "content": "y" * 4_000})
    # 4.8K + 4K characters at that rate = 8.8K < 9K: no compaction yet ...
    kept, _ = compaction.maybe_compact(messages, 10_000, llm, "m", None, measured=measured)
    assert llm.calls == 0
    # ... one more such output (12.8K) is over 90% of 10K.
    messages.append({"role": "tool", "tool_call_id": "later", "content": "z" * 4_000})
    compaction.maybe_compact(messages, 10_000, llm, "m", None, measured=measured)
    assert llm.calls == 1
