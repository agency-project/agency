"""Tests for agllm — LLM client wrapper, streaming call, kwarg building, and compaction."""

from unittest.mock import MagicMock, patch

from agency.llm.agllm import agllm
from agency.agconfig import agConfig
from agency.llm.agllm import _AgLLMFields
from agency.llm import agOpenAIBackendConfig


def _cfg(**fields) -> agConfig:
    """Test helper: wrap agllm_backend fields in an agConfig."""
    return agConfig({"agllm_backend": fields})


_COMPACT_THRESHOLD = _AgLLMFields.COMPACT_THRESHOLD
_TAIL_MAX_TOKENS = _AgLLMFields.TAIL_MAX_TOKENS
_TAIL_MIN_TOKENS = _AgLLMFields.TAIL_MIN_TOKENS
_TAIL_FRACTION = _AgLLMFields.TAIL_FRACTION
_TOOL_OUTPUT_MAX_CHARS = _AgLLMFields.TOOL_OUTPUT_MAX_CHARS
_PRUNE_MIN_FREE_TOKENS = _AgLLMFields.PRUNE_MIN_FREE_TOKENS

TAIL_TURNS = _AgLLMFields.tail_turns.default
DEFAULT_CONTEXT_LIMIT = _AgLLMFields.default_context_limit.default

build_assistant_msg = agllm.build_assistant_msg
build_llm_kwargs = agllm.build_llm_kwargs
fetch_context_limit = agllm.fetch_context_limit


LLM_COMPACT_CONFIG = {"api_key": "test", "model": "", "base_url": "http://localhost/v1"}


def test_build_llm_kwargs_includes_reasoning_effort():
    cfg = agConfig(agOpenAIBackendConfig(model="gpt-5.6-luna", reasoning_effort="none"))

    kwargs = agllm.build_llm_kwargs(cfg, [{"role": "user", "content": "hi"}], None)

    assert kwargs["reasoning_effort"] == "none"


BIG_CTX = 100_000
LLM_COMPACT = agllm(_cfg(**LLM_COMPACT_CONFIG), context_limit=BIG_CTX)


# ---------------------------------------------------------------------------
# build_llm_kwargs
# ---------------------------------------------------------------------------


def test_build_llm_kwargs_includes_model():
    cfg = _cfg(model="", api_key="x")
    kw = build_llm_kwargs(cfg, [], None)
    assert kw["model"] == ""


def test_build_llm_kwargs_default_model():
    kw = build_llm_kwargs(_cfg(), [], None)
    assert kw["model"] == ""


def test_build_llm_kwargs_messages_included():
    msgs = [{"role": "user", "content": "hi"}]
    kw = build_llm_kwargs(_cfg(), msgs, None)
    assert kw["messages"] == msgs


def test_build_llm_kwargs_strips_underscore_keys_from_messages():
    msgs = [{"role": "user", "content": "hi", "_thinking": "internal"}]
    kw = build_llm_kwargs(_cfg(), msgs, None)
    assert "_thinking" not in kw["messages"][0]
    assert kw["messages"][0]["content"] == "hi"


def test_build_llm_kwargs_no_tools_key_when_none():
    kw = build_llm_kwargs(_cfg(), [], None)
    assert "tools" not in kw


def test_build_llm_kwargs_tools_included():
    tools = [{"type": "function", "function": {"name": "f"}}]
    kw = build_llm_kwargs(_cfg(), [], tools)
    assert kw["tools"] == tools


def test_build_llm_kwargs_openai_gen_params_forwarded():
    cfg = _cfg(model="m", temperature=0.7, max_completion_tokens=512)
    kw = build_llm_kwargs(cfg, [], None)
    assert kw["temperature"] == 0.7
    assert kw["max_completion_tokens"] == 512


def test_build_llm_kwargs_max_tokens_translated_with_warning(capsys):
    kw = build_llm_kwargs(_cfg(model="m", max_tokens=256), [], None)
    assert kw["max_completion_tokens"] == 256
    assert "max_tokens" not in kw
    assert "deprecated" in capsys.readouterr().out


def test_build_llm_kwargs_max_completion_tokens_wins_when_both_present(capsys):
    cfg = _cfg(model="m", max_tokens=256, max_completion_tokens=512)
    kw = build_llm_kwargs(cfg, [], None)
    assert kw["max_completion_tokens"] == 512
    assert "deprecated" in capsys.readouterr().out


def test_build_llm_kwargs_unknown_params_not_forwarded():
    cfg = _cfg(model="m", custom_param="ignored")
    kw = build_llm_kwargs(cfg, [], None)
    assert "custom_param" not in kw


def test_build_llm_kwargs_extra_body_params():
    cfg = _cfg(model="m", top_k=50, guided_json={"type": "object"})
    kw = build_llm_kwargs(cfg, [], None)
    assert kw["extra_body"]["top_k"] == 50
    assert kw["extra_body"]["guided_json"] == {"type": "object"}


def test_build_llm_kwargs_explicit_extra_body_merged():
    cfg = _cfg(model="m", extra_body={"stream_options": True}, top_k=10)
    kw = build_llm_kwargs(cfg, [], None)
    assert kw["extra_body"]["stream_options"] is True
    assert kw["extra_body"]["top_k"] == 10


def test_build_llm_kwargs_no_extra_body_when_empty():
    kw = build_llm_kwargs(_cfg(model="m"), [], None)
    assert "extra_body" not in kw


# ---------------------------------------------------------------------------
# change_config / get_config_copy
# ---------------------------------------------------------------------------


def test_llm_change_config_reaches_backend():
    """Mutating a cloned agconfig's field alone never reaches llm.backend --
    change_config is the supported way to push a live update through."""
    llm = agllm(_cfg(temperature=0.7), context_limit=BIG_CTX)
    llm.change_config(_cfg(temperature=0.2))
    assert llm.backend.temperature == 0.2


def test_llm_change_config_clones_given_agconfig():
    llm = agllm(_cfg(), context_limit=BIG_CTX)
    new_cfg = _cfg(temperature=0.2)
    llm.change_config(new_cfg)
    new_cfg.agllm_backend.temperature = 0.9
    assert llm.backend.temperature == 0.2


def test_llm_get_config_copy_returns_clone_not_same_object():
    llm = agllm(_cfg(temperature=0.7), context_limit=BIG_CTX)
    copy = llm.get_config_copy()
    assert copy is not llm._agconfig


def test_llm_get_config_copy_reflects_current_values():
    llm = agllm(_cfg(temperature=0.7), context_limit=BIG_CTX)
    assert llm.get_config_copy().agllm_backend.temperature == 0.7


def test_llm_get_config_copy_after_change_config_reflects_new_values():
    llm = agllm(_cfg(temperature=0.7), context_limit=BIG_CTX)
    llm.change_config(_cfg(temperature=0.2))
    assert llm.get_config_copy().agllm_backend.temperature == 0.2


def test_mutating_llm_get_config_copy_does_not_affect_llm():
    llm = agllm(_cfg(temperature=0.7), context_limit=BIG_CTX)
    copy = llm.get_config_copy()
    copy.agllm_backend.temperature = 0.1
    assert llm.backend.temperature == 0.7


# ---------------------------------------------------------------------------
# build_assistant_msg
# ---------------------------------------------------------------------------


def test_build_assistant_msg_plain_content():
    msg = build_assistant_msg(["hello ", "world"], [], {})
    assert msg["role"] == "assistant"
    assert msg["content"] == "hello world"
    assert "_thinking" not in msg


def test_build_assistant_msg_no_content_no_thinking_minimal():
    msg = build_assistant_msg([], [], {})
    assert msg == {"role": "assistant"}


def test_build_assistant_msg_reasoning_parts():
    msg = build_assistant_msg(["answer"], ["step1 ", "step2"], {})
    assert msg["_thinking"] == "step1 step2"
    assert msg["content"] == "answer"


def test_build_assistant_msg_reasoning_without_content():
    msg = build_assistant_msg([], ["thinking..."], {})
    assert msg["_thinking"] == "thinking..."
    assert "content" not in msg


def test_build_assistant_msg_think_tag_extracted():
    msg = build_assistant_msg(["<think>my plan</think>final answer"], [], {})
    assert msg["_thinking"] == "my plan"
    assert msg["content"] == "final answer"


def test_build_assistant_msg_thinking_tag_variant():
    msg = build_assistant_msg(["<thinking>reasoning here</thinking>result"], [], {})
    assert msg["_thinking"] == "reasoning here"
    assert msg["content"] == "result"


def test_build_assistant_msg_no_think_tag_no_thinking_key():
    msg = build_assistant_msg(["plain text, no tags"], [], {})
    assert "_thinking" not in msg


def test_build_assistant_msg_tool_calls_ordered():
    tc_raw = {
        1: {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        0: {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
    }
    msg = build_assistant_msg([], [], tc_raw)
    assert msg["tool_calls"][0]["id"] == "c1"
    assert msg["tool_calls"][1]["id"] == "c2"


def test_build_assistant_msg_tool_calls_no_content():
    tc_raw = {0: {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}}
    msg = build_assistant_msg([], [], tc_raw)
    assert "content" not in msg
    assert len(msg["tool_calls"]) == 1


# LLMCallResult.ok / the llm_call() shim (agllm(cfg).call(...)) and every
# test_llm_call_* / test_agllm_call_returns_llm_call_result test were
# retired here: they all tested agllm.py's own `call()` method -- its
# streaming reassembly, retry-with-backoff on transient/rate-limit/SSL/
# timeout errors, and `LLMCallResult` itself -- whose only production
# caller was execute_react(). The terminus does its own single-attempt
# streaming dispatch (never calls agllm.call()); native's entrypoint
# dispatches via its own _dispatch_via_terminus with its own, differently-
# scoped retry policy (see that function's docstring) -- covered by
# test_native.py's real-Docker test_dispatch_retries_transient_terminus_error_and_recovers.

# ===========================================================================
# Compaction — token estimation, pruning, tail selection, compact, maybe_compact
# ===========================================================================

# agllm._estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_content():
    msg = {"role": "assistant", "content": "a" * 400}  # 400 chars → 100 tokens
    assert agllm._estimate_tokens(msg) == 100


def test_estimate_tokens_tool_args():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"arguments": "x" * 800}}],
    }
    assert agllm._estimate_tokens(msg) == 200


def test_estimate_tokens_minimum_one():
    assert agllm._estimate_tokens({"role": "user", "content": ""}) == 1


# ---------------------------------------------------------------------------
# agllm._prune_tool_outputs
# ---------------------------------------------------------------------------


def _big_tool_msg(chars: int) -> dict:
    return {"role": "tool", "content": "x" * chars, "tool_call_id": "t1"}


def test_prune_does_nothing_when_savings_below_threshold():
    # One tool result slightly over limit but savings < 20K tokens
    msg = _big_tool_msg(_TOOL_OUTPUT_MAX_CHARS + 100)
    msgs = [msg]
    result = agllm._prune_tool_outputs(msgs)
    assert result[0]["content"] == msg["content"]  # unchanged


def test_prune_trims_when_savings_above_threshold():
    # Many large tool results — total savings > 20K tokens
    big = _TOOL_OUTPUT_MAX_CHARS + _PRUNE_MIN_FREE_TOKENS * 4 + 1000
    msgs = [_big_tool_msg(big)]
    result = agllm._prune_tool_outputs(msgs)
    assert len(result[0]["content"]) == _TOOL_OUTPUT_MAX_CHARS + len("\n[truncated]")


def test_prune_leaves_small_tool_results_intact():
    small = {"role": "tool", "content": "small result", "tool_call_id": "t2"}
    # Add enough large results to cross the threshold, but keep one small
    big = _TOOL_OUTPUT_MAX_CHARS + _PRUNE_MIN_FREE_TOKENS * 4 + 1000
    msgs = [_big_tool_msg(big), small]
    result = agllm._prune_tool_outputs(msgs)
    assert result[1]["content"] == "small result"


def test_prune_leaves_non_tool_messages_intact():
    big = _TOOL_OUTPUT_MAX_CHARS + _PRUNE_MIN_FREE_TOKENS * 4 + 1000
    asst = {"role": "assistant", "content": "x" * big}
    msgs = [asst, _big_tool_msg(big)]
    result = agllm._prune_tool_outputs(msgs)
    assert result[0]["content"] == asst["content"]  # assistant untouched


# ---------------------------------------------------------------------------
# agllm._tail_start
# ---------------------------------------------------------------------------


def _conv(*roles: str) -> list[dict]:
    return [{"role": r, "content": f"msg-{i}"} for i, r in enumerate(roles)]


def test_tail_start_empty():
    assert agllm._tail_start([], BIG_CTX) == 0


def test_tail_start_fewer_turns_than_requested():
    # Two assistant turns exist but tail_turns=3 requested — keep all ReAct turns.
    # In compact(), this means head = conv[1:1] = [] → no compaction.
    conv = _conv("user", "assistant", "user", "assistant")
    ts = agllm._tail_start(conv, BIG_CTX, tail_turns=3)
    # tail starts at or before index 1 (the first assistant), meaning nothing left to summarise
    assert ts <= 1


def test_tail_start_exact_one_turn():
    conv = _conv("user", "assistant", "user", "assistant")
    ts = agllm._tail_start(conv, BIG_CTX, tail_turns=1)
    # Keep only the last assistant turn (index 3)
    assert ts == 3
    assert conv[ts]["role"] == "assistant"


def test_tail_start_two_turns():
    conv = _conv("user", "assistant", "user", "assistant", "user", "assistant")
    ts = agllm._tail_start(conv, BIG_CTX, tail_turns=2)
    # Keep last two assistant turns; tail starts at second-to-last assistant (index 3)
    assert ts == 3
    assert conv[ts]["role"] == "assistant"


def test_tail_start_with_tool_messages():
    conv = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{}]},
        {"role": "tool", "content": "r1"},
        {"role": "assistant", "content": None, "tool_calls": [{}]},
        {"role": "tool", "content": "r2"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "done"},
    ]
    ts = agllm._tail_start(conv, BIG_CTX, tail_turns=2)
    # Second-to-last assistant is at index 3
    assert ts == 3
    assert conv[ts]["role"] == "assistant"


def test_tail_start_respects_token_budget():
    # Create a turn whose token estimate exceeds _TAIL_MAX_TOKENS on its own.
    # It should still be kept as the first (and only) turn — the budget cap
    # only applies when a second turn would be added.
    huge_content = "x" * (_TAIL_MAX_TOKENS * 4 * 2)  # >> _TAIL_MAX_TOKENS tokens
    conv = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "small first turn"},
        {"role": "assistant", "content": huge_content},
    ]
    ts = agllm._tail_start(conv, BIG_CTX, tail_turns=2)
    # The huge turn is always kept (single turn always accepted).
    # Whether the small first turn is also kept depends on budget.
    # With budget=8000 and huge turn >> 8000, only the huge turn fits → ts=2.
    assert ts == 2


# ---------------------------------------------------------------------------
# agllm.should_compact
# ---------------------------------------------------------------------------


def test_should_compact_below_threshold():
    threshold = int(BIG_CTX * _COMPACT_THRESHOLD)
    assert not agllm.should_compact(threshold - 1, BIG_CTX)


def test_should_compact_above_threshold():
    threshold = int(BIG_CTX * _COMPACT_THRESHOLD)
    assert agllm.should_compact(threshold + 1, BIG_CTX)


def test_should_compact_at_exact_threshold():
    threshold = int(BIG_CTX * _COMPACT_THRESHOLD)
    assert agllm.should_compact(threshold, BIG_CTX)


def test_should_compact_small_model():
    small_ctx = 10_000
    threshold = int(small_ctx * _COMPACT_THRESHOLD)
    assert not agllm.should_compact(threshold - 1, small_ctx)
    assert agllm.should_compact(threshold, small_ctx)


def test_fetch_context_limit_model_with_slash_in_name():
    """Model names like 'nvidia/foo' must not trigger a 404 via retrieve()."""
    cfg = _cfg(**{**LLM_COMPACT_CONFIG, "model": "nvidia/MiniMax-M2.7-NVFP4"})
    mock_info = MagicMock()
    mock_info.id = "nvidia/MiniMax-M2.7-NVFP4"
    mock_info.model_extra = {"max_model_len": 196000}
    mock_client = MagicMock()
    mock_client.models.list.return_value = [mock_info]

    with patch("agency.llm.agllm.openai.OpenAI", return_value=mock_client):
        result = agllm.fetch_context_limit(cfg)
    assert result == 196000
    mock_client.models.retrieve.assert_not_called()


def test_fetch_context_limit_config_wins_over_vllm():
    cfg = _cfg(**{**LLM_COMPACT_CONFIG, "context_limit": 8192})
    mock_info = MagicMock()
    mock_info.model_extra = {"max_model_len": 131072}
    mock_client = MagicMock()
    mock_client.models.list.return_value = [mock_info]

    with patch("agency.llm.agllm.openai.OpenAI", return_value=mock_client):
        result = agllm.fetch_context_limit(cfg)
    assert result == 8192


# Everything from here to the end of this file was retired: compact()'s
# own tests (return-shorter-list, task-input/tail preservation, incremental
# summary, tool-output pruning, and its retry-on-timeout/connection-error/
# bad-request/rate-limit paths), maybe_compact()'s force-parameter tests,
# and the agskill-integration tests that drove compaction through
# execute_react() (triggers-over-threshold, skips-under-threshold,
# passes-context-limit, context-exceeded forced compaction/no-retry-double-
# count). All of them tested agllm.py's `compact()`/`maybe_compact()`
# methods, whose only production caller was execute_react() -- native's own
# compaction (`_native_in_container_entrypoint.py`'s `_maybe_compact`) uses
# the same underlying algorithm via the still-alive, still-tested
# `agllm_pure.py` (see tests/test_agllm_pure.py and this file's own
# estimate_tokens/prune/tail_start/should_compact tests above, all of which
# are thin delegates to that module and remain fully covered).
