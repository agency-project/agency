"""Tests for agllm — LLM client wrapper, streaming call, kwarg building, and compaction."""

import ssl
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agency.agllm import LLMCallResult, agllm
import json

from agency.agconfig import agConfig
from agency.agdata import agdata as _agdata, agerror as _agerror
from agency.agcontext import agcontext
from agency.agllm import _AgLLMFields
from agency.profiler import agprof as _agprof


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


def llm_call(kwargs, cfg, messages, *args, **kwargs2):
    """Test shim: create a temporary agllm instance and call .call()."""
    return agllm(cfg).call(kwargs, messages, *args, **kwargs2)


LLM_COMPACT_CONFIG = {"api_key": "test", "model": "", "base_url": "http://localhost/v1"}
BIG_CTX = 100_000
LLM_COMPACT = agllm(_cfg(**LLM_COMPACT_CONFIG), context_limit=BIG_CTX)


def _make_mock_agent(llm=None, sandbox=None):
    from agency.agent import agent as _agent_cls

    class _Cls:
        agresource_pool = MagicMock()
        ping_interval_s = 300
        poll_interval_s = 5
        agconfig = None
        _drain_inbox = _agent_cls._drain_inbox
        _check_pause = _agent_cls._check_pause

    ag = _Cls()
    from agency.agent import agent as _agent_cls, agent_state as _agent_state_cls

    ag._state = _agent_state_cls("test")
    ag.llm = llm or LLM_COMPACT
    if sandbox is not None:
        ag.sandbox = sandbox
    else:
        ag.sandbox = MagicMock()
        # A bare MagicMock()'s _has_pending_background_work() would
        # otherwise auto-mock to a truthy value, making agtool.py's
        # dispatch_tools() defer stop() forever -- default to "nothing
        # pending" so tests get the common case without configuring it.
        ag.sandbox._has_pending_background_work.return_value = False
    ag.terminal = MagicMock()
    ag.log = MagicMock()
    ag.log.token_usage = {}
    ag.agname = "test"
    ag._set_ui_state = MagicMock()
    ag._push_live_messages = MagicMock()
    ag._append_full_history = MagicMock()
    ag._next_inbox_msg = MagicMock(return_value=None)
    ag.push_token_count_update_to_ui = MagicMock()
    return ag


# ---------------------------------------------------------------------------
# Streaming chunk helpers
# ---------------------------------------------------------------------------


class _Delta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.model_extra = {"reasoning_content": reasoning_content} if reasoning_content else {}
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt=5, completion=3):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None, reasoning=None):
        delta = _Delta(content=content, tool_calls=tool_calls, reasoning_content=reasoning)
        self.choices = [_Choice(delta)] if (content is not None or tool_calls or reasoning) else []
        self.usage = usage


class _TCDelta:
    def __init__(self, name, args_json, call_id, index=0):
        self.id = call_id
        self.index = index
        self.function = _TCFn(name, args_json)


class _TCFn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


def _text_chunks(text: str) -> list:
    """Simulate a streaming text response."""
    return [_Chunk(content=text), _Chunk(usage=_Usage())]


def _tool_chunks(name: str, args: str, call_id: str = "c1") -> list:
    import json

    tc = _TCDelta(name, json.dumps(args), call_id)
    return [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]


def _run_call(chunks, llm_config=None):
    """Run llm_call with mocked OpenAI and return the result."""
    cfg = llm_config or _cfg(base_url="http://x", api_key="k", model="m")
    msgs: list[dict] = [{"role": "user", "content": "hi"}]
    with patch("agency.agllm.openai.OpenAI") as MockCls:
        MockCls.return_value.chat.completions.create.return_value = iter(chunks)
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "test_skill",
        )
    return result


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


# ---------------------------------------------------------------------------
# LLMCallResult.ok
# ---------------------------------------------------------------------------


def test_llm_call_result_ok_when_clean():
    r = LLMCallResult()
    assert r.ok is True


def test_llm_call_result_not_ok_when_conn_error():
    r = LLMCallResult(conn_error=RuntimeError("fail"))
    assert r.ok is False


def test_llm_call_result_not_ok_when_context_exceeded():
    r = LLMCallResult(context_exceeded=True)
    assert r.ok is False


# ---------------------------------------------------------------------------
# fetch_context_limit
# ---------------------------------------------------------------------------


def test_fetch_context_limit_uses_config_key():
    limit = fetch_context_limit(_cfg(context_limit=65536))
    assert limit == 65536


def test_fetch_context_limit_int_coercion():
    limit = fetch_context_limit(_cfg(context_limit="32000"))
    assert limit == 32000


def test_fetch_context_limit_falls_back_to_default(capsys):
    with patch("agency.agllm.openai.OpenAI") as MockCls:
        MockCls.return_value.models.list.side_effect = RuntimeError("offline")
        limit = fetch_context_limit(_cfg(model="m"))
    assert limit == _AgLLMFields.default_context_limit.default


def test_fetch_context_limit_reads_max_model_len():
    mock_model = MagicMock()
    mock_model.id = "my-model"
    mock_model.model_extra = {"max_model_len": 200_000}
    with patch("agency.agllm.openai.OpenAI") as MockCls:
        MockCls.return_value.models.list.return_value = [mock_model]
        limit = fetch_context_limit(_cfg(model="my-model"))
    assert limit == 200_000


def test_fetch_context_limit_reads_max_input_tokens_for_anthropic_provider():
    """The first-party Anthropic API exposes max_input_tokens as a typed
    field (not a model_extra vLLM-style extension) — see
    agllm_backends/anthropic.py's _AnthropicBackend.list_models()."""
    mock_model = MagicMock()
    mock_model.id = "claude-sonnet-5"
    mock_model.model_extra = {}
    mock_model.max_input_tokens = 1_000_000
    mock_sdk = MagicMock()
    mock_sdk.Anthropic.return_value.models.list.return_value = [mock_model]
    with patch("agency.agllm_backends.anthropic._anthropic_sdk", mock_sdk):
        limit = fetch_context_limit(_cfg(provider="anthropic", model="claude-sonnet-5"))
    assert limit == 1_000_000


# ---------------------------------------------------------------------------
# agllm class
# ---------------------------------------------------------------------------


def test_agllm_explicit_context_limit_skips_fetch():
    with patch.object(agllm, "fetch_context_limit") as mock_fetch:
        llm = agllm(_cfg(model="m"), context_limit=128_000)
    mock_fetch.assert_not_called()
    assert llm.context_limit == 128_000


def test_agllm_no_context_limit_calls_fetch():
    with patch.object(agllm, "fetch_context_limit", return_value=32_000) as mock_fetch:
        llm = agllm(_cfg(model="m"))
    mock_fetch.assert_called_once()
    assert llm.context_limit == 32_000


def test_agllm_config_stored():
    cfg = _cfg(model="", temperature=0.5)
    llm = agllm(cfg, context_limit=128_000)
    assert llm.backend.model == ""
    assert llm.backend.temperature == 0.5


def test_agllm_build_kwargs_delegates():
    llm = agllm(_cfg(model="m"), context_limit=128_000)
    msgs = [{"role": "user", "content": "hi"}]
    kw = llm.build_kwargs(msgs)
    assert kw["model"] == "m"
    assert kw["messages"] == msgs


def test_agllm_build_kwargs_with_tools():
    llm = agllm(_cfg(model="m"), context_limit=128_000)
    tools = [{"type": "function", "function": {"name": "f"}}]
    kw = llm.build_kwargs([], tools)
    assert kw["tools"] == tools


# ---------------------------------------------------------------------------
# llm_call — success path
# ---------------------------------------------------------------------------


def test_llm_call_returns_content_parts():
    result = _run_call(_text_chunks("hello world"))
    assert result.ok
    assert "".join(result.content_parts) == "hello world"


def test_llm_call_empty_content_ok():
    result = _run_call(_text_chunks(""))
    assert result.ok


def test_llm_call_accumulates_token_counts():
    chunks = [_Chunk(content="hi"), _Chunk(usage=_Usage(prompt=10, completion=5))]
    result = _run_call(chunks)
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_input_tokens == 10
    assert result.total_output_tokens == 5
    assert result.ttft_ms is not None
    assert result.ttft_ms >= 0
    assert result.generation_ms is not None
    assert result.generation_ms >= 0


def test_llm_attempt_profiler_metadata_includes_tokens_ttft_and_outcome(monkeypatch):
    class FakeRecordFunction:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(_agprof, "_records", [])
    monkeypatch.setattr(_agprof, "_open_spans", {})
    monkeypatch.setattr(_agprof, "_session", _agprof._TorchSession(FakeRecordFunction))

    chunks = [_Chunk(content="hi"), _Chunk(usage=_Usage(prompt=10, completion=5))]
    _run_call(chunks)

    attempt = next(record for record in _agprof._records if record[1] == "llm:attempt[0]")
    metadata = attempt[6]
    assert metadata["outcome"] == "success"
    assert metadata["input_tokens"] == 10
    assert metadata["output_tokens"] == 5
    assert metadata["ttft_ms"] is not None
    assert metadata["generation_ms"] is not None


def test_llm_call_elapsed_ms_set():
    result = _run_call(_text_chunks("ok"))
    assert isinstance(result.elapsed_ms, int)
    assert result.elapsed_ms >= 0


def test_llm_call_prompt_tokens_from_usage_chunk():
    chunks = [_Chunk(content="hi"), _Chunk(usage=_Usage(prompt=42))]
    result = _run_call(chunks)
    assert result.prompt_tokens == 42


def test_llm_call_tool_call_accumulated():
    import json

    tc = _TCDelta("my_tool", json.dumps({"x": 1}), "call_abc")
    chunks = [_Chunk(tool_calls=[tc]), _Chunk(usage=_Usage())]
    result = _run_call(chunks)
    assert result.ok
    assert 0 in result.tool_calls_raw
    assert result.tool_calls_raw[0]["function"]["name"] == "my_tool"
    assert result.tool_calls_raw[0]["id"] == "call_abc"


def test_llm_call_tool_call_arguments_concatenated():
    tc1 = _TCDelta("f", '{"a":', "c1")
    tc2 = _TCDelta("", '"val"}', "")
    tc2.function.name = ""
    chunks = [_Chunk(tool_calls=[tc1]), _Chunk(tool_calls=[tc2]), _Chunk(usage=_Usage())]
    result = _run_call(chunks)
    assert result.tool_calls_raw[0]["function"]["arguments"] == '{"a":"val"}'


def test_llm_call_removes_partial_message_on_success():
    msgs: list[dict] = [{"role": "user", "content": "hi"}]
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    with patch("agency.agllm.openai.OpenAI") as MockCls:
        MockCls.return_value.chat.completions.create.return_value = iter(_text_chunks("ok"))
        llm_call(build_llm_kwargs(cfg, msgs, None), cfg, msgs, None, None, None, None, 0, 0, "sk")
    # partial placeholder must be removed; original user msg stays
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_llm_call_reasoning_parts_accumulated():
    chunks = [
        _Chunk(reasoning="step 1 "),
        _Chunk(reasoning="step 2"),
        _Chunk(usage=_Usage()),
    ]
    result = _run_call(chunks)
    assert result.ok
    assert "".join(result.reasoning_parts) == "step 1 step 2"


def test_llm_call_token_update_fn_called():
    called_with = []

    def _update(inp, out):
        called_with.append((inp, out))

    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    with patch("agency.agllm.openai.OpenAI") as MockCls:
        MockCls.return_value.chat.completions.create.return_value = iter(
            [_Chunk(content="hi"), _Chunk(usage=_Usage(prompt=7, completion=3))]
        )
        llm_call(
            build_llm_kwargs(cfg, msgs, None), cfg, msgs, None, None, None, _update, 0, 0, "sk"
        )
    assert len(called_with) == 1
    inp, out = called_with[0]
    assert inp == 7
    assert out == 3


# ---------------------------------------------------------------------------
# llm_call — error paths
# ---------------------------------------------------------------------------


def test_llm_call_context_exceeded_on_bad_request():
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.BadRequestError(
        message="context_length_exceeded: too long",
        response=MagicMock(status_code=400),
        body=None,
    )
    with patch("agency.agllm.openai.OpenAI") as MockCls:
        MockCls.return_value.chat.completions.create.side_effect = err
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
        )
    assert result.context_exceeded
    assert not result.ok


def test_llm_call_bad_request_not_context_gives_conn_error():
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.BadRequestError(
        message="invalid_request_error",
        response=MagicMock(status_code=400),
        body=None,
    )
    with patch("agency.agllm.openai.OpenAI") as MockCls:
        MockCls.return_value.chat.completions.create.side_effect = err
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
        )
    assert not result.context_exceeded
    assert result.conn_error is err


def test_llm_call_transient_error_retries_and_exhausts():
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = ssl.SSLError("handshake failed")
    with patch("agency.agllm.openai.OpenAI") as MockCls, patch("agency.agllm.time.sleep"):
        MockCls.return_value.chat.completions.create.side_effect = err
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
        )
    assert not result.ok
    assert result.conn_error is err


def test_llm_call_transient_error_notifies_full_history_fn():
    from agency.agllm import _AgLLMFields

    LLM_MAX_RETRIES = _AgLLMFields.max_retries.default
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = OSError("connection reset")
    events: list = []
    with patch("agency.agllm.openai.OpenAI") as MockCls, patch("agency.agllm.time.sleep"):
        MockCls.return_value.chat.completions.create.side_effect = err
        llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
            events.append,
        )
    retry_events = [e for e in events if e.get("type") == "llm_retry"]
    assert len(retry_events) == LLM_MAX_RETRIES - 1


def test_llm_call_bare_api_error_retries_and_exhausts():
    """A bare openai.APIError (e.g. a mid-stream server error frame with no
    HTTP status to build a more specific subclass from) must retry like any
    other transient error, not propagate uncaught and crash the skill."""
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.APIError(
        "The server had an error while processing your request. Sorry about that!",
        httpx.Request("POST", "http://x"),
        body=None,
    )
    with patch("agency.agllm.openai.OpenAI") as MockCls, patch("agency.agllm.time.sleep"):
        MockCls.return_value.chat.completions.create.side_effect = err
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
        )
    assert not result.ok
    assert result.conn_error is err


def test_llm_call_bare_api_error_notifies_full_history_fn():
    from agency.agllm import _AgLLMFields

    LLM_MAX_RETRIES = _AgLLMFields.max_retries.default
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.APIError("server error", httpx.Request("POST", "http://x"), body=None)
    events: list = []
    with patch("agency.agllm.openai.OpenAI") as MockCls, patch("agency.agllm.time.sleep"):
        MockCls.return_value.chat.completions.create.side_effect = err
        llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
            events.append,
        )
    retry_events = [e for e in events if e.get("type") == "llm_retry"]
    assert len(retry_events) == LLM_MAX_RETRIES - 1


def test_llm_call_bad_request_still_immediate_despite_being_an_api_error():
    """BadRequestError is itself an openai.APIError subclass — the broadened
    retry-on-APIError clause must not shadow the more specific BadRequestError
    handling (which returns immediately, no retry) since it's checked first."""
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.BadRequestError(
        message="invalid_request_error",
        response=MagicMock(status_code=400),
        body=None,
    )
    with (
        patch("agency.agllm.openai.OpenAI") as MockCls,
        patch("agency.agllm.time.sleep") as mock_sleep,
    ):
        MockCls.return_value.chat.completions.create.side_effect = err
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
        )
    assert result.conn_error is err
    mock_sleep.assert_not_called()


def test_llm_call_rate_limit_honors_retry_after_header():
    """A 429 must retry (not crash the skill) and never sleep for less than
    the server-provided Retry-After duration when present — jitter is added
    on top, never subtracted, so this asserts a floor rather than equality."""
    from agency.agllm import _AgLLMFields

    LLM_RATE_LIMIT_RETRY_AFTER_JITTER_S = _AgLLMFields.rate_limit_retry_after_jitter_s.default
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.RateLimitError(
        message="rate_limit_error",
        response=MagicMock(status_code=429, headers={"retry-after": "3"}),
        body=None,
    )
    with (
        patch("agency.agllm.openai.OpenAI") as MockCls,
        patch("agency.agllm.time.sleep") as mock_sleep,
    ):
        MockCls.return_value.chat.completions.create.side_effect = err
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
        )
    assert not result.ok
    assert result.conn_error is err
    for retry_call in mock_sleep.call_args_list:
        sleep_s = retry_call.args[0]
        assert 3.0 <= sleep_s <= 3.0 + LLM_RATE_LIMIT_RETRY_AFTER_JITTER_S


def test_llm_call_rate_limit_retry_after_jitter_decorrelates_calls():
    """Two agents that receive the identical Retry-After value must not be
    guaranteed to sleep for the identical duration — otherwise concurrently
    throttled agents sharing one org-wide window would all wake up and
    retry in the same instant."""
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.RateLimitError(
        message="rate_limit_error",
        response=MagicMock(status_code=429, headers={"retry-after": "3"}),
        body=None,
    )
    sleeps: list[float] = []
    with (
        patch("agency.agllm.openai.OpenAI") as MockCls,
        patch("agency.agllm.time.sleep", side_effect=lambda s: sleeps.append(s)),
    ):
        MockCls.return_value.chat.completions.create.side_effect = err
        llm_call(build_llm_kwargs(cfg, msgs, None), cfg, msgs, None, None, None, None, 0, 0, "sk")
    # Across the many retries within a single run, jitter must vary the sleep
    # duration rather than collapsing to the bare Retry-After value every time.
    assert len(set(sleeps)) > 1


def test_llm_call_rate_limit_falls_back_to_exponential_backoff_without_header():
    """Missing/unparseable Retry-After must not crash — fall back to bounded,
    jittered exponential backoff instead of raising a TypeError/ValueError."""
    from agency.agllm import _AgLLMFields

    LLM_RATE_LIMIT_MAX_BACKOFF_S = _AgLLMFields.rate_limit_max_backoff_s.default
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.RateLimitError(
        message="rate_limit_error",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    with (
        patch("agency.agllm.openai.OpenAI") as MockCls,
        patch("agency.agllm.time.sleep") as mock_sleep,
    ):
        MockCls.return_value.chat.completions.create.side_effect = err
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
        )
    assert not result.ok
    assert result.conn_error is err
    for retry_call in mock_sleep.call_args_list:
        sleep_s = retry_call.args[0]
        assert 0 <= sleep_s <= LLM_RATE_LIMIT_MAX_BACKOFF_S


def test_llm_call_rate_limit_exhausts_retries_without_raising():
    """The 429 must never propagate uncaught — this was the original bug
    (agskill.py crashing on anthropic.RateLimitError)."""
    from agency.agllm import _AgLLMFields

    LLM_MAX_RETRIES = _AgLLMFields.max_retries.default
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = openai.RateLimitError(
        message="rate_limit_error",
        response=MagicMock(status_code=429, headers={"retry-after": "0"}),
        body=None,
    )
    with (
        patch("agency.agllm.openai.OpenAI") as MockCls,
        patch("agency.agllm.time.sleep") as mock_sleep,
    ):
        MockCls.return_value.chat.completions.create.side_effect = err
        result = llm_call(
            build_llm_kwargs(cfg, msgs, None),
            cfg,
            msgs,
            None,
            None,
            None,
            None,
            0,
            0,
            "sk",
        )
    assert not result.ok
    assert result.conn_error is err
    assert mock_sleep.call_count == LLM_MAX_RETRIES - 1


def test_llm_call_partial_placeholder_removed_on_error():
    cfg = _cfg(base_url="http://x", api_key="k", model="m")
    msgs = [{"role": "user", "content": "hi"}]
    err = ssl.SSLError("fail")
    with patch("agency.agllm.openai.OpenAI") as MockCls, patch("agency.agllm.time.sleep"):
        MockCls.return_value.chat.completions.create.side_effect = err
        llm_call(build_llm_kwargs(cfg, msgs, None), cfg, msgs, None, None, None, None, 0, 0, "sk")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


# ---------------------------------------------------------------------------
# agllm.call — instance method
# ---------------------------------------------------------------------------


def test_agllm_call_returns_llm_call_result():
    llm = agllm(_cfg(base_url="http://x", api_key="k", model="m"), context_limit=128_000)
    msgs = [{"role": "user", "content": "hi"}]
    kw = llm.build_kwargs(msgs)
    with patch("agency.agllm.openai.OpenAI") as MockCls:
        MockCls.return_value.chat.completions.create.return_value = iter(_text_chunks("pong"))
        result = llm.call(kw, msgs, None, None, None, None, 0, 0, "sk")
    assert isinstance(result, LLMCallResult)
    assert result.ok
    assert "pong" in "".join(result.content_parts)


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

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        result = agllm.fetch_context_limit(cfg)
    assert result == 196000
    mock_client.models.retrieve.assert_not_called()


def test_fetch_context_limit_config_wins_over_vllm():
    cfg = _cfg(**{**LLM_COMPACT_CONFIG, "context_limit": 8192})
    mock_info = MagicMock()
    mock_info.model_extra = {"max_model_len": 131072}
    mock_client = MagicMock()
    mock_client.models.list.return_value = [mock_info]

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        result = agllm.fetch_context_limit(cfg)
    assert result == 8192


# ---------------------------------------------------------------------------
# compact()
# ---------------------------------------------------------------------------


def _mock_compact_response(summary_text: str):
    """compact() now delegates to call(), which always streams -- so a
    successful summarisation response must look like a chunk stream (see
    _text_chunks), not a flat non-streaming completion object."""
    return _text_chunks(summary_text)


def _make_messages(n_turns: int, *, with_tools: bool = False) -> list[dict]:
    """Build: system + user(task) + N assistant turns (optionally with tool calls)."""
    msgs: list[dict] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": '{"task": "do the thing"}'},
    ]
    for i in range(n_turns):
        if with_tools:
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {"name": "bash", "arguments": f'{{"command":"cmd{i}"}}'},
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "content": f"result {i}", "tool_call_id": f"c{i}"})
        else:
            msgs.append({"role": "assistant", "content": f"assistant reply {i}"})
    return msgs


def test_compact_returns_shorter_list():
    messages = _make_messages(6)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("## Goal\nTest.")

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        new_msgs, summary = LLM_COMPACT.compact(messages)

    assert summary == "## Goal\nTest."
    assert len(new_msgs) < len(messages)
    assert new_msgs[0]["role"] == "system"
    assert any("summary" in m.get("content", "").lower() for m in new_msgs)


def test_compact_preserves_task_input():
    """conv[0] (the skill task input) must always appear verbatim after the system msg."""
    messages = _make_messages(6)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("summary")

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        new_msgs, _ = LLM_COMPACT.compact(messages)

    # system at [0], task user input at [1]
    assert new_msgs[0]["role"] == "system"
    assert new_msgs[1]["role"] == "user"
    assert new_msgs[1]["content"] == '{"task": "do the thing"}'


def test_compact_preserves_tail_turns():
    messages = _make_messages(6)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("summary")

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        new_msgs, _ = LLM_COMPACT.compact(messages, tail_turns=2)

    assistant_bodies = [
        m["content"]
        for m in new_msgs
        if m["role"] == "assistant" and "Understood" not in (m.get("content") or "")
    ]
    assert "assistant reply 4" in assistant_bodies
    assert "assistant reply 5" in assistant_bodies


def test_compact_nothing_to_summarise():
    # Only 1 assistant turn → tail_turns=2 means nothing in head → no-op
    messages = _make_messages(1)
    mock_client = MagicMock()

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        new_msgs, summary = LLM_COMPACT.compact(messages, tail_turns=2)

    assert new_msgs == messages
    assert summary == ""
    mock_client.chat.completions.create.assert_not_called()


def test_compact_incremental_with_previous_summary():
    messages = _make_messages(4)
    prev = "## Goal\nPrevious task."
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("updated summary")

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        _, summary = LLM_COMPACT.compact(messages, previous_summary=prev)

    assert summary == "updated summary"
    user_content = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert prev in user_content


def test_compact_task_input_included_in_summarisation_prompt():
    """The task input message should appear in the summarisation prompt so the
    LLM knows what goal the compacted turns were working toward."""
    messages = _make_messages(4)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("s")

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        LLM_COMPACT.compact(messages)

    user_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "do the thing" in user_prompt  # task content appears in prompt


def test_compact_prunes_large_tool_outputs():
    """When head contains large tool results that hit the prune threshold,
    they should be trimmed before being sent to the summariser."""
    # Build messages with a giant tool result in the head
    huge = _TOOL_OUTPUT_MAX_CHARS + _PRUNE_MIN_FREE_TOKENS * 4 + 100
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": '{"task": "x"}'},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "y" * huge, "tool_call_id": "c0"},
        # Two more turns kept as tail
        {"role": "assistant", "content": "turn 1"},
        {"role": "assistant", "content": "turn 2"},
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_compact_response("s")

    with patch("agency.agllm.openai.OpenAI", return_value=mock_client):
        LLM_COMPACT.compact(messages, tail_turns=2)

    user_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    # Pruned content appears truncated in the prompt
    assert "[truncated]" in user_prompt or len(user_prompt) < huge // 2


# ---------------------------------------------------------------------------
# compact() retries transient LLM errors
#
# compact()'s summarisation request used to be a single bare call with no
# retry at all — a lone read-timeout against the backend would propagate all
# the way up through maybe_compact() -> execute_react() and crash the whole
# agent, discarding any already-completed work higher in the call stack (a
# harness run whose real metrics_output was already finished, in the incident
# that prompted this). These mirror the existing call()-retry tests above,
# adapted to compact()'s contract: it has no LLMCallResult-style return value,
# so exhausted retries raise the underlying error rather than being swallowed.
# ---------------------------------------------------------------------------


def test_compact_retries_api_timeout_then_succeeds():
    """The exact failure mode from the incident: openai.APITimeoutError on the
    summarisation call must be retried, not propagated on the first attempt."""
    messages = _make_messages(6)
    mock_client = MagicMock()
    timeout_err = openai.APITimeoutError(request=httpx.Request("POST", "http://x"))
    mock_client.chat.completions.create.side_effect = [
        timeout_err,
        timeout_err,
        _mock_compact_response("summary after retries"),
    ]

    with (
        patch("agency.agllm.openai.OpenAI", return_value=mock_client),
        patch("agency.agllm.time.sleep"),
    ):
        _, summary = LLM_COMPACT.compact(messages)

    assert summary == "summary after retries"
    assert mock_client.chat.completions.create.call_count == 3


def test_compact_retries_connection_error_then_succeeds():
    messages = _make_messages(6)
    mock_client = MagicMock()
    conn_err = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    mock_client.chat.completions.create.side_effect = [conn_err, _mock_compact_response("ok")]

    with (
        patch("agency.agllm.openai.OpenAI", return_value=mock_client),
        patch("agency.agllm.time.sleep"),
    ):
        _, summary = LLM_COMPACT.compact(messages)

    assert summary == "ok"
    assert mock_client.chat.completions.create.call_count == 2


def test_compact_bare_api_error_retries_then_succeeds():
    """A bare openai.APIError (no HTTP status to build a more specific
    subclass from, e.g. a mid-stream server error frame) must retry like any
    other transient error rather than propagate uncaught."""
    messages = _make_messages(6)
    mock_client = MagicMock()
    err = openai.APIError("server error", httpx.Request("POST", "http://x"), body=None)
    mock_client.chat.completions.create.side_effect = [err, _mock_compact_response("ok")]

    with (
        patch("agency.agllm.openai.OpenAI", return_value=mock_client),
        patch("agency.agllm.time.sleep"),
    ):
        _, summary = LLM_COMPACT.compact(messages)

    assert summary == "ok"
    assert mock_client.chat.completions.create.call_count == 2


def test_compact_exhausts_retries_and_raises_last_error():
    """Unlike call() (which swallows exhaustion into LLMCallResult.conn_error),
    compact() has no result-object convention — after max_retries identical
    failures it must raise the underlying error rather than looping forever
    or returning a summary built from no successful response."""
    from agency.agllm import _AgLLMFields

    LLM_MAX_RETRIES = _AgLLMFields.max_retries.default
    messages = _make_messages(6)
    mock_client = MagicMock()
    timeout_err = openai.APITimeoutError(request=httpx.Request("POST", "http://x"))
    mock_client.chat.completions.create.side_effect = timeout_err

    with (
        patch("agency.agllm.openai.OpenAI", return_value=mock_client),
        patch("agency.agllm.time.sleep"),
    ):
        with pytest.raises(openai.APITimeoutError):
            LLM_COMPACT.compact(messages)

    assert mock_client.chat.completions.create.call_count == LLM_MAX_RETRIES


def test_compact_bad_request_error_not_retried():
    """BadRequestError means the request itself is malformed — retrying an
    identical malformed request can never succeed, so this must fail fast
    with zero retries/sleeps, mirroring call()'s handling. BadRequestError is
    itself an openai.APIError subclass, so this also guards against the
    broadened retry-on-APIError clause shadowing the more specific handling."""
    messages = _make_messages(6)
    mock_client = MagicMock()
    err = openai.BadRequestError(
        message="invalid_request_error",
        response=MagicMock(status_code=400),
        body=None,
    )
    mock_client.chat.completions.create.side_effect = err

    with (
        patch("agency.agllm.openai.OpenAI", return_value=mock_client),
        patch("agency.agllm.time.sleep") as mock_sleep,
    ):
        with pytest.raises(openai.BadRequestError):
            LLM_COMPACT.compact(messages)

    assert mock_client.chat.completions.create.call_count == 1
    mock_sleep.assert_not_called()


def test_compact_rate_limit_honors_retry_after_header():
    """A 429 during compaction must retry (not crash the agent) and never
    sleep for less than the server-provided Retry-After duration — jitter is
    added on top, never subtracted, so this asserts a floor, not equality."""
    from agency.agllm import _AgLLMFields

    LLM_RATE_LIMIT_RETRY_AFTER_JITTER_S = _AgLLMFields.rate_limit_retry_after_jitter_s.default
    messages = _make_messages(6)
    mock_client = MagicMock()
    err = openai.RateLimitError(
        message="rate_limit_error",
        response=MagicMock(status_code=429, headers={"retry-after": "3"}),
        body=None,
    )
    mock_client.chat.completions.create.side_effect = [err, _mock_compact_response("ok")]

    with (
        patch("agency.agllm.openai.OpenAI", return_value=mock_client),
        patch("agency.agllm.time.sleep") as mock_sleep,
    ):
        _, summary = LLM_COMPACT.compact(messages)

    assert summary == "ok"
    # The retry backoff is the first sleep call; a successful streamed response
    # also sleeps once per stream-batch drain interval (agutil._iter_batched),
    # which is unrelated to retrying and is patched to 0s in tests (conftest.py).
    sleep_s = mock_sleep.call_args_list[0].args[0]
    assert 3.0 <= sleep_s <= 3.0 + LLM_RATE_LIMIT_RETRY_AFTER_JITTER_S


def test_compact_rate_limit_falls_back_to_exponential_backoff_without_header():
    """Missing/unparseable Retry-After must not crash with a TypeError/ValueError
    — fall back to bounded, jittered exponential backoff instead."""
    from agency.agllm import _AgLLMFields

    LLM_RATE_LIMIT_MAX_BACKOFF_S = _AgLLMFields.rate_limit_max_backoff_s.default
    messages = _make_messages(6)
    mock_client = MagicMock()
    err = openai.RateLimitError(
        message="rate_limit_error",
        response=MagicMock(status_code=429, headers={}),
        body=None,
    )
    mock_client.chat.completions.create.side_effect = [err, _mock_compact_response("ok")]

    with (
        patch("agency.agllm.openai.OpenAI", return_value=mock_client),
        patch("agency.agllm.time.sleep") as mock_sleep,
    ):
        _, summary = LLM_COMPACT.compact(messages)

    assert summary == "ok"
    for retry_call in mock_sleep.call_args_list:
        sleep_s = retry_call.args[0]
        assert 0 <= sleep_s <= LLM_RATE_LIMIT_MAX_BACKOFF_S


# ---------------------------------------------------------------------------
# Integration: agskill triggers compaction when threshold exceeded
# ---------------------------------------------------------------------------


def _make_stream(content: str, prompt_tokens: int) -> list:
    """Build a minimal list of streaming chunks that agskill.run() can iterate."""
    # Chunk 1: content delta
    delta1 = MagicMock()
    delta1.content = content
    delta1.tool_calls = None
    choice1 = MagicMock()
    choice1.delta = delta1
    choice1.finish_reason = None
    chunk1 = MagicMock()
    chunk1.choices = [choice1]
    chunk1.usage = None

    # Chunk 2: final chunk carrying usage, no content
    delta2 = MagicMock()
    delta2.content = None
    delta2.tool_calls = None
    choice2 = MagicMock()
    choice2.delta = delta2
    choice2.finish_reason = "stop"
    chunk2 = MagicMock()
    chunk2.choices = [choice2]
    chunk2.usage = MagicMock(prompt_tokens=prompt_tokens)

    return [chunk1, chunk2]


def test_agskill_triggers_compaction_when_over_threshold():
    from agency.agskill import agskill

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    limit = BIG_CTX
    over = int(limit * _COMPACT_THRESHOLD) + 1

    compact_calls = []

    def fake_compact(messages, **kw):
        compact_calls.append(len(messages))
        return messages, "summary"

    with (
        patch("agency.agllm.openai.OpenAI") as MockClient,
        patch.object(agllm, "compact", side_effect=fake_compact),
    ):
        MockClient.return_value = MagicMock()
        MockClient.return_value.chat.completions.create.return_value = _make_stream(
            '{"result": "done"}', over
        )
        skill.execute_react(
            _make_mock_agent(agllm(_cfg(**LLM_COMPACT_CONFIG), context_limit=limit)),
            agcontext(),
            _agdata(task="x"),
        )

    assert len(compact_calls) == 1


def test_agskill_skips_compaction_when_under_threshold():
    from agency.agskill import agskill

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    limit = BIG_CTX
    under = int(limit * _COMPACT_THRESHOLD) - 1

    compact_calls = []

    def fake_compact(messages, **kw):
        compact_calls.append(True)
        return messages, ""

    with (
        patch("agency.agllm.openai.OpenAI") as MockClient,
        patch.object(agllm, "compact", side_effect=fake_compact),
    ):
        MockClient.return_value = MagicMock()
        MockClient.return_value.chat.completions.create.return_value = _make_stream(
            '{"result": "ok"}', under
        )
        skill.execute_react(
            _make_mock_agent(agllm(_cfg(**LLM_COMPACT_CONFIG), context_limit=limit)),
            agcontext(),
            _agdata(task="x"),
        )

    assert len(compact_calls) == 0


def test_agskill_passes_context_limit_to_compact():
    """compact() must receive the context_limit kwarg so tail sizing is correct."""
    from agency.agskill import agskill

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])
    limit = BIG_CTX
    over = int(limit * _COMPACT_THRESHOLD) + 1

    received_kwargs = {}

    def fake_compact(messages, **kw):
        received_kwargs.update(kw)
        return messages, "summary"

    with (
        patch("agency.agllm.openai.OpenAI") as MockClient,
        patch.object(agllm, "compact", side_effect=fake_compact),
    ):
        MockClient.return_value = MagicMock()
        MockClient.return_value.chat.completions.create.return_value = _make_stream(
            '{"result": "done"}', over
        )
        skill.execute_react(
            _make_mock_agent(agllm(_cfg(**LLM_COMPACT_CONFIG), context_limit=limit)),
            agcontext(),
            _agdata(task="x"),
        )

    assert received_kwargs.get("context_limit") == limit


# ---------------------------------------------------------------------------
# Helpers for tool-call streaming mocks
# ---------------------------------------------------------------------------


class _TC:
    def __init__(self, name: str, args_json: str, call_id: str = "c1") -> None:
        self.id = call_id
        self.index = 0
        self.function = _TCFn(name, args_json)


def _make_tool_call_stream(name: str, args: dict, call_id: str = "c1") -> list:
    """Minimal streaming chunks representing a tool-call LLM response."""
    tc = _TC(name, json.dumps(args), call_id)

    delta1 = MagicMock()
    delta1.content = None
    delta1.tool_calls = [tc]
    delta1.model_extra = {}
    delta1.reasoning_content = None
    choice1 = MagicMock()
    choice1.delta = delta1
    chunk1 = MagicMock()
    chunk1.choices = [choice1]
    chunk1.usage = None

    delta2 = MagicMock()
    delta2.content = None
    delta2.tool_calls = None
    delta2.model_extra = {}
    delta2.reasoning_content = None
    choice2 = MagicMock()
    choice2.delta = delta2
    chunk2 = MagicMock()
    chunk2.choices = [choice2]
    chunk2.usage = MagicMock(prompt_tokens=500)

    return [chunk1, chunk2]


# ---------------------------------------------------------------------------
# maybe_compact force parameter
# ---------------------------------------------------------------------------


def test_maybe_compact_force_bypasses_threshold():
    """force=True must trigger compaction even when token estimate is below threshold."""

    # Tiny messages — well below threshold
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "reply"},
    ]
    compact_calls = []

    def fake_compact(messages, **kw):
        compact_calls.append(True)
        return messages, "summary"

    with patch.object(agllm, "compact", side_effect=fake_compact):
        LLM_COMPACT.maybe_compact(agcontext(), messages, None, skill_name="test", force=True)

    assert len(compact_calls) == 1


def test_maybe_compact_no_force_below_threshold_does_nothing():
    """Without force, maybe_compact must not fire when estimate is below threshold."""

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "reply"},
    ]
    compact_calls = []

    def fake_compact(messages, **kw):
        compact_calls.append(True)
        return messages, "summary"

    with patch.object(agllm, "compact", side_effect=fake_compact):
        LLM_COMPACT.maybe_compact(agcontext(), messages, None, skill_name="test", force=False)

    assert len(compact_calls) == 0


# ---------------------------------------------------------------------------
# Reactive compaction: context-exceeded 400 error
# ---------------------------------------------------------------------------


def test_agskill_context_exceeded_triggers_forced_compaction():
    """A context_exceeded result must call compact() (bypassing the threshold) and retry the LLM.

    force=True is an maybe_compact parameter, not forwarded to compact() itself.
    We verify forced compaction fired by checking that compact() was called even
    though the message estimate is far below the threshold (proactive path would
    not fire on tiny messages).
    """
    from agency.agskill import agskill
    from agency.agllm import LLMCallResult

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    compact_calls: list[int] = []

    def fake_compact(messages, **kw):
        compact_calls.append(len(messages))
        return messages, "summary"

    call_count = 0

    def fake_llm_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMCallResult(context_exceeded=True)
        result = LLMCallResult()
        result.content_parts = ['{"result": "done"}']
        result.prompt_tokens = 100
        result.total_input_tokens = 100
        result.total_output_tokens = 10
        result.elapsed_ms = 50
        return result

    with (
        patch.object(agllm, "compact", side_effect=fake_compact),
        patch.object(agllm, "call", side_effect=fake_llm_call),
    ):
        skill.execute_react(_make_mock_agent(LLM_COMPACT), agcontext(), _agdata(task="x"))

    # compact() must have been called — proactive path won't fire on tiny messages,
    # so any call means the reactive (force=True) path triggered it.
    assert len(compact_calls) >= 1, (
        "compact() must be called after context_exceeded (forced, bypassing threshold)"
    )
    assert call_count == 2, "LLM must be retried after forced compaction"


def test_agskill_context_exceeded_does_not_count_as_retry():
    """context_exceeded compaction must not consume a connection-retry slot."""
    from agency.agskill import agskill
    from agency.agllm import LLMCallResult

    skill = agskill(name="test", system_prompt="You are helpful.", replace_tools=[])

    call_count = 0

    def fake_llm_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMCallResult(context_exceeded=True)
        result = LLMCallResult()
        result.content_parts = ['{"result": "done"}']
        result.prompt_tokens = 100
        result.total_input_tokens = 100
        result.total_output_tokens = 10
        result.elapsed_ms = 50
        return result

    def fake_compact(messages, **kw):
        return messages, "summary"

    with (
        patch.object(agllm, "compact", side_effect=fake_compact),
        patch.object(agllm, "call", side_effect=fake_llm_call),
    ):
        result, *_ = skill.execute_react(
            _make_mock_agent(LLM_COMPACT), agcontext(), _agdata(task="x")
        )

    # Must succeed — context_exceeded is not a connection error
    assert not isinstance(result, _agerror)
