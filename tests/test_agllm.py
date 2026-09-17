"""Tests for agllm — LLM client wrapper and kwarg building."""

from unittest.mock import MagicMock, patch

from agency.llm.agllm import agllm
from agency.configs.agconfig import agconfig, llmconfig


def _cfg(**fields) -> agconfig:
    """Test helper: build an agconfig with the given llmconfig fields.
    Defaults base_url since the OpenAI-compatible backend (the default
    fallthrough for an unspecified provider) now requires one explicitly."""
    fields.setdefault("base_url", "http://localhost/v1")
    return agconfig(llmconfig(**fields))


def build_llm_kwargs(cfg, messages, openai_tools=None):
    return agllm.for_config(cfg).build_kwargs(messages, openai_tools)


LLM_COMPACT_CONFIG = {"api_key": "test", "model": "", "base_url": "http://localhost/v1"}


def test_build_llm_kwargs_includes_reasoning_effort():
    cfg = agconfig(
        llmconfig(model="gpt-5.6-luna", reasoning_effort="none", base_url="http://localhost/v1")
    )

    kwargs = build_llm_kwargs(
        cfg, [{"role": "user", "blocks": [{"type": "text", "index": 0, "text": "hi"}]}], None
    )

    assert kwargs["reasoning_effort"] == "none"


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
    msgs = [{"role": "user", "blocks": [{"type": "text", "index": 0, "text": "hi"}]}]
    kw = build_llm_kwargs(_cfg(), msgs, None)
    assert kw["messages"] == [{"role": "user", "content": "hi"}]


def test_build_llm_kwargs_strips_underscore_keys_from_messages():
    msgs = [
        {
            "role": "user",
            "blocks": [{"type": "text", "index": 0, "text": "hi"}],
            "_thinking": "internal",
        }
    ]
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


def test_build_llm_kwargs_falls_back_to_default_max_tokens_when_unset():
    """An OpenAI-compatible server has no server-side default the way
    Anthropic's/Bedrock's native APIs do -- litellm itself falls back to a
    hardcoded 4096 whenever a request omits this, silently truncating any
    model capable of more. agconfig's own default_max_tokens must always be
    sent explicitly instead of leaving it up to the backend in front of us."""
    cfg = _cfg(model="m")
    assert cfg.llm.max_completion_tokens is None
    assert cfg.llm.max_tokens is None
    kw = build_llm_kwargs(cfg, [], None)
    assert kw["max_completion_tokens"] == cfg.llm.default_max_tokens


def test_build_llm_kwargs_explicit_max_completion_tokens_not_overridden():
    cfg = _cfg(model="m", max_completion_tokens=512)
    kw = build_llm_kwargs(cfg, [], None)
    assert kw["max_completion_tokens"] == 512


def _sys_msg(text):
    return {"role": "system", "blocks": [{"type": "text", "index": 0, "text": text}]}


def _msg(role, text):
    return {"role": role, "blocks": [{"type": "text", "index": 0, "text": text}]}


def test_build_llm_kwargs_leading_system_message_stays_system(capsys):
    """The wire schema itself allows role:"system" anywhere, but not every
    server's own chat template does -- Qwen3's (via
    transformers.apply_chat_template inside vLLM) rejects a non-leading one
    outright. Matches anthropic.py's/bedrock.py's own handling: only a
    *leading* system message is exempt."""
    kw = build_llm_kwargs(
        _cfg(),
        [_sys_msg("You are helpful."), _msg("user", "hi")],
        None,
    )
    assert kw["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert kw["messages"][1] == {"role": "user", "content": "hi"}
    assert "mid-conversation" not in capsys.readouterr().out


def test_build_llm_kwargs_mid_conversation_system_message_becomes_user(capsys):
    kw = build_llm_kwargs(
        _cfg(),
        [
            _sys_msg("You are helpful."),
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _sys_msg("reminder: be concise"),
            _msg("user", "ok"),
        ],
        None,
    )
    assert kw["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "reminder: be concise"},
        {"role": "user", "content": "ok"},
    ]
    assert "second system-class" in capsys.readouterr().out


def test_build_llm_kwargs_empty_mid_conversation_system_message_dropped():
    kw = build_llm_kwargs(
        _cfg(),
        [
            _msg("user", "hi"),
            {"role": "system", "blocks": []},
            _msg("user", "ok"),
        ],
        None,
    )
    assert kw["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "ok"},
    ]


def test_build_llm_kwargs_leading_developer_after_system_becomes_user(capsys):
    """Reproduces a real vLLM/Qwen3 failure: codex.py's generic role
    passthrough emits a separate role:"developer" message (OpenAI's
    Responses API companion to "system") right after the leading system
    message, both still ahead of any real turn. Qwen3's chat template
    rejects it as a *second* system-class message ("System message must be
    at the beginning") even though it's not "mid-conversation" in the
    human sense -- only the very first system-class message may keep its
    role, regardless of what comes between it and the real conversation."""
    kw = build_llm_kwargs(
        _cfg(),
        [
            _sys_msg("You are Codex."),
            _msg("developer", "Some developer-level instructions."),
            _msg("user", "do X"),
        ],
        None,
    )
    assert kw["messages"] == [
        {"role": "system", "content": "You are Codex."},
        {"role": "user", "content": "Some developer-level instructions."},
        {"role": "user", "content": "do X"},
    ]
    assert "second system-class" in capsys.readouterr().out


def test_build_llm_kwargs_unknown_params_not_forwarded():
    """build_kwargs only forwards the known OpenAI-style generation params --
    other agconfig fields (provider selection, transport config, ...) don't
    leak into the wire kwargs."""
    cfg = _cfg(model="m", provider="openai", base_url="http://x")
    kw = build_llm_kwargs(cfg, [], None)
    assert "provider" not in kw
    assert "base_url" not in kw


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
    """Mutating a cloned agconfig's field alone never reaches the instance --
    change_config is the supported way to push a live update through."""
    llm = agllm.for_config(_cfg(temperature=0.7))
    llm.change_config(_cfg(temperature=0.2))
    assert llm.agconfig.llm.temperature == 0.2


def test_llm_change_config_clones_given_agconfig():
    llm = agllm.for_config(_cfg())
    new_cfg = _cfg(temperature=0.2)
    llm.change_config(new_cfg)
    new_cfg.llm.temperature = 0.9
    assert llm.agconfig.llm.temperature == 0.2


def test_llm_get_config_copy_returns_clone_not_same_object():
    llm = agllm.for_config(_cfg(temperature=0.7))
    copy = llm.get_config_copy()
    assert copy is not llm.agconfig


def test_llm_get_config_copy_reflects_current_values():
    llm = agllm.for_config(_cfg(temperature=0.7))
    assert llm.get_config_copy().llm.temperature == 0.7


def test_llm_get_config_copy_after_change_config_reflects_new_values():
    llm = agllm.for_config(_cfg(temperature=0.7))
    llm.change_config(_cfg(temperature=0.2))
    assert llm.get_config_copy().llm.temperature == 0.2


def test_mutating_llm_get_config_copy_does_not_affect_llm():
    llm = agllm.for_config(_cfg(temperature=0.7))
    copy = llm.get_config_copy()
    copy.llm.temperature = 0.1
    assert llm.agconfig.llm.temperature == 0.7


# ---------------------------------------------------------------------------
# fetch_context_limit
# ---------------------------------------------------------------------------


def test_fetch_context_limit_model_with_slash_in_name():
    """Model names like 'nvidia/foo' must not trigger a 404 via retrieve()."""
    cfg = _cfg(**{**LLM_COMPACT_CONFIG, "model": "nvidia/MiniMax-M2.7-NVFP4"})
    mock_info = MagicMock()
    mock_info.id = "nvidia/MiniMax-M2.7-NVFP4"
    mock_info.model_extra = {"max_model_len": 196000}
    mock_client = MagicMock()
    mock_client.models.list.return_value = [mock_info]

    with patch("agency.llm.agllm.openai.OpenAI", return_value=mock_client):
        result = agllm.for_config(cfg).fetch_context_limit()
    assert result == 196000
    mock_client.models.retrieve.assert_not_called()


def test_fetch_context_limit_config_wins_over_vllm():
    cfg = _cfg(**{**LLM_COMPACT_CONFIG, "context_limit": 8192})
    mock_info = MagicMock()
    mock_info.model_extra = {"max_model_len": 131072}
    mock_client = MagicMock()
    mock_client.models.list.return_value = [mock_info]

    with patch("agency.llm.agllm.openai.OpenAI", return_value=mock_client):
        result = agllm.for_config(cfg).fetch_context_limit()
    assert result == 8192
