"""Tests for the Claude backend (agency.llm.anthropic) -- direct agency
<-> Anthropic-native translation, both directions, both streaming and
non-streaming, no OpenAI-shape intermediate."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import httpx

from agency.configs.agconfig import agconfig, llmconfig
from agency.llm.anthropic import (
    _AnthropicBackend,
    _agency_messages_to_anthropic,
    _agency_tool_choice_to_anthropic,
    _agency_tools_to_anthropic,
    _known_anthropic_context_window,
)


def _cfg(**fields) -> agconfig:
    """Test helper: build an agconfig with the given llmconfig fields."""
    return agconfig(llmconfig(**fields))


def _ev(**kwargs):
    return SimpleNamespace(**kwargs)


def _metadata_block(raw, *, index, stop_reason, prompt_tokens=0, completion_tokens=0):
    return {
        "type": "metadata",
        "index": index,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "stop_reason": stop_reason,
        "data": raw,
    }


class _SdkObj(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# _AnthropicBackend -- client construction, model listing, context window
# ---------------------------------------------------------------------------


class TestAnthropicBackend:
    def test_make_client_raises_when_anthropic_sdk_missing(self):
        backend = _AnthropicBackend(_cfg())
        with patch("agency.llm.anthropic._anthropic_sdk", None):
            with pytest.raises(RuntimeError, match="pip install anthropic"):
                backend.make_client(httpx.Timeout(5.0))

    def test_make_client_uses_config_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-from-config"))
        mock_sdk = MagicMock()
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            client = backend.make_client(httpx.Timeout(30.0))
        mock_sdk.Anthropic.assert_called_once_with(
            api_key="sk-ant-from-config", timeout=httpx.Timeout(30.0)
        )
        assert client is mock_sdk.Anthropic.return_value

    def test_make_client_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg())
        mock_sdk = MagicMock()
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        mock_sdk.Anthropic.assert_called_once_with(
            api_key="sk-ant-from-env", timeout=httpx.Timeout(5.0)
        )

    def test_config_api_key_takes_priority_over_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-from-config"))
        mock_sdk = MagicMock()
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["api_key"] == "sk-ant-from-config"

    def test_list_models_returns_empty_when_sdk_missing(self):
        backend = _AnthropicBackend(_cfg())
        with patch("agency.llm.anthropic._anthropic_sdk", None):
            assert backend.list_models() == []

    def test_list_models_calls_raw_client(self):
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-x"))
        mock_sdk = MagicMock()
        mock_raw_client = MagicMock()
        mock_raw_client.models.list.return_value = ["claude-sonnet-5"]
        mock_sdk.Anthropic.return_value = mock_raw_client
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            result = backend.list_models()
        assert result == ["claude-sonnet-5"]

    def test_tokenize_url_is_none(self):
        assert _AnthropicBackend(_cfg()).tokenize_url() is None

    def test_known_context_limit_delegates_to_lookup(self):
        backend = _AnthropicBackend(_cfg())
        assert backend.known_context_limit("claude-sonnet-5") == 1_000_000
        assert backend.known_context_limit("claude-nonexistent-model") is None

    def test_no_workspace_id_omits_default_headers(self, monkeypatch):
        """Claude Platform on AWS requires the header; plain api.anthropic.com
        doesn't use it — omit default_headers entirely rather than sending an
        empty/None header when no workspace ID is configured."""
        monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-x"))
        mock_sdk = MagicMock()
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert "default_headers" not in kwargs

    def test_config_workspace_id_sent_as_header(self):
        """Claude Platform on AWS (short-term API key + ANTHROPIC_BASE_URL
        override) rejects requests with 400 'Missing anthropic-workspace-id
        header' unless this is sent explicitly — the plain client does not
        read ANTHROPIC_WORKSPACE_ID into a header on its own."""
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-x", workspace_id="wrkspc_from_config"))
        mock_sdk = MagicMock()
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["default_headers"] == {"anthropic-workspace-id": "wrkspc_from_config"}

    def test_workspace_id_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_from_env")
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-x"))
        mock_sdk = MagicMock()
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["default_headers"] == {"anthropic-workspace-id": "wrkspc_from_env"}

    def test_config_workspace_id_takes_priority_over_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_from_env")
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-x", workspace_id="wrkspc_from_config"))
        mock_sdk = MagicMock()
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            backend.make_client(httpx.Timeout(5.0))
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["default_headers"] == {"anthropic-workspace-id": "wrkspc_from_config"}

    def test_list_models_also_sends_workspace_header(self):
        backend = _AnthropicBackend(_cfg(api_key="sk-ant-x", workspace_id="wrkspc_from_config"))
        mock_sdk = MagicMock()
        mock_sdk.Anthropic.return_value.models.list.return_value = []
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            backend.list_models()
        _, kwargs = mock_sdk.Anthropic.call_args
        assert kwargs["default_headers"] == {"anthropic-workspace-id": "wrkspc_from_config"}


# ---------------------------------------------------------------------------
# _known_anthropic_context_window
# ---------------------------------------------------------------------------


class TestKnownAnthropicContextWindow:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("us.anthropic.claude-sonnet-5", 1_000_000),
            ("anthropic.claude-sonnet-5", 1_000_000),
            ("eu.anthropic.claude-opus-4-8", 1_000_000),
            ("global.anthropic.claude-fable-5", 1_000_000),
            ("apac.anthropic.claude-haiku-4-5", 200_000),
            ("anthropic.claude-haiku-4-5-20251001-v1:0", 200_000),
            ("anthropic.claude-opus-4-5-20251101-v1:0", 1_000_000),
            ("claude-sonnet-5", 1_000_000),  # bare first-party ID, no Bedrock prefix
            ("claude-haiku-4-5", 200_000),
        ],
    )
    def test_known_models_resolve(self, model, expected):
        assert _known_anthropic_context_window(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-instant-v1",
            "",
            None,
        ],
    )
    def test_unknown_models_return_none(self, model):
        assert _known_anthropic_context_window(model) is None

    def test_does_not_prefix_match_unrelated_longer_name(self):
        """'claude-sonnet-5' must not accidentally match a model that merely
        starts with the same characters without a '-' boundary."""
        assert _known_anthropic_context_window("anthropic.claude-sonnet-50000") is None


def _text_msg(role, text):
    return {"role": role, "blocks": [{"type": "text", "index": 0, "text": text}]}


class TestAgencyMessagesToAnthropic:
    def test_system_message_extracted(self):
        system, msgs = _agency_messages_to_anthropic(
            [_text_msg("system", "You are helpful."), _text_msg("user", "hi")]
        )
        assert system == "You are helpful."
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_multiple_system_messages_joined(self):
        system, _ = _agency_messages_to_anthropic(
            [
                _text_msg("system", "Part 1."),
                _text_msg("system", "Part 2."),
                _text_msg("user", "hi"),
            ]
        )
        assert system == "Part 1.\n\nPart 2."

    def test_no_system_message_returns_none(self):
        system, _ = _agency_messages_to_anthropic([_text_msg("user", "hi")])
        assert system is None

    def test_empty_system_content_not_appended(self):
        system, _ = _agency_messages_to_anthropic(
            [{"role": "system", "blocks": []}, _text_msg("user", "hi")]
        )
        assert system is None

    def test_assistant_text_citations_reconstructed(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "x"),
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "text",
                            "index": 0,
                            "text": "see source",
                            "citations": [{"url": "http://x"}],
                        }
                    ],
                },
            ]
        )
        assert msgs[1]["content"] == [
            {"type": "text", "text": "see source", "citations": [{"url": "http://x"}]}
        ]

    def test_plain_assistant_text(self):
        _, msgs = _agency_messages_to_anthropic(
            [_text_msg("user", "hi"), _text_msg("assistant", "hello")]
        )
        assert msgs[1] == {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}

    def test_assistant_with_tool_call_becomes_tool_use_block(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "weather?"),
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "tool_use",
                            "index": 0,
                            "id": "call_1",
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        }
                    ],
                },
            ]
        )
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == [
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}}
        ]

    def test_assistant_with_text_and_tool_call_both_present(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "weather?"),
                {
                    "role": "assistant",
                    "blocks": [
                        {"type": "text", "index": 0, "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "index": 1,
                            "id": "call_1",
                            "name": "get_weather",
                            "arguments": "{}",
                        },
                    ],
                },
            ]
        )
        blocks = msgs[1]["content"]
        assert blocks[0] == {"type": "text", "text": "Let me check."}
        assert blocks[1]["type"] == "tool_use"

    def test_malformed_tool_call_arguments_become_empty_dict(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "x"),
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "tool_use",
                            "index": 0,
                            "id": "call_1",
                            "name": "f",
                            "arguments": "not json",
                        }
                    ],
                },
            ]
        )
        assert msgs[1]["content"][0]["input"] == {}

    def test_assistant_no_content_no_tools_becomes_empty_string(self):
        _, msgs = _agency_messages_to_anthropic(
            [_text_msg("user", "x"), {"role": "assistant", "blocks": []}]
        )
        assert msgs[1] == {"role": "assistant", "content": ""}

    def test_tool_result_becomes_user_message_with_tool_result_block(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "weather?"),
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "tool_use",
                            "index": 0,
                            "id": "call_1",
                            "name": "get_weather",
                            "arguments": "{}",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "blocks": [
                        {
                            "type": "tool_result",
                            "index": 0,
                            "tool_call_id": "call_1",
                            "text": "Sunny, 20C",
                        }
                    ],
                },
            ]
        )
        assert msgs[2] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny, 20C"}],
        }

    def test_consecutive_tool_results_merge_into_one_user_message(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "x"),
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "tool_use",
                            "index": 0,
                            "id": "c1",
                            "name": "a",
                            "arguments": "{}",
                        },
                        {
                            "type": "tool_use",
                            "index": 1,
                            "id": "c2",
                            "name": "b",
                            "arguments": "{}",
                        },
                    ],
                },
                {
                    "role": "tool",
                    "blocks": [
                        {
                            "type": "tool_result",
                            "index": 0,
                            "tool_call_id": "c1",
                            "text": "result a",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "blocks": [
                        {
                            "type": "tool_result",
                            "index": 0,
                            "tool_call_id": "c2",
                            "text": "result b",
                        }
                    ],
                },
            ]
        )
        tool_result_msgs = [
            m for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
        ]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]["content"] == [
            {"type": "tool_result", "tool_use_id": "c1", "content": "result a"},
            {"type": "tool_result", "tool_use_id": "c2", "content": "result b"},
        ]

    def test_tool_result_after_assistant_creates_new_user_message(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "x"),
                _text_msg("assistant", "thinking out loud"),
                {
                    "role": "tool",
                    "blocks": [
                        {"type": "tool_result", "index": 0, "tool_call_id": "c1", "text": "result"}
                    ],
                },
            ]
        )
        assert msgs[2] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "result"}],
        }

    def test_tool_result_raw_content_takes_priority_over_text(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "x"),
                {
                    "role": "tool",
                    "blocks": [
                        {
                            "type": "tool_result",
                            "index": 0,
                            "tool_call_id": "c1",
                            "text": "a photo",
                            "raw_content": [{"type": "image", "source": {"data": "abc"}}],
                        }
                    ],
                },
            ]
        )
        assert msgs[1]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "c1",
                "content": [{"type": "image", "source": {"data": "abc"}}],
            }
        ]

    def test_unknown_block_in_user_message_reconstructed(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                {
                    "role": "user",
                    "blocks": [
                        {"type": "text", "index": 0, "text": "look at this"},
                        {
                            "type": "anthropic_image",
                            "index": 1,
                            "data": {"source": {"type": "base64", "data": "abc"}},
                        },
                    ],
                }
            ]
        )
        assert msgs[0]["content"] == [
            {"type": "text", "text": "look at this"},
            {"type": "image", "source": {"type": "base64", "data": "abc"}},
        ]

    def test_unknown_block_in_assistant_message_reconstructed(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "x"),
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "anthropic_redacted_thinking",
                            "index": 0,
                            "data": {"data": "encrypted-blob"},
                        }
                    ],
                },
            ]
        )
        assert msgs[1]["content"] == [{"type": "redacted_thinking", "data": "encrypted-blob"}]

    def test_foreign_origin_unknown_block_dropped_not_sent_to_anthropic(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "x"),
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "openai_responses_local_shell_call",
                            "index": 0,
                            "data": {"call_id": "c1"},
                        }
                    ],
                },
            ]
        )
        assert msgs[1]["content"] == ""

    def test_streaming_origin_unknown_block_flattened_on_reconstruction(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "x"),
                {
                    "role": "assistant",
                    "blocks": [
                        {
                            "type": "anthropic_server_tool_use",
                            "index": 0,
                            "data": [
                                {
                                    "start": {"id": "st1", "name": "web_search"},
                                    "deltas": [
                                        {"partial_json": '{"q": '},
                                        {"partial_json": '"x"}'},
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ]
        )
        assert msgs[1]["content"] == [
            {
                "type": "server_tool_use",
                "id": "st1",
                "name": "web_search",
                "partial_json": '{"q": "x"}',
            }
        ]

    def test_unrecognized_role_dropped(self):
        _, msgs = _agency_messages_to_anthropic(
            [_text_msg("user", "x"), _text_msg("function_call_result_legacy", "should be dropped")]
        )
        assert len(msgs) == 1


class TestAgencyToolsToAnthropic:
    def test_none_returns_none(self):
        assert _agency_tools_to_anthropic(None) is None

    def test_empty_list_returns_none(self):
        assert _agency_tools_to_anthropic([]) is None

    def test_converts_openai_function_tool_shape(self):
        result = _agency_tools_to_anthropic(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ]
        )
        assert result == [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]

    def test_missing_parameters_defaults_to_empty_object_schema(self):
        result = _agency_tools_to_anthropic([{"type": "function", "function": {"name": "f"}}])
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_flat_tool_shape_without_function_wrapper(self):
        result = _agency_tools_to_anthropic([{"name": "f", "description": "d"}])
        assert result == [
            {"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {}}}
        ]

    def test_multiple_tools_converted_in_order(self):
        result = _agency_tools_to_anthropic(
            [
                {"type": "function", "function": {"name": "a"}},
                {"type": "function", "function": {"name": "b"}},
            ]
        )
        assert [t["name"] for t in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# _agency_tool_choice_to_anthropic
# ---------------------------------------------------------------------------


class TestAgencyToolChoiceToAnthropic:
    def test_none_returns_none(self):
        assert _agency_tool_choice_to_anthropic(None) is None

    def test_auto(self):
        assert _agency_tool_choice_to_anthropic("auto") == {"type": "auto"}

    def test_required_maps_to_any(self):
        assert _agency_tool_choice_to_anthropic("required") == {"type": "any"}

    def test_named_function_choice(self):
        result = _agency_tool_choice_to_anthropic({"type": "function", "function": {"name": "f"}})
        assert result == {"type": "tool", "name": "f"}

    def test_none_choice_maps_to_anthropic_none(self):
        assert _agency_tool_choice_to_anthropic("none") == {"type": "none"}

    def test_unrecognized_string_returns_none(self):
        assert _agency_tool_choice_to_anthropic("nonsense") is None


# ---------------------------------------------------------------------------
# _AnthropicBackend._format_context_agency_to_backend
# ---------------------------------------------------------------------------


class TestFormatContextAgencyToBackend:
    def test_builds_kwargs_with_system_and_cache_control(self):
        backend = _AnthropicBackend(
            _cfg(
                model="m",
                max_completion_tokens=256,
                temperature=0.5,
                top_p=0.9,
                extra_body={"top_k": 40},
            )
        )
        kwargs = backend._format_context_agency_to_backend(
            {
                "messages": [_text_msg("system", "Be terse."), _text_msg("user", "hi")],
                "tools": [{"type": "function", "function": {"name": "f"}}],
            }
        )
        assert kwargs == {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                    ],
                }
            ],
            "max_tokens": 256,
            "system": [
                {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}
            ],
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "tools": [
                {
                    "name": "f",
                    "description": "",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        }

    def test_no_messages_omits_last_message_cache_control(self):
        backend = _AnthropicBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend(
            {"messages": [_text_msg("system", "Be terse.")]}
        )
        assert kwargs["messages"] == []
        assert kwargs["system"] == [
            {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}
        ]

    def test_cache_control_lands_on_last_tool_result_block_not_first(self):
        backend = _AnthropicBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend(
            {
                "messages": [
                    _text_msg("user", "call the tool"),
                    {
                        "role": "assistant",
                        "blocks": [
                            {
                                "type": "tool_use",
                                "index": 0,
                                "id": "t1",
                                "name": "f",
                                "arguments": "{}",
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "blocks": [
                            {
                                "type": "tool_result",
                                "index": 0,
                                "tool_call_id": "t1",
                                "text": "result-1",
                            }
                        ],
                    },
                ]
            }
        )
        tool_result_message = kwargs["messages"][-1]
        assert tool_result_message["content"][-1] == {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "result-1",
            "cache_control": {"type": "ephemeral"},
        }

    def test_cache_control_does_not_mutate_caller_messages(self):
        backend = _AnthropicBackend(_cfg(model="m"))
        original_messages = [_text_msg("user", "ping")]
        backend._format_context_agency_to_backend({"messages": original_messages})
        assert original_messages == [_text_msg("user", "ping")]

    def test_max_tokens_defaults_to_128000_when_omitted(self):
        backend = _AnthropicBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend({"messages": [_text_msg("user", "x")]})
        assert kwargs["max_tokens"] == 128000

    def test_extra_body_without_top_k_is_ignored(self):
        backend = _AnthropicBackend(_cfg(model="m", extra_body={"repetition_penalty": 1.1}))
        kwargs = backend._format_context_agency_to_backend({"messages": [_text_msg("user", "x")]})
        assert "top_k" not in kwargs

    def test_no_system_message_omits_system_kwarg(self):
        backend = _AnthropicBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend({"messages": [_text_msg("user", "x")]})
        assert "system" not in kwargs

    def test_tool_choice_included_when_given(self):
        backend = _AnthropicBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend(
            {"messages": [_text_msg("user", "x")], "tool_choice": "auto"}
        )
        assert kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tool_choice_omits_kwarg(self):
        backend = _AnthropicBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend({"messages": [_text_msg("user", "x")]})
        assert "tool_choice" not in kwargs


# ---------------------------------------------------------------------------
# _AnthropicBackend._call_backend / _call_backend_stream
# ---------------------------------------------------------------------------


class TestCallBackend:
    def test_call_backend_calls_messages_create_and_closes_client(self):
        mock_sdk = MagicMock()
        mock_raw_client = MagicMock()
        mock_raw_client.messages.create.return_value = _ev(content=[], usage=None, stop_reason=None)
        mock_sdk.Anthropic.return_value = mock_raw_client
        backend = _AnthropicBackend(_cfg(api_key="k"))
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            backend._call_backend({"model": "m", "messages": []})
        mock_raw_client.messages.create.assert_called_once_with(model="m", messages=[])
        mock_raw_client.close.assert_called_once()

    def test_call_backend_stream_returns_raw_stream_and_client(self):
        mock_sdk = MagicMock()
        mock_raw_client = MagicMock()
        mock_raw_client.messages.create.return_value = iter([])
        mock_sdk.Anthropic.return_value = mock_raw_client
        backend = _AnthropicBackend(_cfg(api_key="k"))
        with patch("agency.llm.anthropic._anthropic_sdk", mock_sdk):
            raw_stream, client = backend._call_backend_stream({"model": "m", "messages": []})
        mock_raw_client.messages.create.assert_called_once_with(model="m", messages=[], stream=True)
        assert client is mock_raw_client
        assert list(raw_stream) == []


# ---------------------------------------------------------------------------
# _AnthropicBackend._format_context_backend_to_agency
# ---------------------------------------------------------------------------


class TestFormatContextBackendToAgency:
    def test_extracts_text_blocks(self):
        raw = _ev(
            content=[_ev(type="text", text="Hello "), _ev(type="text", text="world")],
            usage=None,
            stop_reason="end_turn",
        )
        result = _AnthropicBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {"type": "text", "index": 0, "text": "Hello "},
            {"type": "text", "index": 1, "text": "world"},
            _metadata_block(raw, index=2, stop_reason="end_turn"),
        ]

    def test_text_block_citations_preserved(self):
        raw = _ev(
            content=[_ev(type="text", text="see source", citations=[_SdkObj(url="http://x")])],
            usage=None,
            stop_reason="end_turn",
        )
        result = _AnthropicBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {"type": "text", "index": 0, "text": "see source", "citations": [{"url": "http://x"}]},
            _metadata_block(raw, index=1, stop_reason="end_turn"),
        ]

    def test_tool_use_block_preserved(self):
        raw = _ev(
            content=[_ev(type="tool_use", id="t1", name="f", input={})],
            usage=None,
            stop_reason="tool_use",
        )
        result = _AnthropicBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {"type": "tool_use", "index": 0, "id": "t1", "name": "f", "arguments": "{}"},
            _metadata_block(raw, index=1, stop_reason="tool_use"),
        ]

    def test_thinking_block_preserved_with_signature(self):
        raw = _ev(
            content=[_ev(type="thinking", thinking="pondering", signature="sig123")],
            usage=None,
            stop_reason="end_turn",
        )
        result = _AnthropicBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {"type": "thinking", "index": 0, "text": "pondering", "signature": "sig123"},
            _metadata_block(raw, index=1, stop_reason="end_turn"),
        ]

    def test_no_blocks_when_content_empty(self):
        raw = _ev(content=[], usage=None, stop_reason="end_turn")
        result = _AnthropicBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            _metadata_block(raw, index=0, stop_reason="end_turn")
        ]

    def test_usage_and_stop_reason_extracted(self):
        raw = _ev(content=[], usage=_ev(input_tokens=10, output_tokens=5), stop_reason="end_turn")
        result = _AnthropicBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert result["stop_reason"] == "end_turn"

    def test_unrecognized_content_block_preserved_as_unknown(self):
        raw = _ev(
            content=[_SdkObj(type="redacted_thinking", data="encrypted-blob")],
            usage=None,
            stop_reason="end_turn",
        )
        result = _AnthropicBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {
                "type": "anthropic_redacted_thinking",
                "index": 0,
                "data": {"type": "redacted_thinking", "data": "encrypted-blob"},
            },
            _metadata_block(raw, index=1, stop_reason="end_turn"),
        ]

    def test_unrecognized_block_without_model_dump_stored_as_is(self):
        raw_block = _ev(type="server_tool_use", id="st1")
        raw = _ev(content=[raw_block], usage=None, stop_reason="end_turn")
        result = _AnthropicBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {"type": "anthropic_server_tool_use", "index": 0, "data": raw_block},
            _metadata_block(raw, index=1, stop_reason="end_turn"),
        ]


# ---------------------------------------------------------------------------
# _AnthropicBackend._format_stream_to_agency
# ---------------------------------------------------------------------------


class TestFormatStreamToAgency:
    def test_text_only_stream(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=10))),
            _ev(type="content_block_start", index=0, content_block=_ev(type="text", text="")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="text_delta", text="Hello")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="text_delta", text=", world")),
            _ev(type="content_block_stop", index=0),
            _ev(
                type="message_delta", delta=_ev(stop_reason="end_turn"), usage=_ev(output_tokens=5)
            ),
            _ev(type="message_stop"),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        text_items = [i for i in items if i["type"] == "block_delta" and i["block_type"] == "text"]
        assert [i["text"] for i in text_items] == ["Hello", ", world"]
        usage_item = items[-1]
        assert usage_item["type"] == "usage"
        assert usage_item["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        assert usage_item["stop_reason"] == "end_turn"

    def test_citations_delta_emitted_for_text_block(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(type="content_block_start", index=0, content_block=_ev(type="text", text="")),
            _ev(
                type="content_block_delta", index=0, delta=_ev(type="text_delta", text="see source")
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_SdkObj(type="citations_delta", citation=_SdkObj(url="http://x")),
            ),
            _ev(type="content_block_stop", index=0),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        citation_items = [i for i in items if i.get("citations")]
        assert len(citation_items) == 1
        assert citation_items[0]["block_type"] == "text"
        assert citation_items[0]["citations"] == [{"url": "http://x"}]

    def test_thinking_delta_emits_reasoning_item(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=None)),
            _ev(
                type="content_block_start", index=0, content_block=_ev(type="thinking", thinking="")
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="thinking_delta", thinking="pondering"),
            ),
            _ev(type="content_block_stop", index=0),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        thinking_items = [
            i for i in items if i["type"] == "block_delta" and i["block_type"] == "thinking"
        ]
        assert [i.get("text") for i in thinking_items] == ["pondering"]

    def test_signature_delta_emits_signature_item(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=None)),
            _ev(
                type="content_block_start", index=0, content_block=_ev(type="thinking", thinking="")
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="thinking_delta", thinking="pondering"),
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="signature_delta", signature="sig-abc"),
            ),
            _ev(type="content_block_stop", index=0),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        sig_items = [i for i in items if i["type"] == "block_delta" and i.get("signature")]
        assert len(sig_items) == 1
        assert sig_items[0]["signature"] == "sig-abc"
        assert sig_items[0]["index"] == 0

    def test_tool_use_emits_single_item_with_full_arguments(self):
        """Regression test: tool-call JSON must arrive as ONE item with the
        complete concatenated arguments, not streamed fragment-by-fragment."""
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(
                type="content_block_start",
                index=0,
                content_block=_ev(type="tool_use", id="toolu_1", name="get_weather"),
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="input_json_delta", partial_json=""),
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="input_json_delta", partial_json='{"city": '),
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="input_json_delta", partial_json='"Paris"}'),
            ),
            _ev(type="content_block_stop", index=0),
            _ev(
                type="message_delta", delta=_ev(stop_reason="tool_use"), usage=_ev(output_tokens=1)
            ),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        tool_items = [
            i for i in items if i["type"] == "block_delta" and i["block_type"] == "tool_use"
        ]
        assert len(tool_items) == 1
        assert tool_items[0]["index"] == 0
        assert tool_items[0]["id"] == "toolu_1"
        assert tool_items[0]["name"] == "get_weather"
        assert tool_items[0]["arguments"] == '{"city": "Paris"}'

    def test_truncated_tool_use_is_flushed_not_dropped(self):
        """Regression: if the stream ends (e.g. stop_reason="max_tokens")
        while a tool_use block is still open, content_block_stop never fires
        for it. The partial JSON must still be flushed so the caller sees a
        (possibly unparseable) tool call attempt instead of silence."""
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(
                type="content_block_start",
                index=0,
                content_block=_ev(type="tool_use", id="toolu_1", name="return_env_requirements"),
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="input_json_delta", partial_json='{"foo": '),
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="input_json_delta", partial_json='"bar'),
            ),
            # stream ends here — no content_block_stop, no message_stop event needed
            _ev(
                type="message_delta",
                delta=_ev(stop_reason="max_tokens"),
                usage=_ev(output_tokens=1),
            ),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        tool_items = [
            i for i in items if i["type"] == "block_delta" and i["block_type"] == "tool_use"
        ]
        assert len(tool_items) == 1
        assert tool_items[0]["id"] == "toolu_1"
        assert tool_items[0]["name"] == "return_env_requirements"
        assert tool_items[0]["arguments"] == '{"foo": "bar'  # truncated, but present

    def test_unrecognized_content_block_buffered_and_emitted_on_stop(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(
                type="content_block_start",
                index=0,
                content_block=_SdkObj(type="server_tool_use", id="st1", name="web_search"),
            ),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_SdkObj(type="input_json_delta", partial_json="{}"),
            ),
            _ev(type="content_block_stop", index=0),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        unknown_items = [
            i
            for i in items
            if i["type"] == "block_delta" and i["block_type"] == "anthropic_server_tool_use"
        ]
        assert len(unknown_items) == 1
        assert unknown_items[0]["data"]["start"] == {
            "type": "server_tool_use",
            "id": "st1",
            "name": "web_search",
        }
        assert unknown_items[0]["data"]["deltas"] == [
            {"type": "input_json_delta", "partial_json": "{}"}
        ]

    def test_unrecognized_content_block_flushed_if_stream_ends_without_stop(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(
                type="content_block_start",
                index=0,
                content_block=_SdkObj(type="server_tool_use", id="st1", name="web_search"),
            ),
            _ev(
                type="message_delta",
                delta=_ev(stop_reason="max_tokens"),
                usage=_ev(output_tokens=1),
            ),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        unknown_items = [
            i
            for i in items
            if i["type"] == "block_delta" and i["block_type"] == "anthropic_server_tool_use"
        ]
        assert len(unknown_items) == 1

    def test_text_then_tool_use_at_nonzero_index(self):
        stream = [
            _ev(type="message_start", message=_ev(usage=_ev(input_tokens=1))),
            _ev(type="content_block_start", index=0, content_block=_ev(type="text", text="")),
            _ev(
                type="content_block_delta",
                index=0,
                delta=_ev(type="text_delta", text="Checking..."),
            ),
            _ev(type="content_block_stop", index=0),
            _ev(
                type="content_block_start",
                index=1,
                content_block=_ev(type="tool_use", id="toolu_2", name="get_weather"),
            ),
            _ev(
                type="content_block_delta",
                index=1,
                delta=_ev(type="input_json_delta", partial_json='{"city":"NYC"}'),
            ),
            _ev(type="content_block_stop", index=1),
            _ev(
                type="message_delta", delta=_ev(stop_reason="tool_use"), usage=_ev(output_tokens=1)
            ),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        text_items = [i for i in items if i["type"] == "block_delta" and i["block_type"] == "text"]
        tool_items = [
            i for i in items if i["type"] == "block_delta" and i["block_type"] == "tool_use"
        ]
        assert [i["text"] for i in text_items] == ["Checking..."]
        assert len(tool_items) == 1
        assert tool_items[0]["index"] == 1
        assert tool_items[0]["arguments"] == '{"city":"NYC"}'

    def test_content_block_stop_without_prior_tool_use_emits_nothing(self):
        """content_block_stop for a text block (never registered in
        tool_blocks) must not emit a spurious tool_use item."""
        stream = [
            _ev(type="message_start", message=_ev(usage=None)),
            _ev(type="content_block_start", index=0, content_block=_ev(type="text", text="")),
            _ev(type="content_block_delta", index=0, delta=_ev(type="text_delta", text="hi")),
            _ev(type="content_block_stop", index=0),
        ]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        tool_items = [
            i for i in items if i["type"] == "block_delta" and i["block_type"] == "tool_use"
        ]
        assert tool_items == []

    def test_empty_stream_still_yields_final_usage_item(self):
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter([])))
        assert [i["type"] for i in items] == ["block_delta", "usage"]
        assert items[0]["block_type"] == "metadata"
        assert items[-1]["type"] == "usage"
        assert items[-1]["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_missing_usage_on_message_start_defaults_to_zero(self):
        stream = [_ev(type="message_start", message=_ev(usage=None))]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        assert items[-1]["usage"]["prompt_tokens"] == 0
