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
    _AnthropicModelInfo,
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


def _metadata_block(
    raw,
    *,
    index,
    stop_reason,
    prompt_tokens=0,
    completion_tokens=0,
    cache_read_tokens=0,
    cache_write_tokens=0,
):
    return {
        "type": "metadata",
        "index": index,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
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
# fetch_context_limit -- aliases resolve through GET /v1/models/{id}
# ---------------------------------------------------------------------------


def _model_info(model_id, max_input_tokens):
    return SimpleNamespace(id=model_id, max_input_tokens=max_input_tokens, model_extra={})


_LIVE_LISTING = [
    _model_info("claude-sonnet-5-5", 1_000_000),
    _model_info("claude-opus-5-5", 1_000_000),
    _model_info("claude-opus-4-5-20251101", 200_000),
    _model_info("claude-haiku-4-5-20251001", 200_000),
]
_ALIASES = {
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-opus-4-5": "claude-opus-4-5-20251101",
}


def _fake_anthropic_client():
    import anthropic

    def retrieve(model_id):
        target = _ALIASES.get(model_id, model_id)
        for info in _LIVE_LISTING:
            if info.id == target:
                return info
        raise anthropic.NotFoundError(
            "not found",
            response=httpx.Response(404, request=httpx.Request("GET", "http://x")),
            body=None,
        )

    client = MagicMock()
    client.models.list.return_value = list(_LIVE_LISTING)
    client.models.retrieve.side_effect = retrieve
    return client


class TestAnthropicFetchContextLimit:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("claude-opus-5-5", 1_000_000),  # exact listing match
            ("claude-haiku-4-5-20251001", 200_000),  # exact listing match
            ("claude-haiku-4-5", 200_000),  # alias: was 1M (first listed model)
            ("claude-opus-4-5", 200_000),  # alias: was 1M
        ],
    )
    def test_listing_and_alias_resolution(self, model, expected):
        backend = _AnthropicBackend(_cfg(model=model, api_key="k"))
        with patch.object(backend, "make_client", return_value=_fake_anthropic_client()):
            assert backend.fetch_context_limit() == expected

    def test_exact_match_does_not_call_retrieve(self):
        backend = _AnthropicBackend(_cfg(model="claude-opus-5-5", api_key="k"))
        client = _fake_anthropic_client()
        with patch.object(backend, "make_client", return_value=client):
            backend.fetch_context_limit()
        client.models.retrieve.assert_not_called()

    def test_typo_model_gets_default_not_first_listed_window(self):
        backend = _AnthropicBackend(_cfg(model="claude-typo-model", api_key="k"))
        with patch.object(backend, "make_client", return_value=_fake_anthropic_client()):
            assert backend.fetch_context_limit() == 200_000  # default_context_limit

    def test_unlisted_model_with_static_entry_uses_table(self):
        backend = _AnthropicBackend(_cfg(model="claude-mythos-5", api_key="k"))
        with patch.object(backend, "make_client", return_value=_fake_anthropic_client()):
            assert backend.fetch_context_limit() == 1_000_000

    def test_bedrock_has_no_retrieve_and_uses_table(self):
        from agency.llm.bedrock import _AnthropicBedrockBackend

        backend = _AnthropicBedrockBackend(_cfg(model="us.anthropic.claude-haiku-4-5", api_key="k"))
        assert backend.retrieve_model("us.anthropic.claude-haiku-4-5") is None
        assert backend.fetch_context_limit() == 200_000


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
            # First-party Models API reports max_input_tokens=200000 for Opus 4.5.
            ("anthropic.claude-opus-4-5-20251101-v1:0", 200_000),
            ("claude-opus-4-5-20251101", 200_000),
            ("claude-sonnet-5", 1_000_000),  # bare first-party ID, no Bedrock prefix
            ("claude-haiku-4-5", 200_000),
            ("claude-fable-5-1", 1_000_000),
            ("claude-opus-5-5", 1_000_000),
            ("claude-sonnet-5-5", 1_000_000),
            ("claude-opus-5", 1_000_000),
            ("us.anthropic.claude-opus-5-5", 1_000_000),
            ("global.anthropic.claude-fable-5-1-v1:0", 1_000_000),
            ("anthropic.claude-sonnet-5-5-v1", 1_000_000),
        ],
    )
    def test_known_models_resolve(self, model, expected):
        assert _known_anthropic_context_window(model) == expected

    @pytest.mark.parametrize(
        "new_model,older_model",
        [
            ("claude-fable-5-1", "claude-fable-5"),
            ("claude-opus-5-5", "claude-opus-5"),
            ("claude-sonnet-5-5", "claude-sonnet-5"),
        ],
    )
    def test_new_point_release_uses_its_own_entry_not_older_model(
        self, monkeypatch, new_model, older_model
    ):
        """'claude-fable-5-1' is its own model, not a dated snapshot of
        'claude-fable-5' -- it must resolve through its own entry even when
        the older model's entry is listed (or iterated) first."""
        windows = {older_model: _AnthropicModelInfo(123), new_model: _AnthropicModelInfo(456)}
        monkeypatch.setattr("agency.llm.anthropic._ANTHROPIC_MODELS", windows)
        assert _known_anthropic_context_window(new_model) == 456
        assert _known_anthropic_context_window(f"us.anthropic.{new_model}-v1:0") == 456
        assert _known_anthropic_context_window(older_model) == 123
        assert _known_anthropic_context_window(f"{older_model}-20260101") == 123

    def test_unlisted_point_release_does_not_inherit_older_model_window(self, monkeypatch):
        """A model newer than the table must fall through to None (so the
        caller uses the live listing or default_context_limit) instead of
        silently inheriting an older model's window via prefix matching."""
        monkeypatch.setattr(
            "agency.llm.anthropic._ANTHROPIC_MODELS", {"claude-fable-5": _AnthropicModelInfo(123)}
        )
        assert _known_anthropic_context_window("claude-fable-5-1") is None
        assert _known_anthropic_context_window("anthropic.claude-fable-5-2-v1:0") is None

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

    def test_mid_conversation_system_message_becomes_user_message_in_place(self):
        system, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("system", "You are helpful."),
                _text_msg("user", "hi"),
                _text_msg("assistant", "hello"),
                _text_msg("system", "reminder: be concise"),
                _text_msg("user", "ok"),
            ]
        )
        assert system == "You are helpful."
        assert msgs == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            {"role": "user", "content": "reminder: be concise"},
            {"role": "user", "content": "ok"},
        ]

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

    @pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"])
    def test_default_config_request_is_valid_for_current_claude_models(self, model):
        """These models 400 on explicit `thinking` configs other than
        adaptive, on non-default temperature/top_p/top_k, and on forced
        tool_choice. Omitting all of them (thinking then runs adaptive, effort
        at the model's default) is the valid default request -- the model ID
        itself must pass through untouched."""
        backend = _AnthropicBackend(_cfg(model=model))
        kwargs = backend._format_context_agency_to_backend(
            {
                "messages": [_text_msg("system", "Be terse."), _text_msg("user", "hi")],
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": "auto",
            }
        )
        assert kwargs["model"] == model
        assert kwargs["tool_choice"] == {"type": "auto"}
        for rejected in ("thinking", "temperature", "top_p", "top_k", "output_config"):
            assert rejected not in kwargs

    def test_thinking_block_replayed_with_signature_for_tool_loop(self):
        """Current Claude models return (possibly empty-text) thinking blocks
        whose signature must be echoed back unchanged on the next turn."""
        backend = _AnthropicBackend(_cfg(model="claude-opus-5-5"))
        kwargs = backend._format_context_agency_to_backend(
            {
                "messages": [
                    _text_msg("user", "call f"),
                    {
                        "role": "assistant",
                        "blocks": [
                            {"type": "thinking", "index": 0, "text": "", "signature": "sig"},
                            {
                                "type": "tool_use",
                                "index": 1,
                                "id": "t1",
                                "name": "f",
                                "arguments": "{}",
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "blocks": [
                            {"type": "tool_result", "index": 0, "tool_call_id": "t1", "text": "ok"}
                        ],
                    },
                ]
            }
        )
        assert kwargs["messages"][1]["content"][0] == {
            "type": "thinking",
            "thinking": "",
            "signature": "sig",
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

    def test_max_tokens_defaults_to_32000_when_omitted(self):
        backend = _AnthropicBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend({"messages": [_text_msg("user", "x")]})
        assert kwargs["max_tokens"] == 32000

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
        assert result["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
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
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
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
        assert items[-1]["usage"] == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    def test_missing_usage_on_message_start_defaults_to_zero(self):
        stream = [_ev(type="message_start", message=_ev(usage=None))]
        items = list(_AnthropicBackend(_cfg())._format_stream_to_agency(iter(stream)))
        assert items[-1]["usage"]["prompt_tokens"] == 0


# ---------------------------------------------------------------------------
# Model-scoped request policy: forced tool_choice and sampling parameters.
# Model sets mirror the live 2026-09-29 compatibility matrix.
# ---------------------------------------------------------------------------

# Reject tool_choice {"type": "any"} / {"type": "tool"} with a 400.
_NO_FORCED_CHOICE = [
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-mythos-5-1",
    "us.anthropic.claude-opus-5-5",  # Bedrock inference-profile ID
    "global.anthropic.claude-sonnet-5-5-v1:0",
]
# Accept forced tool_choice, including with adaptive thinking on.
_FORCED_CHOICE_OK = [
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5",
]
# Not in the table, or in it with unverified restrictions: pass through.
_UNVERIFIED = ["m", "claude-future-9", "claude-opus-4-1", "claude-sonnet-4-0"]
# 400 on temperature != 1.0 and on any top_p / top_k.
_NO_SAMPLING = [
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
]
# Each sampling field accepted alone; temperature + top_p together is a 400.
_EXCLUSIVE_SAMPLING = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-haiku-4-5-20251001",
]

_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
_NAMED = {"type": "function", "function": {"name": "f"}}
_SAMPLING_KEYS = ("temperature", "top_p", "top_k")


@pytest.fixture
def fresh_warnings(monkeypatch):
    monkeypatch.setattr("agency.llm.anthropic._WARNED_ONCE", set())


def _kwargs(model, tool_choice=None, **llm_fields):
    backend = _AnthropicBackend(_cfg(model=model, **llm_fields))
    request = {"messages": [_text_msg("user", "call f")], "tools": _TOOLS}
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    return backend._format_context_agency_to_backend(request)


class TestForcedToolChoicePolicy:
    @pytest.mark.parametrize("model", _NO_FORCED_CHOICE)
    @pytest.mark.parametrize(
        "tool_choice,requested", [("required", "required"), (_NAMED, "named tool 'f'")]
    )
    def test_restricted_models_degrade_to_auto_with_warning(
        self, model, tool_choice, requested, capsys
    ):
        kwargs = _kwargs(model, tool_choice)
        assert kwargs["tool_choice"] == {"type": "auto"}
        warning = capsys.readouterr().out
        assert f"{model} does not support provider-level tool enforcement" in warning
        assert f"({requested}) was converted to `auto`" in warning
        assert "no longer guaranteed by the provider" in warning

    @pytest.mark.parametrize("model", _NO_FORCED_CHOICE[:3])
    def test_degradation_warns_on_every_request(self, model, capsys):
        _kwargs(model, "required")
        _kwargs(model, "required")
        assert capsys.readouterr().out.count("provider-level tool enforcement") == 2

    @pytest.mark.parametrize("model", _NO_FORCED_CHOICE[:3])
    def test_degradation_injects_no_messages(self, model):
        assert (
            _kwargs(model, "required")["messages"]
            == _kwargs(model, "auto")["messages"]
            == _kwargs(model)["messages"]
        )

    @pytest.mark.parametrize("model", _NO_FORCED_CHOICE[:3])
    @pytest.mark.parametrize("tool_choice,expected", [("auto", "auto"), ("none", "none")])
    def test_restricted_models_keep_auto_and_none(self, model, tool_choice, expected, capsys):
        assert _kwargs(model, tool_choice)["tool_choice"] == {"type": expected}
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("model", _FORCED_CHOICE_OK + _UNVERIFIED)
    def test_other_models_keep_forced_choice(self, model, capsys):
        assert _kwargs(model, "required")["tool_choice"] == {"type": "any"}
        assert _kwargs(model, _NAMED)["tool_choice"] == {"type": "tool", "name": "f"}
        assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("fresh_warnings")
class TestSamplingPolicy:
    @pytest.mark.parametrize("model", _NO_SAMPLING)
    @pytest.mark.parametrize(
        "fields,expected",
        [
            ({"temperature": 0.2}, {}),
            ({"temperature": 0.0}, {}),
            ({"temperature": 1.0}, {"temperature": 1.0}),
            ({"temperature": 1}, {"temperature": 1}),
            ({"top_p": 0.9}, {}),
            ({"top_p": 1.0}, {}),  # rejected even at its default
            ({"extra_body": {"top_k": 40}}, {}),
            ({"temperature": 0.2, "top_p": 0.9}, {}),
            ({"temperature": 1.0, "top_p": 0.9}, {"temperature": 1.0}),
        ],
    )
    def test_no_sampling_models(self, model, fields, expected):
        kwargs = _kwargs(model, **fields)
        assert {k: kwargs[k] for k in _SAMPLING_KEYS if k in kwargs} == expected

    @pytest.mark.parametrize("model", _EXCLUSIVE_SAMPLING)
    @pytest.mark.parametrize(
        "fields,expected",
        [
            ({"temperature": 0.2}, {"temperature": 0.2}),
            ({"temperature": 1.0}, {"temperature": 1.0}),
            ({"top_p": 0.9}, {"top_p": 0.9}),
            ({"extra_body": {"top_k": 40}}, {"top_k": 40}),
            ({"temperature": 0.2, "top_p": 0.9}, {"temperature": 0.2}),
            (
                {"temperature": 0.2, "top_p": 0.9, "extra_body": {"top_k": 40}},
                {"temperature": 0.2, "top_k": 40},
            ),
        ],
    )
    def test_exclusive_sampling_models(self, model, fields, expected):
        kwargs = _kwargs(model, **fields)
        assert {k: kwargs[k] for k in _SAMPLING_KEYS if k in kwargs} == expected

    @pytest.mark.parametrize("model", _UNVERIFIED)
    def test_unverified_models_pass_everything_through(self, model, capsys):
        kwargs = _kwargs(model, temperature=0.2, top_p=0.9, extra_body={"top_k": 40})
        assert {k: kwargs[k] for k in _SAMPLING_KEYS} == {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
        }
        assert capsys.readouterr().out == ""

    def test_dropped_sampling_warns_once_per_model_and_cause(self, capsys):
        for _ in range(3):
            _kwargs("claude-opus-5-5", temperature=0.2, top_p=0.9)
        out = capsys.readouterr().out
        assert out.count("WARNING") == 1
        assert "claude-opus-5-5 rejects sampling parameters" in out
        assert "not sending temperature, top_p" in out

    def test_exclusive_conflict_warns(self, capsys):
        _kwargs("claude-haiku-4-5", temperature=0.2, top_p=0.9)
        assert "sending temperature and not top_p" in capsys.readouterr().out

    def test_no_warning_when_nothing_dropped(self, capsys):
        _kwargs("claude-opus-5-5", temperature=1.0)
        _kwargs("claude-haiku-4-5", top_p=0.9)
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# developer role (Codex / Responses-style operator instructions)
# ---------------------------------------------------------------------------


def _assistant_tool_calls(*ids):
    return {
        "role": "assistant",
        "blocks": [
            {"type": "tool_use", "index": i, "id": tid, "name": "f", "arguments": "{}"}
            for i, tid in enumerate(ids)
        ],
    }


def _tool_result(tid, text="ok"):
    return {
        "role": "tool",
        "blocks": [{"type": "tool_result", "index": 0, "tool_call_id": tid, "text": text}],
    }


class TestDeveloperRole:
    def test_initial_developer_message_joins_top_level_system(self):
        system, msgs = _agency_messages_to_anthropic(
            [_text_msg("developer", "Sandbox is read-only."), _text_msg("user", "hi")]
        )
        assert system == "Sandbox is read-only."
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_system_then_developer_both_kept_in_order(self):
        system, msgs = _agency_messages_to_anthropic(
            [_text_msg("system", "Base."), _text_msg("developer", "Dev."), _text_msg("user", "hi")]
        )
        assert system == "Base.\n\nDev."
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_mid_conversation_developer_becomes_user_message_in_place(self):
        system, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "hi"),
                _text_msg("assistant", "hello"),
                _text_msg("developer", "Now answer in French."),
                _text_msg("user", "again"),
            ]
        )
        assert system is None
        assert msgs[2] == {"role": "user", "content": "Now answer in French."}
        assert msgs[3] == {"role": "user", "content": "again"}

    @pytest.mark.parametrize("role", ["system", "developer"])
    def test_system_class_text_never_splits_tool_use_from_its_result(self, role):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "go"),
                _assistant_tool_calls("t1"),
                _text_msg(role, "Reminder."),
                _tool_result("t1"),
                _text_msg("assistant", "done"),
            ]
        )
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
        assert msgs[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            {"type": "text", "text": "Reminder."},
        ]

    def test_parallel_results_stay_together_around_system_class_text(self):
        _, msgs = _agency_messages_to_anthropic(
            [
                _text_msg("user", "go"),
                _assistant_tool_calls("t1", "t2"),
                _tool_result("t1", "a"),
                _text_msg("developer", "Reminder."),
                _tool_result("t2", "b"),
            ]
        )
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert [b.get("tool_use_id") or b["text"] for b in msgs[2]["content"]] == [
            "t1",
            "t2",
            "Reminder.",
        ]

    def test_codex_developer_instructions_reach_anthropic(self):
        from agency.harness.adapters.codex import CodexAdapter

        context = CodexAdapter(agconfig())._format_context_harness_to_agency(
            {
                "instructions": "BASE",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "DEVELOPER RULES"}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    },
                ],
            }
        )
        kwargs = _AnthropicBackend(_cfg(model="claude-opus-5-5"))._format_context_agency_to_backend(
            context
        )
        assert kwargs["system"][0]["text"] == "BASE\n\nDEVELOPER RULES"
        assert [m["role"] for m in kwargs["messages"]] == ["user"]


# ---------------------------------------------------------------------------
# Thinking replay: only Claude-signed blocks are sent back
# ---------------------------------------------------------------------------


def _thinking(text, index, **fields):
    return {"type": "thinking", "index": index, "text": text, **fields}


def _replayed_assistant(blocks):
    _, msgs = _agency_messages_to_anthropic(
        [_text_msg("user", "go"), {"role": "assistant", "blocks": blocks}]
    )
    return msgs[1:]


@pytest.mark.usefixtures("fresh_warnings")
class TestThinkingReplay:
    def test_signed_blocks_replay_byte_identical_in_order_around_tool_calls(self):
        blocks = [
            _thinking("", 0, signature="sig-1"),
            {"type": "text", "index": 1, "text": "Checking."},
            _thinking("progress", 2, signature="sig-2"),
            {"type": "tool_use", "index": 3, "id": "t1", "name": "f", "arguments": '{"a": 1}'},
        ]
        assert _replayed_assistant(blocks) == [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig-1"},
                    {"type": "text", "text": "Checking."},
                    {"type": "thinking", "thinking": "progress", "signature": "sig-2"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}},
                ],
            }
        ]

    @pytest.mark.parametrize(
        "block",
        [
            _thinking("reasoned", 0, signature=""),  # empty signature
            _thinking("reasoned", 0),  # missing signature
            _thinking("", 0, signature="openai_responses:gAAAAB"),  # OpenAI Responses
        ],
        ids=["empty_signature", "missing_signature", "openai_responses"],
    )
    def test_unsigned_and_foreign_blocks_are_not_sent(self, block, capsys):
        tool_use = {"type": "tool_use", "index": 1, "id": "t1", "name": "f", "arguments": "{}"}
        assert _replayed_assistant([block, {**tool_use}]) == [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            }
        ]
        assert "without a Claude signature" in capsys.readouterr().out

    def test_foreign_provider_reasoning_content_is_never_replayed_as_thinking(self):
        """vLLM / Chat Completions `reasoning_content` becomes an unsigned
        agency thinking block; after a provider switch it must not reach
        Anthropic as a thinking block."""
        from agency.llm.openai import _OpenAICompatibleBackend

        raw = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content="answer",
                        reasoning_content="private chain of thought",
                        tool_calls=None,
                        function_call=None,
                        model_extra={},
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        agency = _OpenAICompatibleBackend(
            _cfg(provider="vllm", base_url="http://x/v1", model="m")
        )._format_context_backend_to_agency(raw)
        thinking = [b for b in agency["message"]["blocks"] if b["type"] == "thinking"]
        assert thinking and not thinking[0].get("signature")
        replayed = _replayed_assistant(
            [b for b in agency["message"]["blocks"] if b["type"] != "metadata"]
        )
        assert replayed == [{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]

    def test_turn_of_only_foreign_reasoning_is_omitted_not_sent_empty(self):
        assert _replayed_assistant([_thinking("reasoned", 0, signature="")]) == []

    def test_redacted_thinking_still_round_trips(self):
        blocks = [
            {"type": "anthropic_redacted_thinking", "index": 0, "data": {"data": "opaque"}},
            {"type": "text", "index": 1, "text": "ok"},
        ]
        assert _replayed_assistant(blocks)[0]["content"] == [
            {"data": "opaque", "type": "redacted_thinking"},
            {"type": "text", "text": "ok"},
        ]


class TestStopSequences:
    @pytest.mark.parametrize(
        "stop,expected", [("END", ["END"]), (["a", "b"], ["a", "b"]), (("x",), ["x"])]
    )
    def test_stop_becomes_stop_sequences(self, stop, expected):
        assert _kwargs("claude-opus-5-5", stop=stop)["stop_sequences"] == expected

    @pytest.mark.parametrize("stop", [None, [], ""])
    def test_no_stop_omits_stop_sequences(self, stop):
        assert "stop_sequences" not in _kwargs("claude-opus-5-5", stop=stop)


@pytest.mark.usefixtures("fresh_warnings")
class TestReasoningEffort:
    @pytest.mark.parametrize(
        "model,effort",
        [
            ("claude-opus-5-5", "low"),
            ("claude-opus-5-5", "xhigh"),
            ("claude-fable-5-1", "max"),
            ("claude-sonnet-5-5", "medium"),
            ("claude-opus-4-8", "xhigh"),
            ("claude-opus-4-6", "max"),
            ("claude-opus-4-5", "high"),
            ("us.anthropic.claude-opus-5-5", "high"),
        ],
    )
    def test_supported_level_maps_to_output_config(self, model, effort, capsys):
        assert _kwargs(model, reasoning_effort=effort)["output_config"] == {"effort": effort}
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "model,effort,reason",
        [
            ("claude-haiku-4-5", "low", "does not support the effort parameter"),
            ("claude-sonnet-4-5", "high", "does not support the effort parameter"),
            ("claude-opus-4-6", "xhigh", "accepts only"),
            ("claude-opus-4-5", "max", "accepts only"),
            ("claude-opus-5-5", "none", "accepts only"),  # OpenAI-only value
            ("claude-opus-5-5", "minimal", "accepts only"),
            ("claude-opus-4-1", "high", "unverified"),
            ("claude-future-9", "high", "unverified"),
        ],
    )
    def test_unsupported_or_unverified_effort_is_omitted_with_warning(
        self, model, effort, reason, capsys
    ):
        assert "output_config" not in _kwargs(model, reasoning_effort=effort)
        out = capsys.readouterr().out
        assert f"not sending reasoning_effort={effort!r} to {model}" in out
        assert reason in out

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-haiku-4-5", "m"])
    def test_unset_effort_keeps_model_default(self, model, capsys):
        assert "output_config" not in _kwargs(model)
        assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("fresh_warnings")
class TestFirstClassTopK:
    @pytest.mark.parametrize("model", _EXCLUSIVE_SAMPLING + _UNVERIFIED)
    def test_llmconfig_top_k_is_sent(self, model):
        assert _kwargs(model, top_k=40)["top_k"] == 40

    def test_llmconfig_top_k_wins_over_extra_body(self):
        assert _kwargs("claude-haiku-4-5", top_k=40, extra_body={"top_k": 5})["top_k"] == 40

    @pytest.mark.parametrize("model", _NO_SAMPLING)
    def test_llmconfig_top_k_dropped_where_sampling_is_rejected(self, model):
        assert "top_k" not in _kwargs(model, top_k=40)


class TestToolStrict:
    _SCHEMA = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }

    def test_chat_completions_strict_is_preserved(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "w",
                    "description": "d",
                    "parameters": self._SCHEMA,
                    "strict": True,
                },
            }
        ]
        assert _agency_tools_to_anthropic(tools) == [
            {"name": "w", "description": "d", "input_schema": self._SCHEMA, "strict": True}
        ]

    def test_flat_tool_strict_is_preserved(self):
        converted = _agency_tools_to_anthropic(
            [{"name": "w", "parameters": self._SCHEMA, "strict": True}]
        )
        assert converted[0]["strict"] is True

    @pytest.mark.parametrize("strict", [False, None, "true"])
    def test_non_true_strict_is_omitted(self, strict):
        fn = {"name": "w", "parameters": self._SCHEMA}
        if strict is not None:
            fn["strict"] = strict
        assert "strict" not in _agency_tools_to_anthropic([{"type": "function", "function": fn}])[0]
