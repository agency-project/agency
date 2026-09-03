"""Tests for the OpenAI(-compatible) LLM backend (agency.llm.openai)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import httpx

from agency.agconfig import agConfig
from agency.llm.openai import _OpenAICompatibleBackend


def _cfg(**fields) -> agConfig:
    """Test helper: wrap agllm_backend fields in an agConfig."""
    return agConfig({"agllm_backend": fields})


def _ev(**kwargs):
    return SimpleNamespace(**kwargs)


class _SdkObj(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


def _metadata_block(raw, *, index, stop_reason, usage=None):
    return {
        "type": "metadata",
        "index": index,
        "usage": usage,
        "stop_reason": stop_reason,
        "data": raw,
    }


class TestOpenAICompatibleBackend:
    def test_make_client_passes_config_through(self):
        backend = _OpenAICompatibleBackend(_cfg(api_key="k", base_url="http://x/v1"))
        with patch("agency.llm.openai.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="k", base_url="http://x/v1", timeout=httpx.Timeout(5.0)
        )

    def test_make_client_defaults_api_key_to_empty(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url="http://x/v1"))
        with patch("agency.llm.openai.openai.OpenAI") as MockCls:
            backend.make_client(httpx.Timeout(5.0))
        MockCls.assert_called_once_with(
            api_key="EMPTY", base_url="http://x/v1", timeout=httpx.Timeout(5.0)
        )

    def test_tokenize_url_strips_v1_suffix(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url="http://x:8000/v1"))
        assert backend.tokenize_url() == "http://x:8000"

    def test_tokenize_url_strips_trailing_slash(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url="http://x:8000/v1/"))
        assert backend.tokenize_url() == "http://x:8000"

    def test_tokenize_url_none_when_no_base_url(self):
        assert _OpenAICompatibleBackend(_cfg()).tokenize_url() is None

    def test_tokenize_url_preserves_non_v1_path(self):
        backend = _OpenAICompatibleBackend(_cfg(base_url="http://x:8000/custom"))
        assert backend.tokenize_url() == "http://x:8000/custom"


class TestFormatContextAgencyToBackend:
    def test_recognized_string_tool_choice_passed_through(self):
        backend = _OpenAICompatibleBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend(
            {"messages": [], "tool_choice": "required"}
        )
        assert kwargs["tool_choice"] == "required"

    def test_recognized_function_tool_choice_passed_through(self):
        backend = _OpenAICompatibleBackend(_cfg(model="m"))
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        kwargs = backend._format_context_agency_to_backend(
            {"messages": [], "tool_choice": tool_choice}
        )
        assert kwargs["tool_choice"] == tool_choice

    def test_foreign_origin_tool_choice_dropped(self):
        backend = _OpenAICompatibleBackend(_cfg(model="m"))
        tool_choice = {"type": "openai_responses_file_search", "data": {"type": "file_search"}}
        kwargs = backend._format_context_agency_to_backend(
            {"messages": [], "tool_choice": tool_choice}
        )
        assert "tool_choice" not in kwargs

    def test_no_tool_choice_omits_kwarg(self):
        backend = _OpenAICompatibleBackend(_cfg(model="m"))
        kwargs = backend._format_context_agency_to_backend({"messages": []})
        assert "tool_choice" not in kwargs


class TestFormatContextBackendToAgency:
    def _raw(self, message, finish_reason="stop"):
        choice = _ev(message=message, finish_reason=finish_reason)
        return _ev(choices=[choice], usage=None)

    def test_extracts_text_block(self):
        raw = self._raw(_SdkObj(content="hello"))
        result = _OpenAICompatibleBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {"type": "text", "index": 0, "text": "hello"},
            _metadata_block(raw, index=1, stop_reason="stop"),
        ]

    def test_legacy_function_call_becomes_tool_use(self):
        raw = self._raw(
            _SdkObj(content=None, function_call=_ev(name="get_weather", arguments="{}"))
        )
        result = _OpenAICompatibleBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {"type": "tool_use", "index": 0, "id": "", "name": "get_weather", "arguments": "{}"},
            _metadata_block(raw, index=1, stop_reason="stop"),
        ]

    def test_refusal_field_preserved_with_named_type(self):
        raw = self._raw(_SdkObj(content=None, refusal="I can't help with that"))
        result = _OpenAICompatibleBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {
                "type": "openai_chatcompletions_refusal",
                "index": 0,
                "data": "I can't help with that",
            },
            _metadata_block(raw, index=1, stop_reason="stop"),
        ]

    def test_annotations_and_audio_both_preserved(self):
        raw = self._raw(
            _SdkObj(
                content="see sources",
                annotations=[{"type": "url_citation", "url": "http://x"}],
                audio={"id": "a1", "data": "base64..."},
            )
        )
        result = _OpenAICompatibleBackend(_cfg())._format_context_backend_to_agency(raw)
        types = {
            b["type"]
            for b in result["message"]["blocks"]
            if b["type"].startswith("openai_chatcompletions_")
        }
        assert types == {"openai_chatcompletions_annotations", "openai_chatcompletions_audio"}

    def test_message_without_model_dump_only_gets_known_fields(self):
        raw = self._raw(_ev(content="hi", tool_calls=None, reasoning_content=None))
        result = _OpenAICompatibleBackend(_cfg())._format_context_backend_to_agency(raw)
        assert result["message"]["blocks"] == [
            {"type": "text", "index": 0, "text": "hi"},
            _metadata_block(raw, index=1, stop_reason="stop"),
        ]


class TestFormatStreamToAgency:
    def test_unrecognized_delta_field_streamed_with_named_type(self):
        chunk = _ev(
            usage=None,
            choices=[_ev(delta=_SdkObj(content=None, refusal="no"), finish_reason=None)],
        )
        items = list(_OpenAICompatibleBackend(_cfg())._format_stream_to_agency(iter([chunk])))
        unknown_items = [
            i for i in items if i.get("block_type") == "openai_chatcompletions_refusal"
        ]
        assert len(unknown_items) == 1
        assert unknown_items[0]["data"] == "no"

    def test_repeated_unrecognized_field_reuses_same_index(self):
        chunks = [
            _ev(
                usage=None,
                choices=[_ev(delta=_SdkObj(content=None, refusal="a"), finish_reason=None)],
            ),
            _ev(
                usage=None,
                choices=[_ev(delta=_SdkObj(content=None, refusal="b"), finish_reason=None)],
            ),
        ]
        items = list(_OpenAICompatibleBackend(_cfg())._format_stream_to_agency(iter(chunks)))
        unknown_items = [
            i for i in items if i.get("block_type") == "openai_chatcompletions_refusal"
        ]
        assert len(unknown_items) == 2
        assert unknown_items[0]["index"] == unknown_items[1]["index"]
