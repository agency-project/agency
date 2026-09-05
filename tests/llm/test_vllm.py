"""Tests for the vLLM LLM backend config/dispatch (agency.llm.vllm).

vLLM has no backend class of its own -- it speaks the same OpenAI-compatible
chat.completions API as `.openai._OpenAICompatibleBackend`, which
`agllm.for_config()` routes to directly (after checking base_url is
set, since -- unlike real OpenAI -- there's no well-known default URL for a
self-hosted vLLM endpoint). See test_openai.py for _OpenAICompatibleBackend's
own behavior.
"""

from __future__ import annotations

import pytest

from agency.configs.agconfig import agconfig, llmconfig
from agency.llm.agllm import agllm
from agency.llm.openai import _OpenAICompatibleBackend


class TestVllmDispatch:
    def test_vllm_with_base_url_returns_openai_compatible_backend(self):
        backend = agllm.for_config(
            agconfig(llmconfig(provider="vllm", base_url="http://localhost:8000/v1", model="m"))
        )
        assert isinstance(backend, _OpenAICompatibleBackend)

    def test_vllm_without_base_url_raises_value_error(self):
        with pytest.raises(ValueError, match="missing required field.*base_url"):
            agllm.for_config(agconfig(llmconfig(provider="vllm", model="m")))

    def test_config_fixes_provider_to_vllm(self):
        cfg = agconfig(llmconfig(provider="vllm", model="m", base_url="http://localhost:8000/v1"))
        assert cfg.llm.provider == "vllm"

    def test_config_rejects_disallowed_field(self):
        with pytest.raises(TypeError):
            agconfig(llmconfig(provider="vllm", not_a_real_field="w"))
