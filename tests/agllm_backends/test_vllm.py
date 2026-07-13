"""Tests for the vLLM LLM backend config/dispatch (agency.agllm_backends.vllm).

vLLM has no backend class of its own -- it speaks the same OpenAI-compatible
chat.completions API as `.openai._OpenAICompatibleBackend`, which
`agllm_backend.for_config()` routes to directly (after checking base_url is
set, since -- unlike real OpenAI -- there's no well-known default URL for a
self-hosted vLLM endpoint). See test_openai.py for _OpenAICompatibleBackend's
own behavior.
"""

from __future__ import annotations

import pytest

from agency.agconfig import agConfig
from agency.agllm_backends import agllm_backend
from agency.agllm_backends.openai import _OpenAICompatibleBackend
from agency.agllm_backends.vllm import agVLLMBackendConfig


def _cfg(**fields) -> agConfig:
    return agConfig({"agllm_backend": fields})


class TestVllmDispatch:
    def test_vllm_with_base_url_returns_openai_compatible_backend(self):
        backend = agllm_backend.for_config(
            _cfg(provider="vllm", base_url="http://localhost:8000/v1", model="m")
        )
        assert isinstance(backend, _OpenAICompatibleBackend)

    def test_vllm_without_base_url_raises_value_error(self):
        with pytest.raises(ValueError, match="requires base_url"):
            agllm_backend.for_config(_cfg(provider="vllm", model="m"))

    def test_config_fixes_provider_to_vllm(self):
        cfg = agVLLMBackendConfig(model="m", base_url="http://localhost:8000/v1").agconfig
        assert cfg.agllm_backend.provider == "vllm"

    def test_config_rejects_disallowed_field(self):
        with pytest.raises(TypeError):
            agVLLMBackendConfig(workspace_id="w")
