"""Shared LLM configuration for examples.

The examples keep the historical VLLM_* environment variable names because the
same config can target vLLM, local OpenAI-compatible servers, or OpenAI itself.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse


def _is_openai_endpoint(base_url: str) -> bool:
    if not base_url:
        return True
    return urlparse(base_url).netloc.lower() == "api.openai.com"


def _is_gpt5_family(model: str) -> bool:
    return model.lower().startswith("gpt-5")


def make_llm_config(
    *,
    max_tokens: int = 8000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> dict:
    base_url = os.environ.get("VLLM_BASE_URL", "")
    model = os.environ.get("VLLM_MODEL", "")
    openai_endpoint = _is_openai_endpoint(base_url)

    cfg = {
        "base_url": base_url,
        "api_key": os.environ.get("VLLM_API_KEY", ""),
        "model": model,
        "max_tokens": max_tokens,
    }

    if not (openai_endpoint and _is_gpt5_family(model)):
        cfg.update({
            "temperature": temperature,
            "top_p": top_p,
        })

    if not openai_endpoint:
        cfg.update({
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        })

    return cfg
