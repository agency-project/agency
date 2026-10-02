"""native_harness and tandem_harness LLMClient keep reasoning deltas on the
assistant message they return, so the next request sends them back."""

from __future__ import annotations

import json

import httpx
import pytest

from agency.native_harness.llm_client import LLMClient as NativeLLMClient
from agency.tandem_harness.llm_client import LLMClient as TandemLLMClient


def _sse(deltas):
    lines = [
        "data: " + json.dumps({"choices": [{"index": 0, "delta": d, "finish_reason": None}]})
        for d in deltas
    ]
    return "\n\n".join(lines + ["data: [DONE]"]) + "\n\n"


def _client(cls, body):
    client = cls("http://bridge", "token")
    client._client = httpx.Client(
        base_url="http://bridge",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
    )
    return client


@pytest.mark.parametrize("cls", [NativeLLMClient, TandemLLMClient])
def test_reasoning_deltas_kept_on_message(cls):
    body = _sse(
        [
            {"reasoning_content": "pl"},
            {"reasoning_content": "an"},
            {"reasoning_signature": "sig"},
            {"content": "answer"},
        ]
    )
    message = _client(cls, body).dispatch("m", [{"role": "user", "content": "q"}])["message"]
    assert message == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "plan",
        "reasoning_signature": "sig",
    }


@pytest.mark.parametrize("cls", [NativeLLMClient, TandemLLMClient])
def test_no_reasoning_deltas_leaves_message_unchanged(cls):
    message = _client(cls, _sse([{"content": "answer"}])).dispatch(
        "m", [{"role": "user", "content": "q"}]
    )["message"]
    assert message == {"role": "assistant", "content": "answer"}
