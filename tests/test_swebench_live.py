"""Offline checks for the real-workload runner's shared provider budget."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agency.configs.agconfig import agconfig, llmconfig
from agency.llm.openai import _OpenAICompatibleBackend


@pytest.fixture
def runner(monkeypatch):
    examples = Path(__file__).parent.parent / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("swebench_live", examples / "12_swebench_live.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_two_model_clients_and_sdk_retry_share_request_budget(monkeypatch, runner):

    clock = [100.0]
    starts = []

    def sleep(delay):
        clock[0] += delay

    def respond(request):
        starts.append(clock[0])
        if len(starts) == 1:
            return httpx.Response(429, json={"error": {"message": "temporary token limit"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    class OfflineClient(httpx.Client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runner.time, "sleep", sleep)
    monkeypatch.setattr(httpx, "Client", OfflineClient)
    # Register the original method for restoration after the example override.
    monkeypatch.setattr(
        _OpenAICompatibleBackend, "make_client", _OpenAICompatibleBackend.make_client
    )
    runner.pace_provider_requests(10)
    config = agconfig(
        llmconfig(
            provider="openai",
            model="test",
            api_key="offline",
            base_url="https://example.invalid/v1",
        )
    )
    clients = [_OpenAICompatibleBackend(config).make_client(httpx.Timeout(30)) for _ in range(2)]
    try:
        for model_client in clients:
            reply = model_client.chat.completions.create(
                model="test", messages=[{"role": "user", "content": "hello"}]
            )
            assert reply.choices[0].message.content == "ok"
    finally:
        for model_client in clients:
            model_client.close()
    assert starts == [100, 110, 120]


def test_stripped_sandbox_patch_can_be_applied_to_other_workspace(runner, tmp_path):
    (tmp_path / "example.txt").write_text("old\n")
    patch = "diff --git a/example.txt b/example.txt\n--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-old\n+new"
    actor = SimpleNamespace(sandbox=SimpleNamespace(exec=lambda *args, **kwargs: (patch, 0)))
    subprocess.run(
        ["git", "apply", "-"],
        input=runner.extract_patch(actor),
        text=True,
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert (tmp_path / "example.txt").read_text() == "new\n"
