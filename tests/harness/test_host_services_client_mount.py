from __future__ import annotations

import httpx
import pytest

from agency.harness.clients import HostServicesClient


@pytest.fixture
def client_and_paths(tmp_path, monkeypatch):
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/resolve_model"):
            return httpx.Response(200, json={"model": "m"})
        if request.url.path.endswith("/context_limit"):
            return httpx.Response(200, json={"context_limit": 1000})
        return httpx.Response(200, json={"message": {"blocks": []}})

    client = HostServicesClient(str(tmp_path / "host.sock"))
    client.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://host")
    monkeypatch.setattr(client, "_attempt_headers", lambda _token: {})
    return client, paths


def test_llm_calls_default_to_the_llm_mount(client_and_paths):
    client, paths = client_and_paths
    client.resolve_model("t")
    client.context_limit("t")
    client.dispatch("t", {"messages": []})
    assert paths == ["/llm/resolve_model", "/llm/context_limit", "/llm/dispatch"]


def test_llm_calls_route_to_another_mount(client_and_paths):
    client, paths = client_and_paths
    assert client.resolve_model("t", mount="llm_alt") == "m"
    assert client.context_limit("t", mount="llm_alt") == 1000
    client.dispatch("t", {"messages": []}, mount="llm_alt")
    assert paths == ["/llm_alt/resolve_model", "/llm_alt/context_limit", "/llm_alt/dispatch"]
