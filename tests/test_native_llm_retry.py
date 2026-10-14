import httpx

from agency.native_harness import llm_client
from agency.native_harness.llm_client import LLMClient

OK = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'


def _client(responses, monkeypatch):
    monkeypatch.setattr(llm_client, "_retry_backoff_s", lambda attempt: 0)
    calls = []

    def handler(request):
        calls.append(request)
        r = responses[min(len(calls), len(responses)) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    c = LLMClient("http://x", "k")
    c._client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    return c, calls


def test_transient_gateway_errors_are_retried(monkeypatch):
    c, calls = _client([httpx.Response(524, text="timeout"),
                        httpx.Response(500, text='{"error":{"message":"peer closed","transient":true}}'),
                        httpx.RemoteProtocolError("peer closed connection"),
                        httpx.Response(200, text=OK)], monkeypatch)
    out = c.dispatch("m", [{"role": "user", "content": "x"}])
    assert out["message"]["content"] == "hi" and len(calls) == 4


def test_non_transient_error_is_not_retried(monkeypatch):
    c, calls = _client([httpx.Response(400, text="bad request")], monkeypatch)
    assert "400" in c.dispatch("m", [])["error"] and len(calls) == 1


def test_relayed_gateway_errors_in_stream_are_retried(monkeypatch):
    err = 'data: {"error":{"message":"Error code: 524 - origin timeout"}}\n\n'
    c, calls = _client([httpx.Response(200, text=err), httpx.Response(500, text="Internal Server Error"),
                        httpx.Response(200, text=OK)], monkeypatch)
    assert c.dispatch("m", [])["message"]["content"] == "hi" and len(calls) == 3
