"""Checks that protect the study from invented or double-counted usage."""

import json
from pathlib import Path

import numpy as np
import pytest

from analysis import bootstrap, call_cost, metrics
from recording_proxy import summarize


@pytest.fixture
def prices():
    return {
        "input_cost_per_token": 1e-7,
        "output_cost_per_token": 5e-7,
        "cache_read_input_token_cost": 1e-8,
        "cache_creation_input_token_cost": 1.25e-7,
        "cache_creation_input_token_cost_above_1hr": 2e-7,
        "input_cost_per_token_above_100k_tokens": 5e-7,
        "output_cost_per_token_above_100k_tokens": 2.5e-6,
        "cache_read_input_token_cost_above_100k_tokens": 5e-8,
        "cache_creation_input_token_cost_above_100k_tokens": 6.25e-7,
        "cache_creation_input_token_cost_above_1hr_above_100k_tokens": 1e-6,
    }


def test_cache_neutral_cost_counts_each_input_category_once(prices):
    reference, estimated = call_cost(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 300,
                "cache_creation": {"ephemeral_1h_input_tokens": 100},
            }
        },
        prices,
    )
    assert reference == pytest.approx(600e-7 + 50 * 5e-7)
    assert estimated == pytest.approx(100e-7 + 50 * 5e-7 + 200e-8 + 200 * 1.25e-7 + 100 * 2e-7)


def test_long_context_threshold_includes_cached_tokens(prices):
    reference, _ = call_cost(
        {"usage": {"input_tokens": 2, "cache_read_input_tokens": 100000, "output_tokens": 10}},
        prices,
    )
    assert reference == pytest.approx(100002 * 5e-7 + 10 * 2.5e-6)


def request_fixture(tmp_path, response):
    (tmp_path / "0000.request.json").write_text(json.dumps({"model": "claude-haiku-5-5"}))
    (tmp_path / "0000.metadata.json").write_text(
        json.dumps({"path": "/v1/messages", "status": 200})
    )
    (tmp_path / "0000.response.body").write_text(
        "\n\n".join("data: " + json.dumps(e) for e in response)
    )


def test_cumulative_stream_usage_is_not_summed_twice(tmp_path):
    request_fixture(
        tmp_path,
        [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 12, "output_tokens": 1}},
            },
            {"type": "message_delta", "usage": {"input_tokens": 12, "output_tokens": 30}},
            {"type": "message_stop", "usage": {"input_tokens": 12, "output_tokens": 31}},
        ],
    )
    result = summarize(tmp_path)
    assert result["usage"]["input_tokens"] == 12
    assert result["usage"]["output_tokens"] == 31
    assert result["usage_complete"]


def test_truncated_stream_is_unknown_not_zero_cost(tmp_path):
    request_fixture(
        tmp_path,
        [{"type": "message_start", "message": {"usage": {"input_tokens": 12, "output_tokens": 1}}}],
    )
    assert not summarize(tmp_path)["usage_complete"]


def test_server_error_without_usage_cannot_be_treated_as_zero_cost(prices):
    result = {
        "run_id": "fixture",
        "instance_id": "fixture",
        "arm": "standalone",
        "repetition": 1,
        "attempt": 1,
        "status": "failed",
        "provider": {
            "usage_complete": True,
            "calls": [
                {
                    "status": 200,
                    "usage_valid": True,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
                {"status": 500, "usage_valid": False, "usage": {}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    }
    row = metrics(result, prices, {})
    assert row["reference_cost_usd"] is None
    assert not row["usage_complete"]


def test_verified_local_admission_failure_has_no_provider_usage(prices):
    result = {
        "run_id": "fixture",
        "instance_id": "fixture",
        "arm": "standalone",
        "repetition": 1,
        "attempt": 1,
        "status": "completed",
        "provider": {
            "usage_complete": False,
            "calls": [
                {
                    "status": 200,
                    "usage_valid": True,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
                {"error": "database is locked", "usage_valid": False, "usage": {}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    }
    row = metrics(result, prices, {})
    assert row["reference_cost_usd"] == pytest.approx(10e-7 + 5 * 5e-7)
    assert row["usage_complete"]
    assert row["local_pre_inference_errors"] == 1
    assert row["llm_requests"] == 1
    assert row["native_api_attempts"] == 2


def test_fixed_task_bootstrap_equal_weights_tasks_with_unequal_cost_scales():
    groups = {
        "small": {"standalone": np.arange(1.0, 11.0), "agency": np.arange(2.0, 12.0)},
        "large": {"standalone": 10 * np.arange(1.0, 11.0), "agency": 10 * np.arange(1.0, 11.0) + 3},
    }
    tasks, aggregate = bootstrap(groups, 123, 2000)
    assert aggregate["difference_usd"] == 2
    assert aggregate["geometric_variance_ratio"] == pytest.approx(1)
    assert aggregate == bootstrap(groups, 123, 2000)[1]
    assert len(tasks) == 2


def test_shared_limiter_blocks_other_worker_until_stream_slot_is_released(tmp_path, monkeypatch):
    import concurrent.futures
    import sqlite3
    import recording_proxy

    monkeypatch.setattr(recording_proxy, "ROOT", tmp_path)
    (tmp_path / "configuration.json").write_text(
        json.dumps(
            {
                "admission": {
                    "requests_per_minute": 60,
                    "tokens_per_minute": 10000,
                    "max_inflight": 1,
                }
            }
        )
    )
    first = recording_proxy.admit(100)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        second = pool.submit(recording_proxy.admit, 100)
        with pytest.raises(concurrent.futures.TimeoutError):
            second.result(timeout=0.3)
        with sqlite3.connect(tmp_path / "admission.sqlite3") as db:
            db.execute("DELETE FROM inflight WHERE id=?", (first["reservation_id"],))
        assert second.result(timeout=3)["active_requests_at_admission"] == 1


def test_transient_database_lock_is_waited_out_before_provider_admission(tmp_path, monkeypatch):
    import sqlite3
    import recording_proxy

    monkeypatch.setattr(recording_proxy, "ROOT", tmp_path)
    (tmp_path / "configuration.json").write_text(
        json.dumps(
            {
                "admission": {
                    "requests_per_minute": 60,
                    "tokens_per_minute": 10000,
                    "max_inflight": 1,
                }
            }
        )
    )
    original = sqlite3.connect
    attempts = []

    def connect(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    result = recording_proxy.admit(100)
    assert result["database_lock_retries"] == 1
    assert result["active_requests_at_admission"] == 1
    assert result["reserved_tokens"] == 100


def test_token_count_override_preserves_body_and_rejects_invalid_session(tmp_path, monkeypatch):
    import ast
    from types import SimpleNamespace
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    import httpx

    source = (Path(__file__).parent / "capture/sitecustomize.py").read_text()
    function = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "transparent_register"
    )
    config = tmp_path / "recorder.json"
    config.write_text(
        json.dumps({"base_url": "http://recorder.fixture", "proxy_token": "relay-test-token"})
    )
    observed = []

    class FixtureClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, headers, content):
            observed.append((url, headers, content))
            return httpx.Response(200, json={"input_tokens": 123})

    monkeypatch.setattr(httpx, "AsyncClient", FixtureClient)

    def original(self, app, router):
        @app.post("/v1/messages/count_tokens")
        async def estimated(request: Request):
            return {"input_tokens": 1}

    namespace = {
        "original_register": original,
        "json": json,
        "Path": lambda _: config,
        "claude_module": SimpleNamespace(
            extract_bearer_token=lambda request: request.headers.get("x-api-key")
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "study-counter", "exec"), namespace)
    app = FastAPI()
    namespace["transparent_register"](
        None, app, SimpleNamespace(validate_token=lambda token: token == "valid-session")
    )
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/messages/count_tokens", headers={"x-api-key": "invalid"}, json={}
            ).status_code
            == 401
        )
        body = b'{"model":"claude-haiku-5-5","messages":[{"role":"user","content":"same bytes"}]}'
        response = client.post(
            "/v1/messages/count_tokens",
            headers={"x-api-key": "valid-session", "Content-Type": "application/json"},
            content=body,
        )
        assert response.json() == {"input_tokens": 123}
        assert observed == [
            (
                "http://recorder.fixture/v1/messages/count_tokens",
                {"content-type": "application/json", "x-api-key": "relay-test-token"},
                body,
            )
        ]
