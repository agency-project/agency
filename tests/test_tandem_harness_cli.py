from __future__ import annotations

import argparse
import json

import pytest

from agency.tandem_harness import cli
from agency.tandem_harness.cli import _build_arg_parser, _resolve_supervisor_llm, main
from agency.tandem_harness.llm_client import LLMClient


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(supervisor_model="model-a", worker_model="model-a")
    defaults.update(overrides)
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-llm-base-url", default=None)
    parser.add_argument("--supervisor-llm-api-key", default=None)
    parser.add_argument("--supervisor-model", default=defaults["supervisor_model"])
    parser.add_argument("--worker-model", default=defaults["worker_model"])
    ns = parser.parse_args([])
    for key, value in defaults.items():
        setattr(ns, key, value)
    return ns


class TestResolveSupervisorLlm:
    def test_shares_the_worker_connection_when_models_match(self):
        worker_llm = LLMClient("http://bridge", "token")
        args = _args(supervisor_model="model-a", worker_model="model-a")
        assert _resolve_supervisor_llm(args, worker_llm, "http://bridge", "token") is worker_llm

    def test_raises_when_models_differ_with_no_distinct_endpoint(self):
        worker_llm = LLMClient("http://bridge", "token")
        args = _args(supervisor_model="model-a", worker_model="model-b")
        with pytest.raises(SystemExit, match="model-a.*model-b"):
            _resolve_supervisor_llm(args, worker_llm, "http://bridge", "token")

    def test_uses_a_distinct_client_when_a_supervisor_endpoint_is_given(self):
        worker_llm = LLMClient("http://bridge", "token")
        args = _args(
            supervisor_model="model-a",
            worker_model="model-b",
            supervisor_llm_base_url="https://real-provider/v1",
            supervisor_llm_api_key="real-key",
        )
        resolved = _resolve_supervisor_llm(args, worker_llm, "http://bridge", "token")
        assert resolved is not worker_llm


class TestMainRejectsEmptyModels:
    def test_supervisor_model_flag_accepts_an_empty_string_at_the_parser_level(self):
        args = _build_arg_parser().parse_args(
            [
                "-p",
                "x",
                "--supervisor-model",
                "",
                "--worker-model",
                "w",
                "--worker-history-turns",
                "0",
            ]
        )
        assert args.supervisor_model == ""

    def test_main_still_rejects_an_empty_supervisor_model(self, capsys):
        with pytest.raises(SystemExit):
            main(
                [
                    "-p",
                    "x",
                    "--supervisor-model",
                    "",
                    "--worker-model",
                    "w",
                    "--worker-history-turns",
                    "0",
                ]
            )
        assert "--supervisor-model" in capsys.readouterr().err

    def test_main_still_rejects_an_empty_worker_model(self, capsys):
        with pytest.raises(SystemExit):
            main(
                [
                    "-p",
                    "x",
                    "--supervisor-model",
                    "s",
                    "--worker-model",
                    "",
                    "--worker-history-turns",
                    "0",
                ]
            )
        assert "--worker-model" in capsys.readouterr().err


class _RecordingLLMClient:
    calls: "list[tuple[str, str, str]]" = []

    def __init__(self, base_url, api_key, timeout_s=300):
        self.base_url = base_url
        self.api_key = api_key

    def dispatch(self, model, messages, tools=None, *, internal_kind=None):
        type(self).calls.append((self.base_url, self.api_key, model))
        if model == "supervisor-real-model" and len(type(self).calls) == 1:
            tool_call = {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "smart_tool",
                    "arguments": json.dumps({"task": "do the thing"}),
                },
            }
            message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        else:
            message = {"role": "assistant", "content": "done", "tool_calls": []}
        return {"message": message, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_supervisor_and_worker_actually_dispatch_under_different_models(monkeypatch, tmp_path):
    """The outcome that matters for a two-model harness: the model string
    that reaches the wire for supervisor turns must differ from the
    worker's whenever they're configured to differ -- not just that some
    intermediate config field was set correctly."""
    _RecordingLLMClient.calls = []
    monkeypatch.setattr(cli, "LLMClient", _RecordingLLMClient)

    main(
        [
            "-p",
            "do the task",
            "--supervisor-model",
            "supervisor-real-model",
            "--worker-model",
            "worker-real-model",
            "--worker-history-turns",
            "0",
            "--llm-base-url",
            "http://worker-bridge",
            "--llm-api-key",
            "worker-key",
            "--supervisor-llm-base-url",
            "http://supervisor-real-endpoint",
            "--supervisor-llm-api-key",
            "supervisor-key",
            "--session-dir",
            str(tmp_path / "sessions"),
            "--offload-dir",
            str(tmp_path / "offload"),
            "--max-steps",
            "5",
        ]
    )

    models_dispatched = {model for _, _, model in _RecordingLLMClient.calls}
    assert models_dispatched == {"supervisor-real-model", "worker-real-model"}
    supervisor_endpoints = {
        (base_url, api_key)
        for base_url, api_key, model in _RecordingLLMClient.calls
        if model == "supervisor-real-model"
    }
    worker_endpoints = {
        (base_url, api_key)
        for base_url, api_key, model in _RecordingLLMClient.calls
        if model == "worker-real-model"
    }
    assert supervisor_endpoints == {("http://supervisor-real-endpoint", "supervisor-key")}
    assert worker_endpoints == {("http://worker-bridge", "worker-key")}


@pytest.mark.parametrize("flag", [False, True])
def test_brief_reports_picks_the_supervisor_prompt_for_a_fresh_session(tmp_path, flag):
    from agency.tandem_harness.tandem_loop import SUPERVISOR_SYSTEM, SUPERVISOR_SYSTEM_BRIEF

    argv = ["-p", "x", "--supervisor-model", "s", "--worker-model", "w", "--worker-history-turns", "0"]
    argv += ["--session-dir", str(tmp_path)]
    args = _build_arg_parser().parse_args(argv + (["--brief-reports"] if flag else []))
    _, messages = cli._resolve_session(args)
    assert messages[0]["content"] == (SUPERVISOR_SYSTEM_BRIEF if flag else SUPERVISOR_SYSTEM)
    assert SUPERVISOR_SYSTEM_BRIEF != SUPERVISOR_SYSTEM


def test_workflow_prompt_adds_the_locate_step(tmp_path):
    from agency.tandem_harness.tandem_loop import SUPERVISOR_SYSTEM, SUPERVISOR_SYSTEM_WORKFLOW

    argv = ["-p", "x", "--supervisor-model", "s", "--worker-model", "w", "--worker-history-turns", "0"]
    args = _build_arg_parser().parse_args(argv + ["--session-dir", str(tmp_path), "--workflow-prompt"])
    _, messages = cli._resolve_session(args)
    assert messages[0]["content"] == SUPERVISOR_SYSTEM_WORKFLOW
    assert SUPERVISOR_SYSTEM_WORKFLOW.startswith(SUPERVISOR_SYSTEM.split("Examples:")[0][:200])
    assert "first ask smart_tool to locate" in SUPERVISOR_SYSTEM_WORKFLOW


@pytest.mark.parametrize("mode", ["prompt", "list", "list_cond"])
def test_batch_mode_picks_its_supervisor_prompt(tmp_path, mode):
    from agency.tandem_harness.tandem_loop import BATCH_MODES

    argv = ["-p", "x", "--supervisor-model", "s", "--worker-model", "w", "--worker-history-turns", "0"]
    args = _build_arg_parser().parse_args(argv + ["--session-dir", str(tmp_path), "--batch-mode", mode])
    assert cli._resolve_session(args)[1][0]["content"] == BATCH_MODES[mode]
