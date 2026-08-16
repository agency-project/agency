"""Focused tests for the production Codex CLI harness adapter.

The orchestration tier uses captured Codex 0.140.0 JSONL and a mocked process
boundary.  A final test invokes the installed CLI with the generated config so
strict-config drift is caught without requiring a paid model call.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tomllib
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agency.agconfig import agConfig
from agency.agcontext import agcontext
from agency.agdata import agdata, agerror
from agency.agharness import HarnessMessages
from agency.agharness_internal.agharness_backends import codex as codex_module
from agency.agharness_internal.agharness_backends.codex import (
    _CodexBackend,
    _parse_codex_jsonl,
    codex_available,
)
from agency.agskill import agskill


_FIXTURES = Path(__file__).parents[1] / "fixtures" / "codex"
_REAL_CODEX = shutil.which("codex")
_VERSION = "0.140.0"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text()


def _success_events(
    text: str = "ok",
    session_id: str = "thread-1",
    *,
    input_tokens: int = 1,
    output_tokens: int = 2,
) -> str:
    return "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": session_id}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": text},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": 1,
                        "output_tokens": output_tokens,
                        "reasoning_output_tokens": 1,
                    },
                }
            ),
        ]
    )


def _rollout_blob(
    session_id: str,
    suffix: str = "",
    *,
    cli_version: str = _VERSION,
    cwd: str = "/workspace",
) -> bytes:
    meta = {
        "timestamp": "2026-08-15T00:00:00Z",
        "type": "session_meta",
        "payload": {
            "id": session_id,
            "cwd": cwd,
            "cli_version": cli_version,
        },
    }
    return (json.dumps(meta) + "\n" + suffix).encode()


def _install_rollout(config_home: str, session_id: str, blob: bytes | None = None) -> Path:
    path = (
        Path(config_home)
        / "sessions"
        / "2026"
        / "08"
        / "15"
        / f"rollout-2026-08-15T00-00-00-{session_id}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob or _rollout_blob(session_id))
    return path


def _make_agent(*, with_sandbox: bool = True):
    ag = MagicMock()
    ag.agname = "codex-test-agent"
    ag.agconfig = agConfig()
    ag.llm.backend.model = "test-model"
    ag._harness_sessions = {}
    if with_sandbox:
        ag.sandbox = MagicMock()
        ag.sandbox._backend.IMAGE_KIND = "not-a-container"
    else:
        ag.sandbox = None
    return ag


def _make_handle(stdout: str = "", stderr: str = "", rc: int = 0):
    handle = MagicMock()
    handle.wait.return_value = (stdout, stderr, rc)
    return handle


@pytest.fixture(autouse=True)
def _stable_binary_and_version(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(codex_module, "_detect_codex_version", lambda *_args: _VERSION)


@contextmanager
def _runtime(
    handles,
    *,
    collected=None,
    transcript=None,
    on_launch=None,
    mcp_start_error: Exception | None = None,
):
    if not isinstance(handles, list):
        handles = [handles]
    gateway = MagicMock(base_url="http://127.0.0.1:18080")
    terminus = MagicMock()
    terminus.transcript_for_token.return_value = transcript
    profiler = MagicMock()
    mcp = MagicMock()
    mcp.start.return_value = "http://127.0.0.1:18081"
    mcp.ensure_uds_started.return_value = "/tmp/fake-agency-mcp.sock"
    if mcp_start_error is not None:
        mcp.start.side_effect = mcp_start_error
    if isinstance(collected, list):
        mcp.collected_output.side_effect = collected
    else:
        mcp.collected_output.return_value = collected or {}

    launches = []

    def launch(argv, envp, *, cwd, policy, ag, sandbox=None, stdin=None):
        index = len(launches)
        launches.append(
            {
                "argv": list(argv),
                "envp": dict(envp),
                "cwd": cwd,
                "sandbox": sandbox,
                "stdin": stdin,
            }
        )
        if on_launch is not None:
            on_launch(index, argv, envp, stdin)
        return handles[index]

    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway", return_value=gateway),
        patch(
            "agency.agharness_internal.agllm_terminus.get_shared_terminus",
            return_value=terminus,
        ),
        patch(
            "agency.agharness_internal.agprof_ingest.get_shared_profiler_ingest",
            return_value=profiler,
        ),
        patch("agency.agharness_internal.agmcp_server.get_shared_mcp_server", return_value=mcp),
        patch("agency.agharness_internal.agproxy_ptrace.agProxyPtrace") as ptrace_cls,
        patch("agency.agharness_internal.agproxy_ptrace.wire_to_sandbox") as wire,
    ):
        ptrace_cls.return_value.launch.side_effect = launch
        yield {
            "gateway": gateway,
            "terminus": terminus,
            "profiler": profiler,
            "mcp": mcp,
            "launches": launches,
            "wire": wire,
        }


def test_codex_available_reflects_which(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert codex_available() is False
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/codex")
    assert codex_available() is True


def test_execute_returns_agerror_when_binary_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    result, ctx, delta = backend.execute(_make_agent(), agcontext(), agdata(x=1), None, skill=skill)
    assert isinstance(result, agerror)
    assert "not found" in result.error
    assert ctx.messages == []
    assert delta[0]["role"] == "system"


def test_execute_fails_closed_for_chroot_before_starting_lifecycle():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    ag.sandbox._backend.IMAGE_KIND = "chroot"

    with (
        patch("agency.agharness_internal.agproxy_llm.get_shared_gateway") as gateway,
        patch("agency.agharness.run_harness_cli") as run_cli,
    ):
        result, ctx, delta = backend.execute(ag, agcontext(), agdata(x=1), None, skill=skill)

    assert isinstance(result, agerror)
    assert "does not yet support chroot-backed sandboxes safely" in result.error
    assert ctx.messages == []
    assert delta[0]["role"] == "system"
    gateway.assert_not_called()
    run_cli.assert_not_called()


def test_execute_uses_real_fresh_argv_final_file_and_usage():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="developer policy")
    ag = _make_agent()
    previous = agcontext(
        messages=[{"role": "assistant", "content": "portable history"}], revision=3
    )

    def on_launch(_index, argv, _envp, _stdin):
        final_path = Path(argv[argv.index("--output-last-message") + 1])
        final_path.write_text("from final-message file")

    handle = _make_handle(_fixture("success-0.140.0.jsonl"))
    with _runtime(handle, on_launch=on_launch) as runtime:
        result, ctx, _delta = backend.execute(ag, previous, agdata(task="go"), None, skill=skill)

    assert result.result == "from final-message file"
    assert ctx.total_input_tokens == 37
    # reasoning_output_tokens is already represented by output_tokens.
    assert ctx.total_output_tokens == 11
    launch = runtime["launches"][0]
    assert launch["argv"][:4] == ["/usr/bin/codex", "-C", "/workspace", "exec"]
    assert launch["argv"][-1] == "-"
    assert "--json" in launch["argv"]
    assert "--strict-config" in launch["argv"]
    assert "--skip-git-repo-check" in launch["argv"]
    assert "--ignore-rules" in launch["argv"]
    assert launch["cwd"] == "/workspace"
    assert launch["sandbox"] is ag.sandbox
    assert "[PREVIOUS CONTEXT]" in launch["stdin"]
    assert "portable history" in launch["stdin"]
    assert "developer policy" not in launch["stdin"]
    runtime["wire"].assert_called_once()


def test_generated_config_is_isolated_authenticated_mcp_enabled_and_secret_safe():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt='policy with "quotes"\nand newlines')
    ag = _make_agent()
    captured = {}

    def on_launch(_index, argv, envp, stdin):
        content = (Path(envp["CODEX_HOME"]) / "config.toml").read_text()
        captured.update(content=content, config=tomllib.loads(content), envp=dict(envp))
        captured["argv"] = list(argv)
        captured["stdin"] = stdin

    with _runtime(_make_handle(_success_events()), on_launch=on_launch):
        result, _, _ = backend.execute(ag, agcontext(), agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    config = captured["config"]
    assert config["model"] == "test-model"
    assert config["model_provider"] == "agency-proxy"
    assert config["model_reasoning_effort"] == "none"
    assert config["developer_instructions"] == 'policy with "quotes"\nand newlines'
    assert config["approval_policy"] == "never"
    assert config["sandbox_mode"] == "danger-full-access"
    assert config["web_search"] == "disabled"
    assert config["features"]["apps"] is False
    assert config["features"]["plugins"] is False
    assert config["features"]["multi_agent"] is False
    assert config["features"]["shell_snapshot"] is False, (
        "Codex snapshots its parent environment before applying the shell policy; "
        "enabling snapshots can resurrect the provider and MCP bearer variables"
    )
    assert config["tools"]["web_search"] is False
    assert config["model_providers"]["agency-proxy"]["wire_api"] == "responses"
    assert config["mcp_servers"]["agency"]["required"] is True
    assert config["mcp_servers"]["agency"]["bearer_token_env_var"] == "AGENCY_MCP_TOKEN"
    assert config["shell_environment_policy"]["inherit"] == "none"
    proxy_token = captured["envp"]["AGENCY_PROXY_API_KEY"]
    assert proxy_token == captured["envp"]["AGENCY_MCP_TOKEN"]
    assert proxy_token not in captured["content"]
    assert "OPENAI_API_KEY" not in captured["envp"]
    assert "CODEX_API_KEY" not in captured["envp"]
    assert "policy with" not in captured["stdin"]


def test_terminal_failure_is_error_even_with_partial_agent_message():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    previous = agcontext()
    with _runtime(_make_handle(_fixture("turn-failed-0.140.0.jsonl"))):
        result, ctx, _ = backend.execute(
            _make_agent(), previous, agdata(task="go"), None, skill=skill
        )
    assert isinstance(result, agerror)
    assert "provider stream disconnected" in result.error
    assert ctx.messages == []


def test_malformed_jsonl_is_explicit_protocol_failure():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    with _runtime(_make_handle(_fixture("malformed-0.140.0.jsonl"))):
        result, _, _ = backend.execute(
            _make_agent(), agcontext(), agdata(task="go"), None, skill=skill
        )
    assert isinstance(result, agerror)
    assert "malformed Codex JSONL" in result.error


def test_missing_turn_completed_is_failure():
    stdout = json.dumps(
        {"type": "item.completed", "item": {"type": "agent_message", "text": "partial"}}
    )
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    with _runtime(_make_handle(stdout)):
        result, _, _ = backend.execute(
            _make_agent(), agcontext(), agdata(task="go"), None, skill=skill
        )
    assert isinstance(result, agerror)
    assert "without a turn.completed" in result.error


def test_timeout_is_distinguished_from_normal_exit():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    with _runtime(_make_handle("partial", "killed", -1)):
        result, _, _ = backend.execute(
            _make_agent(), agcontext(), agdata(task="go"), None, skill=skill
        )
    assert isinstance(result, agerror)
    assert "timed out" in result.error


def test_process_failure_diagnostic_keeps_stderr_and_jsonl_error():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    stdout = json.dumps(
        {"type": "turn.failed", "error": {"message": "upstream rejected parameter"}}
    )
    with _runtime(_make_handle(stdout, "startup warning", 1)):
        result, _, _ = backend.execute(
            _make_agent(), agcontext(), agdata(task="go"), None, skill=skill
        )
    assert isinstance(result, agerror)
    assert "startup warning" in result.error
    assert "upstream rejected parameter" in result.error


def test_session_creation_captures_portable_rollout_before_cleanup():
    session_id = "0198ab70-1234-7000-8000-000000000010"
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    captured_home = None

    def on_launch(_index, _argv, envp, _stdin):
        nonlocal captured_home
        captured_home = envp["CODEX_HOME"]
        _install_rollout(captured_home, session_id)

    with _runtime(_make_handle(_success_events(session_id=session_id)), on_launch=on_launch):
        result, _, _ = backend.execute(
            ag, agcontext(revision=7), agdata(task="go"), None, skill=skill
        )

    assert not isinstance(result, agerror)
    record = ag._harness_sessions["codex"]
    assert record["session_id"] == session_id
    assert record["rollout_path"].startswith("sessions/")
    assert base64.b64decode(record["blob_b64"]) == _rollout_blob(session_id)
    assert record["agcontext_revision"] == 8
    assert record["codex_version"] == _VERSION
    assert record["rollout_cli_version"] == _VERSION
    assert captured_home is not None and not Path(captured_home).exists()


def test_valid_rollout_is_restored_and_real_resume_subcommand_omits_history():
    session_id = "0198ab70-1234-7000-8000-000000000011"
    relative = f"sessions/2026/08/15/rollout-2026-08-15-{session_id}.jsonl"
    blob = _rollout_blob(session_id)
    ag = _make_agent()
    ag._harness_sessions = {
        "codex": {
            "session_id": session_id,
            "rollout_path": relative,
            "blob_b64": base64.b64encode(blob).decode(),
            "agcontext_revision": 3,
            "codex_version": _VERSION,
            "rollout_cli_version": _VERSION,
        }
    }
    previous = agcontext(
        messages=[{"role": "assistant", "content": "portable history"}], revision=3
    )

    def on_launch(_index, _argv, envp, _stdin):
        assert (Path(envp["CODEX_HOME"]) / relative).read_bytes() == blob

    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    with _runtime(
        _make_handle(_success_events(session_id=session_id)), on_launch=on_launch
    ) as runtime:
        result, _, _ = backend.execute(ag, previous, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    launch = runtime["launches"][0]
    assert launch["argv"][:6] == [
        "/usr/bin/codex",
        "-C",
        "/workspace",
        "exec",
        "resume",
        session_id,
    ]
    assert "[PREVIOUS CONTEXT]" not in launch["stdin"]
    assert "portable history" not in launch["stdin"]


@pytest.mark.parametrize(
    ("record_revision", "record_version"),
    [(2, _VERSION), (3, "0.139.0")],
)
def test_stale_revision_or_version_falls_back_to_portable_history(record_revision, record_version):
    session_id = "0198ab70-1234-7000-8000-000000000012"
    ag = _make_agent()
    ag._harness_sessions = {
        "codex": {
            "session_id": session_id,
            "rollout_path": f"sessions/x-{session_id}.jsonl",
            "blob_b64": base64.b64encode(_rollout_blob(session_id)).decode(),
            "agcontext_revision": record_revision,
            "codex_version": record_version,
            "rollout_cli_version": _VERSION,
        }
    }
    previous = agcontext(
        messages=[{"role": "assistant", "content": "portable history"}], revision=3
    )
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    with _runtime(_make_handle(_success_events())) as runtime:
        result, _, _ = backend.execute(ag, previous, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    launch = runtime["launches"][0]
    assert "resume" not in launch["argv"]
    assert "[PREVIOUS CONTEXT]" in launch["stdin"]
    assert "portable history" in launch["stdin"]


@pytest.mark.parametrize(
    ("stored_session_id", "blob"),
    [
        (
            "--last",
            _rollout_blob("0198ab70-1234-7000-8000-000000000020"),
        ),
        (
            "0198ab70-1234-7000-8000-000000000021",
            _rollout_blob(
                "0198ab70-1234-7000-8000-000000000021",
                cli_version="different-rollout-build",
            ),
        ),
        (
            "0198ab70-1234-7000-8000-000000000022",
            _rollout_blob(
                "0198ab70-1234-7000-8000-000000000022",
                cwd="/different-workspace",
            ),
        ),
        (
            "0198ab70-1234-7000-8000-000000000023",
            b'{"type":"event","message":"0198ab70-1234-7000-8000-000000000023"}\n',
        ),
    ],
)
def test_invalid_or_mismatched_rollout_metadata_falls_back_safely(stored_session_id, blob):
    ag = _make_agent()
    ag._harness_sessions = {
        "codex": {
            "session_id": stored_session_id,
            "rollout_path": "sessions/2026/08/15/rollout.jsonl",
            "blob_b64": base64.b64encode(blob).decode(),
            "agcontext_revision": 3,
            "codex_version": _VERSION,
            "rollout_cli_version": _VERSION,
        }
    }
    previous = agcontext(
        messages=[{"role": "assistant", "content": "portable history"}], revision=3
    )
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")

    with _runtime(_make_handle(_success_events())) as runtime:
        result, _, _ = backend.execute(ag, previous, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert "resume" not in runtime["launches"][0]["argv"]
    assert "portable history" in runtime["launches"][0]["stdin"]


def test_codex_rejected_resume_retries_fresh_with_portable_history():
    session_id = "0198ab70-1234-7000-8000-000000000013"
    relative = f"sessions/2026/08/15/rollout-{session_id}.jsonl"
    blob = _rollout_blob(session_id)
    ag = _make_agent()
    ag._harness_sessions = {
        "codex": {
            "session_id": session_id,
            "rollout_path": relative,
            "blob_b64": base64.b64encode(blob).decode(),
            "agcontext_revision": 3,
            "codex_version": _VERSION,
            "rollout_cli_version": _VERSION,
        }
    }
    previous = agcontext(
        messages=[{"role": "assistant", "content": "portable history"}], revision=3
    )
    handles = [
        _make_handle(stderr="No saved session found for thread ID", rc=1),
        _make_handle(_success_events(session_id="fresh-thread")),
    ]
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    with _runtime(handles) as runtime:
        result, _, _ = backend.execute(ag, previous, agdata(task="go"), None, skill=skill)

    assert not isinstance(result, agerror)
    assert "resume" in runtime["launches"][0]["argv"]
    assert "resume" not in runtime["launches"][1]["argv"]
    assert "portable history" in runtime["launches"][1]["stdin"]


def test_structured_output_comes_from_common_mcp_submit_output_path():
    backend = _CodexBackend(agConfig())
    skill = agskill(
        name="s",
        system_prompt="do the thing",
        output_schema=agdata(answer=str, count=int),
    )
    collected = {"answer": "done", "count": 2}
    with _runtime(_make_handle(_success_events()), collected=collected) as runtime:
        result, _, _ = backend.execute(
            _make_agent(), agcontext(), agdata(task="go"), None, skill=skill
        )

    assert result == agdata(answer="done", count=2)
    assert "submit_output" in runtime["launches"][0]["stdin"]
    runtime["mcp"].register.assert_called_once()
    runtime["mcp"].unregister.assert_called_once()


def test_incomplete_structured_output_retries_same_native_thread():
    session_id = "0198ab70-1234-7000-8000-000000000014"
    backend = _CodexBackend(agConfig())
    skill = agskill(
        name="s",
        system_prompt="do the thing",
        output_schema=agdata(answer=str, count=int),
        max_output_schema_retries=1,
    )

    def on_launch(index, _argv, envp, _stdin):
        if index == 0:
            _install_rollout(envp["CODEX_HOME"], session_id)

    handles = [
        _make_handle(_success_events("first", session_id)),
        _make_handle(_success_events("second", session_id)),
    ]
    with _runtime(
        handles,
        collected=[{"answer": "done"}, {"answer": "done", "count": 2}],
        on_launch=on_launch,
    ) as runtime:
        result, _, _ = backend.execute(
            _make_agent(), agcontext(), agdata(task="go"), None, skill=skill
        )

    assert result == agdata(answer="done", count=2)
    assert len(runtime["launches"]) == 2
    assert runtime["launches"][1]["argv"][4:6] == ["resume", session_id]
    assert "Still missing: ['count']" in runtime["launches"][1]["stdin"]


def test_failed_structured_output_retry_keeps_completed_attempt_usage():
    session_id = "0198ab70-1234-7000-8000-000000000015"
    backend = _CodexBackend(agConfig())
    skill = agskill(
        name="s",
        system_prompt="do the thing",
        output_schema=agdata(answer=str, count=int),
        max_output_schema_retries=1,
    )
    previous = agcontext(total_input_tokens=10, total_output_tokens=20)

    def on_launch(index, _argv, envp, _stdin):
        if index == 0:
            _install_rollout(envp["CODEX_HOME"], session_id)

    handles = [
        _make_handle(
            _success_events(
                "first",
                session_id,
                input_tokens=5,
                output_tokens=6,
            )
        ),
        _make_handle(stderr="provider unavailable", rc=1),
    ]
    with _runtime(handles, collected={"answer": "done"}, on_launch=on_launch):
        result, ctx, _ = backend.execute(
            _make_agent(), previous, agdata(task="go"), None, skill=skill
        )

    assert isinstance(result, agerror)
    assert ctx.total_input_tokens == 15
    assert ctx.total_output_tokens == 26


def test_terminus_transcript_is_authoritative_history():
    transcript = [
        {"role": "system", "content": "Codex base instructions"},
        {"role": "system", "content": "Agency developer instructions"},
        {"role": "user", "content": "wire user"},
        {"role": "assistant", "content": "wire assistant"},
        {"role": "tool", "content": "tool result"},
    ]
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    previous = agcontext()
    with _runtime(_make_handle(_success_events()), transcript=transcript):
        result, ctx, delta = backend.execute(
            _make_agent(), previous, agdata(task="go"), None, skill=skill
        )
    assert not isinstance(result, agerror)
    assert ctx.messages == transcript[2:]
    assert delta[1:] == transcript[2:]


def test_cleanup_after_partial_service_setup_failure():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    with _runtime(
        _make_handle(_success_events()), mcp_start_error=RuntimeError("MCP failed to bind")
    ) as runtime:
        result, _, _ = backend.execute(
            _make_agent(), agcontext(), agdata(task="go"), None, skill=skill
        )
    assert isinstance(result, agerror)
    assert "MCP failed to bind" in result.error
    runtime["gateway"].unregister.assert_called_once()
    runtime["profiler"].unregister.assert_called_once()
    runtime["mcp"].unregister.assert_called_once()


def test_config_cleanup_failure_does_not_replace_successful_result(capsys):
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")

    from agency import agharness

    real_cleanup = agharness.cleanup_config_home

    def cleanup_then_fail(path):
        real_cleanup(path)
        raise RuntimeError("cleanup transport failed")

    with (
        _runtime(_make_handle(_success_events())),
        patch("agency.agharness.cleanup_config_home", side_effect=cleanup_then_fail),
    ):
        result, _, _ = backend.execute(
            _make_agent(), agcontext(), agdata(task="go"), None, skill=skill
        )

    assert not isinstance(result, agerror)
    assert result.result == "ok"
    assert "failed to clean up Codex config home" in capsys.readouterr().out


def test_container_topology_cleans_services_and_relay_after_config_failure():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    ag.sandbox._backend.IMAGE_KIND = "container"
    ag.sandbox.exec.return_value = ("", 1)
    ag.sandbox.write_file.side_effect = RuntimeError("container config write failed")
    relay_process = MagicMock()

    with (
        _runtime(_make_handle(_success_events())) as runtime,
        patch(
            "agency.agharness.resolve_harness_binary_in_container",
            return_value="/opt/agency_harness_bin/codex",
        ),
        patch(
            "agency.agharness_internal.agproxy_llm_in_container.ensure_agproxy_llm_in_container",
            return_value="http://127.0.0.1:18082",
        ),
        patch(
            "agency.agharness_internal.agproxy_ptrace_internal._in_container_launcher."
            "start_tcp_relay",
            return_value=(relay_process, 18083),
        ),
        patch(
            "agency.agharness_internal.agproxy_ptrace_internal._in_container_launcher."
            "stop_tcp_relay"
        ) as stop_relay,
        patch(
            "agency.agharness.materialize_config_home_in_container",
            return_value="/tmp/agharness-codex-container",
        ),
        patch("agency.agharness.cleanup_config_home_in_container") as cleanup_home,
    ):
        result, _, _ = backend.execute(ag, agcontext(), agdata(task="go"), None, skill=skill)

    assert isinstance(result, agerror)
    assert "container config write failed" in result.error
    runtime["gateway"].register.assert_not_called()
    runtime["terminus"].register.assert_called_once()
    runtime["terminus"].unregister.assert_called_once()
    runtime["profiler"].unregister.assert_called_once()
    runtime["mcp"].unregister.assert_called_once()
    stop_relay.assert_called_once_with(relay_process)
    cleanup_home.assert_called_once_with(ag.sandbox, "/tmp/agharness-codex-container")
    assert runtime["launches"] == []


def test_unsupported_contract_inputs_fail_clearly_before_launch():
    backend = _CodexBackend(agConfig())
    skill = agskill(name="s", system_prompt="do the thing")
    ag = _make_agent()
    result, _, _ = backend.execute(ag, agcontext(), agdata(task="go"), 5, skill=skill)
    assert isinstance(result, agerror)
    assert "max_steps" in result.error

    canonical = HarnessMessages(
        system_instructions="do the thing",
        previous_context=(),
        current_user_input="inspect image",
        attachments=({"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},),
    )
    result, _, _ = backend.execute(
        ag,
        agcontext(),
        agdata(task="go"),
        None,
        skill=skill,
        canonical_input=canonical,
    )
    assert isinstance(result, agerror)
    assert "attachments" in result.error


def test_parser_fixture_matches_captured_success_shape():
    summary = _parse_codex_jsonl(_fixture("success-0.140.0.jsonl"))
    assert summary.completed is True
    assert summary.error is None
    assert summary.final_text == "workspace task complete"
    assert summary.input_tokens == 37
    assert summary.output_tokens == 11


@pytest.mark.skipif(_REAL_CODEX is None, reason="Codex CLI not installed")
def test_real_codex_0140_accepts_generated_strict_config(tmp_path):
    """Codex 0.140 accepts the hardened config, including disabled snapshots."""
    backend = _CodexBackend(agConfig())
    backend._write_codex_config(
        tmp_path,
        "http://127.0.0.1:9",
        "gpt-5.4",
        developer_instructions="return briefly",
        mcp_base_url="http://127.0.0.1:9",
        sandbox_mode="workspace-write",
        workspace=os.getcwd(),
    )
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "CODEX_HOME": str(tmp_path),
        "AGENCY_PROXY_API_KEY": "not-a-real-key",
        "AGENCY_MCP_TOKEN": "not-a-real-key",
    }
    completed = subprocess.run(
        [
            _REAL_CODEX,
            "-C",
            os.getcwd(),
            "exec",
            "--json",
            "--strict-config",
            "--skip-git-repo-check",
            "-",
        ],
        input="Return the word ok.",
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )
    diagnostic = (completed.stdout + "\n" + completed.stderr).lower()
    assert "unknown configuration field" not in diagnostic
    assert "failed to load configuration" not in diagnostic
    assert "configuration error" not in diagnostic
    assert "required mcp servers failed to initialize" in diagnostic
