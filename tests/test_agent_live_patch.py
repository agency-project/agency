# agent.patch_history() landing in a skill run that is active, and the synchronous
# agent.pause() it is meant to be used with (see docs/Message_Patches.md).

from __future__ import annotations

import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx2
import pytest

from agency import agent
from agency.agdata import agdata, agerror
from agency.agpolicy import agpolicy
from agency.agskill import agskill
from agency.configs.agconfig import (
    agconfig,
    agentconfig,
    dataloggerconfig,
    hostserverconfig,
    llmconfig,
)
from agency.engine import AgentEngine
from agency.engine.host_servers.host_server_manager import HostServerManager
from agency.harness.protocol import ATTEMPT_TOKEN_HEADER
from agency.llm.message_store import MessageBlock
from agency.llm.usage_tracker import LlmUsageTracker
from agency.observability.agdatalogger import agDataLogger


def _agent(tmp_path) -> agent:
    config = agconfig(
        agentconfig(log_dir=str(tmp_path)),
        llmconfig(api_key="test", model="m"),
    )
    sandbox = MagicMock()
    sandbox._lock = threading.RLock()
    return agent(sandbox=sandbox, agconfig=config)


def _daemon(ack):
    """A cached daemon handle whose pause RPC answers *ack*."""
    daemon = MagicMock()
    client = daemon.client.return_value.__enter__.return_value
    client.pause_harness.return_value = ack
    return daemon, client


def _text(role, text):
    return {"role": role, "blocks": [{"type": "text", "index": 0, "text": text}]}


def _block(text):
    return {"type": "text", "index": 0, "text": text}


def patch(store, role, old, new_blocks):
    """Add one patch: dicts in the agency block format in, a hash-to-hashes link out."""
    with store:
        return store.add_patch(
            MessageBlock(role, old).hash,
            [store.add_block(MessageBlock(role, b)) for b in new_blocks],
        )


def patch_history(ag, role, old, new_blocks):
    return ag.patch_history(MessageBlock(role, old), [MessageBlock(role, b) for b in new_blocks])


# -- pause is synchronous ------------------------------------------------------------


def test_pause_waits_for_the_harness(tmp_path, monkeypatch):
    ag = _agent(tmp_path)
    daemon, client = _daemon({"ok": True, "parked": True})
    monkeypatch.setattr(ag, "_daemon_handle", lambda: daemon)

    ag.pause(timeout_s=4)

    client.pause_harness.assert_called_once_with(wait_timeout_s=4)
    assert ag.is_paused()


def test_pause_raises_if_the_harness_does_not_stop_but_stays_requested(tmp_path, monkeypatch):
    ag = _agent(tmp_path)
    daemon, client = _daemon({"ok": True, "parked": False})
    monkeypatch.setattr(ag, "_daemon_handle", lambda: daemon)

    with pytest.raises(TimeoutError, match="not stopped"):
        ag.pause(timeout_s=0.1)

    assert ag.is_paused()  # still requested: the harness will stop when it can
    ag.resume()
    assert not ag.is_paused()
    client.resume_harness.assert_called_once()


def test_pause_with_no_harness_process_returns_at_once(tmp_path):
    ag = _agent(tmp_path)
    ag.pause()
    assert ag.is_paused()


def test_async_pause_only_requests_the_stop_and_does_not_wait(tmp_path, monkeypatch):
    ag = _agent(tmp_path)
    daemon, client = _daemon({"ok": True})
    monkeypatch.setattr(ag, "_daemon_handle", lambda: daemon)

    ag.pause(async_=True)

    client.pause_harness.assert_called_once_with()  # no wait_timeout_s
    assert ag.is_paused()


def test_async_pause_never_raises_even_if_the_harness_would_not_stop(tmp_path, monkeypatch):
    ag = _agent(tmp_path)
    daemon, client = _daemon({"ok": True})
    monkeypatch.setattr(ag, "_daemon_handle", lambda: daemon)

    ag.pause(async_=True, timeout_s=0.0)  # timeout_s is unused in async mode

    assert ag.is_paused()


def test_an_unreachable_daemon_is_a_warning_not_a_failure(tmp_path, monkeypatch, capsys):
    ag = _agent(tmp_path)
    daemon, client = _daemon(None)
    client.pause_harness.side_effect = ConnectionError("gone")
    monkeypatch.setattr(ag, "_daemon_handle", lambda: daemon)
    ag.pause()
    assert ag.is_paused()
    assert "pause_harness() failed" in capsys.readouterr().out


# -- a patch made while a skill run is active -------------------------------------------


def _blocking_execute(monkeypatch, *, fail=False):
    """Replace AgentEngine.execute with one that hashes a message into the agcontext
    it is given, then waits until the test releases it."""
    started, release, seen = threading.Event(), threading.Event(), {}

    def execute(self, *, context, **_kwargs):
        context.set_transcript([_text("user", "task"), _text("user", "secret")])
        seen["context"] = context
        started.set()
        assert release.wait(10)
        seen["patched_when_released"] = dict(context.message_store.message_patch_lookup)
        seen["messages"] = [m["blocks"][0]["text"] for m in context.get_resolved_transcript()]
        return agerror("boom") if fail else agdata(ok=True)

    monkeypatch.setattr(AgentEngine, "execute", execute)
    return started, release, seen


def test_a_patch_made_during_a_skill_run_lands_in_its_agcontext(monkeypatch, tmp_path):
    started, release, seen = _blocking_execute(monkeypatch)
    ag = _agent(tmp_path)
    handle = ag.run(agskill("s", ""), agdata())
    assert started.wait(10)

    key = patch_history(ag, "user", _block("secret"), [_block("[redacted]")])
    release.set()
    assert handle.ok

    # The running skill run saw it in its own agcontext, at once ...
    assert key in seen["patched_when_released"]
    assert seen["messages"] == ["task", "[redacted]"]
    # ... and the agcontext it published has it too.
    ag.context.resolve_prev_dependencies()
    assert key in ag.context.message_store.message_patch_lookup


def test_a_patch_made_during_a_failed_skill_run_is_kept(monkeypatch, tmp_path):
    started, release, seen = _blocking_execute(monkeypatch, fail=True)
    ag = _agent(tmp_path)
    handle = ag.run(agskill("s", ""), agdata())
    assert started.wait(10)

    key = patch_history(ag, "user", _block("secret"), [_block("[redacted]")])
    release.set()
    assert "boom" in handle.error

    assert key in seen["patched_when_released"]  # the failed skill run did see it
    ag.context.resolve_prev_dependencies()
    assert key in ag.context.message_store.message_patch_lookup  # and it is not lost


def test_a_patch_made_with_no_skill_run_is_only_queued(monkeypatch, tmp_path):
    ag = _agent(tmp_path)
    key = patch_history(ag, "user", _block("secret"), [_block("[redacted]")])
    ag.context.resolve_prev_dependencies()
    assert ag.context.message_store.message_patch_lookup[key] == [
        ag.context.message_store.add_block(MessageBlock("user", _block("[redacted]")))
    ]


def test_remove_and_clear_also_reach_the_active_skill_run(monkeypatch, tmp_path):
    started, release, seen = _blocking_execute(monkeypatch)
    ag = _agent(tmp_path)
    handle = ag.run(agskill("s", ""), agdata())
    assert started.wait(10)

    key = patch_history(ag, "user", _block("secret"), [_block("[redacted]")])
    running = seen["context"]
    assert key in running.message_store.message_patch_lookup
    ag.remove_history_patch(key)
    assert running.message_store.message_patch_lookup == {}
    patch_history(ag, "user", _block("task"), [_block("job")])
    ag.clear_history_patches()
    assert running.message_store.message_patch_lookup == {}
    release.set()
    assert handle.ok


# -- the whole host path ---------------------------------------------------------------


class _Backend:
    model = "test-model"

    def __init__(self):
        self.requests = []

    def dispatch(self, request):
        self.requests.append(request)
        return {
            "message": {
                "role": "assistant",
                "blocks": [{"type": "text", "index": 0, "text": "ok"}],
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "stop_reason": "stop",
        }


def test_message_patch_lookup_reaches_the_real_host_server_and_the_skill_run_sets_message_hash_list(
    tmp_path,
):
    """agcontext -> HostServerManager(context=...) -> LlmHandlerServer over the
    host UDS: message_patch_lookup is applied to what the backend receives, and
    the skill run's final hash list is set on the same agcontext."""
    from agency.agcontext import agcontext

    uds_path = f"/tmp/patch_e2e_{uuid.uuid4().hex[:8]}.sock"
    cfg = agconfig(
        llmconfig(model="test-model", base_url="http://x/v1"),
        hostserverconfig(uds_path=uds_path),
        dataloggerconfig(db_path=str(tmp_path / "agent.db")),
    )
    fake_agent = SimpleNamespace(
        agname="agent-1",
        agconfig=cfg,
        harness="claude_code",
        data_logger=agDataLogger(cfg),
        llm_usage_tracker=LlmUsageTracker(),
    )
    sandbox = SimpleNamespace(
        ensure_gpu_acquired=lambda _agname, *, is_cancelled: None,
        current_gpu_ids=lambda: None,
    )
    context = agcontext()
    patch(
        context.message_store,
        "user",
        {"type": "text", "index": 0, "text": "secret"},
        [{"type": "text", "index": 0, "text": "[redacted]"}],
    )
    manager = HostServerManager(
        fake_agent,
        sandbox,
        agskill(name="s", prompt="p", policy=agpolicy()),
        SimpleNamespace(),
        context=context,
    )
    backend = _Backend()
    manager.llm_handler_server._backend = backend
    try:
        manager.start()
        manager.bind_attempt_token("tok")
        with httpx2.Client(
            transport=httpx2.HTTPTransport(uds=uds_path), base_url="http://localhost"
        ) as client:
            response = client.post(
                "/llm/dispatch",
                json={"messages": [_text("user", "task"), _text("user", "secret")]},
                headers={ATTEMPT_TOKEN_HEADER: "tok"},
            )
            assert response.status_code == 200
        context.message_hash_list = manager.llm_handler_server.get_main_history("task")
    finally:
        manager.clear_attempt_token("tok")
        manager.stop()

    assert [m["blocks"][0]["text"] for m in backend.requests[0]["messages"]] == [
        "task",
        "[redacted]",
    ]
    assert [m["blocks"][0]["text"] for m in context.get_resolved_transcript()][:2] == [
        "task",
        "[redacted]",
    ]
