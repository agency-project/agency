"""Canonical public lifecycle against real Docker harnesses and replayed LLMs.

CI runs this with normal pytest. On a Linux development host:

    pytest tests/test_golden_execution.py -k "native or claude_code" -v

No image build step is needed: the sandbox base is a fully-qualified registry
image (docker.io/library/python:3.12-slim) that both docker and podman pull
directly.

Alternatively, install the external CLIs on the host and supply an image with
their system runtimes through AGENCY_TEST_HARNESS_IMAGE.
AGENCY_TEST_EXTERNAL_HARNESSES=1 makes missing prerequisites fail locally too.
Profiler artifacts: artifacts/golden-execution/<harness>/agprof.trace.json,
summary.json, summary.md, profile_data.sqlite3, requests.json, and logs/.
Only model responses are synthetic; containers, daemons, PTYs, and controls are real.
Pause/resume assertions cover requested state, not proof that every PID is stopped.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agency import Agent, agconfig, agdata, agDataLogger, agprof, agrawstring, agskill
from agency.configs.agconfig import (
    agentconfig,
    dataloggerconfig,
    llmconfig,
    resourcesconfig,
    sandboxconfig,
)

from agency.engine.host_servers import llm_handler_server
from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.llm.mock import _MockBackend


# for_config() currently uses these strings; there is no harness enum/registry.
HARNESSES = ("native", "claude_code", "codex", "opencode", "grok")
INPUT = "golden invocation input"
QUEUED = "golden queued message"
REDIRECT = "golden redirect"
FUTURE = "golden future-only context"
LATE = "golden completed-execution redirect"
ANSWER = "GOLDEN_OK"
WAIT_SECONDS = 90
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "golden-execution"


CONTAINER_BACKEND = os.environ.get("AGENCY_TEST_CONTAINER_BACKEND", "docker")


@pytest.fixture(scope="module")
def golden_image():
    image = os.environ.get("AGENCY_TEST_HARNESS_IMAGE", "docker.io/library/python:3.12-slim")
    problem = None
    if not sys.platform.startswith("linux"):
        problem = "golden profiling and real external harnesses require Linux"
    else:
        try:
            subprocess.run([CONTAINER_BACKEND, "info"], capture_output=True, check=True, timeout=15)
            subprocess.run(
                [CONTAINER_BACKEND, "image", "inspect", image],
                capture_output=True,
                check=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            problem = (
                f"golden execution requires {CONTAINER_BACKEND} and the local image {image}: {exc}"
            )
    if problem:
        if os.environ.get("CI") or os.environ.get("AGENCY_TEST_EXTERNAL_HARNESSES") == "1":
            pytest.fail(problem)
        pytest.skip(problem)
    return image


@pytest.fixture
def golden_profile(harness, golden_image):
    # Profiler sessions are process-local. Worker directories also protect
    # duplicate cases when xdist's --dist=each runs the matrix on every worker.
    directory = ARTIFACT_ROOT / harness
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        directory /= worker
    with agprof.session(directory):
        yield directory


def _text(messages, role):
    parts = []
    for message in messages:
        if message.get("role") == role:
            parts.extend(
                block.get("text", "")
                for block in message.get("blocks", [])
                if block.get("type") == "text"
            )
    return "\n".join(parts)


def _write_replay(path):
    logger = agDataLogger(agconfig(dataloggerconfig(db_path=str(path))))
    logger.start()
    try:
        for index, answer in enumerate((ANSWER,) * 8):
            logger.record_final_transcript(
                f"golden-{index}",
                type="llm_block",
                payloads=[
                    {"type": "text", "index": 0, "text": answer},
                    {
                        "type": "metadata",
                        "index": 2**31 - 1,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                        "stop_reason": "stop",
                        "data": [],
                    },
                ],
            )
    finally:
        logger.stop()


class _GoldenReplay(_MockBackend):
    """A replay gate that can be released by an actual HTTP client disconnect.

    The production replay hook ignores on_client. Blocking that hook would
    prevent the gateway from joining its producer when Claude interrupts it.
    """

    def __init__(self, config):
        super().__init__(config)
        self.requests = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.interrupted = threading.Event()
        self.block_next = True

    def arm(self):
        self.entered.clear()
        self.release.clear()
        self.interrupted.clear()
        self.block_next = True

    def _gate(self, request, on_client=None):
        self.requests.append(copy.deepcopy(request))
        if not self.block_next:
            return False
        self.block_next = False

        def close():
            self.interrupted.set()
            self.release.set()

        if on_client is not None:
            on_client(SimpleNamespace(close=close))
        self.entered.set()
        assert self.release.wait(WAIT_SECONDS), "model response gate was not released"
        return self.interrupted.is_set()

    def dispatch_stream(self, request, on_client=None):
        if self._gate(request, on_client):
            return
        yield from super().dispatch_stream(request)

    def dispatch(self, request):
        self._gate(request)
        return super().dispatch(request)


@pytest.mark.timeout(420)
@pytest.mark.parametrize("harness", HARNESSES)
def test_golden_execution(harness, golden_image, golden_profile, monkeypatch):
    # Keep the span in the call: pytest does not throw failures into yield fixtures.
    with agprof.span(f"golden-execution:{harness}"):
        _exercise_golden_lifecycle(harness, golden_image, golden_profile, monkeypatch)


def _exercise_golden_lifecycle(harness, golden_image, directory, monkeypatch):
    replay = directory / "replay.sqlite3"
    # Re-running a case must not append exchanges from a previous execution.
    replay.unlink(missing_ok=True)
    _write_replay(replay)
    config = agconfig(
        agentconfig(harness=harness, log_dir=str(directory / "logs")),
        sandboxconfig(backend=CONTAINER_BACKEND, base_image=golden_image),
        resourcesconfig(idle_cpus=1, idle_memory="2g"),
        llmconfig(
            provider="mock",
            model="claude-sonnet-4-6",
            context_limit=200_000,
            replay_db_path=str(replay),
            timing_mode="instant",
        ),
    )
    backend = _GoldenReplay(config)
    ready = threading.Event()
    record_span = HostInteractionServer.record_span

    def observe_ready(server, name, start_ts, end_ts, *args, **kwargs):
        record_span(server, name, start_ts, end_ts, *args, **kwargs)
        if name == "harness:await_cli" and start_ts is not None and end_ts is None:
            ready.set()

    monkeypatch.setattr(HostInteractionServer, "record_span", observe_ready)
    monkeypatch.setattr(llm_handler_server.agllm, "for_config", lambda config: backend)
    skill = agskill(
        "golden_execution",
        "Return a short plain-text answer. No tools are needed.",
        input_schema=agdata(instruction=agrawstring),
        output_schema=agdata(answer=agrawstring),
        max_output_schema_retries=0,
    )
    with agprof.span("Agent()"):
        ag = Agent(agconfig=config)
    result = None
    try:
        with agprof.span("ag.queue_message(): before run"):
            ag.queue_message(QUEUED)
        with agprof.span("ag.run(): first"):
            result = ag.run(skill, agdata(instruction=INPUT), max_steps=4)
        assert backend.entered.wait(WAIT_SECONDS), "first model request never arrived"
        if harness in {"codex", "opencode", "grok"}:
            # The model request can precede native prompt acknowledgment. The
            # await span starts only after that acknowledgment enables controls.
            assert ready.wait(WAIT_SECONDS), "native prompt was not acknowledged"
        first_context = _text(backend.requests[0]["messages"], "user")
        assert INPUT in first_context
        assert QUEUED in first_context
        assert REDIRECT not in first_context

        with agprof.span("ag.queue_message(): during run"):
            ag.queue_message(FUTURE)
        with agprof.span("ag.pause()"):
            ag.pause()
            assert ag.is_paused()
        with agprof.span("ag.resume()"):
            ag.resume()
            assert not ag.is_paused()

        # Claude's filesystem acknowledgment of native prompt submission is
        # collected asynchronously after the model request reaches the gateway.
        # If this grace period is insufficient, the interrupted assertion below
        # fails instead of mistaking queue fallback for live redirect delivery.
        if harness == "claude_code":
            time.sleep(0.25)
        with agprof.span("ag.redirect(): active execution"):
            ag.redirect(result, REDIRECT)
        if harness != "native":
            assert backend.interrupted.wait(5), "redirect did not interrupt the model stream"
        else:
            backend.release.set()
        with agprof.span("result.wait(): first"):
            result.wait(timeout=WAIT_SECONDS)
        assert result.to_dict() == {"answer": ANSWER}
        current_requests = list(backend.requests)
        assert all(FUTURE not in _text(r["messages"], "user") for r in current_requests)
        if harness != "native":
            assert len(current_requests) >= 2
            assert REDIRECT in _text(current_requests[-1]["messages"], "user")
        else:
            assert all(REDIRECT not in _text(r["messages"], "user") for r in current_requests)

        # A completed result keeps its execution identity after wait(). This
        # redirect must enter future context without waking its old daemon.
        with agprof.span("ag.redirect(): completed execution"):
            ag.redirect(result, LATE)
        with agprof.span("ag.run(): future context"):
            result = ag.run(skill, agdata(instruction="golden next run"), max_steps=4)
            result.wait(timeout=WAIT_SECONDS)
        assert result.to_dict() == {"answer": ANSWER}
        future_requests = backend.requests[len(current_requests) :]
        assert future_requests
        context = _text(future_requests[0]["messages"], "user")
        for message in (QUEUED, FUTURE, REDIRECT, LATE):
            assert message in context, (message, context)
        transcript = ag.context.get_resolved_transcript()
        assert ANSWER in _text(transcript, "assistant"), transcript

        backend.arm()
        with agprof.span("ag.run(): cancellation target"):
            result = ag.run(skill, agdata(instruction="golden cancel target"), max_steps=4)
        assert backend.entered.wait(WAIT_SECONDS), "cancellation target never reached the model"
        with agprof.span("ag.cancel(): active execution"):
            ag.cancel(result)
        backend.release.set()
        with agprof.span("result.wait(): canceled"):
            result.wait(timeout=WAIT_SECONDS)
        assert "cancel" in result.to_dict().get("error", "").lower(), result.to_dict()
    finally:
        original_error = sys.exception()
        backend.release.set()
        (directory / "requests.json").write_text(
            json.dumps(backend.requests, indent=2, default=str) + "\n"
        )
        try:
            if result is not None and result.is_pending():
                ag.cancel(result)
                result.wait(timeout=WAIT_SECONDS)
        except Exception as cleanup_error:
            if original_error is None:
                raise
            original_error.add_note(f"Golden cleanup also failed: {cleanup_error!r}")
        finally:
            if ag.sandbox is not None:
                ag.sandbox.destroy()
