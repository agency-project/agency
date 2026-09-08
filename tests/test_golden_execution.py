"""Canonical public lifecycle against real Docker harnesses and replayed LLMs.

CI runs this with normal pytest. On a Linux development host:

    GPU_TYPE=cpu ./images/build.sh
    docker build -t agency-golden:ci -f images/Dockerfile.golden .
    AGENCY_TEST_HARNESS_IMAGE=agency-golden:ci pytest tests/test_golden_execution.py -v

Alternatively, install the external CLIs on the host and supply an image with
their system runtimes through AGENCY_TEST_HARNESS_IMAGE.
AGENCY_TEST_EXTERNAL_HARNESSES=1 makes missing prerequisites fail locally too.
Profiler artifacts: artifacts/golden-execution/<harness>/agprof.trace.json,
summary.json, summary.md, and profile_data.sqlite3.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from agency import Agent, agconfig, agdata, agDataLogger, agprof, agrawstring, agskill
from agency.configs.agconfig import (
    agentconfig,
    dataloggerconfig,
    llmconfig,
    resourcesconfig,
    sandboxconfig,
)


# for_config() currently uses these strings; there is no harness enum/registry.
HARNESSES = ("native", "claude_code", "codex", "opencode", "grok")
INPUT = "golden invocation input"
QUEUED = "golden queued message"
REDIRECT = "golden redirect"
ANSWER = "GOLDEN_OK"
WAIT_SECONDS = 90
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "golden-execution"


@pytest.fixture(scope="module")
def golden_image():
    image = os.environ.get("AGENCY_TEST_HARNESS_IMAGE", "agency-sandbox:latest")
    problem = None
    if not sys.platform.startswith("linux"):
        problem = "golden profiling and real external harnesses require Linux"
    else:
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True, timeout=15)
            subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                check=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            problem = f"golden execution requires Docker and the local image {image}: {exc}"
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
        # The redirect must supersede a distinct draft, then obtain GOLDEN_OK
        # in a generation that actually includes the redirect.
        for index, answer in enumerate(("GOLDEN_DRAFT", ANSWER)):
            logger.finalize_stream(
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


@pytest.mark.timeout(300)
@pytest.mark.parametrize("harness", HARNESSES)
def test_golden_execution(harness, golden_image, golden_profile, tmp_path):
    # Keep the workload span in the test call: pytest does not throw test
    # exceptions back through yield fixtures, which would label failures as
    # successful spans. Session finalization still runs in fixture teardown.
    with agprof.span(f"golden-execution:{harness}"):
        _exercise_golden_lifecycle(harness, golden_image, tmp_path)


def _exercise_golden_lifecycle(harness, golden_image, tmp_path):
    entered = threading.Event()
    release = threading.Event()
    requests = []

    def gate(request):
        requests.append(request)
        if len(requests) == 1:
            entered.set()
            assert release.wait(timeout=WAIT_SECONDS), "first model response was not released"

    replay = tmp_path / "replay.sqlite3"
    _write_replay(replay)
    with agprof.span("agconfig()"):
        config = agconfig(
            agentconfig(harness=harness, log_dir=str(tmp_path / "logs")),
            sandboxconfig(backend="docker", base_image=golden_image),
            resourcesconfig(idle_cpus=1, idle_memory="1g"),
            llmconfig(
                provider="mock",
                model="golden-replay",
                context_limit=200_000,
                replay_db_path=str(replay),
                timing_mode="instant",
                replay_dispatch_hook=gate,
            ),
        )
    with agprof.span("agskill()"):
        skill = agskill(
            "golden_execution",
            "Return a short plain-text answer. No tools are needed.",
            input_schema=agdata(instruction=agrawstring),
            output_schema=agdata(answer=agrawstring),
            max_output_schema_retries=0,
        )
    with agprof.span("Agent()"):
        ag = Agent(agconfig=config)
    inv = None
    try:
        # Explicit spans survive the automatic profiler's short-call filter.
        # Async request calls and their completion waits are distinct intervals.
        with agprof.span("ag.queue_message()"):
            queued = ag.queue_message(QUEUED)
        with agprof.span("ag.run()"):
            inv = ag.run(skill, agdata(instruction=INPUT), max_steps=4)
        with agprof.span("wait: first model request"):
            assert entered.wait(timeout=WAIT_SECONDS), (
                f"model never entered replay; state={inv.state}"
            )
        with agprof.span("queued.wait()"):
            assert queued.wait(timeout=WAIT_SECONDS).state == "SUCCEEDED"
        assert INPUT in _text(requests[0]["messages"], "user")
        assert QUEUED in _text(requests[0]["messages"], "user")
        assert REDIRECT not in _text(requests[0]["messages"], "user")

        with agprof.span("inv.redirect()"):
            inv.redirect(REDIRECT)
        with agprof.span("ag.suspend()"):
            ag.suspend()
        release.set()

        # There is no public wait-for-state API. Use the control condition only
        # for notifications; all predicates/assertions use public lifecycle APIs.
        with agprof.span("wait: suspended safe boundary"), ag._control._condition:
            assert ag._control._condition.wait_for(
                lambda: ag.lifecycle_state == "SUSPENDED" and inv.state == "PAUSED",
                timeout=WAIT_SECONDS,
            ), f"suspension never reached a safe boundary: {ag.lifecycle_state}, {inv.state}"
        assert ag.is_suspended()
        assert ag.is_paused()
        assert inv.is_pending()
        assert len(requests) == 1

        with agprof.span("inv.pause()"):
            inv.pause()
        with agprof.span("ag.resume()"):
            ag.resume()
        assert ag.lifecycle_state == "ACTIVE"
        assert not ag.is_suspended()
        assert inv.is_pause_requested()
        assert inv.state == "PAUSED"
        assert inv.is_pending()
        assert len(requests) == 1

        with agprof.span("inv.resume()"):
            inv.resume()
        with agprof.span("inv.wait()"):
            inv.wait(timeout=WAIT_SECONDS)
        assert inv.state == "SUCCEEDED", inv.to_dict()
        assert inv.to_dict() == {"answer": ANSWER}
        assert not inv.is_cancelled()
        assert not inv.is_destroyed()
        assert len(requests) == 2
        for text in (INPUT, QUEUED, REDIRECT):
            assert text in _text(requests[-1]["messages"], "user")

        transcript = ag.context.get_resolved_transcript()
        for text in (INPUT, QUEUED, REDIRECT):
            assert text in _text(transcript, "user"), transcript
        assert ANSWER in _text(transcript, "assistant"), transcript

        with agprof.span("ag.destroy()"):
            close = ag.destroy()
        with agprof.span("close.wait()"):
            assert close.wait(timeout=WAIT_SECONDS).done()
        assert ag.lifecycle_state == "DESTROYED"
    finally:
        # Release failed assertions' gates before pytest's orchestrator teardown.
        # Never destroy an executing agent, including on the failure path.
        original_error = sys.exception()
        if original_error is not None:
            original_error.add_note(
                f"Observed {len(requests)} model requests; last user context: "
                f"{_text(requests[-1]['messages'], 'user')[-4000:] if requests else '(none)'}"
            )
        release.set()
        try:
            if ag.lifecycle_state != "DESTROYED":
                with agprof.span("failure cleanup"):
                    if inv is not None and inv.is_pending():
                        with agprof.span("inv.cancel()"):
                            inv.cancel()
                        with agprof.span("inv.wait()"):
                            inv.wait(timeout=WAIT_SECONDS)
                    with agprof.span("ag.destroy()"):
                        close = ag.destroy()
                    with agprof.span("close.wait()"):
                        close.wait(timeout=WAIT_SECONDS)
        except Exception as cleanup_error:
            if original_error is None:
                raise
            original_error.add_note(f"Golden cleanup also failed: {cleanup_error!r}")
