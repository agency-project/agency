"""Live ``Invocation.redirect()`` conformance suite: Claude Code on GPT-5.6 Luna.

Unlike ``redirect_test_suite.py``, every model response here comes from the real
OpenAI API through Agency's Claude Code harness. Threading events make redirect
arrival deterministic at the tool, generation, and final-answer boundaries.

Requirements:
  - ``OPENAI_API_KEY`` is set and can access ``gpt-5.6-luna``.
  - The ``claude`` CLI and Agency sandbox image are installed on this host.

Run with::

    uv run python examples/redirect_claude_code_luna.py
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import threading
from contextlib import contextmanager

from agency import Agent, agdata, agrawstring, agskill, agtool
from agency._agent_control import InvocationHandle
from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig
from agency.llm.openai import _OpenAICompatibleBackend


TIMEOUT_SECONDS = 180.0


class GateTicket:
    def __init__(self, label: str) -> None:
        self.label = label
        self.entered = threading.Event()
        self.release = threading.Event()


class ToolGates:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tickets: dict[str, GateTicket] = {}
        self._calls: list[str] = []

    def arm(self, label: str, *, released: bool = False) -> GateTicket:
        ticket = GateTicket(label)
        if released:
            ticket.release.set()
        with self._lock:
            if label in self._tickets:
                raise RuntimeError(f"gate already armed: {label}")
            self._tickets[label] = ticket
        return ticket

    def invoke(self, arguments: agdata) -> agdata:
        label = str(arguments.label)
        with self._lock:
            self._calls.append(label)
            ticket = self._tickets.get(label)
        if ticket is None:
            return agdata(error=f"gate is not armed: {label}")
        ticket.entered.set()
        if not ticket.release.wait(timeout=TIMEOUT_SECONDS):
            return agdata(error=f"gate timed out: {label}")
        return agdata(label=label, released=True)

    def calls(self) -> list[str]:
        with self._lock:
            return list(self._calls)

    def release_all(self) -> None:
        with self._lock:
            tickets = tuple(self._tickets.values())
        for ticket in tickets:
            ticket.release.set()


GATES = ToolGates()

redirect_gate = agtool(
    name="redirect_gate",
    description="Test barrier. Call it with the exact label requested by the skill.",
    fn=GATES.invoke,
    run_in_subprocess=False,
    params={
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    },
)


def _config() -> agconfig_cls:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    if shutil.which("claude") is None:
        raise SystemExit("the claude CLI is required")
    return agconfig_cls(
        llmconfig(
            provider="openai",
            model="gpt-5.6-luna",
            api_key=api_key,
            reasoning_effort="none",
            max_completion_tokens=1024,
        )
    )


FINAL_SKILL = agskill(
    name="redirect_final",
    system_prompt=(
        "Return initial_marker unless an Agency invocation message gives a replacement. "
        "Obey invocation messages in arrival order. Return only the chosen marker."
    ),
    input_schema=agdata(initial_marker=str),
    output_schema=agdata(marker=agrawstring),
)

ONE_TOOL_SKILL = agskill(
    name="redirect_one_tool",
    system_prompt=(
        "Call redirect_gate exactly once with gate_label and wait for its result. "
        "Then return only initial_marker. If an Agency invocation message says to skip "
        "the tool or replace the marker, obey it."
    ),
    input_schema=agdata(gate_label=str, initial_marker=str),
    output_schema=agdata(marker=agrawstring),
    add_host_mcp_tools=[redirect_gate],
)

TWO_TOOL_SKILL = agskill(
    name="redirect_two_tools",
    system_prompt=(
        "Call redirect_gate with first_label and WAIT for its result. Only after it "
        "returns, call redirect_gate with second_label and wait again. Then return only "
        "initial_marker. Agency invocation messages override the remaining steps."
    ),
    input_schema=agdata(first_label=str, second_label=str, initial_marker=str),
    output_schema=agdata(marker=agrawstring),
    add_host_mcp_tools=[redirect_gate],
)


def _new_agent(config: agconfig_cls, name: str) -> Agent:
    return Agent(agname=f"redirect-e2e-{name}", agconfig=config, harness="claude_code")


def _marker(invocation) -> str:
    invocation.wait(timeout=TIMEOUT_SECONDS)
    payload = invocation.result.to_dict()
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    marker = str(payload["marker"]).strip()
    print(f"  accepted Luna response: {marker!r}", flush=True)
    return marker


def _destroy(agent: Agent) -> None:
    GATES.release_all()
    agent.destroy().wait(timeout=TIMEOUT_SECONDS)


class ProviderProbe:
    """Observe, pause, or fail real OpenAI calls without replacing their responses."""

    def __init__(self, *, pause_first_response: bool = False, fail_first: bool = False) -> None:
        self.pause_first_response = pause_first_response
        self.fail_first = fail_first
        self.response_ready = threading.Event()
        self.release_response = threading.Event()
        self._lock = threading.Lock()
        self.requests: list[dict] = []

    def record(self, request: dict) -> int:
        with self._lock:
            self.requests.append(copy.deepcopy(request))
            return len(self.requests)

    def maybe_fail(self, number: int) -> None:
        if self.fail_first and number == 1:
            raise OSError("injected transient provider failure")

    def pause(self, number: int) -> None:
        if not self.pause_first_response or number != 1:
            return
        self.response_ready.set()
        if not self.release_response.wait(timeout=TIMEOUT_SECONDS):
            raise TimeoutError("timed out waiting to release provider response")


@contextmanager
def probe_provider(probe: ProviderProbe):
    original_batch = _OpenAICompatibleBackend._call_backend
    original_stream = _OpenAICompatibleBackend._call_backend_stream

    def batch(backend, request):
        number = probe.record(request)
        probe.maybe_fail(number)
        response = original_batch(backend, request)
        probe.pause(number)
        return response

    def stream(backend, request, on_client=None):
        number = probe.record(request)
        probe.maybe_fail(number)
        raw_stream, client = original_stream(backend, request, on_client=on_client)
        if not probe.pause_first_response or number != 1:
            return raw_stream, client

        def buffered_stream():
            chunks = list(raw_stream)
            probe.pause(number)
            yield from chunks

        return buffered_stream(), client

    _OpenAICompatibleBackend._call_backend = batch
    _OpenAICompatibleBackend._call_backend_stream = stream
    try:
        yield
    finally:
        probe.release_response.set()
        _OpenAICompatibleBackend._call_backend = original_batch
        _OpenAICompatibleBackend._call_backend_stream = original_stream


class FinalCheckpointProbe:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.entered = threading.Event()
        self.release = threading.Event()
        self._used = False
        self._lock = threading.Lock()

    def claim(self) -> bool:
        with self._lock:
            if self._used:
                return False
            self._used = True
            return True

    def wait(self) -> None:
        self.entered.set()
        if not self.release.wait(timeout=TIMEOUT_SECONDS):
            raise TimeoutError("timed out waiting at final checkpoint")


@contextmanager
def probe_final_checkpoint(probe: FinalCheckpointProbe):
    original = InvocationHandle._checkpoint_final_answer

    def checkpoint(invocation, boundary_id, *, abort_event=None):
        should_probe = probe.claim()
        if should_probe and probe.mode == "before":
            probe.wait()
        decision = original(invocation, boundary_id, abort_event=abort_event)
        if should_probe and probe.mode == "after":
            probe.wait()
        return decision

    InvocationHandle._checkpoint_final_answer = checkpoint
    try:
        yield
    finally:
        probe.release.set()
        InvocationHandle._checkpoint_final_answer = original


def case_no_redirect(config: agconfig_cls) -> None:
    ticket = GATES.arm("happy-tool", released=True)
    agent = _new_agent(config, "happy")
    try:
        invocation = agent.run(
            ONE_TOOL_SKILL,
            agdata(gate_label=ticket.label, initial_marker="HAPPY-COMPLETE"),
        )
        assert _marker(invocation) == "HAPPY-COMPLETE"
        assert GATES.calls().count(ticket.label) == 1
    finally:
        _destroy(agent)


def case_redirect_while_tool_runs(config: agconfig_cls) -> None:
    first = GATES.arm("running-A")
    second = GATES.arm("must-not-run-B", released=True)
    calls_before = len(GATES.calls())
    agent = _new_agent(config, "tool-running")
    try:
        invocation = agent.run(
            TWO_TOOL_SKILL,
            agdata(
                first_label=first.label,
                second_label=second.label,
                initial_marker="STALE-TOOLS-DONE",
            ),
        )
        assert first.entered.wait(timeout=TIMEOUT_SECONDS)
        invocation.redirect(
            "Tool A may finish. Do not call the second tool. Return exactly TOOL-REDIRECTED."
        )
        first.release.set()
        assert _marker(invocation) == "TOOL-REDIRECTED"
        assert GATES.calls()[calls_before:] == [first.label]
    finally:
        first.release.set()
        _destroy(agent)


def _case_redirect_during_generation(
    config: agconfig_cls,
    *,
    skill: agskill,
    inputs: agdata,
    stale_tool_label: str | None,
) -> None:
    probe = ProviderProbe(pause_first_response=True)
    calls_before = len(GATES.calls())
    agent = _new_agent(config, f"generation-{stale_tool_label or 'final'}")
    try:
        with probe_provider(probe):
            invocation = agent.run(skill, inputs)
            assert probe.response_ready.wait(timeout=TIMEOUT_SECONDS)
            invocation.redirect(
                "Cancel the previous task and discard that draft, including any planned tool "
                "calls. Skip redirect_gate and all other tools; do not call any tools. "
                "Return exactly GENERATION-REDIRECTED now."
            )
            probe.release_response.set()
            assert _marker(invocation) == "GENERATION-REDIRECTED"
        assert len(probe.requests) >= 2
        if stale_tool_label is not None:
            assert stale_tool_label not in GATES.calls()[calls_before:]
    finally:
        probe.release_response.set()
        _destroy(agent)


def case_redirect_while_model_generates(config: agconfig_cls) -> None:
    stale_ticket = GATES.arm("stale-generated-tool", released=True)
    _case_redirect_during_generation(
        config,
        skill=ONE_TOOL_SKILL,
        inputs=agdata(gate_label=stale_ticket.label, initial_marker="STALE-TOOL-FINAL"),
        stale_tool_label=stale_ticket.label,
    )
    _case_redirect_during_generation(
        config,
        skill=FINAL_SKILL,
        inputs=agdata(initial_marker="STALE-FINAL-ANSWER"),
        stale_tool_label=None,
    )


def case_redirect_races_with_final(config: agconfig_cls) -> None:
    accepted_probe = FinalCheckpointProbe("before")
    accepted_agent = _new_agent(config, "redirect-wins-final")
    try:
        with probe_final_checkpoint(accepted_probe):
            accepted = accepted_agent.run(
                FINAL_SKILL,
                agdata(initial_marker="STALE-BEFORE-CLOSING"),
            )
            assert accepted_probe.entered.wait(timeout=TIMEOUT_SECONDS)
            accepted.redirect("Return exactly REDIRECT-WON-FINAL-RACE.")
            accepted_probe.release.set()
            assert _marker(accepted) == "REDIRECT-WON-FINAL-RACE"
    finally:
        accepted_probe.release.set()
        _destroy(accepted_agent)

    closing_probe = FinalCheckpointProbe("after")
    closing_agent = _new_agent(config, "closing-wins-final")
    try:
        with probe_final_checkpoint(closing_probe):
            closing = closing_agent.run(
                FINAL_SKILL,
                agdata(initial_marker="CLOSING-WON-FINAL-RACE"),
            )
            assert closing_probe.entered.wait(timeout=TIMEOUT_SECONDS)
            try:
                closing.redirect("TOO-LATE")
            except RuntimeError:
                pass
            else:
                raise AssertionError("redirect was accepted after the closing fence")
            closing_probe.release.set()
            assert _marker(closing) == "CLOSING-WON-FINAL-RACE"
    finally:
        closing_probe.release.set()
        _destroy(closing_agent)


def case_failed_model_retains_redirect(config: agconfig_cls) -> None:
    probe = ProviderProbe(fail_first=True)
    agent = _new_agent(config, "failed-model")
    try:
        agent.suspend()
        invocation = agent.run(FINAL_SKILL, agdata(initial_marker="STALE-AFTER-FAILURE"))
        invocation.redirect("Return exactly RETRIED-REDIRECT.")
        with probe_provider(probe):
            agent.resume()
            assert _marker(invocation) == "RETRIED-REDIRECT"
        assert len(probe.requests) >= 2
        assert "RETRIED-REDIRECT" in json.dumps(probe.requests[0])
        assert "RETRIED-REDIRECT" in json.dumps(probe.requests[1])
    finally:
        if agent.is_suspended():
            agent.resume()
        _destroy(agent)


def case_multiple_redirects_fifo(config: agconfig_cls) -> None:
    probe = ProviderProbe(pause_first_response=True)
    agent = _new_agent(config, "fifo")
    try:
        agent.suspend()
        invocation = agent.run(FINAL_SKILL, agdata(initial_marker="STALE-FIFO"))
        invocation.redirect("Redirect A: remember the first fragment FIFO-A.")
        with probe_provider(probe):
            agent.resume()
            assert probe.response_ready.wait(timeout=TIMEOUT_SECONDS)
            invocation.redirect(
                "Redirect B: append FIFO-B after FIFO-A with one |. Return exactly FIFO-A|FIFO-B."
            )
            probe.release_response.set()
            assert _marker(invocation) == "FIFO-A|FIFO-B"
        assert len(probe.requests) >= 2
        first_request = json.dumps(probe.requests[0])
        later_requests = json.dumps(probe.requests[1:])
        assert "Redirect A" in first_request
        assert "Redirect B" not in first_request
        assert "Redirect A" in later_requests and "Redirect B" in later_requests
    finally:
        probe.release_response.set()
        if agent.is_suspended():
            agent.resume()
        _destroy(agent)


CASES = [
    ("no redirects: tool and final succeed", case_no_redirect),
    ("redirect while tool A is running", case_redirect_while_tool_runs),
    ("redirect while model generates stale tool/final", case_redirect_while_model_generates),
    ("redirect versus final closing fence", case_redirect_races_with_final),
    ("failed model call retains redirect for retry", case_failed_model_retains_redirect),
    ("multiple redirects remain FIFO", case_multiple_redirects_fifo),
]


def main() -> None:
    config = _config()
    for label, case in CASES:
        case(config)
        print(f"PASS {label}", flush=True)
    print(f"\n{len(CASES)} live Luna/Claude Code redirect cases passed", flush=True)


if __name__ == "__main__":
    main()
