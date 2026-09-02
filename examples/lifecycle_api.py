"""Live conformance test for Agency's public submission and lifecycle APIs.

This example uses the native harness and a real OpenAI model. It exercises the
public ``Agent``, ``Submission``, ``Invocation``, ``MessageSubmission``, and
``CloseHandle`` surfaces, including ordered context messages, exact-invocation
messages, independent pause gates, cancellation, destruction, awaiting, and result-field
proxying. Host-side events make the concurrency claims observable instead of
inferring them from model prose.

Run with profiling enabled to produce both lifecycle evidence and measured
profiler artifacts::

    export OPENAI_API_KEY="..."
    AGENCY_PROFILE=1 \
    AGENCY_PROFILE_SCOPE=workload \
    AGENCY_PROFILE_DIR="$PWD/runs/lifecycle_api_profile" \
      uv run python examples/lifecycle_api.py --model gpt-5.6-luna

The process exits nonzero on the first failed invariant. ``verification.json``
records every passed check without recording credentials. A profiled run also
validates ``summary.json``, ``summary.md``, and ``agprof.trace.json`` before it
reports success.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agency import (
    Agent,
    AgentDestroyedError,
    CloseHandle,
    Invocation,
    MessageSubmission,
    Submission,
    agdata,
    agprof,
    agrawstring,
    agskill,
    agtool,
)
from agency.agconfig import agConfig
from agency.llm import agOpenAIBackendConfig

ORDERED_MEMORY = "ORDERED-CONTEXT-731"
CANCEL_MEMORY = "CANCEL-CONTEXT-947"
FIFO_FIRST = "FIFO-FIRST"
FIFO_SECOND = "FIFO-SECOND"
FIFO_MARKER = f"{FIFO_FIRST}|{FIFO_SECOND}"
BATCH_MARKER = "BATCH-GATES-OPEN"

# The global orchestrator gives every ordered submission a run-shaped trace
# span: eleven skill invocations plus three host-only message submissions.  The
# ordered and cancellation messages succeed; the destruction message fails
# after the destroy latch, alongside the four intentionally failed invocations.
EXPECTED_RUNS = {
    "started": 14,
    "completed": 14,
    "succeeded": 9,
    "failed": 5,
    "interrupted": 0,
}
EXPECTED_GATE_CALLS = 4
MINIMUM_LLM_CALLS = 11
REQUIRED_PROFILE_SPANS = {
    "e2e:ordered_chain",
    "e2e:batch_gates",
    "e2e:cancellation",
    "e2e:destruction",
    "e2e:tool_gate",
}


class _GateTicket:
    def __init__(self, label: str) -> None:
        self.label = label
        self.entered = threading.Event()
        self.release = threading.Event()


class _ToolGate:
    """Host-observed barriers for model tool calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tickets: dict[str, _GateTicket] = {}
        self._calls: list[str] = []
        self.tool_timeout_s = 300.0

    def arm(self, label: str) -> _GateTicket:
        with self._lock:
            if label in self._tickets:
                raise RuntimeError(f"gate label already armed: {label!r}")
            ticket = _GateTicket(label)
            self._tickets[label] = ticket
            return ticket

    def invoke(self, arg: agdata) -> agdata:
        with agprof.span("tool:lifecycle_gate"):
            with agprof.span("e2e:tool_gate"):
                label = str(arg.label)
                with self._lock:
                    self._calls.append(label)
                    ticket = self._tickets.get(label)
                if ticket is None:
                    return agdata(error=f"gate label was not armed: {label!r}")
                ticket.entered.set()
                if not ticket.release.wait(timeout=self.tool_timeout_s):
                    return agdata(error=f"timed out waiting to release gate {label!r}")
                return agdata(label=label, released=True)

    def calls(self) -> list[str]:
        with self._lock:
            return list(self._calls)

    def release_all(self) -> None:
        with self._lock:
            tickets = tuple(self._tickets.values())
        for ticket in tickets:
            ticket.release.set()


_GATE = _ToolGate()


lifecycle_gate = agtool(
    name="lifecycle_gate",
    description=(
        "Required lifecycle test barrier. Call it exactly once with the label "
        "from the skill input, then wait for the result before answering."
    ),
    fn=_GATE.invoke,
    run_in_subprocess=False,
    params={
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    },
)


def _config(model: str) -> agConfig:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    return agConfig(
        agOpenAIBackendConfig(
            model=model,
            api_key=api_key,
            max_completion_tokens=2048,
            reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "none"),
        )
    )


def _skills() -> tuple[agskill, agskill, agskill]:
    controlled = agskill(
        name="lifecycle_controlled",
        system_prompt=(
            "This is a deterministic API conformance task. You MUST call "
            "lifecycle_gate exactly once with the input label before answering. "
            "Do not call any other tool. Wait for the tool result, then obey all "
            "Agency invocation messages in arrival order. Your final response "
            "must contain only expected_marker, with no quotes, Markdown, or explanation."
        ),
        input_schema=agdata(label=str, expected_marker=str),
        output_schema=agdata(marker=agrawstring),
        add_host_mcp_tools=[lifecycle_gate],
    )
    echo = agskill(
        name="lifecycle_echo",
        system_prompt=(
            "Return the input text verbatim. Output only that text, with no quotes, "
            "Markdown, prefix, suffix, or explanation."
        ),
        input_schema=agdata(result=agrawstring),
        output_schema=agdata(result=agrawstring),
    )
    recall = agskill(
        name="lifecycle_recall",
        system_prompt=(
            "Find the most recent earlier standalone user message beginning with "
            "PUBLIC_API_MEMORY:. Return only the text after that prefix, stripped "
            "of surrounding whitespace. Do not return the prefix or explain."
        ),
        input_schema=agdata(question=agrawstring),
        output_schema=agdata(memory=agrawstring),
    )
    return controlled, echo, recall


async def _await_handle(handle, timeout_s: float):
    return await asyncio.wait_for(handle, timeout=timeout_s)


def _await_sync(handle, timeout_s: float):
    return asyncio.run(_await_handle(handle, timeout_s))


def _wait_until(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _record(evidence: dict, event: str, **details) -> None:
    evidence["events"].append(
        {
            "sequence": len(evidence["events"]) + 1,
            "monotonic_ns": time.monotonic_ns(),
            "event": event,
            **details,
        }
    )


def _check(evidence: dict, condition: bool, label: str, **details) -> None:
    if not condition:
        raise AssertionError(f"{label} failed: {details}")
    _record(evidence, "check_passed", check=label, **details)
    print(f"PASS {label}", flush=True)


def _expect_destroyed(
    evidence: dict,
    label: str,
    operation: Callable[[], object],
) -> None:
    try:
        operation()
    except AgentDestroyedError as exc:
        _record(
            evidence,
            "expected_exception",
            check=label,
            exception=type(exc).__name__,
            message=str(exc),
        )
        print(f"PASS {label}", flush=True)
        return
    raise AssertionError(f"{label} did not raise AgentDestroyedError")


def _expect_runtime_rejection(
    evidence: dict,
    label: str,
    operation: Callable[[], object],
) -> None:
    try:
        operation()
    except RuntimeError as exc:
        _record(
            evidence,
            "expected_exception",
            check=label,
            exception=type(exc).__name__,
            message=str(exc),
        )
        print(f"PASS {label}", flush=True)
        return
    raise AssertionError(f"{label} did not raise RuntimeError")


def _successful(handle: Invocation, label: str, timeout_s: float) -> dict:
    handle.wait(timeout=timeout_s)
    payload = handle.result.to_dict()
    if payload.get("error"):
        raise RuntimeError(f"{label} failed: {payload['error']}")
    return payload


def _terminal_error(
    handle: Invocation | MessageSubmission,
    expected: str,
    label: str,
    timeout_s: float,
) -> dict:
    handle.wait(timeout=timeout_s)
    payload = handle.result.to_dict()
    if payload.get("error") != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {payload!r}")
    return payload


def _raw(payload: dict, field: str) -> str:
    return str(payload[field]).strip()


def _retained_contents(node: Submission) -> list[str]:
    node.output_context.resolve_prev_dependencies()
    return [str(entry["content"]) for entry in node.output_context.retained_messages]


def _wait_for_gate(ticket: _GateTicket, timeout_s: float) -> None:
    if not ticket.entered.wait(timeout=timeout_s):
        raise TimeoutError(f"model did not enter lifecycle_gate({ticket.label!r})")


def _exercise_ordered_chain(
    ag: Agent,
    controlled: agskill,
    echo: agskill,
    recall: agskill,
    evidence: dict,
    timeout_s: float,
) -> None:
    with agprof.span("e2e:ordered_chain"):
        ticket = _GATE.arm("ordered-head")
        initial_context = ag.ctx
        ag.suspend()
        message = ag.queue_message(f"PUBLIC_API_MEMORY: {ORDERED_MEMORY}")
        head = ag.run(
            controlled,
            agdata(label=ticket.label, expected_marker=FIFO_MARKER),
        )
        tail = ag.run(
            recall,
            agdata(question="Recall the public API memory codeword."),
        )
        ready = ag.run(echo, agdata(result="RESULT-FIELD-COLLISION"))

        _check(
            evidence,
            [message.ordering_id, head.ordering_id, tail.ordering_id, ready.ordering_id]
            == [1, 2, 3, 4],
            "ordering IDs span queue_message and run",
        )
        _check(
            evidence,
            all(isinstance(node, Submission) for node in (head, message, tail, ready))
            and all(isinstance(node, Invocation) for node in (head, tail, ready))
            and isinstance(message, MessageSubmission),
            "public submission handle types",
        )
        _check(
            evidence,
            all(
                not hasattr(Agent, name)
                for name in ("prepare", "start", "send", "steer", "cancel", "pause")
            )
            and all(not hasattr(Invocation, name) for name in ("start", "steer")),
            "removed public lifecycle aliases stay absent",
        )
        _check(
            evidence,
            message.predecessor_context is initial_context
            and head.predecessor_context is message.output_context
            and tail.predecessor_context is head.output_context
            and ready.predecessor_context is tail.output_context,
            "context futures are the authoritative chain",
        )
        controls = ("send_message", "pause", "resume", "cancel")
        _check(
            evidence,
            all(not hasattr(message, control) for control in controls),
            "message submission exposes no invocation controls",
        )
        _check(
            evidence,
            all(node.is_pending() for node in (head, tail, ready))
            and message.state in {"QUEUED", "COMPLETING", "SUCCEEDED"},
            "engine-backed ordered handles remain pending behind suspension",
        )

        message_payload = _await_sync(message, timeout_s).to_dict()
        _check(
            evidence,
            message_payload == {} and message.state == "SUCCEEDED" and ag.sandbox is None,
            "queue_message commits while agent dispatch is suspended",
        )

        head.send_message(f"Remember {FIFO_FIRST} as the first marker fragment.")
        head.send_message(f"Append {FIFO_SECOND} after the first fragment using one | separator.")
        _check(
            evidence,
            not ticket.entered.wait(timeout=0.3),
            "agent suspension holds queued invocations without infrastructure",
        )
        ag.resume()
        _wait_for_gate(ticket, timeout_s)
        _check(evidence, head.state == "RUNNING", "head invocation reaches running state")
        head.send_message(f"After the gate, output exactly {FIFO_MARKER}.")
        head.pause()
        ticket.release.set()
        _check(
            evidence,
            _wait_until(lambda: head.state == "PAUSED" and ag.is_paused(), timeout_s),
            "pause parks at the post-tool safe boundary",
        )
        head.resume()

        head_payload = _successful(head, "ordered head", timeout_s)
        _check(
            evidence,
            _raw(head_payload, "marker") == FIFO_MARKER,
            "queued and live invocation messages produce the FIFO marker",
            marker=_raw(head_payload, "marker"),
        )

        tail_payload = _await_sync(tail, timeout_s).to_dict()
        _check(
            evidence,
            _raw(tail_payload, "memory") == ORDERED_MEMORY,
            "later invocation observes ordered queue_message context",
            memory=_raw(tail_payload, "memory"),
        )

        ready.wait(timeout=timeout_s)
        ready.result.wait()
        ready_payload = ready.result.to_dict()
        _check(
            evidence,
            _raw(ready_payload, "result") == "RESULT-FIELD-COLLISION"
            and str(ready.result.result).strip() == "RESULT-FIELD-COLLISION",
            "pending result supports an output field also named result",
        )
        _check(
            evidence,
            f"PUBLIC_API_MEMORY: {ORDERED_MEMORY}" in _retained_contents(tail),
            "resolved output context preserves the sent message",
        )
        _check(
            evidence,
            [head.state, message.state, tail.state, ready.state]
            == ["SUCCEEDED", "SUCCEEDED", "SUCCEEDED", "SUCCEEDED"],
            "ordered chain reaches terminal success states",
        )

        close = ag.destroy()
        awaited_close = _await_sync(close, timeout_s)
        _check(
            evidence,
            isinstance(close, CloseHandle)
            and awaited_close is close
            and close.done()
            and ag.lifecycle_state == "DESTROYED",
            "CloseHandle is awaitable and destruction settles",
        )


def _exercise_independent_gates(
    ag: Agent,
    controlled: agskill,
    echo: agskill,
    evidence: dict,
    timeout_s: float,
) -> None:
    with agprof.span("e2e:batch_gates"):
        ticket = _GATE.arm("batch-one")
        ag.suspend()
        one = ag.run(
            controlled,
            agdata(label=ticket.label, expected_marker=BATCH_MARKER),
        )
        two = ag.run(echo, agdata(result="BATCH-TWO"))
        later = ag.run(echo, agdata(result="BATCH-LATER"))
        one.pause()

        _check(
            evidence,
            [one.ordering_id, two.ordering_id, later.ordering_id] == [1, 2, 3],
            "batch ordering IDs remain stable",
        )
        _check(
            evidence,
            ag.is_suspended() and not ticket.entered.wait(timeout=0.3) and ag.sandbox is None,
            "agent suspension blocks queued infrastructure",
        )

        one.resume()
        _check(
            evidence,
            ag.is_suspended() and not ticket.entered.wait(timeout=0.3),
            "invocation resume does not open the agent suspension gate",
        )
        one.pause()
        ag.resume()
        _check(
            evidence,
            not ag.is_suspended() and not ticket.entered.wait(timeout=0.3),
            "agent resume does not open an invocation pause gate",
        )

        one.resume()
        _wait_for_gate(ticket, timeout_s)
        one.pause()
        ag.suspend()
        ticket.release.set()
        _check(
            evidence,
            _wait_until(lambda: one.state == "PAUSED" and ag.is_paused(), timeout_s),
            "both gates park running work at a safe boundary",
        )

        ag.resume()
        _check(
            evidence,
            not _wait_until(lambda: one.state != "PAUSED", 0.3),
            "opening only the agent gate leaves invocation paused",
        )
        ag.suspend()
        one.resume()
        _check(
            evidence,
            not _wait_until(lambda: one.state != "PAUSED", 0.3),
            "opening only the invocation gate leaves agent suspended",
        )
        ag.resume()

        one_payload = _successful(one, "batch one", timeout_s)
        two_payload = _successful(two, "batch two", timeout_s)
        agdata.wait_all([one, two])
        _check(
            evidence,
            _raw(one_payload, "marker") == BATCH_MARKER
            and _raw(two_payload, "result") == "BATCH-TWO"
            and later.ordering_id == 3,
            "independently gated queued work completes in order",
        )
        later_payload = _await_sync(later, timeout_s).to_dict()
        _check(
            evidence,
            _raw(later_payload, "result") == "BATCH-LATER",
            "later queued invocation follows its predecessors",
        )

        ag.destroy().wait(timeout=timeout_s)
        _check(
            evidence,
            ag.lifecycle_state == "DESTROYED",
            "batch agent cleanup completes",
        )


def _exercise_cancellation(
    ag: Agent,
    controlled: agskill,
    recall: agskill,
    evidence: dict,
    timeout_s: float,
) -> None:
    with agprof.span("e2e:cancellation"):
        message = ag.queue_message(f"PUBLIC_API_MEMORY: {CANCEL_MEMORY}")
        _await_sync(message, timeout_s)
        _check(
            evidence,
            message.state == "SUCCEEDED" and ag.sandbox is None,
            "queue_message commits context without provisioning infrastructure",
        )

        ag.suspend()
        cancelled = ag.run(
            controlled,
            agdata(label="cancel-never", expected_marker="UNREACHABLE"),
        )
        ticket = _GATE.arm("cancel-running")
        survivor = ag.run(
            controlled,
            agdata(label=ticket.label, expected_marker="UNREACHABLE-AFTER-CANCEL"),
        )
        later = ag.run(
            recall,
            agdata(question="Recall the public API memory after cancellation."),
        )
        _check(
            evidence,
            [message.ordering_id, cancelled.ordering_id, survivor.ordering_id, later.ordering_id]
            == [1, 2, 3, 4],
            "cancellation chain preserves submission order",
        )

        cancelled.cancel()
        cancelled_payload = _terminal_error(
            cancelled,
            "agent invocation cancelled",
            "queued cancellation",
            timeout_s,
        )
        _check(
            evidence,
            cancelled_payload["error"] == "agent invocation cancelled"
            and cancelled.state == "CANCELLED"
            and ag.sandbox is None,
            "queued cancellation is terminal without infrastructure",
        )
        cancelled.cancel()
        _expect_runtime_rejection(
            evidence,
            "terminal invocation rejects messages",
            lambda: cancelled.send_message("too late"),
        )
        _expect_runtime_rejection(
            evidence,
            "terminal invocation rejects pause",
            cancelled.pause,
        )
        _expect_runtime_rejection(
            evidence,
            "terminal invocation rejects resume",
            cancelled.resume,
        )

        ag.resume()
        _wait_for_gate(ticket, timeout_s)
        survivor.pause()
        ticket.release.set()
        _check(
            evidence,
            _wait_until(lambda: survivor.state == "PAUSED", timeout_s),
            "running invocation reaches paused safe boundary",
        )
        survivor.cancel()
        survivor_payload = _terminal_error(
            survivor,
            "agent invocation cancelled",
            "paused running cancellation",
            timeout_s,
        )
        _check(
            evidence,
            survivor_payload["error"] == "agent invocation cancelled"
            and survivor.state == "CANCELLED",
            "cancel wakes a paused invocation and latches terminal state",
        )

        later_payload = _successful(later, "later cancellation survivor", timeout_s)
        _check(
            evidence,
            _raw(later_payload, "memory") == CANCEL_MEMORY,
            "later invocation survives both cancellations",
        )
        cancelled_context = _retained_contents(cancelled)
        survivor_context = _retained_contents(survivor)
        _check(
            evidence,
            f"PUBLIC_API_MEMORY: {CANCEL_MEMORY}" in cancelled_context
            and f"PUBLIC_API_MEMORY: {CANCEL_MEMORY}" in survivor_context,
            "cancelled nodes pass predecessor context through",
        )
        calls = _GATE.calls()
        _check(
            evidence,
            calls.count("cancel-never") == 0 and calls.count(ticket.label) == 1,
            "only the started cancellation target reached its tool",
        )

        ag.destroy().wait(timeout=timeout_s)
        _check(
            evidence,
            ag.lifecycle_state == "DESTROYED",
            "cancellation agent cleanup completes",
        )


def _exercise_destruction(
    ag: Agent,
    controlled: agskill,
    echo: agskill,
    evidence: dict,
    timeout_s: float,
) -> None:
    with agprof.span("e2e:destruction"):
        ticket = _GATE.arm("destroy-active")
        active = ag.run(
            controlled,
            agdata(label=ticket.label, expected_marker="UNREACHABLE-AFTER-DESTROY"),
        )
        message = ag.queue_message("PUBLIC_API_MEMORY: DESTROYED-MESSAGE")
        queued = ag.run(echo, agdata(result="UNREACHABLE-QUEUED"))
        _check(
            evidence,
            [active.ordering_id, message.ordering_id, queued.ordering_id] == [1, 2, 3],
            "destruction chain contains active, context-message, and queued nodes",
        )
        _wait_for_gate(ticket, timeout_s)

        close = ag.destroy()
        same_close = ag.destroy()
        _check(
            evidence,
            isinstance(close, CloseHandle)
            and same_close is close
            and not close.done()
            and ag.lifecycle_state == "DESTROYING",
            "destroy is nonblocking, terminal, and idempotent",
        )

        _expect_destroyed(
            evidence,
            "destroy rejects run",
            lambda: ag.run(echo, agdata(result="rejected")),
        )
        _expect_destroyed(
            evidence,
            "destroy rejects queue_message",
            lambda: ag.queue_message("rejected"),
        )
        _expect_destroyed(evidence, "destroy rejects agent suspend", ag.suspend)
        _expect_destroyed(evidence, "destroy rejects agent resume", ag.resume)
        _expect_destroyed(
            evidence,
            "destroy rejects queued invocation pause",
            queued.pause,
        )
        _expect_destroyed(
            evidence,
            "destroy rejects invocation messages",
            lambda: active.send_message("rejected"),
        )
        _expect_destroyed(
            evidence,
            "destroy rejects invocation pause",
            active.pause,
        )
        _expect_destroyed(
            evidence,
            "destroy rejects invocation resume",
            active.resume,
        )
        active.cancel()
        active.cancel()
        _check(
            evidence,
            active.is_destroyed(),
            "cancel remains an idempotent no-op after destroy latch",
        )

        ticket.release.set()
        _terminal_error(active, "agent destroyed", "destroyed active invocation", timeout_s)
        _terminal_error(message, "agent destroyed", "destroyed queued message", timeout_s)
        _terminal_error(queued, "agent destroyed", "destroyed queued invocation", timeout_s)
        _check(
            evidence,
            [active.state, message.state, queued.state] == ["DESTROYED", "DESTROYED", "DESTROYED"],
            "destroy settles every ordered node deterministically",
        )
        close.wait(timeout=timeout_s)
        _check(
            evidence,
            close.done() and ag.lifecycle_state == "DESTROYED",
            "destroy CloseHandle settles after safe-boundary cleanup",
        )


def _exercise(config: agConfig, evidence: dict, timeout_s: float, cleanup_timeout_s: float) -> None:
    controlled, echo, recall = _skills()
    agents: list[Agent] = []

    def new_agent(name: str) -> Agent:
        created = Agent(agname=name, agconfig=config, harness="native")
        agents.append(created)
        return created

    try:
        _exercise_ordered_chain(
            new_agent("public-api-ordered"),
            controlled,
            echo,
            recall,
            evidence,
            timeout_s,
        )
        _exercise_independent_gates(
            new_agent("public-api-batch"),
            controlled,
            echo,
            evidence,
            timeout_s,
        )
        _exercise_cancellation(
            new_agent("public-api-cancel"),
            controlled,
            recall,
            evidence,
            timeout_s,
        )
        _exercise_destruction(
            new_agent("public-api-destroy"),
            controlled,
            echo,
            evidence,
            timeout_s,
        )
        calls = _GATE.calls()
        _check(
            evidence,
            calls == ["ordered-head", "batch-one", "cancel-running", "destroy-active"],
            "host observed the exact four expected gate calls",
            calls=calls,
        )
    finally:
        _GATE.release_all()
        cleanup_errors = []
        for created in agents:
            try:
                created.destroy().wait(timeout=cleanup_timeout_s)
            except Exception as exc:  # cleanup must continue for later agents
                cleanup_errors.append(f"{created.agname}: {exc}")
        if cleanup_errors:
            raise RuntimeError(f"agent cleanup failed: {cleanup_errors}")


def _verify_profile(profile_dir: Path) -> dict:
    required_files = ("summary.json", "summary.md", "agprof.trace.json")
    missing = [
        name
        for name in required_files
        if not (profile_dir / name).is_file() or (profile_dir / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"agprof did not create nonempty required artifacts: {missing}")

    summary = json.loads((profile_dir / "summary.json").read_text())
    trace = json.loads((profile_dir / "agprof.trace.json").read_text())
    run_metrics = summary.get("run_metrics", {})
    llm_metrics = summary.get("llm_metrics", {})
    tool_metrics = summary.get("tool_metrics", {})
    span_rows = summary.get("span_metrics", [])
    span_by_label = {row.get("label"): row for row in span_rows}
    gate_rows = [
        row for row in tool_metrics.get("by_tool", []) if row.get("name") == "lifecycle_gate"
    ]
    trace_events = trace.get("traceEvents", [])
    completed_run_events = [
        event
        for event in trace_events
        if event.get("ph") == "X" and re.match(r"^run\d+:", str(event.get("name", "")))
    ]
    trace_names = {str(event.get("name", "")) for event in trace_events}

    checks = {
        "schema_version_4": summary.get("schema_version") == 4,
        "measured_data": summary.get("data_source") == "measured",
        "positive_duration": float(summary.get("duration_ms", 0)) > 0,
        "raw_samples_present": int(summary.get("sampling", {}).get("raw_samples", 0)) > 0,
        "exact_run_metrics": all(
            run_metrics.get(key) == value for key, value in EXPECTED_RUNS.items()
        ),
        "minimum_llm_calls": int(llm_metrics.get("calls", 0)) >= MINIMUM_LLM_CALLS,
        "exact_gate_tool_metrics": len(gate_rows) == 1
        and all(
            gate_rows[0].get(key) == EXPECTED_GATE_CALLS
            for key in ("started", "completed", "succeeded")
        ),
        "required_e2e_spans": REQUIRED_PROFILE_SPANS.issubset(span_by_label),
        "exact_custom_gate_spans": span_by_label.get("e2e:tool_gate", {}).get("calls")
        == EXPECTED_GATE_CALLS,
        "no_incomplete_spans": summary.get("incomplete_spans") == [],
        "trace_is_milliseconds": trace.get("displayTimeUnit") == "ms",
        "trace_has_events": bool(trace_events),
        "trace_has_exact_completed_runs": len(completed_run_events) == EXPECTED_RUNS["completed"],
        "trace_has_required_e2e_spans": REQUIRED_PROFILE_SPANS.issubset(trace_names),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "agprof evidence checks failed: "
            f"{failed}; runs={run_metrics}; llm_calls={llm_metrics.get('calls')}; "
            f"gate_rows={gate_rows}"
        )
    return {
        "checks": checks,
        "run_metrics": {key: run_metrics.get(key) for key in EXPECTED_RUNS},
        "llm_calls": llm_metrics.get("calls"),
        "gate_tool_metrics": gate_rows[0],
        "raw_samples": summary.get("sampling", {}).get("raw_samples"),
        "completed_run_trace_events": len(completed_run_events),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("LIFECYCLE_MODEL", "gpt-5.6-luna"))
    parser.add_argument(
        "--control-timeout",
        type=float,
        default=180.0,
        help="seconds allowed for each live model or control boundary",
    )
    parser.add_argument(
        "--cleanup-timeout",
        type=float,
        default=120.0,
        help="seconds allowed for deterministic agent cleanup",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            os.environ.get(
                "LIFECYCLE_RUN_ROOT",
                Path(__file__).resolve().parent.parent / "runs",
            )
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    profile_requested = os.environ.get("AGENCY_PROFILE", "").strip().lower() in {
        "1",
        "true",
    }
    if profile_requested and agprof.profile_scope() != "workload":
        raise SystemExit(
            "lifecycle_api.py requires AGENCY_PROFILE_SCOPE=workload so artifacts "
            "exist before in-process verification"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = args.run_root.resolve() / f"{timestamp}_lifecycle_api"
    run_dir.mkdir(parents=True, exist_ok=False)
    Agent.log_dir = run_dir / "logs"
    Agent.output_dir = run_dir / "agent_output"
    verification_path = run_dir / "verification.json"
    profile_dir = Path(os.environ.get("AGENCY_PROFILE_DIR", "agprof_trace")).resolve()
    evidence = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "harness": "native",
        "run_dir": str(run_dir),
        "profile_requested": profile_requested,
        "profile_dir": str(profile_dir) if profile_requested else None,
        "expected_run_metrics": EXPECTED_RUNS,
        "expected_gate_calls": EXPECTED_GATE_CALLS,
        "events": [],
    }

    print(f"Lifecycle run directory: {run_dir}", flush=True)
    try:
        config = _config(args.model)
        with agprof.workload():
            _exercise(config, evidence, args.control_timeout, args.cleanup_timeout)
        if profile_requested:
            profile_evidence = _verify_profile(profile_dir)
            evidence["profile"] = profile_evidence
            _record(evidence, "profile_verified", **profile_evidence)
            print("PASS measured profiler artifacts", flush=True)
        evidence["status"] = "passed"
        return_code = 0
    except BaseException as exc:
        evidence["status"] = "failed"
        evidence["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()
        return_code = 1
    finally:
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(verification_path, evidence)
        print(f"Verification: {verification_path}", flush=True)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
