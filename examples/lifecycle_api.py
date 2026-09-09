"""Live conformance test for Agency's public submission and lifecycle APIs.

This example uses the native harness and a real OpenAI model. It exercises the
public ``Agent`` surface: ``run()`` returns a bare, pending ``agdata`` directly
(no wrapper object), ``queue_message()`` is a plain enqueue with no return
value (ordering into the context chain is already guaranteed synchronously,
before it returns), ``agent.cancel(handle)``, and ``agent.pause()``/``resume()``
as a real admission gate. Host-side tool-call events make the concurrency
claims observable instead of inferring them from model prose.

``agent.redirect()`` is checked only for still raising ``NotImplementedError``
-- it has no live delivery mechanism yet (no harness has a mid-attempt
injection channel today). There is no agent teardown API anymore either
(``destroy()`` was removed -- cleanup is just letting an ``Agent`` go out of
scope, same as any other Python object).

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
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agency import (
    Agent,
    agdata,
    agprof,
    agrawstring,
    agskill,
    agtool,
)
from agency.configs.agconfig import agconfig as agconfig_cls, llmconfig

ORDERED_MEMORY = "ORDERED-CONTEXT-731"
CANCEL_MEMORY = "CANCEL-CONTEXT-947"
ORDERED_MARKER = "ORDERED-MARKER-OK"
CANCEL_MARKER = "CANCEL-RUNNING-OK"
PAUSED_MARKER = "PAUSED-RUN-OK"
REQUIRED_PROFILE_SPANS = {
    "e2e:ordered_chain",
    "e2e:pause_resume",
    "e2e:cancellation",
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


def _config(model: str) -> agconfig_cls:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    return agconfig_cls(
        llmconfig(
            provider="openai",
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
            "Do not call any other tool. Wait for the tool result, then respond "
            "with exactly expected_marker and nothing else -- no quotes, "
            "Markdown, or explanation."
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


def _expect(
    evidence: dict,
    label: str,
    exception_type: type[BaseException],
    operation: Callable[[], object],
) -> None:
    try:
        operation()
    except exception_type as exc:
        _record(
            evidence,
            "expected_exception",
            check=label,
            exception=type(exc).__name__,
            message=str(exc),
        )
        print(f"PASS {label}", flush=True)
        return
    raise AssertionError(f"{label} did not raise {exception_type.__name__}")


def _successful(handle: agdata, label: str, timeout_s: float) -> dict:
    handle.wait(timeout=timeout_s)
    payload = handle.to_dict()
    if payload.get("error"):
        raise RuntimeError(f"{label} failed: {payload['error']}")
    return payload


def _terminal_error(handle: agdata, expected: str, label: str, timeout_s: float) -> dict:
    handle.wait(timeout=timeout_s)
    payload = handle.to_dict()
    if payload.get("error") != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {payload!r}")
    return payload


def _raw(payload: dict, field: str) -> str:
    return str(payload[field]).strip()


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
    """queue_message() ordering and a bare-agdata result handle."""
    with agprof.span("e2e:ordered_chain"):
        ticket = _GATE.arm("ordered-head")

        # A plain enqueue: nothing is returned, ordering into the context
        # chain already happened synchronously before this call returns.
        assert ag.queue_message(f"PUBLIC_API_MEMORY: {ORDERED_MEMORY}") is None

        head = ag.run(controlled, agdata(label=ticket.label, expected_marker=ORDERED_MARKER))
        tail = ag.run(recall, agdata(question="Recall the public API memory codeword."))
        ready = ag.run(echo, agdata(result="RESULT-FIELD-COLLISION"))

        _check(
            evidence,
            type(head) is agdata and type(tail) is agdata and type(ready) is agdata,
            "run() returns a bare agdata -- no wrapper object",
        )
        _check(
            evidence,
            all(node.is_pending() for node in (head, tail, ready)),
            "freshly submitted invocations are pending",
        )

        _wait_for_gate(ticket, timeout_s)
        ticket.release.set()

        head_payload = _successful(head, "ordered head", timeout_s)
        _check(
            evidence,
            _raw(head_payload, "marker") == ORDERED_MARKER,
            "queued invocation reaches the tool gate and answers",
            marker=_raw(head_payload, "marker"),
        )

        tail_payload = _await_sync(tail, timeout_s).to_dict()
        _check(
            evidence,
            _raw(tail_payload, "memory") == ORDERED_MEMORY,
            "later invocation observes the queued message's context",
            memory=_raw(tail_payload, "memory"),
        )

        ready.wait(timeout=timeout_s)
        _check(
            evidence,
            _raw(ready.to_dict(), "result") == "RESULT-FIELD-COLLISION"
            and str(ready.result).strip() == "RESULT-FIELD-COLLISION",
            "a skill output field literally named 'result' proxies through cleanly",
        )


def _exercise_pause_resume(ag: Agent, echo: agskill, evidence: dict, timeout_s: float) -> None:
    """agent.pause()/resume() as a real admission gate for not-yet-launched work.

    This is only ever an admission gate -- there is no way to freeze an
    already-running invocation (that's separately deferred, OS-level/cgroup
    work), so this pauses *before* submitting anything.
    """
    with agprof.span("e2e:pause_resume"):
        ag.pause()
        _check(evidence, ag.is_paused() is True, "pause() sets is_paused()")

        paused = ag.run(echo, agdata(result=PAUSED_MARKER))
        _check(
            evidence,
            not _wait_until(lambda: not paused.is_pending(), 1.0),
            "a paused agent holds a ready run unlaunched",
        )

        ag.resume()
        _check(evidence, ag.is_paused() is False, "resume() clears is_paused()")

        payload = _successful(paused, "paused-then-resumed run", timeout_s)
        _check(
            evidence,
            _raw(payload, "result") == PAUSED_MARKER,
            "resumed run proceeds and completes",
        )


def _exercise_cancellation(
    ag: Agent,
    controlled: agskill,
    recall: agskill,
    evidence: dict,
    timeout_s: float,
) -> None:
    """Cancel a not-yet-started (blocked) invocation, and a running one."""
    with agprof.span("e2e:cancellation"):
        assert ag.queue_message(f"PUBLIC_API_MEMORY: {CANCEL_MEMORY}") is None

        # A single agent only ever has one invocation running at a time --
        # the context-dependency chain alone guarantees this, no admission
        # gate needed. Submitting `never` while `holder` is still running the
        # tool call leaves `never` genuinely blocked (not yet admitted to an
        # engine at all), so cancelling it here is deterministic.
        holder_ticket = _GATE.arm("cancel-holder")
        holder = ag.run(controlled, agdata(label=holder_ticket.label, expected_marker="HOLDER-OK"))
        _wait_for_gate(holder_ticket, timeout_s)

        never = ag.run(
            recall,
            agdata(question="Recall the public API memory during cancellation."),
        )
        ag.cancel(never)
        never_payload = _terminal_error(
            never, "agent invocation cancelled", "blocked cancellation", timeout_s
        )
        _check(
            evidence,
            never_payload["error"] == "agent invocation cancelled",
            "cancelling a not-yet-started invocation is immediate and terminal",
        )
        ag.cancel(never)  # idempotent -- no-op the second time
        _check(evidence, never.to_dict() == never_payload, "cancel is idempotent")

        holder_ticket.release.set()
        _successful(holder, "cancel holder", timeout_s)

        # `later` depends on `never`'s (cancelled) context -- a cancelled
        # invocation still passes its predecessor's context through
        # unchanged, so the chain is never broken by a cancellation.
        later = ag.run(
            recall,
            agdata(question="Recall the public API memory after cancellation."),
        )
        later_payload = _successful(later, "later cancellation survivor", timeout_s)
        _check(
            evidence,
            _raw(later_payload, "memory") == CANCEL_MEMORY,
            "a later invocation survives a cancelled predecessor",
        )

        # Cancelling a *running* invocation is cooperative today (no OS-level
        # kill yet -- deferred to a follow-up cgroup-based redesign): the
        # harness keeps running and the tool call still completes normally,
        # but the engine's post-execution checkpoint discards that real
        # result in favor of the cancellation once it observes the flag.
        running_ticket = _GATE.arm("cancel-running")
        running = ag.run(
            controlled,
            agdata(label=running_ticket.label, expected_marker=CANCEL_MARKER),
        )
        _wait_for_gate(running_ticket, timeout_s)
        ag.cancel(running)
        running_ticket.release.set()
        running_payload = _terminal_error(
            running, "agent invocation cancelled", "cancel during a live tool call", timeout_s
        )
        _check(
            evidence,
            running_payload["error"] == "agent invocation cancelled",
            "a running invocation's real result is discarded once cancelled",
        )

        _expect(
            evidence,
            "agent.redirect() is not yet implemented",
            NotImplementedError,
            lambda: ag.redirect("no live delivery channel exists yet"),
        )


def _exercise(config: agconfig_cls, evidence: dict, timeout_s: float) -> None:
    controlled, echo, recall = _skills()

    def new_agent(name: str) -> Agent:
        return Agent(agname=name, agconfig=config, harness="native")

    try:
        _exercise_ordered_chain(
            new_agent("public-api-ordered"), controlled, echo, recall, evidence, timeout_s
        )
        _exercise_pause_resume(new_agent("public-api-pause"), echo, evidence, timeout_s)
        _exercise_cancellation(
            new_agent("public-api-cancel"), controlled, recall, evidence, timeout_s
        )
        calls = _GATE.calls()
        _check(
            evidence,
            calls == ["ordered-head", "cancel-holder", "cancel-running"],
            "host observed the exact three expected gate calls",
            calls=calls,
        )
    finally:
        # No agent teardown API anymore -- destroy() was removed. Cleanup is
        # just letting each Agent go out of scope, same as any Python object.
        _GATE.release_all()


def _verify_profile(profile_dir: Path, evidence: dict) -> dict:
    """Confirm profiler artifacts exist and are internally consistent.

    Deliberately does not assert exact hardcoded run/LLM-call counts (an
    earlier version of this example did) -- this has no way to execute a live
    model run to confirm such numbers stay correct, so it checks structure
    and cross-references against what this script itself observed instead.
    """
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
    span_by_label = {row.get("label"): row for row in summary.get("span_metrics", [])}
    gate_rows = [
        row for row in tool_metrics.get("by_tool", []) if row.get("name") == "lifecycle_gate"
    ]
    trace_events = trace.get("traceEvents", [])
    gate_calls_observed = len(_GATE.calls())

    checks = {
        "schema_version_4": summary.get("schema_version") == 4,
        "measured_data": summary.get("data_source") == "measured",
        "positive_duration": float(summary.get("duration_ms", 0)) > 0,
        "raw_samples_present": int(summary.get("sampling", {}).get("raw_samples", 0)) > 0,
        "at_least_one_llm_call": int(llm_metrics.get("calls", 0)) >= 1,
        "gate_tool_metrics_match_observed_calls": len(gate_rows) == 1
        and gate_rows[0].get("started") == gate_calls_observed
        and gate_rows[0].get("completed") == gate_calls_observed
        and gate_rows[0].get("succeeded") == gate_calls_observed,
        "required_e2e_spans": REQUIRED_PROFILE_SPANS.issubset(span_by_label),
        "custom_gate_spans_match_observed_calls": span_by_label.get("e2e:tool_gate", {}).get(
            "calls"
        )
        == gate_calls_observed,
        "no_incomplete_spans": summary.get("incomplete_spans") == [],
        "trace_is_milliseconds": trace.get("displayTimeUnit") == "ms",
        "trace_has_events": bool(trace_events),
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
        "run_metrics": run_metrics,
        "llm_calls": llm_metrics.get("calls"),
        "gate_tool_metrics": gate_rows[0],
        "gate_calls_observed": gate_calls_observed,
        "raw_samples": summary.get("sampling", {}).get("raw_samples"),
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
    profile_requested = os.environ.get("AGENCY_PROFILE", "").strip().lower() in {"1", "true"}
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
        "events": [],
    }

    print(f"Lifecycle run directory: {run_dir}", flush=True)
    try:
        config = _config(args.model)
        with agprof.workload():
            _exercise(config, evidence, args.control_timeout)
        if profile_requested:
            profile_evidence = _verify_profile(profile_dir, evidence)
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
