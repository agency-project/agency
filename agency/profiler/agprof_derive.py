"""Derive turn/tool profiler spans from terminus dispatch transcripts.

Tier 1 of docs/Design_profiler_harness_integration.md §5.2: every harness
engine resends its *full* conversation on each LLM dispatch, and
`agllm_terminus.py` already keeps the most recent one per token for its own
history-unification purposes. That means turn and tool structure is
recoverable host-side, for all five engines, with zero container
cooperation:

  - dispatch *n* for a token *is* turn *n* — its own already-measured
    ``llm:attempt[0]`` span is exact per-turn latency, TTFT, and tokens;
  - whatever a dispatch's request-messages array gained since the previous
    dispatch's transcript for that token is that turn's tool-call results —
    diffing two already-recorded message lists, nothing new to capture;
  - tool *duration* has no direct measurement (the tool ran somewhere the
    terminus never observes), so it is approximated as the gap between the
    previous dispatch ending and this one starting, charged to every tool
    call resolved in that gap — exact when there was one call, an upper
    bound when several ran in parallel.

Every span this module emits goes through ``agprof.record_derived_span``,
which tags ``metadata["timing"] = "derived"`` and leaves cpu/runqueue
unset — see that function's docstring for why. Call ``on_dispatch()`` at
each dispatch's actual completion, not from ``_record_transcript()``
directly: that hook fires on every streamed chunk (agllm_terminus.py),
and diffing there would manufacture a turn per chunk instead of per
dispatch.
"""

from __future__ import annotations

import threading

from . import agprof

_lock = threading.Lock()
# token -> {"index", "transcript_len", "response_message", "end_perf_ns", "end_wall_ns"}
_state: "dict[str, dict]" = {}


def forget(token: "str | None") -> None:
    """Drop derivation state for *token*. Call from the terminus's own
    ``unregister()`` so a run's state does not outlive its registration."""
    if token is None:
        return
    with _lock:
        _state.pop(token, None)


def _tool_names_by_call_id(response_message: "dict | None") -> "dict[str, str]":
    """``tool_call_id -> function name`` from an assistant message's
    ``tool_calls`` — absent or malformed entries fall back to "unknown"
    rather than raising, since this is best-effort telemetry, not a
    contract any engine promises to honor exactly."""
    if not response_message:
        return {}
    names = {}
    for call in response_message.get("tool_calls") or []:
        call_id = call.get("id")
        if call_id:
            names[call_id] = call.get("function", {}).get("name") or "unknown"
    return names


def on_dispatch(
    token: "str | None",
    request_messages,
    response_message: "dict | None",
    *,
    start_perf_ns: int,
    start_wall_ns: int,
    end_perf_ns: int,
    end_wall_ns: int,
) -> None:
    """Record one successfully-completed dispatch as ``turn{n}``, deriving
    ``tool:{name}`` spans for whichever of the *previous* dispatch's tool
    calls resolved since it ended. Only call this for a dispatch that
    actually produced a response — there is nothing to diff or to attribute
    a turn to otherwise.
    """
    if not agprof.enabled() or token is None:
        return
    request_messages = list(request_messages or [])
    tool_spans: "list[tuple]" = []
    with _lock:
        prev = _state.get(token)
        index = 0 if prev is None else prev["index"] + 1
        if prev is not None and len(request_messages) >= prev["transcript_len"]:
            tool_names = _tool_names_by_call_id(prev["response_message"])
            gap_ns = max(0, start_perf_ns - prev["end_perf_ns"])
            for message in request_messages[prev["transcript_len"] :]:
                if not isinstance(message, dict) or message.get("role") != "tool":
                    continue
                call_id = message.get("tool_call_id")
                tool_spans.append(
                    (
                        tool_names.get(call_id, "unknown"),
                        prev["end_perf_ns"],
                        prev["end_perf_ns"] + gap_ns,
                        prev["end_wall_ns"],
                        prev["end_wall_ns"] + gap_ns,
                        call_id,
                    )
                )
        # else: the array shrank -- compaction. Nothing before this dispatch
        # is attributable to a diff anymore; the state written below starts
        # a fresh baseline rather than reporting a spurious rewind.
        _state[token] = {
            "index": index,
            "transcript_len": len(request_messages) + 1,  # + this dispatch's own response
            "response_message": response_message,
            "end_perf_ns": end_perf_ns,
            "end_wall_ns": end_wall_ns,
        }

    for name, sp_ns, ep_ns, sw_ns, ew_ns, call_id in tool_spans:
        agprof.record_derived_span(
            f"tool:{name}",
            start_perf_ns=sp_ns,
            end_perf_ns=ep_ns,
            start_wall_ns=sw_ns,
            end_wall_ns=ew_ns,
            metadata={"tool_call_id": call_id} if call_id else {},
        )
    agprof.record_derived_span(
        f"turn{index}",
        start_perf_ns=start_perf_ns,
        end_perf_ns=end_perf_ns,
        start_wall_ns=start_wall_ns,
        end_wall_ns=end_wall_ns,
        metadata={"outcome": "success"},
    )


__all__ = ["on_dispatch", "forget"]
