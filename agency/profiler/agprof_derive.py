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
    parent_context=None,
    span_attributes: "dict | None" = None,
    derive_tools: bool = True,
    skip_tool_call_ids: "set[str] | None" = None,
    before_derive_tools=None,
) -> "set[str]":
    """Record one successfully-completed dispatch as ``turn{n}``, deriving
    ``tool:{name}`` spans for whichever of the *previous* dispatch's tool
    calls resolved since it ended. Only call this for a dispatch that
    actually produced a response — there is nothing to diff or to attribute
    a turn to otherwise.
    """
    if not agprof.enabled() or token is None:
        return set()
    request_messages = list(request_messages or [])
    skip_tool_call_ids = set(skip_tool_call_ids or ())
    tool_spans: "list[tuple]" = []
    with _lock:
        prev = _state.get(token)
        index = 0 if prev is None else prev["index"] + 1
        if derive_tools and prev is not None and len(request_messages) >= prev["transcript_len"]:
            tool_names = _tool_names_by_call_id(prev["response_message"])
            gap_ns = max(0, start_perf_ns - prev["end_perf_ns"])
            for message in request_messages[prev["transcript_len"] :]:
                if not isinstance(message, dict) or message.get("role") != "tool":
                    continue
                call_id = message.get("tool_call_id")
                if call_id in skip_tool_call_ids:
                    continue
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

    correlated_attributes = dict(span_attributes or {})
    candidate_tool_call_ids = {
        call_id
        for _name, _sp, _ep, _sw, _ew, call_id in tool_spans
        if isinstance(call_id, str) and call_id
    }
    if candidate_tool_call_ids and before_derive_tools is not None:
        allowed_tool_call_ids = before_derive_tools(candidate_tool_call_ids)
        if allowed_tool_call_ids is not None:
            allowed_tool_call_ids = set(allowed_tool_call_ids)
            tool_spans = [
                span
                for span in tool_spans
                if not isinstance(span[-1], str) or span[-1] in allowed_tool_call_ids
            ]
    derived_tool_call_ids = {
        call_id
        for _name, _sp, _ep, _sw, _ew, call_id in tool_spans
        if isinstance(call_id, str) and call_id
    }
    for name, sp_ns, ep_ns, sw_ns, ew_ns, call_id in tool_spans:
        metadata = dict(correlated_attributes)
        if call_id:
            metadata["tool_call_id"] = call_id
        agprof.record_derived_span(
            f"tool:{name}",
            start_perf_ns=sp_ns,
            end_perf_ns=ep_ns,
            start_wall_ns=sw_ns,
            end_wall_ns=ew_ns,
            metadata=metadata,
            parent_context=parent_context,
        )
    turn_metadata = {**correlated_attributes, "outcome": "success"}
    agprof.record_derived_span(
        f"turn{index}",
        start_perf_ns=start_perf_ns,
        end_perf_ns=end_perf_ns,
        start_wall_ns=start_wall_ns,
        end_wall_ns=end_wall_ns,
        metadata=turn_metadata,
        parent_context=parent_context,
    )
    return derived_tool_call_ids


__all__ = ["on_dispatch", "forget"]
