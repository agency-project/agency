"""Compact, evidence-only analysis of a completed execution.

Vertices are operation boundaries, not syscalls.  Duration edges and recorded
prerequisites form a temporal DAG; nested intervals remain alternatives rather
than additive weights.  A containing span is an ownership relation, never by
itself proof that a child blocked its parent.  Request envelopes are gates, not
billable work.  Missing instrumentation yields a candidate path and explicit
coverage, rather than a fabricated system-wide execution chain.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
import json
import math
from urllib.parse import quote

from .causal_graph import build_graph

_INCOMPLETE = {"running", "incomplete", "interrupted"}
MAX_PROJECTED_OPERATIONS = 50_000
_METRICS = (
    "cpu_ms",
    "runqueue_ms",
    "blocked_ms",
    "cpu_pct",
    "ttft_ms",
    "streaming_ms",
    "stream_duration_ms",
    "input_tokens",
    "output_tokens",
    "new_prompt_tokens",
    "retry_count",
    "attempt",
    "attempt_index",
    "http_status",
    "error_type",
    "pid",
    "ppid",
    "program",
    "path",
    "executable_path",
    "command_line",
    "rss_mb",
    "memory_mb",
    "io_read_bytes",
    "io_write_bytes",
    "cpu_time_s",
    "exit_status",
    "agency.process_identity",
    "agency.namespace_pid",
    "agency.pid_namespace",
    "agency.process_start_ticks",
    "agency.sandbox_id",
    "stream",
    "streaming",
    "llm.streaming",
    "stop_reason",
    "request_attribution",
    "tool_batch_id",
    "preceding_tool_call_id",
    "preceding_model_tool_call_id",
)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _items(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [value] if value else []
    return value if isinstance(value, list) else []


def _identity(value):
    return f"{value:016x}" if isinstance(value, int) else str(value) if value is not None else None


def _union(intervals):
    total, right = 0.0, -math.inf
    for start, end in sorted(intervals):
        if start > right:
            total += end - start
        elif end > right:
            total += end - right
        right = max(right, end)
    return total


def _band(action):
    kind, name = action.get("kind"), action.get("name", "")
    if name == "sync:submission":
        return "System"
    if kind == "model" or name.startswith("llm:"):
        return "LLM"
    if kind in {"queue", "dependency", "wait"} or name.startswith("sync:"):
        return "Wait"
    if kind in {"process", "system", "setup", "control", "request", "execution"}:
        return "System"
    return "Tool"


def _container(action):
    metadata = action.get("metadata", {})
    return (
        bool(action.get("container"))
        or action.get("kind") in {"request", "execution"}
        or (
            action.get("name") in {"request:submission_to_completion", "engine:execute"}
            or metadata.get("data_span_name") == "request:submission_to_completion"
            or (
                metadata.get("request_id")
                and metadata.get("request_kind")
                and metadata.get("skill")
                and action.get("name", "").startswith("run")
            )
        )
    )


def longest_path(nodes, edges):
    """O(V+E) validated weighted DAG analysis with deterministic tie breaking.

    Inputs use boundary IDs and temporal coordinates.  Cyclic blocking evidence
    is rejected.  Malformed or backwards links are excluded and reported.  Equal
    elapsed paths prefer model/tool evidence to a containing wait interval.
    """
    lookup = {n["id"]: n for n in nodes}
    incoming, outgoing, malformed = defaultdict(list), defaultdict(list), []
    for edge in edges:
        if not edge.get("blocking") or edge.get("confidence") not in {"recorded", "reconstructed"}:
            continue
        source, target = edge.get("source"), edge.get("target")
        reason = None
        if source not in lookup or target not in lookup:
            reason = "Dependency endpoint unavailable"
        elif source == target:
            reason = "Self dependency"
        elif not finite(edge.get("weight")) or edge["weight"] < 0:
            reason = "Invalid dependency weight"
        elif not finite(lookup[source].get("time")) or not finite(lookup[target].get("time")):
            reason = "Dependency timestamp unavailable"
        elif lookup[source]["time"] > lookup[target]["time"] + 1e-6:
            reason = "Dependency contradicts recorded timestamps"
        elif edge["weight"] > lookup[target]["time"] - lookup[source]["time"] + 1e-6:
            reason = "Dependency weight exceeds its elapsed interval"
        if reason:
            malformed.append({"edge_id": edge.get("id"), "reason": reason})
            continue
        incoming[target].append(edge)
        outgoing[source].append(edge)
    degree = {key: len(incoming[key]) for key in lookup}
    ready = deque(sorted(key for key in lookup if degree[key] == 0))
    distance, quality, previous = dict.fromkeys(lookup, 0.0), dict.fromkeys(lookup, 0.0), {}
    order = []
    while ready:
        key = ready.popleft()
        order.append(key)
        for edge in outgoing[key]:
            target = edge["target"]
            value = distance[key] + edge["weight"]
            evidence = quality[key] + edge.get("quality", 0)
            if (
                value > distance[target] + 1e-9
                or (abs(value - distance[target]) <= 1e-9 and evidence > quality[target])
                or target not in previous
            ):
                distance[target], quality[target], previous[target] = value, evidence, edge
            degree[target] -= 1
            if not degree[target]:
                ready.append(target)
    if len(order) != len(lookup):
        return {
            "duration": None,
            "node_ids": [],
            "edge_ids": [],
            "slack": {},
            "malformed_edges": malformed,
            "error": "Cyclic blocking evidence; critical path unavailable",
        }
    tail = dict.fromkeys(lookup, 0.0)
    for key in reversed(order):
        tail[key] = max((e["weight"] + tail[e["target"]] for e in outgoing[key]), default=0)
    end = max(order, key=lambda key: (distance[key], quality[key], key)) if order else None
    duration = distance.get(end, 0.0)
    path_nodes, path_edges = [], []
    while end is not None:
        path_nodes.append(end)
        edge = previous.get(end)
        if edge is None:
            break
        path_edges.append(edge["id"])
        end = edge["source"]
    return {
        "duration": duration,
        "node_ids": path_nodes[::-1],
        "edge_ids": path_edges[::-1],
        "slack": {key: max(0.0, duration - distance[key] - tail[key]) for key in order},
        "malformed_edges": malformed,
    }


def build_profile(run):
    """Project normalized historical spans and generation-qualified metrics.

    ``run.dependencies`` may contain explicit operation IDs and start/end phases.
    Runtime prerequisite request IDs are resolved through instrumented request
    completion and admission boundaries.  Timestamp overlap never creates an
    edge, ownership, or a contention claim.
    """
    actions, issues, limitations = [], [], []
    identities = Counter(
        str(raw.get("id", ""))
        for raw in run.get("actions", [])
        if raw.get("kind") != "syscall" and not raw.get("name", "").startswith("syscall:")
    )
    seen = set()
    for raw in run.get("actions", []):
        if raw.get("kind") == "syscall" or raw.get("name", "").startswith("syscall:"):
            continue
        key = str(raw.get("id", ""))
        if not key or key in seen or identities[key] > 1:
            issues.append(
                {"operation_id": key, "reason": "Missing or duplicate operation identity"}
            )
            continue
        seen.add(key)
        action = {**raw, "id": key, "agent": str(raw.get("agent") or "unattributed")}
        # Existing causal projection assumes numeric starts for identity rejection.
        # Do not repair absent timing into a measured start.
        if not finite(action.get("start")):
            action["start"], action["missing_start"] = 0, True
        if (
            action.get("missing_end")
            or not finite(action.get("duration"))
            or action.get("duration", 0) < 0
        ):
            action["outcome"] = "incomplete"
        action.setdefault("name", key)
        action.setdefault("kind", "tool")
        actions.append(action)
    # Reuse the established canonical exchange/tool-identity correlation.  Only
    # selected operations enter this graph; low-level traces never get copied.
    graph = build_graph(
        {**run, "actions": actions},
        max_unmeasured_tools=max(0, MAX_PROJECTED_OPERATIONS - len(actions)),
    )
    graph_nodes = {n["id"]: n for n in graph["nodes"]}
    source_actions = {a["id"]: a for a in actions}
    operations, boundaries, aliases = {}, {}, {}
    for value in graph["operations"]:
        action = source_actions.get(value["id"], {})
        metadata = action.get("metadata") or {}
        start = graph_nodes.get(value.get("start_node"), {}).get("time")
        end = graph_nodes.get(value.get("end_node"), {}).get("time")
        duration = value.get("duration")
        band = _band(action) if action else value["band"]
        op = {
            "id": value["id"],
            "action_id": value.get("action_id"),
            "agent": value["agent"],
            "kind": action.get("kind", value.get("kind")),
            "band": band,
            "label": value["label"],
            "start": start,
            "end": end,
            "duration": duration,
            "status": action.get("outcome", value.get("outcome", "unknown")),
            "critical": False,
            "slack": None,
            "parent_id": None,
            "child_ids": [],
            "model": action.get("model"),
            "request_id": action.get("request_id")
            or metadata.get("request_id")
            or metadata.get("agency.run_id"),
            "container": _container(action),
            "metrics": {},
            "evidence": [],
            "limitations": [],
            "process_ids": [],
            "links": {},
        }
        for key in ("tokens", "output_tokens", "timing"):
            if action.get(key) is not None:
                op["metrics"][key] = action[key]
        for key in _METRICS:
            if metadata.get(key) is not None:
                op["metrics"][key] = metadata[key]
        if band == "Tool":
            for key, value in (
                ("command_preview", action.get("command") or action.get("arguments")),
                ("result_preview", action.get("result")),
                ("error_message", action.get("error")),
                ("call_id", action.get("call_id") or metadata.get("call_id")),
                ("model_tool_call_id", action.get("model_tool_call_id")),
            ):
                if value:
                    op["metrics"][key] = str(value)[:1200]
        if start is not None and end is not None and duration is not None:
            op["evidence"].append("Recorded operation start and completion")
        else:
            op["limitations"].append("Complete operation timing unavailable")
        if band == "LLM":
            evidence = action.get("causal_evidence") or {}
            op["metrics"]["exchange_complete"] = bool(evidence.get("complete"))
            streaming = metadata.get(
                "streaming", metadata.get("stream", metadata.get("llm.streaming"))
            )
            if streaming is False and finite(metadata.get("ttft_ms")):
                op["metrics"].pop("ttft_ms", None)
                op["metrics"]["response_latency_ms"] = metadata["ttft_ms"]
                op["limitations"].append(
                    "Non-streaming timing records first complete response; first-token latency is unavailable"
                )
            elif (
                finite(metadata.get("ttft_ms"))
                and duration is not None
                and not 0 <= metadata["ttft_ms"] <= duration * 1000 + 1e-6
            ):
                op["metrics"].pop("ttft_ms", None)
                op["metrics"]["recorded_ttft_ms"] = metadata["ttft_ms"]
                op["limitations"].append(
                    "Recorded first-token timing contradicts request duration; timing breakdown unavailable"
                )
            elif finite(metadata.get("ttft_ms")) and duration is not None:
                op["metrics"]["post_first_token_ms"] = max(0, duration * 1000 - metadata["ttft_ms"])
                op["evidence"].append(
                    "Post-first-token elapsed interval computed from request end and recorded TTFT"
                )
                if (
                    metadata.get("stream") is True
                    or metadata.get("streaming") is True
                    or metadata.get("llm.streaming") is True
                ):
                    op["metrics"]["streaming_ms"] = op["metrics"]["post_first_token_ms"]
        if metadata.get("unidentified_dependency_count"):
            count = metadata["unidentified_dependency_count"]
            op["limitations"].append(
                f"{count} runtime prerequisites lack producer execution identities."
            )
            issues.append(
                {
                    "operation_id": op["id"],
                    "reason": "Runtime prerequisites lack producer identities",
                    "count": count,
                }
            )
        if run.get("id"):
            op["links"]["profiler"] = (
                "/profiler?run="
                + quote(str(run["id"]))
                + "&event="
                + quote(str(op["action_id"] or op["id"]))
                + "&agent="
                + quote(op["agent"])
            )
        operations[op["id"]] = op
        aliases[op["id"]] = op["id"]
        for name in ("span_id", "agency.span_id", "trace_span_id"):
            if metadata.get(name) is not None:
                identity = _identity(metadata[name])
                if identity in aliases and aliases[identity] != op["id"]:
                    aliases[identity] = None
                else:
                    aliases[identity] = op["id"]
        for phase, at in (("start", start), ("end", end)):
            if finite(at):
                boundaries[f"{op['id']}:{phase}"] = {
                    "id": f"{op['id']}:{phase}",
                    "operation": op["id"],
                    "time": at,
                }

    edges, edge_lookup = [], {}

    def link(
        source,
        target,
        kind,
        evidence,
        *,
        source_phase="end",
        target_phase="start",
        confidence="recorded",
        blocking=True,
        duration=False,
    ):
        source, target = aliases.get(str(source), source), aliases.get(str(target), target)
        left, right = f"{source}:{source_phase}", f"{target}:{target_phase}"
        key = f"{kind}:{left}->{right}"
        if key in edge_lookup:
            return edge_lookup[key]
        source_time = boundaries.get(left, {}).get("time")
        target_time = boundaries.get(right, {}).get("time")
        valid = left in boundaries and right in boundaries and source != target or duration
        if not valid or not finite(source_time) or not finite(target_time):
            issues.append(
                {
                    "operation_id": target,
                    "reason": "Dependency endpoint or timing unavailable",
                    "source": source,
                }
            )
            return None
        if source_time > target_time + 1e-6:
            confidence, blocking = "unknown", False
            issues.append(
                {
                    "operation_id": target,
                    "reason": "Dependency contradicts recorded timestamps",
                    "source": source,
                }
            )
        if (
            operations[source]["agent"] != operations[target]["agent"]
            and run.get("clock", {}).get("shared") is not True
            and blocking
        ):
            confidence, blocking = "unknown", False
            issues.append(
                {"operation_id": target, "reason": "Cross-agent clock alignment unavailable"}
            )
        edge = {
            "id": key,
            "source": source,
            "target": target,
            "source_phase": source_phase,
            "target_phase": target_phase,
            "source_time": source_time,
            "target_time": target_time,
            "kind": kind,
            "confidence": confidence,
            "blocking": blocking,
            "evidence": evidence,
            "critical": False,
            "_source": left,
            "_target": right,
            "_duration": duration,
            "_weight": max(0, target_time - source_time) if blocking else 0,
        }
        edges.append(edge)
        edge_lookup[key] = edge
        return edge

    for op in operations.values():
        if op["duration"] is not None and not op["container"] and op["kind"] != "process":
            link(
                op["id"],
                op["id"],
                "duration",
                "Recorded operation boundaries",
                source_phase="start",
                target_phase="end",
                duration=True,
            )
    bound_issuers = {
        graph_nodes[edge["target"]]["operation"]: graph_nodes[edge["source"]]
        for edge in graph["edges"]
        if edge["kind"] == "issued_tool"
        and edge["source"] in graph_nodes
        and edge["target"] in graph_nodes
    }
    for edge in graph["edges"]:
        if (
            edge["kind"] == "duration"
            or edge["source"] not in graph_nodes
            or edge["target"] not in graph_nodes
        ):
            continue
        left, right = graph_nodes[edge["source"]], graph_nodes[edge["target"]]
        confidence = "reconstructed" if edge["confidence"] == "verified" else edge["confidence"]
        incomplete_consumer = (
            source_actions.get(right["operation"], {}).get("causal_evidence") or {}
        ).get("complete") is False
        if incomplete_consumer and edge["kind"] in {
            "continuation",
            "tool_result",
            "unmeasured_tool_result",
        }:
            confidence = "provenance"
            reason = "Incomplete prompt evidence cannot identify the latest active prerequisite"
            operations[right["operation"]]["limitations"].append(reason)
            issues.append({"operation_id": right["operation"], "reason": reason})
        if edge["kind"] == "tool_result" and left["timing"] == "consumption_bound":
            # Without tool hooks these are issuance/consumption bounds, not an
            # observed tool completion. The exchange dependency still proves an
            # elapsed interval, whose work remains unknown.
            issuer = bound_issuers.get(left["operation"])
            if issuer:
                link(
                    issuer["operation"],
                    right["operation"],
                    "unmeasured_tool_result",
                    "Recorded tool issuance and consuming prompt; tool execution timing unavailable",
                    source_phase=issuer["phase"],
                    target_phase=right["phase"],
                    confidence="provenance" if incomplete_consumer else "reconstructed",
                    blocking=not incomplete_consumer,
                )
            confidence = "provenance"
        link(
            left["operation"],
            right["operation"],
            edge["kind"],
            edge["evidence"],
            source_phase=left["phase"],
            target_phase=right["phase"],
            confidence=confidence,
            blocking=edge["blocking"] and confidence != "provenance",
        )

    # Explicit span parentage informs the inspector only.  It does not establish
    # waiting, including when parent context crosses a thread or process.
    for action in actions:
        op = operations[action["id"]]
        metadata = action.get("metadata") or {}
        parent = (
            action.get("parent_id")
            or metadata.get("parent_span_id")
            or metadata.get("agency.parent_span_id")
        )
        parent = aliases.get(_identity(parent))
        if parent and parent != op["id"]:
            op["parent_id"] = parent
            operations[parent]["child_ids"].append(op["id"])
            link(
                parent,
                op["id"],
                "containment",
                "Explicit parent span identity; does not prove blocking",
                source_phase="start",
                confidence="provenance",
                blocking=False,
            )

    for dependency in run.get("dependencies", []):
        if not isinstance(dependency, dict):
            continue
        link(
            dependency.get("source"),
            dependency.get("target"),
            dependency.get("kind", "dependency"),
            dependency.get("evidence", "Explicit recorded operation dependency"),
            source_phase=dependency.get("source_phase", "end"),
            target_phase=dependency.get("target_phase", "start"),
            confidence=dependency.get("confidence", "recorded"),
            blocking=dependency.get("blocking", True),
        )
    tool_invocations = defaultdict(list)
    for action in actions:
        if operations[action["id"]]["band"] == "Tool":
            call = action.get("call_id") or (action.get("metadata") or {}).get("call_id")
            if call:
                tool_invocations[(action["agent"], str(call))].append(action)
    for action in actions:
        metadata = action.get("metadata") or {}
        preceding = metadata.get("preceding_tool_call_id")
        if not preceding:
            continue
        candidates = tool_invocations[(action["agent"], str(preceding))]
        batch = metadata.get("tool_batch_id")
        previous_model_tool = metadata.get("preceding_model_tool_call_id")
        if batch:
            candidates = [
                candidate
                for candidate in candidates
                if (candidate.get("metadata") or {}).get("tool_batch_id") == batch
            ]
        if previous_model_tool:
            candidates = [
                candidate
                for candidate in candidates
                if (
                    candidate.get("model_tool_call_id")
                    or (candidate.get("metadata") or {}).get("model_tool_call_id")
                )
                == previous_model_tool
            ]
        model_id = action.get("model_id") or metadata.get("model_id")
        if model_id:
            candidates = [
                candidate
                for candidate in candidates
                if (candidate.get("model_id") or (candidate.get("metadata") or {}).get("model_id"))
                in {None, model_id}
            ]
        if len(candidates) == 1:
            link(
                candidates[0]["id"],
                action["id"],
                "serial_tool",
                "Runtime recorded preceding_tool_call_id in a serial tool executor",
            )
        else:
            issues.append(
                {
                    "operation_id": action["id"],
                    "reason": "Serial predecessor invocation identity absent or ambiguous",
                }
            )
    # LlmHandlerServer emits a stable call_label and consecutive attempt labels
    # from its serial retry loop. This is stronger evidence than equal prompts.
    attempts = defaultdict(lambda: defaultdict(list))
    for action in actions:
        label = (action.get("metadata") or {}).get("call_label")
        name = action.get("name", "")
        if not label or not name.startswith("llm:attempt[") or not name.endswith("]"):
            continue
        try:
            index = int(name[len("llm:attempt[") : -1])
        except ValueError:
            continue
        if index < 0:
            continue
        op = operations[action["id"]]
        op["metrics"].update(attempt_index=index, logical_request_id=label)
        attempts[(action["agent"], op.get("request_id"), label)][index].append(op)
    for by_index in attempts.values():
        recorded = [op for members_ in by_index.values() for op in members_]
        for op in recorded:
            op["metrics"]["observed_attempt_count"] = len(recorded)
            op["metrics"]["retry_count"] = max(by_index)
        for index, current in by_index.items():
            if index == 0:
                continue
            prior = by_index.get(index - 1, [])
            if len(current) == len(prior) == 1 and prior[0]["status"] != "success":
                link(
                    prior[0]["id"],
                    current[0]["id"],
                    "llm_retry",
                    "Same recorded call_label and consecutive LlmHandlerServer attempt indexes",
                    confidence="reconstructed",
                )
            else:
                for op in current:
                    op["limitations"].append(
                        "Retry predecessor missing, ambiguous, or contradicts recorded completion status"
                    )
    for action in actions:
        metadata = action.get("metadata") or {}
        previous_submission = metadata.get("preceding_submission_span_id")
        if previous_submission:
            link(
                previous_submission,
                action["id"],
                "submission_sequence",
                "Explicit preceding synchronous submission span in the same recorded caller context",
            )
        for source in _items(
            metadata.get("depends_on_span_ids") or metadata.get("agency.depends_on_span_ids")
        ):
            link(source, action["id"], "dependency", "Explicit depends_on_span_ids")
        spawn = metadata.get("spawn_span_id") or metadata.get("agency.spawn_span_id")
        if spawn:
            link(
                spawn,
                action["id"],
                "spawn",
                "Explicit spawning span identity",
                source_phase="start",
            )

    requests, execution = defaultdict(list), defaultdict(list)
    members = defaultdict(list)
    for op in operations.values():
        if op.get("request_id"):
            members[op["request_id"]].append(op)
            if op["container"]:
                if op["label"] == "engine:execute" or op["kind"] == "execution":
                    execution[op["request_id"]].append(op)
                else:
                    requests[op["request_id"]].append(op)

    # The runtime defines request:submission_to_completion as the envelope that
    # completes after its engine result.  Boundary gates retain that recorded
    # lifecycle without adding the enclosing duration to descendant work.
    for request, envelopes in requests.items():
        if len(envelopes) != 1:
            issues.append(
                {"reason": "Request envelope identity is ambiguous", "request_id": request}
            )
            continue
        envelope = envelopes[0]
        launcher = source_actions[envelope["id"]].get("metadata", {}).get("launcher_span_id")
        if launcher:
            link(
                launcher,
                envelope["id"],
                "request_submission",
                "Explicit synchronous launcher_span_id on request registration",
                source_phase="start",
            )
        children = [
            op for op in members[request] if not op["container"] and op["kind"] != "process"
        ]
        for child in children:
            if (
                child["start"] is not None
                and envelope["start"] is not None
                and child["start"] >= envelope["start"]
            ):
                link(
                    envelope["id"],
                    child["id"],
                    "request_start",
                    "Recorded request_id and submission boundary",
                    source_phase="start",
                )
            if (
                child["end"] is not None
                and envelope["end"] is not None
                and child["end"] <= envelope["end"]
            ):
                link(
                    child["id"],
                    envelope["id"],
                    "request_complete",
                    "Runtime request completion follows owned operation completion",
                    target_phase="end",
                )
        if not children:
            envelope["limitations"].append("Request has no measured child operations")
        caller = source_actions[envelope["id"]].get("metadata", {}).get("submitted_by_request_id")
        if len(requests.get(caller, [])) == 1:
            parent = requests[caller][0]
            link(
                parent["id"],
                envelope["id"],
                "delegation",
                "Explicit submitted_by_request_id; caller identity is provenance, not a completion prerequisite",
                source_phase="start",
                confidence="provenance",
                blocking=False,
            )

    # Execution admission is a distinct gate: submission is allowed before its
    # prerequisites finish.  Preserve the dependency into the actual body.
    for request, executions in execution.items():
        if len(executions) != 1:
            continue
        gate = executions[0]
        if len(requests.get(request, [])) == 1:
            link(
                requests[request][0]["id"],
                gate["id"],
                "engine_launch",
                "Recorded request registration precedes engine admission",
                source_phase="start",
            )
        for child in members[request]:
            if child["container"] or child["kind"] == "process" or child["band"] == "Wait":
                continue
            if finite(gate["start"]) and finite(child["start"]) and gate["start"] <= child["start"]:
                link(
                    gate["id"],
                    child["id"],
                    "execution_admission",
                    "Recorded request_id and engine admission boundary",
                    source_phase="start",
                )
            if finite(child["end"]) and finite(gate["end"]) and child["end"] <= gate["end"]:
                link(
                    child["id"],
                    gate["id"],
                    "execution_complete",
                    "Recorded request_id and engine completion boundary",
                    target_phase="end",
                )
        if len(requests.get(request, [])) == 1:
            envelope = requests[request][0]
            link(
                gate["id"],
                envelope["id"],
                "request_complete",
                "Recorded engine completion precedes request completion",
                target_phase="end",
            )
    for action in actions:
        submitted = (action.get("metadata") or {}).get("submitted_request_id")
        if submitted and len(requests.get(submitted, [])) == 1:
            link(
                action["id"],
                requests[submitted][0]["id"],
                "request_submission",
                "Explicit submitted_request_id on synchronous launch span",
                source_phase="start",
            )

    request_dependencies = defaultdict(list)
    for record in run.get("request_dependencies", []):
        if not isinstance(record, dict) or not record.get("request_id"):
            continue
        request_dependencies[record["request_id"]].extend(
            _items(record.get("dependency_request_ids"))
        )
        request_dependencies[record["request_id"]].extend(
            dependency["request_id"]
            for dependency in _items(record.get("dependencies"))
            if isinstance(dependency, dict) and dependency.get("request_id")
        )

    for action in actions:
        metadata = action.get("metadata") or {}
        dependent_ids = list(_items(metadata.get("dependency_request_ids")))
        if operations[action["id"]]["container"]:
            dependent_ids.extend(
                request_dependencies.get(operations[action["id"]].get("request_id"), [])
            )
        for dependency in _items(metadata.get("dependencies")):
            if isinstance(dependency, dict) and dependency.get("request_id"):
                dependent_ids.append(dependency["request_id"])
        if not dependent_ids:
            continue
        op = operations[action["id"]]
        target, target_phase = op, "end" if op["band"] == "Wait" else "start"
        if op["container"]:
            waits = [
                item
                for item in members[op["request_id"]]
                if item["label"].startswith("sync:dependency")
            ]
            active = execution.get(op["request_id"], [])
            if len(waits) == 1:
                target, target_phase = waits[0], "end"
            elif len(active) == 1:
                target, target_phase = active[0], "start"
            else:
                issues.append(
                    {
                        "operation_id": op["id"],
                        "reason": "Consumer request admission boundary unavailable",
                    }
                )
                continue
        for request in dict.fromkeys(dependent_ids):
            candidates = requests.get(request, [])
            if len(candidates) == 1 and request != op.get("request_id"):
                link(
                    candidates[0]["id"],
                    target["id"],
                    "request_dependency",
                    f"Recorded Future prerequisite request_id={request}",
                    target_phase=target_phase,
                )
            else:
                issues.append(
                    {
                        "operation_id": op["id"],
                        "reason": "Producer request completion boundary unavailable",
                        "request_id": request,
                    }
                )

    # A recorded blocking dependency wait gates this request's engine admission.
    # The wait interval and its producer execution remain competing DAG branches.
    for request, members_ in members.items():
        if len(execution.get(request, [])) == 1:
            gate = execution[request][0]
            for wait in members_:
                if wait["label"].startswith(("sync:dependency", "sync:scheduler_queue")):
                    link(
                        wait["id"],
                        gate["id"],
                        "dependency_admission",
                        "Runtime engine admitted after recorded request scheduling phase completed",
                    )

    # Agent parentage and messages are provenance unless the recording supplies
    # operation boundary identities and an explicitly blocking relationship.
    for relation in run.get("edges", []):
        if relation.get("source_operation") and relation.get("target_operation"):
            link(
                relation["source_operation"],
                relation["target_operation"],
                relation.get("kind", "dependency"),
                relation.get("evidence", "Recorded cross-agent operation relationship"),
                source_phase=relation.get("source_phase", "end"),
                target_phase=relation.get("target_phase", "start"),
                confidence=relation.get("confidence", "recorded"),
                blocking=relation.get("blocking", False),
            )

    # Every fan-in shares only the tail after the latest prerequisite.  Charging
    # an earlier branch for the entire wait would obscure which worker gated it.
    latest = defaultdict(lambda: -math.inf)
    prerequisites = defaultdict(list)
    for edge in edges:
        if (
            edge["blocking"]
            and edge["confidence"] in {"recorded", "reconstructed"}
            and not edge["_duration"]
        ):
            latest[edge["_target"]] = max(latest[edge["_target"]], edge["source_time"])
            if edge["source_phase"] == "end" and edge["kind"] not in {
                "request_complete",
                "execution_complete",
                "dependency_admission",
                "continuation",
            }:
                prerequisites[edge["_target"]].append(edge)
    gates = []
    for target, candidates in prerequisites.items():
        unique = {edge["_source"]: edge for edge in candidates}
        if len(unique) < 2 and operations[candidates[0]["target"]]["band"] != "Wait":
            continue
        completed = max(edge["source_time"] for edge in unique.values())
        gate = {
            "operation_id": candidates[0]["target"],
            "phase": candidates[0]["target_phase"],
            "latest_completion_time": completed,
            "latest_prerequisite_ids": list(
                dict.fromkeys(
                    edge["source"]
                    for edge in unique.values()
                    if abs(edge["source_time"] - completed) <= 1e-6
                )
            ),
            "prerequisites": [
                {
                    "operation_id": edge["source"],
                    "completed_at": edge["source_time"],
                    "observed_slack": max(0, completed - edge["source_time"]),
                }
                for edge in unique.values()
            ],
            "evidence": "Latest completion among explicitly recorded prerequisites; independent of missing startup causality",
        }
        gates.append(gate)
        operations[gate["operation_id"]].setdefault("joins", []).append(gate)
        for edge in unique.values():
            edge["gating"] = abs(edge["source_time"] - completed) <= 1e-6
            edge["join_slack"] = max(0, completed - edge["source_time"])
            if edge["gating"]:
                operations[edge["source"]]["gating"] = True
    analysis_edges = []
    for edge in edges:
        if (
            not edge["_duration"]
            and edge["blocking"]
            and edge["confidence"] in {"recorded", "reconstructed"}
        ):
            edge["_weight"] = max(0, edge["target_time"] - latest[edge["_target"]])
        op = operations[edge["source"]]
        analysis_edges.append(
            {
                "id": edge["id"],
                "source": edge["_source"],
                "target": edge["_target"],
                "weight": edge["_weight"],
                "confidence": edge["confidence"],
                "blocking": edge["blocking"],
                "quality": edge["_weight"]
                if edge["_duration"] and op["band"] in {"LLM", "Tool"}
                else 0,
            }
        )
    path = longest_path(list(boundaries.values()), analysis_edges)
    chosen, segments, breakdown = (
        set(path["edge_ids"]),
        [],
        dict.fromkeys(("LLM", "Tool", "Wait", "System"), 0.0),
    )
    measured, waiting = 0.0, 0.0
    for edge_id in path["edge_ids"]:
        edge = edge_lookup[edge_id]
        edge["critical"] = True
        for key in (edge["source"], edge["target"]):
            operations[key]["critical"] = True
        elapsed = edge["_weight"]
        if elapsed <= 0:
            continue
        kind = operations[edge["source"]]["band"] if edge["_duration"] else "Wait"
        contribution = edge["source"] if edge["_duration"] else edge["target"]
        if edge["_duration"]:
            measured += elapsed
        else:
            waiting += elapsed
        breakdown[kind] += elapsed
        segments.append(
            {
                "kind": kind,
                "operation_id": contribution,
                "agent": operations[contribution]["agent"],
                "start": edge["target_time"] - elapsed,
                "end": edge["target_time"],
                "duration": elapsed,
                "evidence": "Measured operation interval"
                if edge["_duration"]
                else "Elapsed dependency interval; cause unexplained",
            }
        )
    for op in operations.values():
        op["slack"] = path["slack"].get(f"{op['id']}:start") if op["duration"] is not None else None
    path_operation_set = {boundaries[node]["operation"] for node in path["node_ids"]}
    for gate in gates:
        selected = [
            edge
            for edge in prerequisites[f"{gate['operation_id']}:{gate['phase']}"]
            if edge["id"] in chosen
        ]
        missing_latest = gate["operation_id"] in path_operation_set and not any(
            key in path_operation_set for key in gate["latest_prerequisite_ids"]
        )
        if selected and any(edge.get("gating") is False for edge in selected) or missing_latest:
            gate["candidate_path_matches_gate"] = False
            issues.append(
                {
                    "operation_id": gate["operation_id"],
                    "reason": "Candidate weighted chain differs from the latest recorded prerequisite; startup dependencies are incomplete",
                    "latest_prerequisite_ids": gate["latest_prerequisite_ids"],
                }
            )
        else:
            gate["candidate_path_matches_gate"] = True
    processes = _processes(run, operations, source_actions, issues)
    agents = {
        str(a["id"]): {**a, "operation_ids": []} for a in run.get("agents", []) if a.get("id")
    }
    intervals_by_agent = defaultdict(list)
    for op in operations.values():
        agents.setdefault(
            op["agent"],
            {"id": op["agent"], "label": op["agent"], "parent": None, "operation_ids": []},
        )["operation_ids"].append(op["id"])
        if op["duration"] is not None and not op["container"] and op["kind"] != "process":
            intervals_by_agent[op["agent"]].append((op["start"], op["end"]))
    nodes_by_agent, edges_by_agent = defaultdict(list), defaultdict(list)
    boundary_agent = {}
    for node in boundaries.values():
        agent = operations[node["operation"]]["agent"]
        boundary_agent[node["id"]] = agent
        nodes_by_agent[agent].append(node)
    for edge in analysis_edges:
        agent = boundary_agent[edge["source"]]
        if agent == boundary_agent[edge["target"]]:
            edges_by_agent[agent].append(edge)
    for agent in agents.values():
        local = longest_path(nodes_by_agent[agent["id"]], edges_by_agent[agent["id"]])
        agent["critical_path_duration"] = local["duration"]
    _sandbox_resources(run, operations, agents)
    wall = (
        run.get("execution_wall_time")
        if finite(run.get("execution_wall_time"))
        else run.get("duration", 0)
    )
    wall = max(0, wall) if finite(wall) else 0
    ratio = min(1.0, measured / wall) if wall > 0 else 0
    cross_edges = sum(
        operations[e["source"]]["agent"] != operations[e["target"]]["agent"]
        and e["blocking"]
        and e["confidence"] in {"recorded", "reconstructed"}
        for e in edges
    )
    limitations.extend(
        [
            "The displayed path is the longest recorded dependency chain; missing edges may hide constraints.",
            "Elapsed gaps on dependencies have observed timing but no established resource or scheduling cause.",
            "Resource measurements describe observed behavior; they do not establish why an operation took this long.",
        ]
    )
    input_coverage = run.get("coverage", {})
    limitations.extend(str(value) for value in input_coverage.get("historical_warnings", []))
    if input_coverage.get("omitted_operations"):
        limitations.append(
            f"{input_coverage['omitted_operations']} operation spans omitted by the historical size limit."
        )
    if input_coverage.get("resource_sampling"):
        limitations.append(str(input_coverage["resource_sampling"]))
    for field, label in (
        ("omitted_unmeasured_tools", "unmeasured tool projections"),
        ("omitted_tool_identities", "tool issuance identities"),
    ):
        count = graph.get("coverage", {}).get(field, 0)
        if count:
            limitations.append(f"The projection size limit omitted {count} {label}.")
    if len(agents) > 1 and not cross_edges:
        limitations.append(
            "No blocking cross-agent dependencies recorded; independent agent durations are never summed."
        )
    if any(not gate["candidate_path_matches_gate"] for gate in gates):
        limitations.append(
            "A candidate chain and a recorded join gate differ because branch startup causality is incomplete. Latest-prerequisite markers identify the observed join gate; the candidate chain is not a complete workload path."
        )
    if not processes:
        limitations.append("No generation-qualified process measurements available.")
    if any(o["duration"] is None for o in operations.values()):
        limitations.append("Incomplete operation intervals are excluded from measured work.")
    if len(agents) > 1 and run.get("clock", {}).get("shared") is not True:
        limitations.append(
            "Cross-agent clock alignment unavailable; cross-agent links excluded from analysis."
        )
    for row in graph.get("coverage", {}).get("unknown", []):
        issues.append(
            {
                "operation_id": row.get("operation"),
                "reason": row.get("reason", "Missing causal evidence"),
            }
        )
    if path.get("malformed_edges") or any("contradicts" in row["reason"] for row in issues):
        limitations.append("Malformed dependencies were excluded; inspect the recorded evidence.")
    status = (
        "invalid"
        if path.get("error")
        else "partial"
        if cross_edges or len(agents) == 1
        else "candidate"
    )
    operation_ids = list(dict.fromkeys(boundaries[n]["operation"] for n in path["node_ids"]))
    critical = {
        **{k: v for k, v in path.items() if k not in {"slack", "node_ids", "malformed_edges"}},
        "operation_ids": operation_ids,
        "segments": segments,
        "measured_work": measured,
        "unexplained_wait": waiting,
        "gating_operation_ids": list(
            dict.fromkeys(key for gate in gates for key in gate["latest_prerequisite_ids"])
        ),
    }
    for edge in edges:
        for key in ("_source", "_target", "_duration", "_weight"):
            edge.pop(key, None)
    dominant = sorted(
        (
            o
            for o in operations.values()
            if o["duration"] is not None and not o["container"] and o["kind"] != "process"
        ),
        key=lambda op: (-op["duration"], op["id"]),
    )[:8]
    # Bound diagnostics as well as visual marks. Counts retain the full scope.
    issue_count = len(issues)
    agent_ids = {
        identity
        for identity, row in agents.items()
        if identity not in {"workflow", "unattributed", "system"}
        and row.get("role") != "Host workflow"
    }
    measured_wall = _union(
        (max(0, start), min(wall, end))
        for values in intervals_by_agent.values()
        for start, end in values
        if end > 0 and start < wall
    )
    return {
        "version": 1,
        "run": {k: run.get(k) for k in ("id", "title", "status", "source", "duration")},
        "bounds": {"start": 0, "end": wall},
        "summary": {
            "wall_time": wall,
            "critical_path_duration": path["duration"],
            "critical_path_measured": measured,
            "critical_path_wait": waiting,
            "critical_path_coverage": ratio,
            "aggregate_work": sum(_union(values) for values in intervals_by_agent.values()),
            "agent_count": len(agent_ids),
            "actor_count": len(agents),
            "aggregate_agent_activity": sum(
                _union(intervals_by_agent[identity]) for identity in agent_ids
            ),
            "process_count": len(processes),
            "unattributed_time": max(0, wall - measured_wall),
            "measured_activity_wall_time": measured_wall,
            "path_unattributed_time": max(0, wall - measured),
            "breakdown": breakdown,
            "dominant_operations": [
                {
                    "id": o["id"],
                    "label": o["label"],
                    "duration": o["duration"],
                    "critical": o["critical"],
                }
                for o in dominant
            ],
            "gating_operations": [
                {"id": key, "label": operations[key]["label"], "agent": operations[key]["agent"]}
                for key in critical["gating_operation_ids"]
            ],
        },
        "operations": list(operations.values()),
        "edges": edges,
        "agents": list(agents.values()),
        "processes": processes,
        "critical_path": critical,
        "joins": gates,
        "coverage": {
            "complete": False,
            "status": status,
            "ratio": ratio,
            "limitations": limitations,
            "issues": issues[:200],
            "issue_count": issue_count,
            "malformed_edges": path["malformed_edges"][:200],
            "measured_operations": sum(
                o["duration"] is not None and not o["container"] for o in operations.values()
            ),
            "total_operations": len(operations),
            "cross_agent_dependencies": cross_edges,
            "clock_shared": run.get("clock", {}).get("shared"),
        },
        "links": run.get("profiler_links", {}),
    }


def _processes(run, operations, actions, issues):
    """Attach measured process generations; neither PID nor overlap owns work."""
    rows = [
        row for row in run.get("summary", {}).get("process_metrics", []) if isinstance(row, dict)
    ]
    counts = Counter(str(row["identity"]) for row in rows if row.get("identity"))
    ambiguous = {identity for identity, count in counts.items() if count > 1}
    for identity in ambiguous:
        issues.append(
            {
                "reason": "Duplicate process generation identity; resource attribution excluded",
                "process_identity": identity,
            }
        )
    rows = [row for row in rows if str(row.get("identity")) not in ambiguous]
    row_index = {str(row["identity"]): row for row in rows if row.get("identity")}
    processes, tuples, owners = {}, defaultdict(list), defaultdict(set)
    for row in rows:
        if not isinstance(row, dict) or not row.get("identity"):
            continue
        identity = str(row["identity"])
        start, end = row.get("first_seen_s"), row.get("last_seen_s")
        origin = run.get("clock", {}).get("perf_origin_ns")
        shift = run.get("clock", {}).get("trace_origin_us", 0)
        if finite(origin) and finite(shift):
            if finite(row.get("first_seen_ns")):
                start = (row["first_seen_ns"] - origin) / 1e9 - shift / 1e6
            if finite(row.get("last_seen_ns")):
                end = (row["last_seen_ns"] - origin) / 1e9 - shift / 1e6
        process = {
            "id": "process:" + identity,
            "identity": identity,
            "pid": row.get("pid"),
            "parent_id": "process:" + str(row["parent_identity"])
            if row.get("parent_identity")
            else None,
            "agent": None,
            "owner_ids": [],
            "label": row.get("display_name") or row.get("name") or identity,
            "start": start,
            "end": end,
            "duration": None,
            "lifecycle": "sample_bounds",
            "metrics": {
                key: value
                for key, value in row.items()
                if key not in {"identity", "name", "display_name", "cmdline", "parent_identity"}
                and value is not None
            },
            "attribution": "unattributed",
            "evidence": ["Collector process identity includes PID generation"],
            "limitations": [
                "Sampling bounds are observations, not process birth or exit.",
                "Whole-process resource totals cannot be assigned to a tool interval.",
            ],
        }
        if row.get("cmdline"):
            process["metrics"]["command_line"] = row["cmdline"][:4000]
        processes[identity] = process
        key = (
            row.get("sandbox"),
            row.get("pid_namespace"),
            row.get("namespace_pid"),
            row.get("start_ticks"),
        )
        if all(part is not None for part in key):
            tuples[key].append(identity)
    process_actions = {}

    def contradicts(metadata, identity):
        if identity in ambiguous:
            return True
        row = row_index.get(str(identity), {})
        return any(
            metadata.get(field) is not None
            and row.get(metric) is not None
            and metadata[field] != row[metric]
            for field, metric in (
                ("agency.sandbox_id", "sandbox"),
                ("agency.pid_namespace", "pid_namespace"),
                ("agency.namespace_pid", "namespace_pid"),
                ("agency.process_start_ticks", "start_ticks"),
            )
        )

    for key, action in actions.items():
        if action.get("kind") != "process":
            continue
        metadata = action.get("metadata") or {}
        identity = metadata.get("agency.process_identity") or metadata.get("process_identity")
        if identity is not None:
            identity = str(identity)
        contradiction = contradicts(metadata, identity)
        if contradiction:
            issues.append(
                {
                    "operation_id": key,
                    "reason": "Process identity contradicts namespace or generation evidence; resource join excluded",
                }
            )
            identity = None
        if metadata.get("agency.process_identity_unknown"):
            identity = None
        tuple_key = (
            metadata.get("agency.sandbox_id"),
            metadata.get("agency.pid_namespace"),
            metadata.get("agency.namespace_pid"),
            metadata.get("agency.process_start_ticks"),
        )
        if metadata.get("agency.process_identity_unknown") or contradiction:
            tuple_key = (None, None, None, None)
        if identity not in processes:
            candidates = (
                [] if metadata.get("agency.process_identity_unknown") else tuples.get(tuple_key, [])
            )
            if len(candidates) == 1:
                identity = candidates[0]
            elif identity is None and all(value is not None for value in tuple_key[1:]):
                # A namespace-qualified generation is safe even when host /proc
                # sampling did not observe this short-lived subprocess.
                identity = ":".join(str(value) for value in tuple_key[1:])
            elif identity is None:
                identity = "span:" + key
        process_actions[key] = identity
        op = operations[key]
        if identity not in processes:
            processes[identity] = {
                "id": "process:" + identity,
                "identity": identity,
                "pid": metadata.get("agency.namespace_pid") or metadata.get("pid"),
                "parent_id": None,
                "agent": op["agent"],
                "owner_ids": [],
                "label": op["label"],
                "start": op["start"],
                "end": op["end"],
                "duration": op["duration"],
                "lifecycle": "observed_lifetime"
                if op["duration"] is not None
                else "incomplete_lifetime",
                "metrics": {
                    k: v
                    for k, v in metadata.items()
                    if k in {*_METRICS, "executable", "exit_code", "process_provenance", "timing"}
                },
                "attribution": "recorded_process_span",
                "evidence": ["Instrumented process launch and completion boundaries"],
                "limitations": ["No process resource samples recorded."],
            }
        else:
            processes[identity].update(
                start=op["start"],
                end=op["end"],
                duration=op["duration"],
                lifecycle="observed_lifetime"
                if op["duration"] is not None
                else "incomplete_lifetime",
            )
            processes[identity]["evidence"].append(
                "Instrumented process lifetime joined to exact collector generation"
            )
            processes[identity]["limitations"] = [
                value
                for value in processes[identity]["limitations"]
                if not value.startswith("Sampling bounds")
            ]
        if all(value is not None for value in tuple_key):
            tuples[tuple_key] = list(dict.fromkeys([*tuples[tuple_key], identity]))
    counters = run.get("counters", {})
    by_process = defaultdict(dict)
    # Counter values already have bounded historical sampling; do not duplicate
    # all series into the response. Temporal queries use binary search.
    for name, points in counters.items():
        if not name.startswith("process:"):
            continue
        parts = name.split(":", 2)
        if len(parts) != 3:
            continue
        _, identity, metric = parts
        if identity not in processes or not isinstance(points, list):
            continue
        values = sorted(
            (float(p[0]), float(p[1]))
            for p in points
            if isinstance(p, (list, tuple)) and len(p) == 2 and finite(p[0]) and finite(p[1])
        )
        if not values:
            continue
        by_process[identity][metric] = ([p[0] for p in values], values)
        process = processes[identity]
        if process["lifecycle"] == "sample_bounds":
            process["start"] = (
                min(process["start"], values[0][0]) if finite(process["start"]) else values[0][0]
            )
            process["end"] = (
                max(process["end"], values[-1][0]) if finite(process["end"]) else values[-1][0]
            )
    tool_calls = defaultdict(list)
    for key, action in actions.items():
        if operations[key]["band"] == "Tool":
            metadata = action.get("metadata") or {}
            call_id = action.get("call_id") or metadata.get("call_id")
            if call_id:
                tool_calls[(operations[key]["agent"], str(call_id))].append(key)

    def attach_tool(identity, key, evidence):
        op, process = operations[key], processes[identity]
        process["owner_ids"] = list(dict.fromkeys([*process["owner_ids"], key]))
        op["process_ids"] = list(dict.fromkeys([*op["process_ids"], process["id"]]))
        op["evidence"].append(evidence)
        owners[identity].add(op["agent"])
        window = {}
        if finite(op["start"]) and finite(op["end"]) and run.get("clock", {}).get("shared") is True:
            for metric, (times, values) in by_process.get(identity, {}).items():
                samples = values[bisect_left(times, op["start"]) : bisect_right(times, op["end"])]
                if samples:
                    window[metric] = {
                        "observed_peak": max(v for _, v in samples),
                        "observed_min": min(v for _, v in samples),
                        "samples": len(samples),
                    }
            if window:
                resources = op["metrics"].setdefault("process_resources", [])
                if not any(row["process_id"] == process["id"] for row in resources):
                    resources.append(
                        {
                            "process_id": process["id"],
                            "start": op["start"],
                            "end": op["end"],
                            "sampled_window": window,
                            "basis": "Recorded samples inside tool interval; no whole-process totals allocated",
                        }
                    )

    for key, action in actions.items():
        metadata = action.get("metadata") or {}
        identity = (
            process_actions.get(key)
            or metadata.get("agency.process_identity")
            or metadata.get("process_identity")
        )
        if identity is not None:
            identity = str(identity)
        if metadata.get("agency.process_identity_unknown") and key not in process_actions:
            continue
        if key not in process_actions and contradicts(metadata, identity):
            issues.append(
                {
                    "operation_id": key,
                    "reason": "Process identity contradicts namespace or generation evidence; resource join excluded",
                }
            )
            continue
        evidence = "Explicit collector process identity on operation"
        if identity not in processes:
            tuple_key = (
                metadata.get("agency.sandbox_id"),
                metadata.get("agency.pid_namespace"),
                metadata.get("agency.namespace_pid"),
                metadata.get("agency.process_start_ticks"),
            )
            candidates = tuples.get(tuple_key, [])
            identity = candidates[0] if len(candidates) == 1 else None
            evidence = "Exact sandbox / PID namespace / namespace PID / process start-ticks match"
        if identity is None or identity not in processes:
            continue
        op, process = operations[key], processes[identity]
        owners[identity].add(op["agent"])
        if op["band"] == "Tool":
            attach_tool(identity, key, evidence)
        if action.get("kind") == "process":
            parent = metadata.get("parent_process_identity")
            if parent in processes:
                process["parent_id"] = processes[parent]["id"]
                process["evidence"].append("Explicit generation-qualified parent process identity")
            explicit_tool = metadata.get("tool_action_id")
            candidates = (
                [explicit_tool]
                if explicit_tool in operations
                else tool_calls.get((op["agent"], str(metadata.get("tool_call_id"))), [])
            )
            model_tool_id = action.get("model_tool_call_id") or metadata.get("model_tool_call_id")
            if len(candidates) > 1 and model_tool_id:
                candidates = [
                    candidate
                    for candidate in candidates
                    if actions[candidate].get("model_tool_call_id") == model_tool_id
                ]
            if len(candidates) == 1 and operations[candidates[0]]["band"] == "Tool":
                attach_tool(
                    identity,
                    candidates[0],
                    "Explicit tool invocation ID on process launch span, scoped to agent",
                )
                process["attribution"] = "explicit_tool_owner"
            elif metadata.get("tool_call_id") or explicit_tool:
                process["limitations"].append(
                    "Tool invocation identity absent or ambiguous; ownership not inferred"
                )
        process["evidence"].append(evidence)
        if process["attribution"] != "explicit_tool_owner":
            process["attribution"] = "recorded_operation_identity"
    for identity, process in processes.items():
        explicit_tool = row_index.get(identity, {}).get("tool_action_id")
        if explicit_tool in operations and operations[explicit_tool]["band"] == "Tool":
            attach_tool(identity, explicit_tool, "Explicit tool_action_id on process record")
            process["attribution"] = "explicit_tool_owner"
            process["evidence"].append("Explicit tool_action_id on process record")
        if len(owners[identity]) == 1:
            process["agent"] = next(iter(owners[identity]))
        elif len(owners[identity]) > 1:
            process["attribution"] = "shared_recorded_process"
        if not owners[identity]:
            sandbox = process["metrics"].get("sandbox")
            agent_owners = run.get("sandbox_owners", {}).get(sandbox, []) if sandbox else []
            if len(agent_owners) == 1:
                process["agent"], process["attribution"] = agent_owners[0], "sandbox_owner"
                process["evidence"].append(
                    "Explicit sandbox ownership; no tool ownership established"
                )
        if finite(process["start"]) and finite(process["end"]):
            process["duration"] = max(0, process["end"] - process["start"])
        process["evidence"] = list(dict.fromkeys(process["evidence"]))
    # Host helper processes may have been sampled in the profiler's broader
    # cgroup. They are not workload processes without identity/ownership proof.
    process_by_id = {process["id"]: process for process in processes.values()}
    resolved = set()
    for process in processes.values():
        chain, visited, current = [], set(), process
        while current["id"] not in resolved and current["id"] not in visited:
            visited.add(current["id"])
            chain.append(current)
            parent = process_by_id.get(current.get("parent_id"))
            if parent is None:
                break
            current = parent
        if current["id"] in visited and current.get("parent_id") in visited:
            process["limitations"].append(
                "Cyclic recorded process ancestry; inherited tool attribution excluded"
            )
            issues.append(
                {"reason": "Cyclic recorded process ancestry", "process_id": process["id"]}
            )
            resolved.update(visited)
            continue
        for child in reversed(chain):
            parent = process_by_id.get(child.get("parent_id"))
            if parent and parent["attribution"] in {
                "explicit_tool_owner",
                "inherited_tool_ancestry",
            }:
                for tool in parent["owner_ids"]:
                    attach_tool(
                        child["identity"],
                        tool,
                        "Generation-qualified process ancestry descends from explicitly owned tool subprocess",
                    )
                child["attribution"] = "inherited_tool_ancestry"
                child["evidence"].append(
                    "Generation-qualified parent process with explicit tool invocation ownership"
                )
                child["limitations"].append(
                    "Process ancestry establishes origin; it does not prove the tool waited for this descendant."
                )
                child["agent"] = parent["agent"]
            resolved.add(child["id"])
    observed_identities = set(process_actions.values())
    return [
        process
        for identity, process in processes.items()
        if owners[identity] or process["metrics"].get("sandbox") or identity in observed_identities
    ]


def _sandbox_resources(run, operations, agents):
    """Contextual shared-resource samples, never tool-owned work or causality."""
    by_sandbox = defaultdict(dict)
    for name, points in run.get("counters", {}).items():
        if not name.startswith("sandbox:") or not isinstance(points, list):
            continue
        parts = name.split(":", 2)
        if len(parts) != 3:
            continue
        _, sandbox, metric = parts
        values = sorted(
            (float(point[0]), float(point[1]))
            for point in points
            if isinstance(point, (list, tuple))
            and len(point) == 2
            and finite(point[0])
            and finite(point[1])
        )
        if values:
            by_sandbox[sandbox][metric] = ([point[0] for point in values], values)

    def observation(sandbox, start, end):
        if not finite(start) or not finite(end) or run.get("clock", {}).get("shared") is not True:
            return None
        sampled = {}
        for metric, (times, values) in by_sandbox.get(sandbox, {}).items():
            window = values[bisect_left(times, start) : bisect_right(times, end)]
            if window:
                sampled[metric] = {
                    "observed_peak": max(value for _, value in window),
                    "observed_min": min(value for _, value in window),
                    "samples": len(window),
                }
        if not sampled:
            return None
        return {
            "sandbox_id": sandbox,
            "scope": "shared_sandbox",
            "shared_with": run.get("sandbox_owners", {}).get(sandbox, []),
            "start": start,
            "end": end,
            "sampled_window": sampled,
            "basis": "Retained recorded samples inside interval; shared sandbox activity is not tool-owned and does not establish latency causality",
        }

    agent_intervals = defaultdict(list)
    agent_sandboxes = defaultdict(set)
    for op in operations.values():
        sandbox = op["metrics"].get("agency.sandbox_id")
        if sandbox:
            agent_sandboxes[op["agent"]].add(sandbox)
            if op["band"] == "Tool":
                sample = observation(sandbox, op["start"], op["end"])
                if sample:
                    op["metrics"]["sandbox_resources"] = [sample]
        if finite(op["start"]) and finite(op["end"]):
            agent_intervals[op["agent"]].append((op["start"], op["end"]))
    for identity, agent in agents.items():
        intervals = agent_intervals.get(identity, [])
        if not intervals:
            continue
        start, end = min(value[0] for value in intervals), max(value[1] for value in intervals)
        samples = [
            observation(sandbox, start, end) for sandbox in sorted(agent_sandboxes[identity])
        ]
        samples = [sample for sample in samples if sample is not None]
        if samples:
            agent.setdefault("metrics", {})["sandbox_resources"] = samples
