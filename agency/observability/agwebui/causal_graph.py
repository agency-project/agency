"""Evidence-only causal projection shared by live, saved and replay views.

DAG weights combine measured work with explicitly unexplained elapsed waits
on verified dependencies. Both contributions are reported separately; a wait
interval establishes timing, not its cause. Missing links still prevent a claim
about a complete system-wide critical path. History references are
provenance; only the latest response/results consumed by a prompt block execution.
"""

from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
import math


def _digest(block):
    # Role is normalized separately; providers may change role spelling at the boundary.
    value = {k: v for k, v in block.items() if k != "role"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:24]


def exchange_evidence(prompt, response, *, complete=True):
    """Retain identities for ALL blocks without exposing prompt text/private reasoning."""

    def refs(blocks):
        return [
            {
                "hash": _digest(b),
                "type": b.get("type"),
                "role": b.get("role"),
                "tool_id": b.get("tool_call_id") or b.get("tool_use_id") or b.get("call_id")
                if b.get("type") in {"tool_result", "function_call_output"}
                else b.get("id"),
                "name": b.get("name"),
            }
            for b in blocks
            if isinstance(b, dict)
            and b.get("type") not in {"metadata", "thinking", "reasoning", "redacted_thinking"}
        ]

    return {"prompt": refs(prompt), "response": refs(response), "complete": complete}


def messages_evidence(raw, response):
    """Saved profiler spans can carry bounded JSON; never repair truncated JSON."""
    try:
        messages = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        messages = None
    prompt = []
    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, dict):
            content = message.get("blocks", message.get("content", []))
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if isinstance(content, list):
                prompt.extend(
                    {"role": message.get("role"), **b} for b in content if isinstance(b, dict)
                )
    return exchange_evidence(
        prompt, (response or {}).get("blocks", []), complete=isinstance(messages, list)
    )


def _lineage(refs, initial=""):
    states = []
    state = initial
    for ref in refs:
        if ref.get("role") != "tool_schema":
            state = hashlib.sha256((state + str(ref.get("hash"))).encode()).hexdigest()[:24]
        states.append(state)
    return states


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def critical_path(nodes, edges):
    """O(V+E) longest weighted path and slack over verified blocking DAG edges.

    Provenance, guesses and process ancestry never participate. Reject cyclic
    evidence rather than silently picking an arbitrary path through it.
    """
    lookup = {n["id"]: n for n in nodes}
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for e in edges:
        if e["confidence"] == "verified" and e["blocking"]:
            incoming[e["target"]].append(e)
            outgoing[e["source"]].append(e)
    degree = {key: len(incoming[key]) for key in lookup}
    ready = deque(key for key in lookup if degree[key] == 0)
    order, distance, previous = [], dict.fromkeys(lookup, 0.0), {}
    while ready:
        key = ready.popleft()
        order.append(key)
        for e in outgoing[key]:
            target, value = e["target"], distance[key] + e["weight"]
            if target not in previous or value > distance[target]:
                distance[target], previous[target] = value, e
            degree[target] -= 1
            if degree[target] == 0:
                ready.append(target)
    if len(order) != len(lookup):
        return {
            "duration": None,
            "node_ids": [],
            "edge_ids": [],
            "slack": {},
            "error": "Cyclic blocking evidence; analysis unavailable",
        }
    tail = dict.fromkeys(lookup, 0.0)
    for key in reversed(order):
        tail[key] = max((e["weight"] + tail[e["target"]] for e in outgoing[key]), default=0)
    end = max(order, key=lambda k: distance[k]) if order else None
    duration = distance.get(end, 0)
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
        "slack": {key: max(0, duration - distance[key] - tail[key]) for key in order},
        "slack_basis": "Observed dependency-chain elapsed time; missing dependencies may hide constraints",
    }


def build_graph(run, *, max_unmeasured_tools=None):
    actions = sorted(run.get("actions", []), key=lambda a: (a.get("start", 0), a["id"]))
    nodes, edges, operations, unknown = {}, {}, {}, []
    totals = dict.fromkeys(("LLM", "Tool", "System"), 0.0)
    clock = run.get("duration", 0)
    unknown.extend(
        {"reason": gap}
        for gap in run.get("coverage", {}).get("causal_evidence", {}).get("gaps", [])
    )

    def node(key, operation, phase, at, timing="observed"):
        nodes[key] = {
            "id": key,
            "operation": operation["id"],
            "action_id": operation.get("action_id"),
            "agent": operation["agent"],
            "band": operation["band"],
            "label": operation["label"],
            "phase": phase,
            "time": at,
            "timing": timing,
            "request_id": operation.get("request_id"),
            "tool_id": operation.get("tool_id"),
        }
        return key

    def edge(source, target, kind, evidence, *, confidence="verified", blocking=True, weight=0):
        if source not in nodes or target not in nodes or source == target:
            return
        left, right = nodes[source]["time"], nodes[target]["time"]
        if finite(left) and finite(right) and left > right + 1e-6:
            confidence, blocking = "unknown", False
            evidence = "Inconsistent recorded timestamps: " + evidence
        key = f"{kind}:{source}->{target}"
        edges[key] = {
            "id": key,
            "source": source,
            "target": target,
            "kind": kind,
            "confidence": confidence,
            "blocking": blocking,
            "evidence": evidence,
            "weight": weight if confidence == "verified" and blocking else 0,
            "gap": max(0, right - left)
            if kind != "duration" and finite(left) and finite(right)
            else 0,
        }
        return key

    def operation(key, agent, band, label, start, end, *, action=None, measured=True, tool_id=None):
        op = {
            "id": key,
            "agent": agent,
            "band": band,
            "kind": action.get("kind") if action else "tool",
            "label": label,
            "model": action.get("model") if action else None,
            "command_preview": str(action.get("command") or action.get("arguments") or "")[:240]
            if action
            else "",
            "action_id": action.get("id") if action else None,
            "request_id": action.get("request_id") if action else None,
            "tool_id": tool_id,
            "duration": max(0, end - start) if measured and finite(start) and finite(end) else None,
            "outcome": action.get("outcome", "unknown") if action else "unobserved",
            "metrics": {
                k: action.get(k) for k in ("tokens", "output_tokens", "timing", "event_ids")
            }
            if action
            else {},
        }
        if action:
            for metric in (
                "cpu_ms",
                "runqueue_ms",
                "blocked_ms",
                "cpu_pct",
                "ttft_ms",
                "new_prompt_tokens",
                "pid",
                "program",
                "path",
                "return_value",
                "ppid",
                "executable_path",
                "command_line",
                "argv",
                "rss_mb",
                "memory_mb",
                "io_read_bytes",
                "io_write_bytes",
                "cpu_time_s",
                "exit_status",
            ):
                value = action.get("metadata", {}).get(metric)
                if value is not None:
                    op["metrics"][metric] = value
            op["exchange_id"] = (
                action.get("call_id") or action.get("metadata", {}).get("call_label")
                if band == "LLM"
                else None
            )
        operations[key] = op
        if finite(start):
            op["start_node"] = node(
                f"{key}:start", op, "start", start, "observed" if measured else "issuance_bound"
            )
        if finite(end):
            op["end_node"] = node(
                f"{key}:end", op, "end", end, "observed" if measured else "consumption_bound"
            )
        if op.get("start_node") and op.get("end_node"):
            edge(
                op["start_node"],
                op["end_node"],
                "duration",
                "Recorded operation boundaries"
                if measured
                else "Tool execution not observed; issuance/consumption are bounds",
                confidence="verified" if measured else "unknown",
                blocking=measured,
                weight=op["duration"] or 0,
            )
        if op["duration"] is not None:
            totals[band] += op["duration"]
        return op

    for a in actions:
        timed = (
            finite(a.get("start"))
            and not a.get("missing_start")
            and "timestamp_missing" not in str(a.get("source", ""))
        )
        start = a.get("start") if timed else None
        complete = a.get("outcome") not in {"running", "incomplete", "interrupted"}
        end = start + a["duration"] if timed and complete and finite(a.get("duration")) else None
        if (
            end is None
            and complete
            and finite(a.get("end_ts"))
            and finite(run.get("time_origin"))
            and "timestamp_missing" not in str(a.get("source", ""))
        ):
            end = a["end_ts"] - run["time_origin"]
        band = (
            "LLM"
            if a["kind"] == "model"
            else "System"
            if a["kind"]
            in {"queue", "dependency", "setup", "syscall", "system", "process", "input", "control"}
            else "Tool"
        )
        op = operation(
            a["id"],
            a["agent"],
            band,
            a["name"],
            start,
            end,
            action=a,
            tool_id=a.get("model_tool_call_id"),
        )
        if end is None:
            unknown.append(
                {
                    "operation": op["id"],
                    "reason": "Operation end missing; in-flight work excluded from measured path",
                }
            )
            if finite(start) and a.get("outcome") == "running" and finite(clock):
                op["open_through"] = clock
                op["open_node"] = node(f"{op['id']}:open", op, "open", clock, "ongoing_boundary")
                edge(
                    op["start_node"],
                    op["open_node"],
                    "in_flight",
                    "Started operation; completion not yet observed",
                    confidence="unknown",
                    blocking=False,
                )
        if start is None:
            unknown.append({"operation": op["id"], "reason": "Operation start unavailable"})

    # Index identities by agent and emitting exchange. Never resolve reused IDs globally.
    issued, by_hash, by_tool = {}, defaultdict(list), defaultdict(list)
    by_lineage = defaultdict(list)
    failed_prompts = defaultdict(list)
    models = [a for a in actions if a["kind"] == "model"]
    omitted_tool_identities = 0
    identity_limit = len(actions) + max_unmeasured_tools if max_unmeasured_tools is not None else None
    for a in models:
        evidence = a.get("causal_evidence") or {}
        refs = evidence.get("response") or []
        if not refs:
            refs = [{"type": "tool_use", "tool_id": key} for key in a.get("tool_call_ids", [])]
        prompt_states = _lineage(evidence.get("prompt", []))
        response_states = _lineage(refs, prompt_states[-1] if prompt_states else "")
        if a.get("outcome") == "failed" and prompt_states:
            failed_prompts[(a["agent"], prompt_states[-1])].append(a)
        for index, ref in enumerate(refs):
            by_lineage[(a["agent"], ref.get("hash"), response_states[index])].append((a, index))
            if ref.get("hash"):
                by_hash[(a["agent"], ref["hash"])].append((a, index))
            if ref.get("type") not in {"tool_use", "function_call"} or not ref.get("tool_id"):
                continue
            if identity_limit is not None and len(issued) >= identity_limit:
                omitted_tool_identities += 1
                continue
            key = f"issued:{a['id']}:{index}:{ref['tool_id']}"
            item = {"id": key, "issuer": a, "ref": ref, "host": None, "consumer": None}
            issued[key] = item
            by_tool[(a["agent"], ref["tool_id"])].append(item)

    dependencies = []
    for a in models:
        refs = (a.get("causal_evidence") or {}).get("prompt", [])
        matches = []
        scoped = {}
        results = []
        prompt_states = _lineage(refs)
        for index, ref in enumerate(refs):
            lineage = by_lineage.get((a["agent"], ref.get("hash"), prompt_states[index]), [])
            identities = lineage or by_hash.get((a["agent"], ref.get("hash")), [])
            candidates = [
                (m, i)
                for m, i in identities
                if m["id"] != a["id"] and m["start"] < a["start"] and ref.get("role") == "assistant"
            ]
            # Identical responses across retries cannot be disambiguated by proximity.
            if len({m["id"] for m, _ in candidates}) == 1:
                m, response_index = candidates[0]
                matches.append((index, m))
                if ref.get("type") in {"tool_use", "function_call"}:
                    scoped[ref.get("tool_id")] = issued.get(
                        f"issued:{m['id']}:{response_index}:{ref.get('tool_id')}"
                    )
            if ref.get("type") in {"tool_result", "function_call_output"} and ref.get("tool_id"):
                item = scoped.get(ref["tool_id"])
                if item is None:
                    candidates = [
                        t
                        for t in by_tool[(a["agent"], ref["tool_id"])]
                        if t["issuer"]["start"] < a["start"]
                    ]
                    item = candidates[0] if len(candidates) == 1 else None
                if item:
                    results.append((index, item))
                else:
                    unknown.append(
                        {
                            "operation": a["id"],
                            "tool_id": ref["tool_id"],
                            "reason": "Prompt result has no unique recorded tool issuer",
                        }
                    )
        last = matches[-1] if matches else None
        latest_issuer = last[1]["id"] if last else None
        latest_position = last[0] if last else -1
        for _, m in matches:
            dependencies.append(
                (
                    m["id"],
                    a["id"],
                    "continuation" if m["id"] == latest_issuer else "history",
                    m["id"] == latest_issuer,
                    "Response content hash present in consuming prompt",
                )
            )
        for index, item in results:
            active = index > latest_position and (
                latest_issuer is None or item["issuer"]["id"] == latest_issuer
            )
            dependencies.append(
                (
                    item["id"],
                    a["id"],
                    "tool_result" if active else "history",
                    active,
                    f"Prompt tool_result ID {item['ref']['tool_id']} scoped to issuing exchange",
                )
            )
            if item["consumer"] is None or a["start"] < item["consumer"]["start"]:
                item["consumer"] = a
        if prompt_states:
            retries = [
                m
                for m in failed_prompts[(a["agent"], prompt_states[-1])]
                if m["start"] < a["start"]
            ]
            if len(retries) == 1:
                left, right = operations[retries[0]["id"]], operations[a["id"]]
                if left.get("end_node") and right.get("start_node"):
                    edge(
                        left["end_node"],
                        right["start_node"],
                        "retry",
                        "Same recorded prompt as a failed attempt; retry not explicitly recorded",
                        confidence="inferred",
                        blocking=False,
                    )
        if not last and not results:
            unknown.append(
                {
                    "operation": a["id"],
                    "reason": "No recorded incoming dependency (root, retry, compaction or missing history)",
                }
            )

    for a in actions:
        tool_id = a.get("model_tool_call_id")
        if not tool_id or a["kind"] == "model":
            continue
        candidates = by_tool[(a["agent"], tool_id)]
        # Explicit model_id wins. Otherwise require a unique occurrence whose
        # recorded issuance / first-consumption window contains the call.
        # Identity and prompt evidence establish the scope; timing only rejects
        # impossible matches, never establishes ownership on its own.
        if a.get("model_id"):
            candidates = [
                t
                for t in candidates
                if a["model_id"]
                in {
                    t["issuer"]["id"],
                    t["issuer"].get("call_id"),
                    t["issuer"].get("metadata", {}).get("call_label"),
                }
            ]
        else:
            candidates = [
                t
                for t in candidates
                if t["issuer"]["start"] + t["issuer"].get("duration", 0) <= a["start"] + 1e-6
                and (not t["consumer"] or a["start"] <= t["consumer"]["start"])
            ]
        if len(candidates) == 1 and candidates[0]["host"] is None:
            candidates[0]["host"] = a
        else:
            unknown.append(
                {"operation": a["id"], "reason": "Tool ID has no unique issuing exchange"}
            )

    aliases = {}
    omitted_unmeasured_tools = 0
    unmeasured_tools = 0
    for key, item in issued.items():
        origin = operations[item["issuer"]["id"]]
        host, consumer = item["host"], item["consumer"]
        if host:
            op = operations[host["id"]]
        else:
            if max_unmeasured_tools is not None and unmeasured_tools >= max_unmeasured_tools:
                aliases[key] = None
                omitted_unmeasured_tools += 1
                continue
            unmeasured_tools += 1
            at = nodes[origin["end_node"]]["time"] if origin.get("end_node") else None
            op = operation(
                key,
                origin["agent"],
                "Tool",
                item["ref"].get("name") or item["ref"]["tool_id"],
                at,
                consumer["start"] if consumer else None,
                measured=False,
                tool_id=item["ref"]["tool_id"],
            )
            unknown.append(
                {
                    "operation": key,
                    "reason": "Tool timing unavailable; model issuance/result consumption bounds only",
                }
            )
        aliases[key] = op["id"]
        if origin.get("end_node") and op.get("start_node"):
            edge(
                origin["end_node"],
                op["start_node"],
                "issued_tool",
                "Model tool_use ID in recorded response",
            )
        if not consumer:
            unknown.append(
                {"operation": op["id"], "reason": "Tool result not observed in a later prompt"}
            )

    for source, target, kind, blocking, evidence in dependencies:
        resolved = aliases.get(source, source)
        if resolved is None:
            resolved = issued[source]["issuer"]["id"]
            kind = "unmeasured_tool_result" if blocking else "history"
            evidence += "; unmeasured tool projection omitted by size limit"
        left, right = operations[resolved], operations[target]
        if left.get("end_node") and right.get("start_node"):
            edge(
                left["end_node"],
                right["start_node"],
                kind,
                evidence,
                confidence="verified" if blocking else "provenance",
                blocking=blocking,
            )
    if omitted_unmeasured_tools or omitted_tool_identities:
        unknown.append({"reason": f"Projection size limit omitted {omitted_unmeasured_tools} unmeasured tool spans and {omitted_tool_identities} tool issuance identities"})
    # A successful execve proves execution of a process image, not its birth,
    # lifetime, exit status or syscall-to-tool ownership. Preserve the syscall
    # and project a separate process observation with unknown lifecycle bounds.
    processes = []
    for a in actions:
        metadata = a.get("metadata", {})
        if a["kind"] == "process":
            processes.append((a, operations[a["id"]]))
        elif (
            a["kind"] == "syscall"
            and a["name"] == "execve"
            and metadata.get("return_value") == 0
            and metadata.get("pid") is not None
            and metadata.get("path")
        ):
            observed = a.get("end_ts")
            if finite(observed) and finite(run.get("time_origin")):
                observed -= run["time_origin"]
            elif finite(a.get("start")) and not a.get("missing_start"):
                observed = a["start"]
            else:
                observed = None
            image = {**a, "kind": "process", "outcome": "execution_observed"}
            op = operation(
                f"process:{a['id']}",
                a["agent"],
                "System",
                metadata["path"].rsplit("/", 1)[-1],
                observed,
                None,
                action=image,
                measured=False,
            )
            op["duration"] = None
            op["metrics"].pop("return_value", None)
            for metric in ("cpu_ms", "cpu_pct", "runqueue_ms", "blocked_ms", "cpu_time_s"):
                op["metrics"].pop(metric, None)
            op["metrics"]["execution_observation"] = "Successful execve; lifecycle not recorded"
            try:
                arguments = json.loads(a.get("arguments", "{}"))
                if isinstance(arguments, dict) and arguments.get("argv") is not None:
                    op["metrics"]["argv"] = arguments["argv"]
            except (ValueError, TypeError):
                pass
            if op.get("start_node"):
                nodes[op["start_node"]]["timing"] = "execve_observation"
                nodes[op["start_node"]]["phase"] = "observation"
            processes.append((a, op))
    process_actions = {a["id"]: op for a, op in processes}
    process_pids = defaultdict(list)
    for a, op in processes:
        pid = op["metrics"].get("pid")
        if pid is not None:
            process_pids[(a["agent"], pid)].append(op)
    for a, op in processes:
        metadata = a.get("metadata", {})
        parent = process_actions.get(metadata.get("parent_process_action_id"))
        evidence = "Explicit parent_process_action_id"
        if parent is None and metadata.get("ppid") is not None:
            candidates = process_pids[(a["agent"], metadata["ppid"])]
            if len(candidates) == 1:
                parent = candidates[0]
                evidence = "Captured PPID and unique process identity in the same actor scope"
        if parent and parent.get("start_node") and op.get("start_node"):
            edge(
                parent["start_node"],
                op["start_node"],
                "process_parent",
                evidence,
                confidence="provenance",
                blocking=False,
            )
        owner = metadata.get("tool_action_id")
        if owner in operations:
            left = operations[owner]
            if left.get("start_node") and op.get("start_node"):
                edge(
                    left["start_node"],
                    op["start_node"],
                    "process_owner",
                    "Explicit tool_action_id",
                    confidence="provenance",
                    blocking=False,
                )

    node_list, edge_list = list(nodes.values()), list(edges.values())
    # The wait after the latest observed prerequisite is elapsed time on a
    # verified dependency. Include it explicitly so a late-starting short tool
    # can determine the join's completion; never hide that unexplained segment.
    latest = defaultdict(lambda: -math.inf)
    for e in edge_list:
        if (
            e["blocking"]
            and e["confidence"] == "verified"
            and e["kind"] != "duration"
            and nodes[e["source"]]["timing"] == "observed"
        ):
            latest[e["target"]] = max(latest[e["target"]], nodes[e["source"]]["time"])
    for e in edge_list:
        e["wait"] = e["gap"]
        if e["blocking"] and e["confidence"] == "verified" and e["kind"] != "duration":
            before = latest[e["target"]]
            e["gap"] = max(0, nodes[e["target"]]["time"] - before) if math.isfinite(before) else 0
            e["weight"] = e["gap"]
            e["segment_confidence"] = "unexplained_wait" if e["gap"] else "verified"
    path = critical_path(node_list, edge_list)
    # A presentation path omits instrumentation vertices, without contracting
    # them or inventing dependencies between their neighbours. Keep the full
    # analysis available to the system profiler and evidence inspector.
    operation_nodes = [
        n
        for n in node_list
        if operations[n["operation"]]["band"] in {"LLM", "Tool"}
        or operations[n["operation"]]["kind"] == "process"
    ]
    operation_ids = {n["id"] for n in operation_nodes}
    operation_edges = [
        e for e in edge_list if e["source"] in operation_ids and e["target"] in operation_ids
    ]
    operation_path = critical_path(operation_nodes, operation_edges)
    operation_path_edges = set(operation_path["edge_ids"])
    operation_path["measured_work"] = sum(
        e["weight"]
        for e in operation_edges
        if e["id"] in operation_path_edges and e["kind"] == "duration"
    )
    operation_path["unexplained_wait"] = sum(
        e["gap"]
        for e in operation_edges
        if e["id"] in operation_path_edges and e["kind"] != "duration"
    )
    # Each agent's chain is computed over its own operations only. Recorded
    # dependencies are scoped to one agent, so a cross-agent edge is evidence
    # we do not interpret; the multi-agent critical path stays unavailable.
    owner = {n["id"]: n["agent"] for n in operation_nodes}
    by_agent = defaultdict(list)
    for n in operation_nodes:
        by_agent[n["agent"]].append(n)
    agent_paths = {}
    for agent, members in by_agent.items():
        local = [
            e
            for e in operation_edges
            if owner[e["source"]] == agent and owner[e["target"]] == agent
        ]
        chain = critical_path(members, local)
        chosen = set(chain["edge_ids"])
        chain["measured_work"] = sum(
            e["weight"] for e in local if e["id"] in chosen and e["kind"] == "duration"
        )
        chain["unexplained_wait"] = sum(
            e["gap"] for e in local if e["id"] in chosen and e["kind"] != "duration"
        )
        agent_paths[agent] = chain
    cross_agent = sum(owner[e["source"]] != owner[e["target"]] for e in operation_edges)
    path_edges = set(path["edge_ids"])
    path["measured_work"] = sum(
        e["weight"] for e in edge_list if e["id"] in path_edges and e["kind"] == "duration"
    )
    path["unexplained_wait"] = sum(
        e["gap"] for e in edge_list if e["id"] in path_edges and e["kind"] != "duration"
    )
    critical_edges, critical_nodes = set(path["edge_ids"]), set(path["node_ids"])
    for n in node_list:
        n["critical"] = n["id"] in critical_nodes
        n["slack"] = (
            path["slack"].get(n["id"])
            if n["timing"] == "observed" and operations[n["operation"]]["duration"] is not None
            else None
        )
    gaps = []
    gap_targets = set()
    for e in edge_list:
        e["critical"] = e["id"] in critical_edges
        if (
            e["blocking"]
            and e["kind"] != "duration"
            and e["gap"] > 1e-6
            and e["target"] not in gap_targets
        ):
            gap_targets.add(e["target"])
            gaps.append(
                {
                    "edge": e["id"],
                    "duration": e["gap"],
                    "reason": "Elapsed dependency wait; timing observed, cause unexplained",
                }
            )
    incoming = defaultdict(list)
    for e in edge_list:
        if e["blocking"] and e["kind"] == "tool_result":
            incoming[e["target"]].append(e["source"])
    return {
        "version": 1,
        "nodes": node_list,
        "edges": edge_list,
        "operations": list(operations.values()),
        "critical_path": path,
        "operation_critical_path": operation_path,
        "agent_critical_paths": agent_paths,
        "system_critical_path": {
            "available": False,
            "reason": "Cross-agent logical dependencies are not captured; each agent's "
            "known chain is shown separately and none of them is the system-wide path.",
            "cross_agent_edges": cross_agent,
        },
        "joins": [{"node": k, "dependencies": v} for k, v in incoming.items() if len(v) > 1],
        "summary": {
            "execution_duration": run.get("execution_wall_time", clock)
            if run.get("execution_started_at") is not None
            else clock,
            "critical_path_duration": path["duration"],
            "critical_path_measured_work": path["measured_work"],
            "critical_path_unexplained_wait": path["unexplained_wait"],
            "operation_time": totals,
            "time_basis": "Sum of measured operation durations; concurrent operations overlap",
        },
        "coverage": {
            "complete": False,
            "confidence": "partial",
            "unknown": unknown,
            "gaps": gaps,
            "measured_operations": sum(o["duration"] is not None for o in operations.values()),
            "total_operations": len(operations),
            "native_process_attribution": False,
            "omitted_unmeasured_tools": omitted_unmeasured_tools,
            "omitted_tool_identities": omitted_tool_identities,
            "verified_edges": sum(e["confidence"] == "verified" for e in edge_list),
            "inferred_edges": sum(e["confidence"] == "inferred" for e in edge_list),
            "explanation": "Partial recorded dependency path. Unexplained waits are marked; unobserved work and cross-agent dependencies remain unknown. System-wide critical path is not established.",
        },
    }


def enrich_saved_evidence(run, trace_path):
    """Join a saved span's call_label to its sibling canonical SQLite exchange.

    Keep the saved profiler's own clocks/actions/metrics unchanged. Prompt previews
    and truncated span attributes never replace complete canonical identity evidence.
    """
    import sqlite3
    from pathlib import Path
    from .trajectory import LiveSource

    by_label = defaultdict(list)
    for action in run.get("actions", []):
        if action["kind"] == "model" and action.get("metadata", {}).get("call_label"):
            by_label[(action["agent"], action["metadata"]["call_label"])].append(action)
    if not by_label:
        return run
    path = Path(trace_path)
    databases = list((path.parent.parent / "logs").glob("*data.sqlite3"))
    profile = path.parent / "profile_data.sqlite3"
    if profile.is_file():
        databases.append(profile)
    observed = {}
    failures = []
    for database in databases:
        try:
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=1) as con:
                tables = {
                    row[0]
                    for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                if not {"exchanges", "exchange_chain", "blocks"} <= tables:
                    continue
                for label, agent, kind in con.execute("SELECT call_label,name,type FROM exchanges"):
                    key = (agent, label)
                    if key not in by_label:
                        continue
                    payload = LiveSource._exchange(con, label, kind)
                    evidence = payload["causal_evidence"]
                    if key in observed and observed[key] != evidence:
                        observed[key] = None  # conflicting source identities stay unknown
                    else:
                        observed[key] = evidence
        except (sqlite3.Error, OSError, ValueError):
            failures.append(f"{database.name}: canonical exchange evidence unavailable")
    count = 0
    for key, evidence in observed.items():
        if evidence is None or len(by_label[key]) != 1:
            continue
        action = by_label[key][0]
        action["causal_evidence"] = evidence
        action["tool_call_ids"] = [
            r["tool_id"] for r in evidence["response"] if r["type"] == "tool_use" and r["tool_id"]
        ]
        count += 1
    run.setdefault("coverage", {})["causal_evidence"] = {
        "canonical_exchanges": count,
        "gaps": failures,
    }
    return run
