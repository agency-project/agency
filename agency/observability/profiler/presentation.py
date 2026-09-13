"""Version 1 of the object-owned timeline, independent of Perfetto's UI.

Ownership comes from instrumentation or the explicit parent graph. Physical
thread identity and causal edges remain separate from the presentation lane.
Older traces without ownership annotations retain their original layout.
"""

import json
import heapq

OWNER_KEYS = (
    "agency.agent_id",
    "agency.sandbox_id",
    "agency.component",
    "agency.execution_side",
    "agency.reporter_id",
    "agency.namespace_pid",
    "agency.pid_namespace",
    "agency.process_start_ticks",
)
PROCESS_KEYS = ("agency.namespace_pid", "agency.pid_namespace", "agency.process_start_ticks")
COUNTER_PREFIX = "agency-counter-v1:"


def _id(value):
    return f"{value:016x}" if isinstance(value, int) else value


def _merge_owner(parent, metadata):
    owner = dict(parent)
    side = metadata.get("agency.execution_side")
    if side is not None and side != owner.get("agency.execution_side"):
        for key in PROCESS_KEYS:
            owner.pop(key, None)
    owner.update({k: metadata[k] for k in OWNER_KEYS if metadata.get(k) is not None})
    return owner


def _pack_intervals(groups):
    """Pack connected source stacks as indivisible intervals, never by slice.

    Only compact timing/index tuples are kept; transcript/tool details can
    continue streaming on the second pass through the source events.
    """
    assignments, labels = {}, {}
    next_tid = 2_000_000_000
    for path_key, sources in sorted(groups.items()):
        blocks = []
        for intervals in sources.values():
            block = None
            for start, end, ordinal in sorted(intervals):
                if block is None or start >= block[1]:
                    block = [start, end, [ordinal]]
                    blocks.append(block)
                else:
                    block[1] = max(block[1], end)
                    block[2].append(ordinal)
        active, free, tids = [], [], []
        for start, end, ordinals in sorted(blocks):
            while active and active[0][0] <= start:
                _, lane = heapq.heappop(active)
                heapq.heappush(free, lane)
            if free:
                lane = heapq.heappop(free)
            else:
                lane = len(tids)
                tids.append(next_tid)
                prefix = "Host" if json.loads(path_key)[-1][0] == "host" else "Execution"
                labels[next_tid] = f"{prefix} · Lane {lane + 1}"
                next_tid += 1
            heapq.heappush(active, (end, lane))
            for ordinal in ordinals:
                assignments[ordinal] = tids[lane]
    return assignments, labels


def present_events(events, records, root_pid, *, process_info=None):
    by_id = {_id(r[7]): r for r in records if len(r) > 7 and r[7] is not None}
    enabled = any(
        (r[6] or {}).get(k)
        for r in records
        if len(r) > 6
        for k in ("agency.agent_id", "agency.component")
    )
    if not enabled:
        for event in events():
            event.pop("agency_resource", None)
            yield event
        return
    owners = {}

    def resolve(span_id):
        chain, seen = [], set()
        while span_id in by_id and span_id not in owners and span_id not in seen:
            seen.add(span_id)
            record = by_id[span_id]
            chain.append((span_id, record[6] or {}))
            span_id = _id(record[8]) if len(record) > 8 else None
        result = owners.get(span_id, {})
        for key, metadata in reversed(chain):
            result = _merge_owner(result, metadata)
            owners[key] = result
        return result

    process_matches = {}
    for process in (process_info or {}).values():
        key = (
            process.get("sandbox"),
            process.get("pid_namespace"),
            process.get("namespace_pid"),
            process.get("start_ticks"),
        )
        if all(value is not None for value in key):
            process_matches.setdefault(key, []).append(process)

    def mapped_process(owner):
        if owner.get("agency.execution_side") != "sandbox":
            return None
        key = (
            owner.get("agency.sandbox_id"),
            owner.get("agency.pid_namespace"),
            owner.get("agency.namespace_pid"),
            owner.get("agency.process_start_ticks"),
        )
        matches = process_matches.get(key, [])
        return matches[0] if len(matches) == 1 else None

    sandbox_agents = {}
    for span_id in by_id:
        owner = resolve(span_id)
        agent, sandbox = owner.get("agency.agent_id"), owner.get("agency.sandbox_id")
        if agent and sandbox:
            sandbox_agents.setdefault(sandbox, set()).add(agent)

    def sandbox_path(sandbox):
        agents = sandbox_agents.get(sandbox, set())
        base = (
            [["agent:" + next(iter(agents)), "Agent " + next(iter(agents))]]
            if len(agents) == 1
            else [["shared", "Shared resources"]]
        )
        return base + [["sandbox:" + sandbox, "Sandbox " + sandbox]]

    def path_for(owner, tid):
        agent = owner.get("agency.agent_id")
        sandbox = owner.get("agency.sandbox_id")
        if owner.get("agency.execution_side") == "sandbox":
            base = (
                sandbox_path(sandbox)
                if sandbox
                else [["unattributed", "Unattributed"], ["sandbox", "Sandbox (unknown)"]]
            )
            reporter = str(owner.get("agency.reporter_id", "unknown"))
            process = mapped_process(owner)
            if process is not None:
                return base + [
                    ["processes", "Processes"],
                    ["process:" + process["identity"], process["display_name"]],
                ]
            return base + [
                ["processes", "Processes"],
                ["reporter:" + reporter, "Harness attempt " + reporter],
            ]
        if agent:
            return [["agent:" + agent, "Agent " + agent], ["host", "Host"]]
        if owner.get("agency.component") == "orchestrator":
            return [["orchestrator", "Global orchestrator"]]
        if owner.get("agency.component") == "workflow" or tid == root_pid:
            return [["workflow", "Workflow / Main"]]
        return [["unattributed", "Unattributed"]]

    def route(event):
        args = event.setdefault("args", {})
        span_id = _id(args.get("span_id"))
        owner = resolve(
            span_id
            if span_id in by_id
            else _id(args.get("agency.context_span_id", args.get("parent_span_id")))
        )
        owner = _merge_owner(owner, args)
        args.update(owner)
        if "agency.context_span_id" in args:
            args["agency.context_span_id"] = _id(args["agency.context_span_id"])
        source_tid = args.setdefault("agency.source_tid", event["tid"])
        source_pid = args.setdefault("agency.source_pid", root_pid)
        path = path_for(owner, source_tid)
        process = mapped_process(owner)
        if process is not None:
            args["agency.host_pid"] = process["pid"]
            args["agency.process_identity"] = process["identity"]
        if event.get("cat") == "gpu_lease":
            path = [["shared", "Shared resources"], ["gpu", "GPUs"]]
        return path, (source_pid, source_tid, args.get("agency.reporter_id"))

    def descriptor(path, **extra):
        spec = {"version": 1, "path": path, **extra}
        if path[0][0] == "shared" and len(path) > 1 and path[1][0].startswith("sandbox:"):
            agents = sandbox_agents.get(path[1][0][len("sandbox:") :], set())
            if len(agents) > 1:
                spec["shared_with"] = sorted(agents)
        return json.dumps(spec)

    groups, original_names = {}, {}
    ordinal = 0
    for event in events():
        if event["ph"] == "X":
            path, source = route(event)
            groups.setdefault(json.dumps(path), {}).setdefault(source, []).append(
                (event["ts"], event["ts"] + event["dur"], ordinal)
            )
            ordinal += 1
        elif event["ph"] == "M" and event["name"] == "thread_name":
            original_names[event["tid"]] = event["args"]["name"]
    assignments, labels = _pack_intervals(groups)
    endpoints = {}
    trace_pid = root_pid
    ordinal = 0
    for event in events():
        trace_pid = event["pid"]
        phase = event["ph"]
        if phase == "M" and event["name"].startswith("thread_"):
            continue
        if phase == "X":
            path, _ = route(event)
            args = event["args"]
            args["agency.source_thread_name"] = original_names.get(event["tid"], "Unknown thread")
            event["tid"] = assignments[ordinal]
            ordinal += 1
            args["agency_presentation"] = descriptor(path)
            if "trace_span_id" in args:
                endpoints[args["trace_span_id"]] = (event["tid"], path)
            yield event
        elif phase in ("s", "f") and event.get("cat") == "agprof.relationship":
            args = event["args"]
            parent = endpoints.get(args["parent_trace_span_id"])
            child = endpoints.get(args["child_trace_span_id"])
            if not parent or not child:
                continue
            if args["relationship"] == "same_track_nesting" and parent[1] != child[1]:
                continue
            event["tid"] = parent[0] if phase == "s" else child[0]
            yield event
        elif phase == "C":
            resource = event.pop("agency_resource", {})
            process = resource.get("process", {})
            name = resource.get("name", event["name"])
            sandbox = process.get("sandbox")
            if sandbox:
                path = sandbox_path(sandbox) + [
                    ["processes", "Processes"],
                    ["process:" + process["identity"], process["display_name"]],
                    ["counters", "Counters"],
                ]
            elif name.startswith("sandbox:"):
                sandbox = name.split(":")[1]
                path = sandbox_path(sandbox) + [["counters", "Sandbox counters"]]
            elif process.get("pid") == root_pid:
                path = [["workflow", "Workflow / Main"], ["counters", "Host process counters"]]
            else:
                path = [
                    ["shared", "Shared resources"],
                    ["counters", "Workload / hardware counters"],
                ]
            event["name"] = COUNTER_PREFIX + descriptor(path, label=event["name"])
            yield event
        else:
            yield event
    for tid, label in labels.items():
        yield {
            "ph": "M",
            "pid": trace_pid,
            "tid": tid,
            "name": "thread_name",
            "args": {"name": label},
        }
