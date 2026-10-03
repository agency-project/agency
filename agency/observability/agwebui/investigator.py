"""Read-only projections of saved Agency traces for the execution investigator.

No runtime instrumentation is changed. Episodes and workflow obligations are
inferences; times, counters, tool output and prompt blocks remain observations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EPISODE_TITLES = {
    "search": "Explore the repository",
    "read": "Inspect implementation",
    "edit": "Implement the change",
    "test": "Run validation",
    "review": "Review the result",
    "tool": "Execute tools",
    "setup": "Prepare execution",
    "model": "Reason and plan",
    "queue": "Await scheduler admission",
    "dependency": "Wait for dependencies",
}


def object_value(value):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def number(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def run_key(path):
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


@lru_cache(maxsize=256)
def recorded_model(path: str, modified: float):
    # Read a bounded prefix for catalog filtering rather than loading every trace.
    # The selected run still exposes all model identities in its agent metadata.
    with Path(path).open() as stream:
        prefix = stream.read(1024 * 1024)
    match = re.search(r'"model"\s*:\s*"([^"\\]+)"', prefix)
    return match.group(1) if match else "Recorded in trace"


@lru_cache(maxsize=4)
def catalog(profile_dir: str, bucket: int = 0):
    entries = []
    experiment = ROOT / "artifacts/e1a1/trajectory-catalog"
    index = experiment / "trajectory-index.csv"
    if index.exists():
        with index.open() as stream:
            for row in csv.DictReader(stream):
                path = experiment / "profile-traces" / Path(row["profiler_trace"]).name
                if not path.exists():
                    continue
                entries.append(
                    {
                        "id": run_key(path),
                        "title": row["instance_id"],
                        "subtitle": f"{row['condition'].replace('_', ' ')} · repetition {row['repetition']}",
                        "condition": row["condition"],
                        "task": row["instance_id"],
                        "repetition": row["repetition"],
                        "source": "recorded",
                        "path": str(path),
                        "duration": float(row["worker_wall_seconds"]),
                        "resolved": row["official_resolved"] == "True",
                        "model": "gpt-5.6-luna",
                        "harness": "codex",
                    }
                )
    configured = Path(profile_dir) / "agprof.trace.json"
    paths = [configured] if configured.is_file() else []
    for base in (ROOT / "agency_runs", ROOT / "artifacts"):
        if base.exists():
            paths.extend(sorted(base.rglob("agprof.trace.json")))
    seen = set()
    for path in paths:
        key = run_key(path)
        if key in seen:
            continue
        seen.add(key)
        summary = {}
        summary_path = path.parent / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, ValueError):
                pass  # A partially written summary does not hide a usable trace.
        engines = summary.get("coverage", {}).get("engines", {})
        entries.append(
            {
                "id": key,
                "title": "Current run" if path == configured else path.parent.name,
                "subtitle": str(path.parent.relative_to(ROOT))
                if path.is_relative_to(ROOT)
                else "Configured profiler directory",
                "source": "recorded",
                "path": str(path),
                "duration": (summary.get("duration_ms") or 0) / 1000,
                "model": recorded_model(str(path), path.stat().st_mtime),
                "harness": ", ".join(engines) or "unknown",
            }
        )
    annotation_root = ROOT / "artifacts/tool-annotation-ec2"
    if annotation_root.exists():
        for path in sorted(annotation_root.rglob("events.jsonl")):
            assignment_path = path.parent.parent / "assignment.json"
            if not assignment_path.exists():
                continue
            try:
                assignment = json.loads(assignment_path.read_text())
                verifier_path = path.parent.parent / "verifier.json"
                verifier = json.loads(verifier_path.read_text()) if verifier_path.exists() else {}
                entries.append(
                    {
                        "id": run_key(path),
                        "title": assignment.get("task_id", "Tool annotation experiment"),
                        "subtitle": f"{assignment.get('arm', 'baseline')} · repetition {assignment.get('repetition', 0)} · {path.parents[4].name}",
                        "source": "recorded",
                        "format": "native_events",
                        "path": str(path),
                        "task": assignment.get("task_id"),
                        "condition": assignment.get("arm"),
                        "repetition": str(assignment.get("repetition", 0)),
                        "model": assignment.get("model_id", "unknown"),
                        "harness": "native",
                        "resolved": verifier.get("success"),
                    }
                )
            except (OSError, ValueError):
                continue  # Incomplete trials remain available in their source directory.
    return entries


def classify(name, command="", intent=""):
    text = f"{name} {command} {intent}".lower()
    if "llm:" in name:
        return "model"
    if "scheduler_queue" in name:
        return "queue"
    if "sync:" in name and ("wait" in name or "depend" in name):
        return "dependency"
    if not name.startswith("tool:"):
        return "setup"
    tool_name = name.removeprefix("tool:").lower()
    if tool_name in ("write", "edit", "apply_patch"):
        return "edit"
    if tool_name in ("read", "read_file"):
        return "read"
    if tool_name in ("retrieve", "search"):
        return "search"
    if re.search(r"apply_patch|write_file|edit_file|\bsed\s+-i|\.write_text|patch\s+<<", text):
        return "edit"
    if re.search(r"pytest|npm test|cargo test|go test|unittest|run.*tests|validate", text):
        return "test"
    if re.search(r"\brg\b|\bgrep\b|\bfind\b|\bls\b|search", text):
        return "search"
    if re.search(r"\bcat\b|\bsed\b|read_file|read|inspect", text):
        return "read"
    if re.search(r"git diff|git status|review", text):
        return "review"
    return "tool"


def context_blocks(raw, previous):
    messages = object_value(raw)
    partial = False
    if not isinstance(messages, list) and isinstance(raw, str) and raw.lstrip().startswith("["):
        # Traces cap prompt attributes. Recover only complete JSON messages;
        # never repair a cut string or infer that missing suffixes were dropped.
        messages = []
        offset = raw.index("[") + 1
        decoder = json.JSONDecoder()
        while offset < len(raw):
            while offset < len(raw) and raw[offset] in " ,\n\t":
                offset += 1
            try:
                message, offset = decoder.raw_decode(raw, offset)
            except ValueError:
                break
            if isinstance(message, dict):
                messages.append(message)
        partial = True
    if not isinstance(messages, list) or not messages:
        return None, previous
    blocks = []
    current = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "unknown")
        content = message.get("blocks", message.get("content", []))
        if isinstance(content, str):
            content = [{"text": content, "type": "text"}]
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text") or block.get("content") or block.get("arguments") or ""
            if not isinstance(text, str):
                text = json.dumps(text)
            if not text:
                continue
            digest = hashlib.sha256(f"{role}:{text}".encode()).hexdigest()[:16]
            source = (
                "System instructions"
                if role in ("system", "developer")
                else "Original task"
                if role == "user"
                else "Conversation"
            )
            if block.get("type") == "tool_result" or role == "tool":
                source = "Tool results"
            if "summary" in text[:100].lower() or "compacted" in text[:100].lower():
                source = "Summary"
            item = {
                "id": digest,
                "source": source,
                "role": role,
                "chars": len(text),
                "change": "repeated" if digest in previous else "added",
                "preview": text[:1800],
            }
            blocks.append(item)
            current[digest] = item
    dropped = (
        []
        if partial
        else [{**item, "change": "dropped"} for key, item in previous.items() if key not in current]
    )
    next_previous = {**previous, **current} if partial else current
    return {
        "blocks": blocks,
        "dropped": dropped,
        "chars": sum(b["chars"] for b in blocks),
        "source": "recorded",
        "truncated": partial,
    }, next_previous


def group_episodes(actions):
    episodes = []
    lookup = {a["id"]: a for a in actions}
    by_agent = defaultdict(list)
    for action in actions:
        by_agent[action["agent"]].append(action)

    def new_episode(agent, kind, items):
        episode = {
            "id": f"episode-{len(episodes)}",
            "agent": agent,
            "kind": kind,
            "title": EPISODE_TITLES[kind],
            "actions": [a["id"] for a in items],
            "start": min(a["start"] for a in items),
            "end": max(a["start"] + a["duration"] for a in items),
            "status": "unknown",
            "inferred": True,
        }
        episodes.append(episode)
        return episode

    for agent, items in by_agent.items():
        episode = None
        pending_models = []
        for action in items:
            if (
                pending_models
                and action["start"] - (pending_models[-1]["start"] + pending_models[-1]["duration"])
                > 12
            ):
                new_episode(agent, "model", pending_models)
                pending_models = []
                episode = None
            if action["kind"] == "model":
                pending_models.append(action)
                continue
            kind = action["kind"]
            recovered = episode is not None and any(
                lookup[id]["outcome"] == "failed" for id in episode["actions"]
            )
            batch = pending_models + [action]
            if (
                episode is None
                or episode["kind"] != kind
                or len(episode["actions"]) >= 8
                or action["start"] - episode["end"] > 12
                or recovered
            ):
                episode = new_episode(agent, kind, batch)
                if recovered and kind == "test":
                    episode["title"] = "Revalidate after failure"
            else:
                episode["actions"].extend(a["id"] for a in batch)
                episode["start"] = min(episode["start"], *(a["start"] for a in batch))
                episode["end"] = max(episode["end"], *(a["start"] + a["duration"] for a in batch))
            pending_models = []
        if pending_models:
            new_episode(agent, "model", pending_models)
    for episode in episodes:
        statuses = [lookup[id]["outcome"] for id in episode["actions"]]
        if "failed" in statuses:
            episode["status"] = "failed"
        elif "interrupted" in statuses:
            episode["status"] = "interrupted"
        elif all(status == "success" for status in statuses):
            episode["status"] = "completed"
        for action_id in episode["actions"]:
            lookup[action_id]["episode"] = episode["id"]
    return sorted(episodes, key=lambda e: e["start"])


def obligations(actions, resolved=None):
    rows = []
    for kind, title in [
        ("search", "Locate relevant implementation"),
        ("read", "Inspect the behavior"),
        ("edit", "Implement a change"),
        ("test", "Validate with tests"),
        ("review", "Review final artifacts"),
    ]:
        evidence = [a for a in actions if a["kind"] == kind]
        successful = [a for a in evidence if a["outcome"] == "success"]
        status = "unvalidated"
        if evidence:
            status = (
                "completed"
                if successful and evidence[-1]["outcome"] == "success"
                else "failed"
                if evidence[-1]["outcome"] == "failed"
                else "in progress"
                if evidence[-1]["outcome"] == "interrupted"
                else "unvalidated"
            )
        rows.append(
            {
                "id": f"obligation-{kind}",
                "title": title,
                "status": status,
                "evidence": [a["id"] for a in evidence],
                "inferred": True,
                "note": "Tool completion is observed; semantic correctness needs review."
                if successful
                else "No conclusive supporting event was recorded.",
            }
        )
    rows.append(
        {
            "id": "obligation-outcome",
            "title": "Confirm task outcome",
            "status": "completed"
            if resolved is True
            else "failed"
            if resolved is False
            else "unvalidated",
            "evidence": [],
            "inferred": False,
            "note": "Saved experiment evaluator result."
            if resolved is not None
            else "No independent evaluator result is available.",
        }
    )
    return rows


def model_output(raw):
    response = object_value(raw)
    if not isinstance(response, dict):
        return str(raw or "")
    content = response.get("content")
    parts = [content] if isinstance(content, str) else []
    for block in response.get("blocks", []):
        if block.get("text"):
            parts.append(block["text"])
        if block.get("type") == "tool_use":
            parts.append(f"{block.get('name', 'tool')}({block.get('arguments', '')})")
    for call in response.get("tool_calls", []):
        function = call.get("function", {})
        parts.append(f"{function.get('name', 'tool')}({function.get('arguments', '')})")
    return "\n\n".join(parts)


def reduce_samples(samples, limit=800):
    if len(samples) <= limit:
        return samples
    # Keep bucket extrema so the overview does not erase short sampled peaks.
    bucket_size = math.ceil(len(samples) / (limit // 2))
    result = []
    for offset in range(0, len(samples), bucket_size):
        bucket = samples[offset : offset + bucket_size]
        low = min(bucket, key=lambda point: point[1])
        high = max(bucket, key=lambda point: point[1])
        result.extend(sorted([low, high], key=lambda point: point[0]))
    return result


def normalize_trace(trace, entry, summary=None):
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    if not isinstance(events, list):
        raise ValueError("Expected a Chrome traceEvents array.")
    spans = sorted(
        [
            e
            for e in events
            if isinstance(e, dict) and e.get("ph") == "X" and number(e.get("ts")) is not None
        ],
        key=lambda e: e["ts"],
    )
    origin = min((e["ts"] for e in spans), default=0)
    duration = max(
        ((e["ts"] - origin + max(0, number(e.get("dur")) or 0)) / 1e6 for e in spans), default=0
    )
    agents = {}
    models = defaultdict(dict)
    actions = []
    intervals = []
    counters = defaultdict(list)
    for span in spans:
        attrs = span.get("args") if isinstance(span.get("args"), dict) else {}
        agent = attrs.get("agency.agent_id") or "workflow"
        if agent not in agents:
            agents[agent] = {
                "id": agent,
                "label": agent.removeprefix("agent_") if agent != "workflow" else "Host workflow",
                "parent": attrs.get("agency.parent_agent_id"),
                "role": "Agent" if agent != "workflow" else "Host workflow",
                "model": attrs.get("model"),
                "harness": attrs.get("harness"),
            }
        for key in ("model", "harness"):
            if attrs.get(key):
                agents[agent][key] = attrs[key]
        if attrs.get("agency.parent_agent_id"):
            agents[agent]["parent"] = attrs["agency.parent_agent_id"]
        name = span.get("name", "")
        if not (
            name.startswith(
                (
                    "tool:",
                    "llm:attempt",
                    "sync:scheduler_queue",
                    "sync:dependency",
                    "sync:result_wait",
                    "runtime.harness_start",
                    "sandbox:commit",
                    "sandbox:restore",
                )
            )
        ):
            continue
        arguments = object_value(attrs.get("tool.arguments")) or {}
        command = (
            arguments.get("command", arguments.get("cmd", json.dumps(arguments)))
            if isinstance(arguments, dict)
            else str(arguments)
        )
        intent = attrs.get("tool.intent") or attrs.get("intent") or attrs.get("purpose") or ""
        kind = classify(name, command, intent)
        start = (span["ts"] - origin) / 1e6
        elapsed = max(0, number(span.get("dur")) or 0) / 1e6
        outcome = attrs.get("outcome") or "unknown"
        outcome = "failed" if outcome in ("failure", "error") else outcome
        context = None
        if kind == "model":
            context, models[agent] = context_blocks(attrs.get("llm.messages"), models[agent])
            if context:
                context["truncated"] = context["truncated"] or bool(
                    attrs.get("llm.messages_truncated")
                )
        action = {
            "id": attrs.get("span_id") or attrs.get("agency.span_id") or f"action-{len(actions)}",
            "agent": agent,
            "kind": kind,
            "name": name.removeprefix("tool:"),
            "intent": intent,
            "start": start,
            "duration": elapsed,
            "outcome": outcome,
            "command": command if name.startswith("tool:") else "",
            "result": model_output(attrs.get("llm.response"))
            if kind == "model"
            else str(attrs.get("tool.result") or ""),
            "context": context,
            "files": sorted(
                set(re.findall(r"[\w./-]+\.(?:py|ts|tsx|js|json|md|rs|go|css)", command))
            )[:30],
            "metadata": {
                k: v
                for k, v in attrs.items()
                if k
                not in (
                    "llm.messages",
                    "llm.response",
                    "tool.result",
                    "tool.arguments",
                    "agency_presentation",
                )
            },
            "tokens": number(attrs.get("input_tokens")),
            "output_tokens": number(attrs.get("output_tokens")),
            "source": "recorded",
            "timing": attrs.get("timing", "host_span"),
        }
        actions.append(action)
        intervals.append(
            {
                "id": action["id"],
                "agent": agent,
                "start": start,
                "duration": elapsed,
                "kind": "model"
                if kind == "model"
                else "queue"
                if kind == "queue"
                else "dependency"
                if kind == "dependency"
                else "tool",
                "label": intent or name,
                "action": action["id"],
                "source": "recorded",
            }
        )
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "C" or number(event.get("ts")) is None:
            continue
        name = event.get("name", "")
        if name.startswith("agency-counter-v1:"):
            info = object_value(name.removeprefix("agency-counter-v1:")) or {}
            name = info.get("label", name)
        for key, value in (event.get("args") or {}).items():
            if number(value) is not None:
                counters[name if key == "value" else f"{name} {key}"].append(
                    [(event["ts"] - origin) / 1e6, value]
                )
    # Unknown gaps are explicit coverage gaps, never inferred CPU or idle work.
    for agent in agents:
        covered = sorted(
            (i["start"], i["start"] + i["duration"]) for i in intervals if i["agent"] == agent
        )
        cursor = 0
        for start, end in covered:
            if start - cursor > 0.05:
                intervals.append(
                    {
                        "id": f"gap-{len(intervals)}",
                        "agent": agent,
                        "start": cursor,
                        "duration": start - cursor,
                        "kind": "unknown",
                        "label": "Unattributed interval",
                        "action": None,
                        "source": "coverage gap",
                    }
                )
            cursor = max(cursor, end)
        if duration - cursor > 0.05:
            intervals.append(
                {
                    "id": f"gap-{len(intervals)}",
                    "agent": agent,
                    "start": cursor,
                    "duration": duration - cursor,
                    "kind": "unknown",
                    "label": "Unattributed interval",
                    "action": None,
                    "source": "coverage gap",
                }
            )
    episodes = group_episodes(actions)
    if entry.get("model") == "Recorded in trace" or not entry.get("model"):
        observed_models = sorted({a["model"] for a in agents.values() if a["model"]})
        if observed_models:
            entry = {**entry, "model": ", ".join(observed_models)}
    edges = [
        {
            "from": a["parent"],
            "to": a["id"],
            "kind": "delegation",
            "label": "Recorded parent relationship",
            "source": "recorded",
        }
        for a in agents.values()
        if a["parent"]
    ]
    return {
        **{k: v for k, v in entry.items() if k != "path"},
        "duration": max(duration, entry.get("duration", 0)),
        "agents": list(agents.values()),
        "actions": actions,
        "episodes": episodes,
        "intervals": sorted(intervals, key=lambda i: i["start"]),
        "edges": edges,
        "counters": {key: reduce_samples(values) for key, values in counters.items()},
        "obligations": obligations(actions, entry.get("resolved")),
        "summary": summary or {},
        "coverage": {
            "spans": len(spans),
            "events": len(events),
            "context": "Recorded prompt blocks; sizes in characters, token totals where reported.",
            "relationships": "Only explicit parent relationships; missing messages are not inferred.",
        },
    }


@lru_cache(maxsize=4)
def load_run(path: str, modified: float, entry_json: str):
    trace_path = Path(path)
    summary_path = trace_path.parent / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError):
            pass  # A damaged optional summary must not block trace inspection.
    entry = json.loads(entry_json)
    if entry.get("format") == "native_events":
        return normalize_native_events(trace_path, entry)
    return normalize_trace(json.loads(trace_path.read_text()), entry, summary)


def normalize_native_events(path, entry):
    """Join model-stated purpose to completion by native event_id, never by order."""
    events = []
    warnings = 0
    for line in Path(path).read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            warnings += 1
            continue
        if isinstance(event, dict) and number(event.get("timestamp_ns")) is not None:
            events.append(event)
    origin = min((e["timestamp_ns"] for e in events), default=0)
    annotations = {e["event_id"]: e for e in events if e.get("kind") == "tool_annotation"}
    trace = []
    for index, event in enumerate(events):
        kind = event.get("kind")
        if kind not in ("tool_result", "model_exchange"):
            continue
        duration_ns = number(event.get("duration_ns"))
        if duration_ns is None:
            continue
        attrs = {
            "agency.agent_id": event.get("agent_id", "agent"),
            "harness": "native",
            "model": entry.get("model"),
            "span_id": event.get("event_id") or f"native-model-{index}",
            "timing": "observer_duration",
            "provenance": "native_observer",
            "outcome": "success",
        }
        if kind == "tool_result":
            annotation = annotations.get(event.get("event_id"), {})
            record = annotation.get("annotation", {})
            raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
            attrs.update(
                {
                    "tool.arguments": json.dumps(annotation.get("arguments", {})),
                    "tool.result": event.get("result", ""),
                    "call_id": event.get("call_id"),
                    "annotation_status": record.get("status"),
                    "tool.intent": raw.get("purpose", "")
                    if record.get("status") == "valid"
                    else "",
                    "workstream_ids": raw.get("workstream_ids", [])
                    if isinstance(raw, dict)
                    else [],
                    "outcome": "failed" if event.get("category") == "error" else "success",
                }
            )
            name = "tool:" + event.get("tool_name", "unknown")
        else:
            usage = event.get("response", {}).get("usage") or {}
            attrs.update(
                {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "llm.response": json.dumps(event.get("response", {}).get("message", {})),
                }
            )
            name = "llm:attempt[0]"
        trace.append(
            {
                "ph": "X",
                "name": name,
                "ts": (event["timestamp_ns"] - origin - duration_ns) / 1000,
                "dur": duration_ns / 1000,
                "args": attrs,
            }
        )
    run = normalize_trace({"traceEvents": trace}, entry)
    run["coverage"]["events"] = len(events)
    run["coverage"]["partial_lines"] = warnings
    run["coverage"]["context"] = (
        "This observer stores responses and usage, but not full model prompts."
    )
    run["coverage"]["relationships"] = (
        "Native tool annotations are model-stated purposes, not verified internal reasoning."
    )
    return run
