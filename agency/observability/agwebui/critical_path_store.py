"""Compact, read-only historical input for the critical path profiler.

The trace is streamed one event at a time. Python stacks and syscalls are
counted, never retained as graph nodes. Transcripts are converted to identity
evidence before retention; the existing detailed investigator owns their UI.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sqlite3
import threading

from .causal_graph import exchange_evidence, messages_evidence
from .investigator import normalize_trace, object_value, reduce_samples, visible_model_response
from .investigator import response_tool_ids

MAX_OPERATIONS = 50_000
MAX_EVENT_BYTES = 32 * 1024 * 1024
COUNTER_POINTS = 400
MAX_ANCESTRY = 100_000
MAX_COUNTER_SERIES = 4096
MAX_COUNTER_POINTS = 250_000
MAX_EVIDENCE_REFS = 250_000


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value {value!r} in historical trace.")


class _JSONStream:
    def __init__(self, stream, chunk_size):
        self.stream, self.chunk_size = stream, chunk_size
        self.buffer, self.pos, self.eof = "", 0, False
        self.decoder = json.JSONDecoder(parse_constant=_reject_constant)

    def more(self):
        self.buffer = self.buffer[self.pos :]
        self.pos = 0
        chunk = self.stream.read(self.chunk_size)
        self.eof = not chunk
        self.buffer += chunk
        if len(self.buffer) > MAX_EVENT_BYTES:
            raise ValueError("A trace event exceeds the 32 MiB historical parsing limit.")

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.more()

    def take(self, char):
        if self.peek() != char:
            raise ValueError(f"Malformed trace JSON: expected {char!r}.")
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                # A number at a chunk boundary may have more digits in the
                # next chunk. Complete it before accepting the value.
                numeric_suffix = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and end < len(self.buffer)
                    and self.buffer[end] in ".eE0123456789+-"
                )
                if (end == len(self.buffer) or numeric_suffix) and not self.eof:
                    self.more()
                    continue
                self.pos = end
                return value
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError("Malformed or truncated historical trace JSON.") from exc
                self.more()

    def array(self):
        self.take("[")
        if self.peek() == "]":
            self.take("]")
            return
        while True:
            yield self.value()
            if self.peek() == "]":
                self.take("]")
                return
            self.take(",")
            if self.peek() == "]":
                raise ValueError("Malformed trace JSON: trailing array comma.")


def iter_trace_events(path, metadata=None, *, chunk_size=256 * 1024):
    """Read Chrome object/array traces without materializing the event array.

    Validate the whole document, including trailing fields, so truncation cannot
    be presented as a complete execution. Optional metadata receives otherData.
    """
    with Path(path).open(encoding="utf-8") as stream:
        source = _JSONStream(stream, chunk_size)
        if source.peek() == "[":
            yield from source.array()
        else:
            source.take("{")
            found = False
            if source.peek() != "}":
                while True:
                    key = source.value()
                    if not isinstance(key, str):
                        raise ValueError("Trace document keys must be strings.")
                    source.take(":")
                    if key == "traceEvents":
                        if found:
                            raise ValueError("Duplicate traceEvents arrays are ambiguous.")
                        found = True
                        yield from source.array()
                    else:
                        value = source.value()
                        if key == "otherData" and metadata is not None and isinstance(value, dict):
                            metadata.update(value)
                    if source.peek() == "}":
                        break
                    source.take(",")
            source.take("}")
            if not found:
                raise ValueError("Expected a Chrome traceEvents array.")
        if source.peek():
            raise ValueError("Unexpected data after historical trace document.")


def _selected(name, attrs):
    return (
        name.startswith(("llm:attempt", "tool:", "sync:", "process:", "agsync:"))
        or name
        in {
            "engine:execute",
            "request:submission_to_completion",
            "agent:create",
        }
        or (
            attrs.get("request_kind") is not None
            and attrs.get("request_id") is not None
            and name.startswith(str(attrs["request_id"]) + ":")
        )
    )


def _small_attrs(attrs):
    result = {}
    request_extra = object_value(attrs.get("llm.request_extra"))
    if isinstance(request_extra, dict) and isinstance(request_extra.get("stream"), bool):
        result["streaming"] = request_extra["stream"]
    for key, value in attrs.items():
        if key in {
            "llm.messages",
            "llm.response",
            "llm.tools",
            "llm.request_extra",
            "agency_presentation",
        }:
            continue
        if isinstance(value, str):
            value = value[:1200]
        result[key] = value
    return result


def _counter_name(event):
    name = event.get("name", "")
    if name.startswith("agency-counter-v1:"):
        info = object_value(name.removeprefix("agency-counter-v1:"))
        if isinstance(info, dict):
            name = info.get("label", name)
            identities = {
                item[0].removeprefix("process:")
                for item in info.get("path", [])
                if isinstance(item, list)
                and item
                and isinstance(item[0], str)
                and item[0].startswith("process:")
            }
            if len(identities) == 1:
                name = f"process:{next(iter(identities))}:{name}"
            else:
                for prefix in ("sandbox:", "agent:"):
                    owners = {
                        item[0]
                        for item in info.get("path", [])
                        if isinstance(item, list)
                        and item
                        and isinstance(item[0], str)
                        and item[0].startswith(prefix)
                    }
                    if len(owners) == 1:
                        name = f"{next(iter(owners))}:{name}"
                        break
    return name


def _canonical_evidence(run, trace_path):
    """Recover complete fingerprints from SQLite, without retaining transcripts.

    Span call_label + agent is the join key. Wall-clock SQLite timestamps never
    align the monotonic trace. Conflicting exchange identities remain unknown.
    """
    by_label = defaultdict(list)
    request_ids = set()
    for action in run["actions"]:
        attrs = action.get("metadata", {})
        request = attrs.get("request_id") or attrs.get("agency.run_id")
        if request:
            request_ids.add(request)
        label = action.get("metadata", {}).get("call_label")
        if action["kind"] == "model" and label:
            by_label[(action["agent"], label)].append(action)
    databases = list((trace_path.parent.parent / "logs").glob("*data.sqlite3"))
    profile = trace_path.parent / "profile_data.sqlite3"
    if profile.is_file():
        databases.append(profile)
    observed, gaps, dependencies = {}, [], []
    retained_refs = omitted_refs = 0
    for database in databases:
        try:
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=1) as con:
                tables = {
                    row[0]
                    for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                if {"exchanges", "exchange_chain", "blocks"} <= tables and by_label:
                    for label, agent in con.execute("SELECT call_label,name FROM exchanges"):
                        key = (agent, label)
                        if key not in by_label:
                            continue
                        evidence = {"prompt": [], "response": [], "complete": True}
                        metrics = {}
                        for kind, raw, block_hash in con.execute(
                            "SELECT c.kind,CASE WHEN length(CAST(b.payload AS BLOB))<=? THEN b.payload END,b.hash "
                            "FROM exchange_chain c LEFT JOIN blocks b ON b.hash=c.hash "
                            "WHERE c.call_label=? ORDER BY c.kind,c.seq",
                            (MAX_EVENT_BYTES, label),
                        ):
                            if raw is None:
                                evidence["complete"] = False
                                gaps.append(
                                    "Canonical exchange references a missing block"
                                    if block_hash is None
                                    else "Canonical block exceeds the historical parsing memory limit"
                                )
                                continue
                            block = json.loads(raw, parse_constant=_reject_constant)
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") == "metadata" and kind == "response":
                                usage = block.get("usage") or {}
                                if isinstance(usage, dict):
                                    metrics.update(
                                        tokens=usage.get("prompt_tokens"),
                                        output_tokens=usage.get("completion_tokens"),
                                    )
                                metrics.update(
                                    {
                                        k: block[k]
                                        for k in ("ttft_ms", "stop_reason", "new_prompt_tokens")
                                        if k in block
                                    }
                                )
                            elif kind in evidence:
                                refs = exchange_evidence([block], [])["prompt"]
                                local_count = len(evidence["prompt"]) + len(evidence["response"])
                                if retained_refs + local_count + len(refs) <= MAX_EVIDENCE_REFS:
                                    evidence[kind].extend(refs)
                                else:
                                    evidence["complete"] = False
                                    omitted_refs += len(refs)
                        retained_refs += len(evidence["prompt"]) + len(evidence["response"])
                        value = (evidence, metrics)
                        if (
                            key in observed
                            and observed[key] is not None
                            and observed[key][0] != evidence
                        ):
                            observed[key] = None
                        elif key not in observed:
                            observed[key] = value
                if "events" in tables:
                    for (raw,) in con.execute(
                        "SELECT payload FROM events WHERE type='request_dependencies'"
                    ):
                        value = json.loads(raw, parse_constant=_reject_constant)
                        if isinstance(value, dict) and value.get("request_id") in request_ids:
                            dependencies.append(value)
        except (sqlite3.Error, OSError, ValueError) as exc:
            gaps.append(f"{database.name}: canonical evidence unavailable ({type(exc).__name__})")
    count = 0
    for key, value in observed.items():
        if value is None or len(by_label[key]) != 1:
            gaps.append("Ambiguous canonical exchange identity")
            continue
        evidence, metrics = value
        action = by_label[key][0]
        action["causal_evidence"] = evidence
        action["tool_call_ids"] = [
            r["tool_id"]
            for r in evidence["response"]
            if r["type"] in {"tool_use", "function_call"} and r.get("tool_id")
        ]
        for field in ("tokens", "output_tokens"):
            if finite(metrics.get(field)):
                action[field] = metrics[field]
        action["metadata"].update(
            {k: v for k, v in metrics.items() if k not in {"tokens", "output_tokens"}}
        )
        count += 1
    if omitted_refs:
        gaps.append(
            f"{omitted_refs} canonical block references omitted by the historical evidence memory limit"
        )
    run["request_dependencies"] = dependencies
    run["coverage"]["causal_evidence"] = {"canonical_exchanges": count, "gaps": gaps}


def load_compact_run(path, entry):
    path = Path(path)
    summary, warnings = {}, []
    summary_path = path.parent / "summary.json"
    if summary_path.is_file():
        try:
            if summary_path.stat().st_size > MAX_EVENT_BYTES:
                raise ValueError("Optional summary exceeds the historical parsing memory limit")
            constants = []
            summary = json.loads(
                summary_path.read_text(), parse_constant=lambda value: constants.append(value)
            )
            if constants:
                warnings.append("Non-finite summary metrics are unavailable.")
            if not isinstance(summary, dict):
                summary = {}
                warnings.append("Saved summary is not a JSON object.")
        except (OSError, ValueError):
            warnings.append("Saved summary is unavailable; resource aggregates may be absent.")
    selected, evidence, counters, counter_counts = [], {}, defaultdict(list), defaultdict(int)
    ancestry, omitted_ancestry = {}, 0
    metadata, event_count, omitted, untimed = {}, 0, 0, 0
    retained_evidence = omitted_evidence = 0
    counter_samples = retained_points = omitted_counter_samples = 0
    origin, end = math.inf, -math.inf
    for index, event in enumerate(iter_trace_events(path, metadata)):
        event_count += 1
        if not isinstance(event, dict):
            continue
        timestamp = event.get("ts")
        if not finite(timestamp):
            if event.get("ph") == "X" and _selected(
                str(event.get("name", "")),
                event.get("args") if isinstance(event.get("args"), dict) else {},
            ):
                untimed += 1
            continue
        phase, name = event.get("ph"), str(event.get("name", ""))
        if phase == "X":
            origin = min(origin, timestamp)
            duration = event.get("dur")
            end = max(end, timestamp + (max(0, duration) if finite(duration) else 0))
            attrs = event.get("args") if isinstance(event.get("args"), dict) else {}
            span_id = attrs.get("span_id") or attrs.get("agency.span_id")
            if (
                span_id is not None
                and not name.startswith("syscall:")
                and event.get("cat") != "python.auto"
            ):
                span_id = str(span_id)
                parent = attrs.get("parent_span_id") or attrs.get("agency.parent_span_id")
                identity = (
                    str(parent) if parent is not None else None,
                    attrs.get("request_id") or attrs.get("agency.run_id"),
                )
                if span_id in ancestry:
                    if ancestry[span_id] != identity:
                        ancestry[span_id] = None
                elif len(ancestry) < MAX_ANCESTRY:
                    ancestry[span_id] = identity
                else:
                    omitted_ancestry += 1
            if not _selected(name, attrs):
                continue
            if len(selected) >= MAX_OPERATIONS:
                omitted += 1
                continue
            attrs = dict(attrs)
            identifier = (
                attrs.get("span_id")
                or attrs.get("agency.span_id")
                or attrs.get("trace_span_id")
                or f"trace-event:{index}"
            )
            identifier = str(identifier)
            attrs["span_id"] = identifier
            if name.startswith("llm:"):
                response = visible_model_response(attrs.get("llm.response"))
                refs = messages_evidence(attrs.get("llm.messages"), response)
                if attrs.get("llm.messages_truncated"):
                    refs["complete"] = False
                count = len(refs["prompt"]) + len(refs["response"])
                if retained_evidence + count > MAX_EVIDENCE_REFS:
                    omitted_evidence += count
                    refs = {"prompt": [], "response": [], "complete": False}
                else:
                    retained_evidence += count
                evidence[identifier] = refs
            selected.append(
                {
                    **{
                        k: event[k] for k in ("name", "ph", "ts", "dur", "pid", "tid") if k in event
                    },
                    "args": _small_attrs(attrs),
                }
            )
        elif phase == "C":
            key = _counter_name(event)
            for field, value in (event.get("args") or {}).items():
                if not finite(value):
                    continue
                counter_samples += 1
                counter = key if field == "value" else f"{key} {field}"
                if counter not in counters and len(counters) >= MAX_COUNTER_SERIES:
                    omitted_counter_samples += 1
                    continue
                if retained_points >= MAX_COUNTER_POINTS:
                    before = len(counters[counter])
                    if before > 4:
                        counters[counter] = reduce_samples(counters[counter], max(4, before // 2))
                        retained_points -= before - len(counters[counter])
                    else:
                        omitted_counter_samples += 1
                        continue
                counters[counter].append([timestamp / 1e6, value])
                retained_points += 1
                counter_counts[counter] += 1
                if len(counters[counter]) > COUNTER_POINTS * 2:
                    before = len(counters[counter])
                    counters[counter] = reduce_samples(counters[counter], COUNTER_POINTS)
                    retained_points -= before - len(counters[counter])
    origin = origin if math.isfinite(origin) else 0
    if omitted_counter_samples:
        warnings.append(
            f"{omitted_counter_samples} resource counter values omitted by historical memory limits; sample windows are incomplete."
        )
    if omitted_ancestry:
        warnings.append(
            f"{omitted_ancestry} span ancestry records omitted; request ownership may be incomplete."
        )
    if untimed:
        warnings.append(
            f"{untimed} selected operation records have no finite start timestamp and could not be placed on the historical clock."
        )
    if omitted_evidence:
        warnings.append(
            f"{omitted_evidence} span block references omitted by the historical evidence memory limit; causal chains are incomplete."
        )
    clock_shared = metadata.get("agprof_clock") == "perf_counter_ns"
    if not clock_shared:
        warnings.append(
            "Trace has no explicit agprof shared-clock declaration; cross-agent blocking and resource-window alignment remain unverified."
        )
    selected_origin = min((span["ts"] for span in selected), default=origin)
    shift = (selected_origin - origin) / 1e6
    normalized = normalize_trace({"traceEvents": selected}, entry, summary)
    # The existing normalizer selects workflow-level spans for trajectory. Add
    # orchestration/process envelopes explicitly, carrying their recorded IDs.
    known = {action["id"] for action in normalized["actions"]}
    for action in normalized["actions"]:
        action["start"] += shift
        if action["id"] in evidence:
            action["causal_evidence"] = evidence[action["id"]]
            action["tool_call_ids"] = [
                r["tool_id"]
                for r in evidence[action["id"]]["response"]
                if r["type"] in {"tool_use", "function_call"} and r.get("tool_id")
            ]
        action["result"] = (
            str(action.get("metadata", {}).pop("tool.result", ""))[:1200]
            or action.get("result", "")[:1200]
        )
    for span in selected:
        attrs = span["args"]
        if attrs["span_id"] in known:
            continue
        name = span["name"]
        elapsed = span.get("dur")
        kind = (
            "process"
            if name.startswith("process:")
            else "dependency"
            if "wait" in name or "join" in name or "dependency" in name
            else "setup"
        )
        normalized["actions"].append(
            {
                "id": attrs["span_id"],
                "agent": attrs.get("agency.agent_id", "workflow"),
                "kind": kind,
                "name": name,
                "start": (span["ts"] - origin) / 1e6,
                "duration": max(0, elapsed) / 1e6 if finite(elapsed) else None,
                "missing_end": not finite(elapsed) or elapsed < 0,
                "outcome": attrs.get("outcome", "unknown"),
                "metadata": attrs,
                "source": "recorded",
                "timing": attrs.get("timing", "host_span"),
            }
        )
    normalized["actions"].sort(key=lambda action: (action["start"], str(action["id"])))
    # Ownership travels through recorded span identities across bridge/harness
    # boundaries. Time overlap and agent labels never establish request scope.
    for action in normalized["actions"]:
        attrs = action["metadata"]
        if attrs.get("request_id") or attrs.get("agency.run_id"):
            continue
        key, visited = str(action["id"]), set()
        while key and key not in visited:
            visited.add(key)
            identity = ancestry.get(key)
            if identity is None:
                break
            key, request = identity
            if request:
                attrs.update(request_id=request, request_attribution="recorded_parent_span_chain")
                break
    normalized["duration"] = max(0, (end - origin) / 1e6) if math.isfinite(end) else 0
    if finite(summary.get("duration_ms")):
        normalized["duration"] = max(normalized["duration"], summary["duration_ms"] / 1000)
    normalized["counters"] = {
        key: [[t - origin / 1e6, value] for t, value in reduce_samples(points, COUNTER_POINTS)]
        for key, points in counters.items()
    }
    normalized["clock"].update(
        shared=clock_shared,
        origin="First recorded trace span",
        trace_origin_us=origin,
        perf_origin_ns=metadata.get("agprof_time_origin_ns"),
    )
    normalized["coverage"].update(
        events=event_count,
        selected_spans=len(selected),
        omitted_operations=omitted,
        counter_samples=counter_samples,
        omitted_counter_samples=omitted_counter_samples,
        retained_counter_points=sum(map(len, normalized["counters"].values())),
        resource_sampling="Recorded counters reduced to bounded extrema; process summary aggregates retain full-session measurements.",
        historical_warnings=warnings,
        omitted_ancestry=omitted_ancestry,
    )
    normalized.pop("intervals", None)
    normalized.pop("episodes", None)
    normalized.pop("obligations", None)
    normalized["status"] = "recorded"
    _canonical_evidence(normalized, path)
    return normalized


def fingerprint(path):
    """Invalidate analysis when the trace, summary or canonical WAL changes."""
    path = Path(path)
    files = [path, path.parent / "summary.json", path.parent / "profile_data.sqlite3"]
    files.extend(sorted((path.parent.parent / "logs").glob("*data.sqlite3")))
    result = []
    for candidate in files:
        for item in (candidate, Path(str(candidate) + "-wal")):
            try:
                stat = item.stat()
                result.append((str(item), stat.st_mtime_ns, stat.st_size))
            except FileNotFoundError:
                continue
    return tuple(result)


def load_compact_native(path, entry):
    """Stream observer JSONL histories with the same semantic span bound."""
    annotations, spans, model_ids = {}, [], {}
    rejected = count = omitted = untimed = 0
    origin, end = math.inf, -math.inf
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            count += 1
            if len(line.encode("utf-8")) > MAX_EVENT_BYTES:
                raise ValueError("A native observer event exceeds the 32 MiB parsing limit.")
            try:
                event = json.loads(line, parse_constant=_reject_constant)
            except ValueError:
                rejected += 1
                continue
            if not isinstance(event, dict):
                rejected += 1
                continue
            kind = event.get("kind")
            if kind == "tool_annotation":
                identifier = event.get("event_id")
                annotation = event.get("annotation") or {}
                annotation = annotation if isinstance(annotation, dict) else {}
                raw = annotation.get("raw") or {}
                if identifier and len(annotations) < MAX_OPERATIONS:
                    annotations[identifier] = {
                        "tool.arguments": json.dumps(event.get("arguments", {}))[:1200],
                        "model_tool_call_id": event.get("model_tool_call_id"),
                        "annotation_status": annotation.get("status"),
                        "tool.intent": raw.get("purpose", "")
                        if isinstance(raw, dict) and annotation.get("status") == "valid"
                        else "",
                    }
                continue
            if kind not in {"tool_result", "model_exchange"}:
                continue
            at, duration = event.get("timestamp_ns"), event.get("duration_ns")
            if not finite(at) or not finite(duration) or duration < 0:
                untimed += 1
                continue
            origin, end = min(origin, at - duration), max(end, at)
            if len(spans) >= MAX_OPERATIONS:
                omitted += 1
                continue
            identifier = str(event.get("event_id") or f"native-event:{index}")
            attrs = {
                "agency.agent_id": event.get("agent_id", "agent"),
                "harness": "native",
                "model": entry.get("model"),
                "span_id": identifier,
                "timing": "observer_duration",
                "provenance": "native_observer",
                "outcome": "failed" if event.get("category") == "error" else "success",
            }
            if kind == "tool_result":
                attrs.update(annotations.get(event.get("event_id"), {}))
                attrs.update(
                    {
                        "tool.result": str(event.get("result", ""))[:1200],
                        "call_id": event.get("call_id"),
                    }
                )
                if event.get("model_tool_call_id"):
                    attrs["model_tool_call_id"] = event["model_tool_call_id"]
                name = "tool:" + event.get("tool_name", "unknown")
            else:
                response = event.get("response") or {}
                response = response if isinstance(response, dict) else {}
                usage = response.get("usage") or {}
                if isinstance(usage, dict):
                    attrs.update(
                        input_tokens=usage.get("prompt_tokens"),
                        output_tokens=usage.get("completion_tokens"),
                    )
                model_ids[identifier] = response_tool_ids(
                    visible_model_response(response.get("message"))
                )
                name = "llm:attempt[0]"
            spans.append(
                {
                    "ph": "X",
                    "name": name,
                    "ts": (at - duration) / 1000,
                    "dur": duration / 1000,
                    "args": _small_attrs(attrs),
                }
            )
    normalized = normalize_trace({"traceEvents": spans}, entry)
    selected_origin = min((span["ts"] for span in spans), default=0) * 1000
    if math.isfinite(origin):
        for action in normalized["actions"]:
            action["start"] += (selected_origin - origin) / 1e9
        normalized["duration"] = (end - origin) / 1e9
    for action in normalized["actions"]:
        if action["kind"] == "model":
            action["tool_call_ids"] = model_ids.get(action["id"], [])
    normalized["coverage"].update(
        events=count,
        partial_lines=rejected,
        omitted_operations=omitted,
        historical_warnings=[
            "Native observer records responses and tool identities without complete prompt or system-resource evidence."
        ]
        + ([f"{untimed} operation records lack valid timing."] if untimed else []),
    )
    normalized["status"] = "recorded"
    for key in ("intervals", "episodes", "obligations"):
        normalized.pop(key, None)
    return normalized


@lru_cache(maxsize=2)
def _cached_profile(path, revision, entry_json):
    """Cache only compact analysis, never a second full trace or trajectory."""
    from .critical_path import build_profile

    entry = json.loads(entry_json)
    if entry.get("format") == "native_events":
        run = load_compact_native(path, entry)
    else:
        run = load_compact_run(path, entry)
    if Path(path).name == "agprof.trace.json":
        run["profiler_links"] = {
            "trace": f"/api/investigator/runs/{entry['id']}/trace",
            "perfetto": "https://ui.perfetto.dev/",
        }
    return build_profile(run)


_profile_lock = threading.Lock()


def load_profile(path, revision, entry_json):
    # Prevent simultaneous requests from parsing/retaining duplicate analyses.
    with _profile_lock:
        return _cached_profile(path, revision, entry_json)
