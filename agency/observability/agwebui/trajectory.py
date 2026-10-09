"""Incremental, evidence-preserving trajectory projection of canonical logger events.

This module never drives an agent. The same reducer consumes live SQLite rows
and replay prefixes; raw rows stay in their original databases or trace files.
"""

from __future__ import annotations

import copy
import bisect
import math
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from .investigator import classify, object_value

EVENT_TYPES = (
    "resource_sample",
    "workload_started",
    "done",
    "agent_registered",
    "agent_created",
    "agent_forked",
    "agent_state",
    "agent_paused",
    "agent_resumed",
    "agent_stop_requested",
    "tool_call",
    "tool_result",
    "skill_start",
    "skill_call",
    "skill_error",
    "skill_cancelled",
    "request_submitted",
    "request_started",
    "request_blocked",
    "request_ready",
    "request_completed",
    "request_failed",
    "request_cancelled",
    "input_required",
    "input_resolved",
    "delegation",
    "handoff",
    "tool_annotation",
)
TITLES = {
    "investigate": "Inspecting the behavior",
    "edit": "Attempting a change",
    "test": "Checking the behavior",
    "review": "Reviewing the result",
    "tool": "Running tools",
    "model": "Waiting for a model response",
    "queue": "Awaiting scheduler admission",
    "dependency": "Waiting for dependencies",
    "setup": "Executing the requested skill",
    "input": "Waiting for your input",
}
TERMINAL = {"completed", "failed", "cancelled"}


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def fingerprint(value):
    return hashlib.sha256(text(value).encode()).hexdigest()[:20]


def tool_outcome(payload):
    result = object_value(payload.get("result", payload.get("output")))
    if payload.get("error"):
        return "failed"
    if isinstance(result, dict):
        if result.get("error") or result.get("isError") is True:
            return "failed"
        if any(result.get(key) not in (None, 0) for key in ("returncode", "exit_code")):
            return "failed"
    return payload.get("outcome", "success")


def returned_preview(value):
    parsed = object_value(value)
    if isinstance(parsed, dict):
        output = text(parsed.get("stdout") or parsed.get("stderr") or "")
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        failures = [line for line in lines if re.search(r"AssertionError|^FAILED|Error:", line)]
        exit_code = parsed.get("exit_code", parsed.get("returncode"))
        if exit_code is not None:
            detail = " · ".join(failures[:2] or lines[-2:])
            return f"exit_code={exit_code} · {detail}"[:240]
        if isinstance(parsed.get("artifacts"), list) and parsed["artifacts"]:
            return (
                "Tool reported artifacts: "
                + ", ".join(
                    str(item.get("path", "unattributed"))
                    for item in parsed["artifacts"]
                    if isinstance(item, dict)
                )[:200]
            )
    return text(value)[:240]


class Trajectory:
    def __init__(self, run_id, title, *, mode="live", source="recorded"):
        self.run = {
            "id": run_id,
            "title": title,
            "subtitle": "Canonical Agency event stream",
            "source": source,
            "mode": mode,
            "model": "Recorded in event metadata",
            "harness": "Agency",
            "duration": 0,
            "status": "unknown",
            "resolved": None,
            "actions": [],
            "episodes": [],
            "agents": [],
            "edges": [],
            "intervals": [],
            "counters": {},
            "obligations": [],
            "signals": [],
            "messages": [],
            "coverage": {
                "events": 0,
                "spans": 0,
                "gaps": [],
                "catching_up": False,
                "context": "Only recorded visible messages and linked model prompts are exposed.",
                "relationships": "Only explicitly recorded delegation and dependency links.",
            },
        }
        # Saved traces already use run-relative time; preserve alignment with
        # their resource lanes even if the first observable call starts later.
        self.origin = None if mode == "live" else 0
        self.seen = set()
        self.actions = {}
        self.episodes = {}
        self.agents = {}
        self.current = {}
        self.actor_request = {}
        self.actor_clock = {}
        self.checks = {}
        self.edit_version = {}
        self.failed_checks = {}
        self.model_links = {}
        self.pending_model_links = {}
        self.dirty = {key: set() for key in ("actions", "episodes", "agents", "edges")}
        self.signals = {}
        self.dirty["signals"] = set()
        self.finished_at = None
        self.running = set()
        self.intervals = {}
        self.dirty["intervals"] = set()
        self.pauses = {}
        self.counter_updates = {}

    def _agent(self, actor):
        if actor not in self.agents:
            agent = {
                "id": actor,
                "label": actor,
                "role": "Fixture actor" if self.run["source"] == "synthetic" else "Recorded actor",
                "parent": None,
                "model": None,
                "harness": None,
                "status": "unknown",
            }
            self.agents[actor] = agent
            self.run["agents"].append(agent)
        self.dirty["agents"].add(actor)
        return self.agents[actor]

    def _signal(self, key, title, explanation, evidence, severity="inspect", active=True):
        self.signals[key] = {
            "id": key,
            "title": title,
            "explanation": explanation,
            "evidence": evidence,
            "severity": severity,
            "active": active,
        }
        self.dirty["signals"].add(key)

    def _agent_status(self, agent, status):
        agent["execution_state"] = status
        agent["status"] = "paused" if agent.get("paused") else status

    def _execution_duration(self, action, end):
        """Exclude recorded control intervals without changing the trace's time axis."""
        start = action["started_ts"]
        paused = sum(
            max(0, min(end, right if right is not None else end) - max(start, left))
            for left, right in self.pauses.get(action["agent"], [])
        )
        return max(0, end - start - paused)

    def _close_signal(self, key):
        if key in self.signals and self.signals[key]["active"]:
            self.signals[key]["active"] = False
            self.dirty["signals"].add(key)

    def _episode(self, action, late=False):
        actor = action["agent"]
        kind = "investigate" if action["kind"] in {"read", "search"} else action["kind"]
        workstreams = action.get("workstream_ids") or []
        group = (action["request_id"], tuple(workstreams) if workstreams else kind)
        previous = self.current.get(actor)
        episode = self.episodes.get(previous)
        if (
            late
            or not episode
            or episode["group"] != group
            or episode["closed"]
            or len(episode["actions"]) >= 8
        ):
            if episode and not late:
                episode["closed"] = True
                self.dirty["episodes"].add(episode["id"])
            episode_id = f"episode:{action['id']}"
            episode = {
                "id": episode_id,
                "agent": actor,
                "kind": kind,
                "title": action["intent"] or TITLES.get(kind, "Unclassified activity"),
                "label_source": "declared" if action["intent"] else "deterministic",
                "intent": action["intent"],
                "actions": [],
                "start": action["start"],
                "end": action["start"],
                "status": "running",
                "inferred": True,
                "latest_result": "No result recorded yet.",
                "group": group,
                "closed": False,
                "late": late,
                "annotation": None,
            }
            self.episodes[episode_id] = episode
            self.run["episodes"].append(episode)
            if not late:
                self.current[actor] = episode_id
        episode["actions"].append(action["id"])
        action["episode"] = episode["id"]
        self.dirty["episodes"].add(episode["id"])

    def _refresh_episode(self, action):
        episode = self.episodes[action["episode"]]
        calls = [self.actions[key] for key in episode["actions"]]
        episode["start"] = min(a["start"] for a in calls)
        episode["end"] = max(a["start"] + a["duration"] for a in calls)
        statuses = {a["outcome"] for a in calls}
        episode["status"] = (
            "running"
            if "running" in statuses
            else "failed"
            if "failed" in statuses
            else "incomplete"
            if "incomplete" in statuses
            else "cancelled"
            if "cancelled" in statuses
            else "completed"
            if statuses == {"success"}
            else "unknown"
        )
        if action.get("result") or action.get("error"):
            episode["latest_result"] = (
                action.get("result_preview") or (action.get("error") or action["result"])[:240]
            )
            episode["latest_action"] = action["id"]
        self.dirty["episodes"].add(episode["id"])

    def _call(self, event, kind, key, *, ending=False):
        actor, payload, timestamp = event["actor"], event["payload"], event["ts"]
        action_id = payload.get("action_id") or f"{actor}:{key}"
        action = self.actions.get(action_id)
        if action is None:
            arguments = payload.get("arguments") or {}
            command = (
                arguments.get("command", text(arguments))
                if isinstance(arguments, dict)
                else text(arguments)
            )
            annotation = payload.get("annotation") or {}
            declared = annotation.get("raw", {}) if annotation.get("status") == "valid" else {}
            declared = declared if isinstance(declared, dict) else {}
            intent = payload.get("intent") or declared.get("purpose") or ""
            name = payload.get("tool") or payload.get("name") or kind
            category = payload.get("kind") or (
                classify(f"tool:{name}", command) if kind == "tool" else kind
            )
            late = timestamp < self.actor_clock.get(actor, timestamp)
            self.actor_clock[actor] = max(timestamp, self.actor_clock.get(actor, timestamp))
            action = {
                "id": action_id,
                "agent": actor,
                "name": name,
                "kind": category,
                "intent": intent,
                "command": command[:8000],
                "arguments": text(arguments)[:8000],
                "input_truncated": payload.get("input_truncated", False)
                or len(text(arguments)) > 8000,
                "result": "",
                "error": "",
                "start": max(0, timestamp - self.origin),
                "started_ts": timestamp,
                "duration": 0,
                "outcome": "unknown" if ending else "running",
                "timing": "host_event_boundary",
                "source": event.get("source", "canonical_event"),
                "tokens": None,
                "output_tokens": None,
                "files": [],
                "artifacts": [],
                "metadata": {},
                "context": None,
                "event_ids": [],
                "call_id": payload.get("call_id") or event.get("call_label"),
                "model_tool_call_id": annotation.get("model_tool_call_id"),
                "model_id": self.model_links.get((actor, annotation.get("model_tool_call_id"))),
                "parent_id": payload.get("parent_id"),
                "request_id": payload.get("request_id") or self.actor_request.get(actor),
                "workstream_ids": declared.get("workstream_ids", []),
                "missing_start": ending,
                "result_hash": None,
                "result_preview": "",
                "edit_version": self.edit_version.get(actor, 0),
            }
            self.actions[action_id] = action
            self.run["actions"].append(action)
            if action["model_tool_call_id"] and not action["model_id"]:
                self.pending_model_links.setdefault(
                    (actor, action["model_tool_call_id"]), []
                ).append(action_id)
            self._episode(action, late)
        action["event_ids"].append(event["id"])
        if not ending:
            missing_start = action["missing_start"]
            action["missing_start"] = False
            action["start"] = max(0, timestamp - self.origin)
            action["started_ts"] = timestamp
            if missing_start:
                arguments = payload.get("arguments") or {}
                action["arguments"] = text(arguments)[:8000]
                action["command"] = (
                    arguments.get("command", text(arguments))
                    if isinstance(arguments, dict)
                    else text(arguments)
                )[:8000]
                action["name"] = payload.get("tool", action["name"])
                action["duration"] = max(0, action.get("end_ts", timestamp) - timestamp)
            if not action["result_hash"]:
                action["outcome"] = "failed" if payload.get("allowed") is False else "running"
                if payload.get("allowed") is False:
                    action["error"] = payload.get("reason") or "Policy denied tool admission."
                    action["result_preview"] = action["error"]
                    self._attention(action)
        else:
            result = payload.get("result", payload.get("output", ""))
            action["result"] = text(result)[:8000]
            action["output_truncated"] = (
                payload.get("output_truncated", False)
                or len(text(result)) > 8000
                or len(text(payload.get("error", ""))) > 8000
            )
            action["error"] = text(payload["error"])[:8000] if payload.get("error") else ""
            action["result_hash"] = fingerprint(action["error"] or result)
            action["result_preview"] = (
                action["error"][:240]
                or payload.get("recorded_result_preview")
                or returned_preview(result)
            )
            first_result = action["end_ts"] if "end_ts" in action else None
            action["end_ts"] = timestamp
            action["duration"] = max(0, timestamp - self.origin - action["start"])
            action["outcome"] = tool_outcome(payload)
            action["tokens"] = payload.get("tokens") if kind == "model" else None
            action["output_tokens"] = payload.get("output_tokens") if kind == "model" else None
            action["context"] = payload.get("context")
            action["files"] = payload.get("files") or []
            action["timing"] = payload.get("timing", action["timing"])
            parsed_result = object_value(result)
            artifacts = payload.get("artifacts") or (
                parsed_result.get("artifacts", []) if isinstance(parsed_result, dict) else []
            )
            action["artifacts"] = (
                [item for item in artifacts if isinstance(item, dict)]
                if isinstance(artifacts, list)
                else []
            )
            if first_result is None:
                self._attention(action)
        if action["outcome"] == "running" and self.finished_at is not None:
            action["outcome"] = "incomplete"
        action["execution_duration"] = self._execution_duration(
            action, action.get("end_ts", timestamp)
        )
        if action["outcome"] == "running":
            self.running.add(action_id)
        else:
            self.running.discard(action_id)
        action["metadata"] = {
            **payload.get("metadata", {}),
            "event_ids": action["event_ids"],
            "call_id": action["call_id"],
            "request_id": action["request_id"],
            "missing_start": action["missing_start"],
            "parent_id": action["parent_id"],
            "source": action["source"],
        }
        self.dirty["actions"].add(action_id)
        self._interval(action)
        self._refresh_episode(action)
        return action

    def _interval(self, action):
        interval = {
            "id": action["id"],
            "agent": action["agent"],
            "action": action["id"],
            "kind": action["kind"]
            if action["kind"] in {"model", "queue", "dependency"}
            else "tool",
            "start": action["start"],
            "duration": action["duration"],
            "label": action["intent"] or action["name"],
            "source": action["source"],
        }
        if action["id"] not in self.intervals:
            self.run["intervals"].append(interval)
        else:
            self.intervals[action["id"]].update(interval)
            interval = self.intervals[action["id"]]
        self.intervals[action["id"]] = interval
        self.dirty["intervals"].add(action["id"])

    def _attention(self, action):
        actor = action["agent"]
        if action["kind"] == "edit" and action["outcome"] == "success":
            self.edit_version[actor] = self.edit_version.get(actor, 0) + 1
            # A recorded edit is sufficient to break an equivalent-check sequence.
            self.checks[actor] = []
            if f"repeat:{actor}" in self.signals:
                self._close_signal(f"repeat:{actor}")
        if action["outcome"] == "failed":
            self._signal(
                f"failure:{action['id']}",
                "Recorded call failure",
                action["result_preview"] or "The call reported a failure.",
                [action["id"]],
            )
        if action["kind"] not in {"test", "review"}:
            return
        failure_key = (actor, action["command"])
        if action["outcome"] == "failed":
            self.failed_checks.setdefault(failure_key, []).append(action["id"])
        elif action["outcome"] == "success":
            for failed_id in self.failed_checks.pop(failure_key, []):
                self._close_signal(f"failure:{failed_id}")
                self.signals[f"failure:{failed_id}"]["followed_by"] = action["id"]
        checks = self.checks.setdefault(actor, [])
        if action["missing_start"]:
            return
        if checks and self.actions[checks[-1][1]].get("end_ts", 0) > action["started_ts"]:
            checks.clear()
            if f"repeat:{actor}" in self.signals:
                self._close_signal(f"repeat:{actor}")
        signature = (
            action["command"],
            fingerprint(action["result_preview"])
            if action["outcome"] == "failed"
            else action["result_hash"],
            action["outcome"],
            action["edit_version"],
        )
        if checks and checks[-1][0] != signature:
            checks.clear()
            if f"repeat:{actor}" in self.signals:
                self._close_signal(f"repeat:{actor}")
        checks.append((signature, action["id"]))
        if len(checks) < 3:
            return
        failed = action["outcome"] == "failed"
        self._signal(
            f"repeat:{actor}",
            "Repeated matching failures" if failed else "Equivalent checks repeated",
            f"{len(checks)} consecutive checks returned the same recorded result. "
            "No intervening edit was recorded for this actor; unobserved changes remain possible.",
            [key for _, key in checks[-5:]],
        )

    def apply(self, event):
        if event["id"] in self.seen:
            return False
        self.seen.add(event["id"])
        self.run["coverage"]["events"] += 1
        if not isinstance(event.get("ts"), (int, float)) or not math.isfinite(event["ts"]):
            event = {
                **event,
                "ts": (self.origin or 0) + self.run["duration"],
                "source": "timestamp_missing_ordered_at_last_observation",
            }
        if self.origin is None:
            self.origin = event["ts"]
            self.run["time_origin"] = self.origin
        self.run["duration"] = max(self.run["duration"], event["ts"] - self.origin)
        actor = event.get("actor") or "unattributed"
        event = {**event, "actor": actor}
        payload, kind = event["payload"], event["type"]
        if kind == "resource_sample":
            at = max(0, event["ts"] - self.origin)
            for name, value in payload.get("values", {}).items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    point = [at, value]
                    series = self.run["counters"].setdefault(name, [])
                    bisect.insort(series, point)
                    self.counter_updates.setdefault(name, []).append(point)
        elif kind == "workload_started":
            self.run["status"] = "running"
            self.run["task"] = payload.get("task")
            for field in ("model", "harness", "dataset"):
                if payload.get(field):
                    self.run[field] = payload[field]
        elif kind == "done":
            self.run["status"] = payload.get("status", "unknown")
            if self.run["status"] == "unknown":
                self._signal(
                    "final-state",
                    "Final execution state is unavailable",
                    "The recording ended without an explicit final execution status. Task success is not established.",
                    [event["id"]],
                )
            self.finished_at = event["ts"]
            if self.running:
                self._signal(
                    "unfinished",
                    "Some calls have no recorded end",
                    "Execution reached its final boundary with unfinished telemetry. These calls are retained as incomplete, not successful.",
                    list(self.running),
                )
            for action_id in list(self.running):
                action = self.actions[action_id]
                if action["outcome"] == "running":
                    action["outcome"] = "incomplete"
                    self.dirty["actions"].add(action["id"])
                    self._refresh_episode(action)
            self.running.clear()
            for episode in self.episodes.values():
                episode["closed"] = True
                self.dirty["episodes"].add(episode["id"])
            for agent in self.agents.values():
                if agent.get("paused"):
                    agent["paused"] = False
                    self._agent_status(agent, self.run["status"])
                    self.dirty["agents"].add(agent["id"])
        else:
            agent = self._agent(actor)
            if kind in {"agent_created", "agent_forked", "agent_registered"}:
                # Process ancestry is deliberately not turned into delegation.
                agent["identity"] = "execution_actor"
            elif kind in {"agent_paused", "agent_resumed"}:
                if event["ts"] >= agent.get("control_ts", 0) and self.finished_at is None:
                    agent["control_ts"] = event["ts"]
                    intervals = self.pauses.setdefault(actor, [])
                    if kind == "agent_paused" and not agent.get("paused"):
                        intervals.append([event["ts"], None])
                        agent["paused"] = True
                    elif kind == "agent_resumed" and agent.get("paused"):
                        intervals[-1][1] = event["ts"]
                        agent["paused"] = False
                    self._agent_status(agent, agent.get("execution_state", "unknown"))
                    self.tick(event["ts"], force=True)
            elif kind == "agent_stop_requested":
                if event["ts"] >= agent.get("stop_ts", 0):
                    agent["stop_ts"] = event["ts"]
                    agent["stop_force"] = bool(payload.get("force"))
                    # This confirms delivery, not process exit. Terminal state
                    # still comes from the scheduler's existing request events.
                    agent["paused"] = False
                    intervals = self.pauses.get(actor, [])
                    if intervals and intervals[-1][1] is None:
                        intervals[-1][1] = event["ts"]
                    self._agent_status(agent, agent.get("execution_state", "unknown"))
            elif kind == "agent_state":
                if event["ts"] >= agent.get("state_ts", 0):
                    self._agent_status(agent, payload.get("state", "unknown"))
                    agent["state_ts"] = event["ts"]
                if payload.get("state") == "waiting_llm" and event.get("call_label"):
                    self._call(event, "model", f"model:{event['call_label']}")
            elif kind in {"tool_call", "tool_result", "model_result"}:
                call_kind = "model" if kind == "model_result" else "tool"
                key = payload.get("call_id") or event.get("call_label") or event["id"]
                if call_kind == "model":
                    key = f"model:{key}"
                self._call(event, call_kind, key, ending=kind.endswith("result"))
                if call_kind == "model":
                    model_id = payload.get("action_id") or f"{actor}:{key}"
                    for tool_id in payload.get("tool_call_ids", []):
                        self.model_links[(actor, tool_id)] = model_id
                        for action_id in self.pending_model_links.pop((actor, tool_id), []):
                            self.actions[action_id]["model_id"] = model_id
                            self.dirty["actions"].add(action_id)
            elif kind.startswith("request_"):
                request = payload.get("request_id") or event["id"]
                self.actor_request[actor] = request
                if kind == "request_submitted":
                    self._call(event, "queue", f"queue:{request}")
                elif kind in {"request_started", "request_cancelled", "request_failed"}:
                    queue = self.actions.get(f"{actor}:queue:{request}")
                    if queue and queue["outcome"] == "running":
                        self._call(
                            {
                                **event,
                                "payload": {
                                    **payload,
                                    "outcome": {
                                        "request_started": "success",
                                        "request_cancelled": "cancelled",
                                        "request_failed": "failed",
                                    }[kind],
                                },
                            },
                            "queue",
                            f"queue:{request}",
                            ending=True,
                        )
                if kind == "request_blocked":
                    self._call(event, "dependency", f"dependency:{request}")
                    self._signal(
                        f"wait:{request}",
                        "Recorded dependency wait",
                        text(
                            payload.get("dependencies")
                            or payload.get("reason")
                            or "Scheduler marked this request blocked."
                        ),
                        [f"{actor}:dependency:{request}"],
                    )
                if kind in {
                    "request_ready",
                    "request_started",
                    "request_completed",
                    "request_cancelled",
                    "request_failed",
                }:
                    wait = self.actions.get(f"{actor}:dependency:{request}")
                    if wait and wait["outcome"] == "running":
                        self._call(
                            {**event, "payload": {**payload, "outcome": "success"}},
                            "dependency",
                            f"dependency:{request}",
                            ending=True,
                        )
                    if f"wait:{request}" in self.signals:
                        self._close_signal(f"wait:{request}")
                if kind == "request_failed":
                    self._signal(
                        f"request:{request}",
                        "Execution request failed",
                        text(payload.get("error") or payload),
                        [event["id"]],
                    )
                self._agent_status(agent, kind.removeprefix("request_"))
            elif kind == "skill_start":
                self._call(
                    {**event, "payload": {**payload, "name": payload.get("skill")}},
                    "setup",
                    f"skill:{payload.get('request_id') or payload.get('ts') or event['id']}",
                )
            elif kind == "skill_call":
                key = f"skill:{payload.get('request_id') or payload.get('ts_start')}"
                self._call(event, "setup", key, ending=True)
            elif kind in {"skill_error", "skill_cancelled"}:
                request = payload.get("request_id") or self.actor_request.get(actor)
                action = self.actions.get(f"{actor}:skill:{request}")
                if action:
                    self._call(
                        {
                            **event,
                            "payload": {
                                **payload,
                                "outcome": "cancelled" if kind == "skill_cancelled" else "failed",
                            },
                        },
                        "setup",
                        f"skill:{request}",
                        ending=True,
                    )
                else:
                    self._signal(
                        event["id"], "Recorded skill interruption", text(payload), [event["id"]]
                    )
            elif kind == "input_required":
                self._signal(
                    f"input:{actor}",
                    "Your input is requested",
                    text(payload.get("message") or payload),
                    [event["id"]],
                    "required",
                )
                self._agent_status(agent, "waiting_input")
            elif kind == "input_resolved" and f"input:{actor}" in self.signals:
                self._close_signal(f"input:{actor}")
            elif kind in {"delegation", "handoff"}:
                edge = {
                    "id": event["id"],
                    "from": actor,
                    "to": payload.get("to", "unattributed"),
                    "kind": kind,
                    "label": payload.get("task") or payload.get("message") or "Recorded handoff",
                    "time": max(0, event["ts"] - self.origin),
                    "source": event.get("source", "canonical_event"),
                    "event_id": event["id"],
                }
                self.run["edges"].append(edge)
                self.dirty["edges"].add(edge["id"])
        if (
            self.run["status"] == "unknown"
            and kind in {"request_started", "skill_start", "tool_call"}
            and self.finished_at is None
        ):
            self.run["status"] = "running"
        if "unfinished" in self.signals:
            missing = [
                action["id"]
                for action in self.actions.values()
                if action["outcome"] == "incomplete" and "end_ts" not in action
            ]
            if missing != self.signals["unfinished"]["evidence"]:
                self.signals["unfinished"]["evidence"] = missing
                self.dirty["signals"].add("unfinished")
            if not missing:
                self._close_signal("unfinished")
        return True

    def tick(self, now, *, force=False):
        if self.finished_at is None and self.origin is not None:
            self.run["duration"] = max(self.run["duration"], now - self.origin)
        for action_id in self.running:
            action = self.actions[action_id]
            if action["outcome"] == "running":
                end = self.origin + self.run["duration"]
                pauses = self.pauses.get(action["agent"], [])
                if pauses and pauses[-1][1] is None:
                    end = min(end, pauses[-1][0])
                duration = max(0, end - action["started_ts"])
                execution_duration = self._execution_duration(action, end)
                if (
                    force
                    or int(duration) != int(action["duration"])
                    or int(execution_duration) != int(action.get("execution_duration", 0))
                ):
                    action["duration"] = duration
                    action["execution_duration"] = execution_duration
                    self.dirty["actions"].add(action["id"])
                    self._interval(action)
                    self._refresh_episode(action)

    def patch(self):
        maps = {
            "actions": self.actions,
            "episodes": self.episodes,
            "agents": self.agents,
            "edges": {edge["id"]: edge for edge in self.run["edges"]},
            "intervals": self.intervals,
            "signals": self.signals,
        }
        patch = {
            key: [
                copy.deepcopy(maps[key][item_id])
                for item_id in sorted(ids, key=lambda item_id: maps[key][item_id].get("start", 0))
            ]
            for key, ids in self.dirty.items()
        }
        patch["meta"] = {
            key: copy.deepcopy(self.run[key])
            for key in (
                "status",
                "duration",
                "coverage",
                "task",
                "model",
                "harness",
                "dataset",
                "time_origin",
            )
            if key in self.run
        }
        patch["counter_samples"] = self.counter_updates
        self.counter_updates = {}
        for ids in self.dirty.values():
            ids.clear()
        return patch

    def snapshot(self):
        self.run["signals"] = list(self.signals.values())
        return copy.deepcopy(self.run)


class LiveSource:
    """One incremental reader per server run directory, shared by all clients."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.model = Trajectory("live", f"Current execution · {self.directory.parent.name}")
        self.paths = {"global": self.directory / "global_data.sqlite3"}
        self.events = []
        self.profile_path = (
            Path(os.environ.get("AGENCY_PROFILE_DIR", self.directory.parent / "profiler"))
            / "profile_data.sqlite3"
        )
        self.cursors = {}
        self.epoch = uuid.uuid4().hex
        self.revision = 0
        self.journal = deque(maxlen=512)
        self.lock = threading.RLock()
        self.last_poll = 0
        self.refiner = None
        if os.environ.get("AGENCY_TRAJECTORY_LABEL_CONFIG"):
            from .trajectory_semantics import SemanticRefiner

            self.refiner = SemanticRefiner(os.environ["AGENCY_TRAJECTORY_LABEL_CONFIG"])
        self.model.run["coverage"]["semantics"] = (
            "Deterministic grouping; optional semantic refinement disabled."
        )

    def _rows(self, source, path):
        if not path.is_file():
            return [], f"{source}: event database unavailable"
        con = None
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
            tables = {
                row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            rows = []
            next_cursors = {}
            for table in ("events", "exchanges"):
                if table not in tables:
                    continue
                cursor_key = (source, table)
                cursor = self.cursors.get(cursor_key, "")
                payload = "payload" if table == "events" else "'{}'"
                types = (
                    f" AND type IN ({','.join('?' for _ in EVENT_TYPES)})"
                    if table == "events"
                    else ""
                )
                parameters = (cursor, *EVENT_TYPES) if table == "events" else (cursor,)
                batch = con.execute(
                    f"SELECT id,type,timestamp,name,call_label,{payload} FROM {table} WHERE id>?{types} ORDER BY id LIMIT 500",
                    parameters,
                ).fetchall()
                if len(batch) == 500:
                    self.model.run["coverage"]["catching_up"] = True
                for row_id, kind, ts, actor, label, raw in batch:
                    body = json.loads(raw)
                    if table == "exchanges":
                        body = self._exchange(con, label, kind)
                        kind = "model_result"
                    event = {
                        "id": f"{source}/{table}/{row_id}",
                        "type": kind,
                        "ts": body.get("sampled_at", ts) if kind == "resource_sample" else ts,
                        "actor": actor,
                        "call_label": label,
                        "payload": body,
                        "source": "canonical_sqlite",
                    }
                    rows.append(event)
                    next_cursors[cursor_key] = row_id
                    if kind == "agent_registered" and body.get("db_path") and actor:
                        self.paths[f"actor:{actor}"] = Path(body["db_path"])
            self.cursors.update(next_cursors)
            return rows, None
        except (sqlite3.Error, ValueError, OSError):
            return [], f"{source}: telemetry could not be read; retrying"
        finally:
            if con is not None:
                con.close()

    @staticmethod
    def _exchange(con, label, kind, *, full=False):
        chains = {"prompt": [], "response": []}
        for chain_kind, raw in con.execute(
            "SELECT c.kind,b.payload FROM exchange_chain c JOIN blocks b ON b.hash=c.hash WHERE c.call_label=? ORDER BY c.kind,c.seq",
            (label,),
        ):
            block = json.loads(raw)
            # Hidden reasoning is not part of the trajectory's evidence surface.
            if block.get("type") not in {"thinking", "reasoning", "redacted_thinking"}:
                chains[chain_kind].append(block)
        metadata = next((b for b in chains["response"] if b.get("type") == "metadata"), {})
        visible = [b.get("text", "") for b in chains["response"] if b.get("type") == "text"]
        markers = next((b for b in chains["response"] if b.get("error") or b.get("cancelled")), {})
        prompt = chains["prompt"] if full else chains["prompt"][:80]
        preview_limit = None if full else 1200
        return {
            "result": "\n".join(visible),
            "error": markers.get("error"),
            "outcome": "failed"
            if kind == "llm_stream_error"
            else "cancelled"
            if kind == "llm_stream_cancelled"
            else "success",
            "tokens": metadata.get("usage", {}).get("prompt_tokens"),
            "output_tokens": metadata.get("usage", {}).get("completion_tokens"),
            "tool_call_ids": [
                b["id"] for b in chains["response"] if b.get("type") == "tool_use" and b.get("id")
            ],
            "context": {
                "source": "recorded",
                "available": bool(chains["prompt"]),
                "blocks": [
                    {
                        "source": b.get("role", "unknown"),
                        "role": b.get("role", "unknown"),
                        "change": "recorded",
                        "preview": text(b)[:preview_limit],
                        "chars": len(text(b)),
                    }
                    for b in prompt
                ],
                "chars": sum(len(text(b)) for b in chains["prompt"]),
                "dropped": [],
                "complete": True,
                "comparison_available": False,
                "truncated": not full
                and (len(chains["prompt"]) > 80 or any(len(text(b)) > 1200 for b in prompt)),
            },
        }

    def refresh(self, *, force=False):
        with self.lock:
            if not force and time.monotonic() - self.last_poll < 0.25:
                return
            self.last_poll = time.monotonic()
            self.model.run["coverage"]["catching_up"] = False
            events, gaps = [], []
            if self.profile_path.is_file():
                self.paths["profiler"] = self.profile_path
            # Read global registration first, including agents created after connection.
            rows, gap = self._rows("global", self.paths["global"])
            events.extend(rows)
            if gap:
                gaps.append(gap)
            for source, path in list(self.paths.items()):
                if source == "global":
                    continue
                rows, gap = self._rows(source, path)
                events.extend(rows)
                if gap:
                    gaps.append(gap)
            for event in events:
                if not isinstance(event.get("ts"), (int, float)) or not math.isfinite(event["ts"]):
                    event["ts"] = (self.model.origin or 0) + self.model.run["duration"]
                    event["source"] = "timestamp_missing_ordered_at_last_observation"
            events.sort(key=lambda e: (e["ts"], e["id"]))
            # Retain bounded replay evidence, not a second copy of huge tool
            # outputs. Immutable originals remain available in their SQLite rows.
            for event in events:
                payload = dict(event["payload"])
                for key, flag in (
                    ("arguments", "input_truncated"),
                    ("result", "output_truncated"),
                    ("output", "output_truncated"),
                    ("error", "output_truncated"),
                ):
                    if key in payload and len(text(payload[key])) > 8000:
                        if key in {"result", "output"}:
                            payload["recorded_result_preview"] = returned_preview(payload[key])
                        payload[key] = text(payload[key])[:8000]
                        payload[flag] = True
                self.events.append({**event, "payload": payload})
            for event in events:
                self.model.apply(event)
            self.model.run["coverage"]["gaps"] = gaps
            self.model.tick(time.time())
            if self.refiner:
                self.refiner.update(self.model)
            self.revision += 1
            self.journal.append((self.revision, self.model.patch()))

    def message(self, after=0, epoch=None):
        with self.lock:
            if (
                epoch != self.epoch
                or not after
                or after > self.revision
                or (self.journal and after < self.journal[0][0] - 1)
            ):
                return {
                    "type": "snapshot",
                    "run": self.model.snapshot(),
                    "cursor": self.revision,
                    "epoch": self.epoch,
                    "resynced": bool(epoch),
                }
            updates = [patch for revision, patch in self.journal if revision > after]
            return {
                "type": "updates",
                "patches": updates,
                "cursor": self.revision,
                "epoch": self.epoch,
            }

    def raw_event(self, event_id):
        with self.lock:
            source, table, row_id = event_id.rsplit("/", 2)
            if source not in self.paths or table not in {"events", "exchanges"}:
                raise KeyError(event_id)
            con = sqlite3.connect(f"file:{self.paths[source]}?mode=ro", uri=True)
            try:
                con.row_factory = sqlite3.Row
                row = con.execute(f"SELECT * FROM {table} WHERE id=?", (row_id,)).fetchone()
                if row is None:
                    raise KeyError(event_id)
                result = dict(row)
                if table == "events":
                    result["payload"] = json.loads(result["payload"])
                else:
                    result["payload"] = self._exchange(
                        con, result["call_label"], result["type"], full=True
                    )
                return result
            finally:
                con.close()


def replay_events(run):
    """Reconstruct start/end observations; never put end evidence in a start."""
    events = []
    for action in run["actions"]:
        actor = action["agent"]
        common = {
            "action_id": action["id"],
            "call_id": action["id"],
            "kind": action["kind"],
            "tool": action["name"],
            "arguments": object_value(action["arguments"])
            if action.get("arguments")
            else {"command": action["command"]},
            "intent": action.get("intent", ""),
            "request_id": action.get("request_id"),
            "metadata": {
                key: value
                for key, value in action.get("metadata", {}).items()
                if key
                in {
                    "agency.agent_id",
                    "agency.span_id",
                    "agency.parent_span_id",
                    "model",
                    "harness",
                    "call_id",
                }
            },
        }
        model = action["kind"] == "model"
        start = {**common, "state": "waiting_llm"} if model else common
        events.append(
            {
                "id": f"replay/start/{action['id']}",
                "actor": actor,
                "ts": action["start"],
                "type": "agent_state" if model else "tool_call",
                "call_label": action["id"],
                "payload": start,
                "source": "reconstructed_trace_boundary",
            }
        )
        if action["outcome"] not in {"interrupted", "incomplete", "running"}:
            events.append(
                {
                    "id": f"replay/end/{action['id']}",
                    "actor": actor,
                    "ts": action["start"] + action["duration"],
                    "type": "model_result" if model else "tool_result",
                    "call_label": action["id"],
                    "payload": {
                        **common,
                        "result": action["result"],
                        "outcome": action["outcome"],
                        "tokens": action.get("tokens"),
                        "output_tokens": action.get("output_tokens"),
                        "context": action.get("context"),
                        "files": action.get("files"),
                        "timing": action.get("timing"),
                        "metadata": action.get("metadata", {}),
                    },
                    "source": "reconstructed_trace_boundary",
                }
            )
    for edge in run.get("edges", []):
        if edge["kind"] in {"delegation", "handoff"}:
            events.append(
                {
                    "id": f"replay/edge/{edge.get('id') or fingerprint(edge)}",
                    "actor": edge["from"],
                    "ts": edge.get("time") or 0,
                    "type": edge["kind"],
                    "payload": {"to": edge["to"], "task": edge["label"]},
                    "source": "recorded_trace_relationship",
                }
            )
    events.sort(key=lambda e: (e["ts"], 0 if "/start/" in e["id"] else 1, e["id"]))
    # Exhausting a saved trace does not establish task success or a complete lifecycle.
    events.append(
        {
            "id": "replay/exhausted",
            "actor": "workflow",
            "ts": run["duration"],
            "type": "done",
            "payload": {"status": run.get("status", "unknown")},
            "source": "replay_boundary_not_recorded_lifecycle",
        }
    )
    return events


def historical_projection(run):
    model = Trajectory(run["id"], run["title"], mode="review", source=run["source"])
    for event in replay_events(run):
        model.apply(event)
    projected = model.snapshot()
    # Existing attribution, resources, evaluator evidence and context stay intact.
    for key in ("actions", "episodes", "signals", "mode", "status"):
        run[key] = projected[key]
    return run
