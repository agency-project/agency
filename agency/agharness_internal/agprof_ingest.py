"""Host-side correlation registry for harness profiler traffic.

Each harness launch already has a unique bearer token used to authenticate
its LLM traffic.  This module resolves that credential to the launch's agent
and active ``run{N}`` span context on the host.  The token remains only a dict
key: it is never copied into span attributes or profiler summaries.

The registry deliberately belongs to its own service rather than reaching
into agLLMTerminus's token map.  Its UDS listener is separate from the LLM
terminus so a telemetry burst cannot stall a streaming dispatch and corrupt
the TTFT measurement the profiler is trying to capture.
"""

from __future__ import annotations

import json
import math
import os
import re
import socketserver
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..profiler import agprof

if TYPE_CHECKING:
    from ..agent import agent


_MAX_EVENT_BYTES = 256 * 1024
_MAX_ATTRIBUTE_CHARS = 16 * 1024
_MAX_CLOCK_OFFSET_DISAGREEMENT_NS = 5_000_000_000
_DURATION_ROUNDING_TOLERANCE_NS = 1_000_000
# A PostToolUse hook can still be in flight when Claude starts the next LLM
# dispatch that exposes the completed tool call in its transcript.  Give the
# authoritative hook a short chance to win before committing the honest
# derived fallback.  The condition releases the ingest lock while waiting;
# healthy/completed hooks take the zero-wait path.
_EXACT_POST_GRACE_S = 0.1
_REDACTED = "[REDACTED]"
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


@dataclass(frozen=True)
class _Registration:
    agent: "agent"
    run_context: object | None
    span_attributes: dict
    exact_events: bool
    exact_tool_events: bool


@dataclass(frozen=True)
class _RemoteSpanStart:
    name: str
    start_perf_ns: int
    start_wall_ns: int
    metadata: dict
    handle: object | None


def _recv_exactly(sock, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("remote profiler emitter disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_framed(sock) -> dict:
    (length,) = struct.unpack(">Q", _recv_exactly(sock, 8))
    if length > _MAX_EVENT_BYTES:
        raise ValueError(f"profiler event exceeds {_MAX_EVENT_BYTES} bytes")
    return json.loads(_recv_exactly(sock, length).decode("utf-8"))


def _send_framed(sock, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(body)) + body)


def _sensitive_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return normalized in {
        "authorization",
        "proxy_authorization",
        "auth",
        "token",
        "api_key",
        "apikey",
        "secret",
        "credential",
        "credentials",
    } or normalized.endswith(("_token", "_api_key", "_secret", "_credential"))


def _redact_sensitive(value, *, secrets: "tuple[str, ...]", depth: int = 0):
    if depth >= 12:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            key: _REDACTED
            if _sensitive_key(key)
            else _redact_sensitive(item, secrets=secrets, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive(item, secrets=secrets, depth=depth + 1) for item in value]
    if not isinstance(value, str):
        return value
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, _REDACTED)
    return _BEARER_RE.sub("Bearer " + _REDACTED, redacted)


def _bounded_metadata(value, *, secrets: "tuple[str, ...]" = ()) -> dict:
    if not isinstance(value, dict):
        return {}
    bounded = {}
    for raw_key, raw_value in list(value.items())[:32]:
        key = str(raw_key)[:128]
        raw_value = (
            _REDACTED if _sensitive_key(raw_key) else _redact_sensitive(raw_value, secrets=secrets)
        )
        if isinstance(raw_value, (bool, int, float)) or raw_value is None:
            bounded[key] = raw_value
            continue
        if not isinstance(raw_value, str):
            raw_value = json.dumps(raw_value, sort_keys=True, default=str)
        if len(raw_value) > _MAX_ATTRIBUTE_CHARS:
            raw_value = raw_value[:_MAX_ATTRIBUTE_CHARS] + "…[truncated]"
        bounded[key] = raw_value
    return bounded


def _claude_tool_span_id(tool_use_id: str) -> str:
    span_id = f"claude-tool:{tool_use_id}"
    if len(span_id) <= 256:
        return span_id
    import hashlib

    return "claude-tool:" + hashlib.sha256(tool_use_id.encode()).hexdigest()


class _IngestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            try:
                event = _recv_framed(self.request)
            except (ConnectionError, json.JSONDecodeError, UnicodeDecodeError, struct.error):
                return
            except ValueError as error:
                _send_framed(self.request, {"ok": False, "error": str(error)})
                return
            response = self.server.ingest._handle_event(event)
            _send_framed(self.request, response)


class _IngestServer(socketserver.ThreadingUnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True


class agProfilerIngest:
    """Token-to-run correlation state shared by all harness backends."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._exact_post_condition = threading.Condition(self._lock)
        self._lifecycle_lock = threading.Lock()
        self._registrations: "dict[str, _Registration]" = {}
        self._clock_offsets: "dict[str, tuple[int, int]]" = {}
        self._clock_sync_estimates: "dict[str, tuple[int, int]]" = {}
        self._open_remote_spans: "dict[tuple[str, str], _RemoteSpanStart]" = {}
        self._tokens_with_exact_tool_events: set[str] = set()
        self._exact_tool_call_ids: "dict[str, set[str]]" = {}
        self._uds_server = None
        self._uds_thread: "threading.Thread | None" = None
        self.uds_path: "str | None" = None

    def register(
        self,
        token: str,
        ag: "agent",
        *,
        run_context=None,
        exact_events: bool = False,
        exact_tool_events: bool = False,
    ) -> None:
        if run_context is None:
            run_context = agprof.current_span_context()
        current_attributes = agprof.current_span_attributes()
        attributes = {
            key: current_attributes[key]
            for key in (
                "agency.run_id",
                "agency.agent_id",
                "agency.parent_agent_id",
            )
            if key in current_attributes
        }
        attributes.setdefault("agency.agent_id", str(ag.agname))
        parent_agent_id = getattr(ag, "_parent_agent_id", None)
        if parent_agent_id is not None:
            attributes.setdefault("agency.parent_agent_id", str(parent_agent_id))
        with self._lock:
            self._registrations[token] = _Registration(
                ag,
                run_context,
                attributes,
                exact_events,
                exact_events or exact_tool_events,
            )

    def unregister(self, token: str) -> None:
        with self._lock:
            self._registrations.pop(token, None)
            self._clock_offsets.pop(token, None)
            self._clock_sync_estimates.pop(token, None)
            self._tokens_with_exact_tool_events.discard(token)
            self._exact_tool_call_ids.pop(token, None)
            stale = [key for key in self._open_remote_spans if key[0] == token]
            open_spans = [self._open_remote_spans.pop(key) for key in stale]
            self._exact_post_condition.notify_all()
        for open_span in open_spans:
            agprof.interrupt_external_span(open_span.handle)

    def agent_for_token(self, token: "str | None"):
        registration = self._registration_for_token(token)
        return registration.agent if registration is not None else None

    def context_for_token(self, token: "str | None"):
        registration = self._registration_for_token(token)
        return registration.run_context if registration is not None else None

    def attributes_for_token(self, token: "str | None") -> dict:
        registration = self._registration_for_token(token)
        return dict(registration.span_attributes) if registration is not None else {}

    def has_exact_events(self, token: "str | None") -> bool:
        registration = self._registration_for_token(token)
        return bool(registration and registration.exact_events)

    def has_exact_tool_events(self, token: "str | None") -> bool:
        registration = self._registration_for_token(token)
        if registration is None:
            return False
        with self._lock:
            return registration.exact_events or token in self._tokens_with_exact_tool_events

    def exact_tool_call_ids(self, token: "str | None") -> set[str]:
        if token is None:
            return set()
        with self._lock:
            return set(self._exact_tool_call_ids.get(token, ()))

    def reconcile_derived_tool_call_ids(
        self, token: "str | None", tool_call_ids: "set[str]"
    ) -> "set[str]":
        """Discard open Claude Pre spans replaced by transcript fallback.

        A missing/malformed Post leaves its Pre interruptible.  If a later
        dispatch's transcript proves that exact call completed, however, the
        derived span becomes its one representation; silently cancel the
        provisional exact handle before deriving so summary counts cannot
        contain both a completed derived tool and an interrupted exact copy.
        """
        if token is None or not tool_call_ids:
            return set()
        derive_ids = set()
        deadline = time.monotonic() + _EXACT_POST_GRACE_S
        with self._exact_post_condition:
            while True:
                completed_ids = self._exact_tool_call_ids.get(token, set())
                pending_exact = any(
                    tool_call_id not in completed_ids
                    and (
                        token,
                        _claude_tool_span_id(tool_call_id),
                    )
                    in self._open_remote_spans
                    for tool_call_id in tool_call_ids
                )
                remaining = deadline - time.monotonic()
                if not pending_exact or remaining <= 0:
                    break
                self._exact_post_condition.wait(remaining)
            completed_ids = self._exact_tool_call_ids.get(token, set())
            starts = []
            for tool_call_id in tool_call_ids:
                # A Post can complete after on_dispatch snapshots the skip
                # set. Completion reserves the ID atomically with removing
                # its Pre below, so this second check closes that window.
                if tool_call_id in completed_ids:
                    continue
                key = (token, _claude_tool_span_id(tool_call_id))
                start = self._open_remote_spans.pop(key, None)
                if start is not None:
                    starts.append(start)
                derive_ids.add(tool_call_id)
        for start in starts:
            agprof.cancel_external_span(start.handle)
        return derive_ids

    def _registration_for_token(self, token: "str | None") -> "_Registration | None":
        if token is None:
            return None
        with self._lock:
            return self._registrations.get(token)

    def ensure_uds_started(self, timeout_s: float = 10) -> str:
        if self.uds_path is not None:
            return self.uds_path
        with self._lifecycle_lock:
            if self.uds_path is not None:
                return self.uds_path
            from ..agutil import agharness_llm_gateway_dir

            sock_path = str(agharness_llm_gateway_dir() / f"agprof-ingest-{uuid.uuid4().hex}.sock")
            server = _IngestServer(sock_path, _IngestHandler)
            server.ingest = self
            os.chmod(sock_path, 0o666)
            self._uds_server = server
            self._uds_thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
                name="agprof-ingest-uds",
            )
            self._uds_thread.start()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and not self._uds_thread.is_alive():
                time.sleep(0.01)
            if not self._uds_thread.is_alive():
                raise RuntimeError("agProfilerIngest UDS server did not start within timeout")
            self.uds_path = sock_path
            return sock_path

    def stop_uds(self) -> None:
        server = self._uds_server
        if server is not None:
            server.shutdown()
            server.server_close()
        if self._uds_thread is not None:
            self._uds_thread.join(timeout=10)
        if self.uds_path and os.path.exists(self.uds_path):
            os.remove(self.uds_path)
        self._uds_server = None
        self._uds_thread = None
        self.uds_path = None

    def _handle_event(self, event: dict) -> dict:
        token = event.get("token")
        registration = self._registration_for_token(token)
        if registration is None:
            return {"ok": False, "error": "unknown or missing token"}

        event_name = event.get("ev")
        if event_name == "clock_sync":
            host_wall_ns = time.time_ns()
            host_perf_ns = time.perf_counter_ns()
            try:
                estimate = (
                    host_wall_ns - int(event["wall_ns"]),
                    host_perf_ns - int(event["perf_ns"]),
                )
            except (KeyError, TypeError, ValueError):
                return {"ok": False, "error": "invalid clock sync"}
            with self._lock:
                if self._registrations.get(token) is not registration:
                    return {"ok": False, "error": "unknown or missing token"}
                self._clock_sync_estimates[token] = estimate
            return {
                "ok": True,
                "host_wall_ns": host_wall_ns,
                "host_perf_ns": host_perf_ns,
            }
        if event_name == "clock_offset":
            try:
                offsets = (int(event["wall_offset_ns"]), int(event["perf_offset_ns"]))
            except (KeyError, TypeError, ValueError):
                return {"ok": False, "error": "invalid clock offset"}
            with self._lock:
                if self._registrations.get(token) is not registration:
                    return {"ok": False, "error": "unknown or missing token"}
                estimate = self._clock_sync_estimates.get(token)
                if estimate is None or any(
                    abs(reported - observed) > _MAX_CLOCK_OFFSET_DISAGREEMENT_NS
                    for reported, observed in zip(offsets, estimate)
                ):
                    return {"ok": False, "error": "clock offset failed host validation"}
                self._clock_offsets[token] = offsets
            return {"ok": True}
        if event_name == "hook":
            return self.handle_hook_event(token, event)
        if event_name not in ("span_start", "span_end"):
            return {"ok": False, "error": "unsupported profiler event"}

        span_id = event.get("span_id")
        if not isinstance(span_id, str) or not span_id or len(span_id) > 256:
            return {"ok": False, "error": "invalid span_id"}
        try:
            perf_ns, wall_ns = self._host_timestamps(token, event)
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "invalid event timestamp"}
        key = (token, span_id)

        if event_name == "span_start":
            name = event.get("name")
            if not isinstance(name, str) or not name or len(name) > 256:
                return {"ok": False, "error": "invalid span name"}
            # Names are exported outside the process just like attributes.
            # Treat them as equally untrusted: a semantic tool name or
            # remote process label must not smuggle the launch credential (or
            # an Authorization value) around metadata redaction.
            name = _redact_sensitive(name, secrets=(token,))[:256]
            metadata = _bounded_metadata(event.get("metadata"), secrets=(token,))
            metadata.update(registration.span_attributes)
            # A hook start is provisional until PostToolUse tells us whether
            # Claude supplied its authoritative duration.  Raw remote spans
            # remain exact; a peer may only downgrade its own classification.
            metadata["timing"] = (
                "hook_boundary" if event.get("timing") == "hook_boundary" else "exact"
            )
            metadata["provenance"] = "container_asserted"
            with self._lock:
                # unregister() shares this lock. Revalidate after all parsing
                # so a request racing launch teardown cannot resurrect an
                # open span under a dead credential.
                if self._registrations.get(token) is not registration:
                    return {"ok": False, "error": "unknown or missing token"}
                if key in self._open_remote_spans:
                    return {"ok": False, "error": "duplicate span_start"}
                handle = agprof.start_external_span(
                    name,
                    start_perf_ns=perf_ns,
                    start_wall_ns=wall_ns,
                    metadata=metadata,
                    parent_context=registration.run_context,
                )
                self._open_remote_spans[key] = _RemoteSpanStart(
                    name,
                    perf_ns,
                    wall_ns,
                    metadata,
                    handle,
                )
            return {"ok": True}

        duration_ns = event.get("authoritative_duration_ns")
        with self._lock:
            if self._registrations.get(token) is not registration:
                return {"ok": False, "error": "unknown or missing token"}
            start = self._open_remote_spans.get(key)
            if start is None:
                return {"ok": False, "error": "span_end without matching span_start"}
            if perf_ns < start.start_perf_ns:
                return {"ok": False, "error": "span_end timestamp precedes span_start"}
            if duration_ns is not None:
                try:
                    duration_ns = int(duration_ns)
                except (TypeError, ValueError):
                    return {"ok": False, "error": "invalid authoritative duration"}
                if duration_ns < 0:
                    return {"ok": False, "error": "invalid authoritative duration"}
                observed_gap_ns = perf_ns - start.start_perf_ns
                if duration_ns > observed_gap_ns + _DURATION_ROUNDING_TOLERANCE_NS:
                    # Keep the Pre open.  The transcript-derived per-ID
                    # fallback remains eligible, and unregister will report
                    # this unmatched semantic span as interrupted.
                    return {
                        "ok": False,
                        "error": "duration_ms exceeds observed PreToolUse/PostToolUse interval",
                    }
                # Claude reports milliseconds, so a rounded duration can be
                # microscopically longer than the host-observed gap. Accept
                # that case but never let it move the child before its Pre.
                duration_ns = min(duration_ns, observed_gap_ns)
            completed_tool_call_id = event.get("completed_tool_call_id")
            if completed_tool_call_id is not None:
                if not isinstance(completed_tool_call_id, str) or not completed_tool_call_id:
                    return {"ok": False, "error": "invalid completed tool_call_id"}
                # Reserve exact completion in the same critical section that
                # removes the Pre. A concurrent transcript fallback must see
                # either the still-open candidate (which it claims/cancels)
                # or this completed marker, never an ambiguous gap.
                self._exact_tool_call_ids.setdefault(token, set()).add(completed_tool_call_id)
            self._open_remote_spans.pop(key)
            self._exact_post_condition.notify_all()
        metadata = {
            **start.metadata,
            **_bounded_metadata(event.get("metadata"), secrets=(token,)),
        }
        # Correlation data is host-owned and wins over remote attributes.
        metadata.update(registration.span_attributes)
        if duration_ns is None:
            effective_start_perf_ns = start.start_perf_ns
            effective_start_wall_ns = start.start_wall_ns
        else:
            effective_start_perf_ns = max(start.start_perf_ns, perf_ns - duration_ns)
            effective_start_wall_ns = max(start.start_wall_ns, wall_ns - duration_ns)
        if start.handle is not None:
            start.handle.end(
                end_perf_ns=max(start.start_perf_ns, perf_ns),
                end_wall_ns=max(start.start_wall_ns, wall_ns),
                metadata=metadata,
                start_perf_ns=(effective_start_perf_ns if duration_ns is not None else None),
                start_wall_ns=(effective_start_wall_ns if duration_ns is not None else None),
            )
            return {"ok": True}
        agprof.ingest_remote_span(
            start.name,
            start_perf_ns=effective_start_perf_ns,
            end_perf_ns=max(effective_start_perf_ns, perf_ns),
            start_wall_ns=effective_start_wall_ns,
            end_wall_ns=max(effective_start_wall_ns, wall_ns),
            metadata=metadata,
            parent_context=registration.run_context,
        )
        return {"ok": True}

    def handle_hook_event(self, token: "str | None", event: dict) -> dict:
        """Translate one authenticated Claude Code hook into a tool span.

        The hook's bearer token is supplied separately by the HTTP bridge and
        is never accepted from, or copied into, the event payload.  Pre/post
        processes correlate through Claude Code's stable ``tool_use_id``.
        A validated Claude ``duration_ms`` is exact. When that optional field
        is absent, the observed Pre/Post interval is explicitly labelled
        ``hook_boundary`` rather than claimed as exact.
        """
        if self._registration_for_token(token) is None:
            return {"ok": False, "error": "unknown or missing token"}
        if not isinstance(event, dict):
            return {"ok": False, "error": "invalid hook event"}

        payload = event.get("payload")
        if not isinstance(payload, dict):
            return {"ok": False, "error": "invalid hook payload"}
        hook_name = event.get("hook_event_name") or payload.get("hook_event_name")
        if hook_name not in ("PreToolUse", "PostToolUse", "PostToolUseFailure"):
            return {"ok": False, "error": "unsupported hook event"}

        tool_use_id = payload.get("tool_use_id")
        if not isinstance(tool_use_id, str) or not tool_use_id:
            return {"ok": False, "error": "missing tool_use_id"}

        try:
            wall_ns = int(event["wall_ns"])
            perf_ns = int(event["perf_ns"])
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "invalid hook timestamp"}

        # Keep hook parsing in one place.  This is the same adapter used by
        # native-hook policy mediation, including Bash argv and file paths.
        from .agharness_backends._native_hooks import hook_payload_to_syscallevent

        syscall_event = hook_payload_to_syscallevent(payload)
        tool_name = syscall_event.tool_name or syscall_event.syscall or "unknown"
        span_id = _claude_tool_span_id(tool_use_id)

        if hook_name == "PreToolUse":
            metadata = {
                "tool_call_id": tool_use_id,
                "arguments": syscall_event.tool_args or {},
            }
            if syscall_event.argv is not None:
                metadata["argv"] = syscall_event.argv
            if syscall_event.path is not None:
                metadata["path"] = syscall_event.path
            result = self._handle_event(
                {
                    "token": token,
                    "ev": "span_start",
                    "span_id": span_id,
                    "name": f"tool:{str(tool_name)[:251]}",
                    "wall_ns": wall_ns,
                    "perf_ns": perf_ns,
                    "timing": "hook_boundary",
                    "metadata": metadata,
                }
            )
            if result.get("ok"):
                with self._lock:
                    if token in self._registrations:
                        self._tokens_with_exact_tool_events.add(token)
            return result

        metadata = {
            "tool_call_id": tool_use_id,
            "outcome": "failure" if hook_name == "PostToolUseFailure" else "success",
        }
        duration_ns = None
        if "duration_ms" in payload:
            try:
                duration_ms = float(payload["duration_ms"])
            except (TypeError, ValueError):
                return {"ok": False, "error": "missing or invalid duration_ms"}
            if not math.isfinite(duration_ms) or duration_ms < 0:
                return {"ok": False, "error": "missing or invalid duration_ms"}
            duration_ns = round(duration_ms * 1_000_000)
            metadata.update(
                {
                    "duration_ms": duration_ms,
                    "timing": "exact",
                    "timing_source": "claude_duration_ms",
                }
            )
        else:
            metadata.update(
                {
                    "timing": "hook_boundary",
                    "timing_source": "pre_post_hooks",
                }
            )
        if "tool_response" in payload:
            metadata["result"] = payload["tool_response"]
        if "error" in payload:
            metadata["error"] = payload["error"]
        span_end_event = {
            "token": token,
            "ev": "span_end",
            "span_id": span_id,
            "wall_ns": wall_ns,
            "perf_ns": perf_ns,
            "completed_tool_call_id": tool_use_id,
            "metadata": metadata,
        }
        if duration_ns is not None:
            span_end_event["authoritative_duration_ns"] = duration_ns
        result = self._handle_event(span_end_event)
        return result

    def _host_timestamps(self, token: str, event: dict) -> tuple[int, int]:
        with self._lock:
            wall_offset, perf_offset = self._clock_offsets.get(token, (0, 0))
        return int(event["perf_ns"]) + perf_offset, int(event["wall_ns"]) + wall_offset


_shared_profiler_ingest: "agProfilerIngest | None" = None
_shared_profiler_ingest_lock = threading.Lock()


def get_shared_profiler_ingest() -> agProfilerIngest:
    global _shared_profiler_ingest
    if _shared_profiler_ingest is not None:
        return _shared_profiler_ingest
    with _shared_profiler_ingest_lock:
        if _shared_profiler_ingest is None:
            _shared_profiler_ingest = agProfilerIngest()
        return _shared_profiler_ingest


__all__ = ["agProfilerIngest", "get_shared_profiler_ingest"]
