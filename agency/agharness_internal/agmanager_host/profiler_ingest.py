"""Profiler correlation/ingestion for `agmanager_host`.

Kept as a SEPARATE listener (its own UDS socket, its own thread) from the
main dispatch app -- same reason the old `agprof_ingest.py` was split out
from `agllm_terminus.py`/`agproxy_llm.py`: a burst of profiler events must
never be able to stall a streaming dispatch and corrupt its TTFT
measurement. See `agmanager_host.py`'s module docstring for the full
two-server design.

Security-sensitive redaction/bounding helpers are duplicated (deliberately,
see that module docstring's reuse policy) from the old `agprof_ingest.py`'s
identical-purpose helpers rather than imported -- kept behaviorally
identical since they exist to keep a bearer token/secret out of an exported
span, not to be "simplified"."""

from __future__ import annotations

import json
import re
import socketserver
import struct
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._native_hooks import hook_payload_to_syscallevent

if TYPE_CHECKING:
    from ...agent import agent
    from .launch_state import LaunchRegistry

_MAX_EVENT_BYTES = 256 * 1024
_MAX_ATTRIBUTE_CHARS = 16 * 1024
_MAX_CLOCK_OFFSET_DISAGREEMENT_NS = 5_000_000_000
_REDACTED = "[REDACTED]"
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


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


def _tool_span_id(tool_call_id: str) -> str:
    span_id = f"tool:{tool_call_id}"
    if len(span_id) <= 256:
        return span_id
    import hashlib

    return "tool:" + hashlib.sha256(tool_call_id.encode()).hexdigest()


def _recv_exactly(sock, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"connection closed with {remaining} bytes still expected")
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


@dataclass
class _RemoteSpanStart:
    name: str
    start_perf_ns: int
    start_wall_ns: int
    metadata: dict
    handle: object | None


class _ProfilerIngestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            try:
                event = _recv_framed(self.request)
            except (ConnectionError, json.JSONDecodeError, UnicodeDecodeError, struct.error):
                return
            except ValueError as error:
                _send_framed(self.request, {"ok": False, "error": str(error)})
                return
            response = self.server.ingest.handle_event(event)
            _send_framed(self.request, response)


class _ProfilerIngestServer(socketserver.ThreadingUnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True


class ProfilerIngest:
    """Owns this agent's profiler UDS listener lifecycle and event
    handling. Constructed with the same `LaunchRegistry` every other
    feature module in this package shares, since span/clock state lives on
    each launch's `_LaunchState`, not a separate registry of its own."""

    def __init__(self, ag: "agent", registry: "LaunchRegistry") -> None:
        self._ag = ag
        self._registry = registry
        self._uds_server = None
        self._uds_thread: "threading.Thread | None" = None
        self.uds_path: "str | None" = None
        self._uds_reserved_path: "str | None" = None

    def ensure_uds_started(self, timeout_s: float = 10) -> str:
        from ...agutil import reserve_uds_path, uds_listener_is_live

        if uds_listener_is_live(self.uds_path, self._uds_thread):
            return self.uds_path
        if self.uds_path is not None:
            self.stop_uds()
        sock_path = reserve_uds_path(
            self._uds_reserved_path, f"agmanager-host-prof-{self._ag.agname}"
        )
        self._uds_reserved_path = sock_path
        server = _ProfilerIngestServer(sock_path, _ProfilerIngestHandler)
        server.ingest = self
        import os

        os.chmod(sock_path, 0o666)
        self._uds_server = server
        self._uds_thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name=f"agmanager_host-prof-uds[{self._ag.agname}]",
        )
        self._uds_thread.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self._uds_thread.is_alive():
            time.sleep(0.01)
        if not self._uds_thread.is_alive():
            raise RuntimeError("ProfilerIngest UDS server did not start within timeout")
        self.uds_path = sock_path
        return sock_path

    def stop_uds(self) -> None:
        import os

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

    def handle_event(self, event: dict) -> dict:
        from ...profiler import agprof

        token = event.get("token")
        launch = self._registry.get(token)
        if launch is None:
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
            with self._registry.lock:
                if token not in self._registry.launches:
                    return {"ok": False, "error": "unknown or missing token"}
                launch.clock_sync_estimate = estimate
            return {"ok": True, "host_wall_ns": host_wall_ns, "host_perf_ns": host_perf_ns}

        if event_name == "clock_offset":
            try:
                offsets = (int(event["wall_offset_ns"]), int(event["perf_offset_ns"]))
            except (KeyError, TypeError, ValueError):
                return {"ok": False, "error": "invalid clock offset"}
            with self._registry.lock:
                if token not in self._registry.launches:
                    return {"ok": False, "error": "unknown or missing token"}
                estimate = launch.clock_sync_estimate
                if estimate is None or any(
                    abs(reported - observed) > _MAX_CLOCK_OFFSET_DISAGREEMENT_NS
                    for reported, observed in zip(offsets, estimate)
                ):
                    return {"ok": False, "error": "clock offset failed host validation"}
                launch.clock_offsets = offsets
            return {"ok": True}

        if event_name == "hook":
            return self._handle_hook_event(token, launch, event)

        if event_name not in ("span_start", "span_end"):
            return {"ok": False, "error": "unsupported profiler event"}

        span_id = event.get("span_id")
        if not isinstance(span_id, str) or not span_id or len(span_id) > 256:
            return {"ok": False, "error": "invalid span_id"}
        wall_offset, perf_offset = launch.clock_offsets or (0, 0)
        try:
            perf_ns = int(event["perf_ns"]) + perf_offset
            wall_ns = int(event["wall_ns"]) + wall_offset
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "error": "invalid event timestamp"}

        if event_name == "span_start":
            name = event.get("name")
            if not isinstance(name, str) or not name or len(name) > 256:
                return {"ok": False, "error": "invalid span name"}
            name = _redact_sensitive(name, secrets=(token,))[:256]
            metadata = _bounded_metadata(event.get("metadata"), secrets=(token,))
            metadata.update(launch.span_attributes)
            metadata["timing"] = (
                "hook_boundary" if event.get("timing") == "hook_boundary" else "exact"
            )
            metadata["provenance"] = "container_asserted"
            with self._registry.lock:
                if token not in self._registry.launches:
                    return {"ok": False, "error": "unknown or missing token"}
                if span_id in launch.open_remote_spans:
                    return {"ok": False, "error": "duplicate span_start"}
                handle = agprof.start_external_span(
                    name,
                    start_perf_ns=perf_ns,
                    start_wall_ns=wall_ns,
                    metadata=metadata,
                    parent_context=launch.run_context,
                )
                launch.open_remote_spans[span_id] = _RemoteSpanStart(
                    name, perf_ns, wall_ns, metadata, handle
                )
            return {"ok": True}

        duration_ns = event.get("authoritative_duration_ns")
        with self._registry.lock:
            if token not in self._registry.launches:
                return {"ok": False, "error": "unknown or missing token"}
            start = launch.open_remote_spans.get(span_id)
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
                if duration_ns > observed_gap_ns + 1_000_000:
                    return {
                        "ok": False,
                        "error": "duration_ms exceeds observed PreToolUse/PostToolUse interval",
                    }
                duration_ns = min(duration_ns, observed_gap_ns)
            completed_tool_call_id = event.get("completed_tool_call_id")
            if completed_tool_call_id is not None:
                if not isinstance(completed_tool_call_id, str) or not completed_tool_call_id:
                    return {"ok": False, "error": "invalid completed tool_call_id"}
                launch.exact_tool_call_ids.add(completed_tool_call_id)
            launch.open_remote_spans.pop(span_id)
        metadata = {**start.metadata, **_bounded_metadata(event.get("metadata"), secrets=(token,))}
        metadata.update(launch.span_attributes)
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
            parent_context=launch.run_context,
        )
        return {"ok": True}

    def _handle_hook_event(self, token: str, launch, event: dict) -> dict:
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

        syscall_event = hook_payload_to_syscallevent(payload)
        tool_name = syscall_event.tool_name or syscall_event.syscall or "unknown"
        span_id = _tool_span_id(tool_use_id)

        if hook_name == "PreToolUse":
            metadata = {"tool_call_id": tool_use_id, "arguments": syscall_event.tool_args or {}}
            if syscall_event.argv is not None:
                metadata["argv"] = syscall_event.argv
            if syscall_event.path is not None:
                metadata["path"] = syscall_event.path
            return self.handle_event(
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
            if duration_ms < 0:
                return {"ok": False, "error": "missing or invalid duration_ms"}
            duration_ns = round(duration_ms * 1_000_000)
            metadata.update({"duration_ms": duration_ms, "timing": "exact"})
        else:
            metadata.update({"timing": "hook_boundary"})
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
        return self.handle_event(span_end_event)


__all__ = ["ProfilerIngest"]
