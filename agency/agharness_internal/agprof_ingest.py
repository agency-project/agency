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
import os
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


@dataclass(frozen=True)
class _Registration:
    agent: "agent"
    run_context: object | None
    span_attributes: dict
    exact_events: bool


@dataclass(frozen=True)
class _RemoteSpanStart:
    name: str
    start_perf_ns: int
    start_wall_ns: int
    metadata: dict


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


def _bounded_metadata(value) -> dict:
    if not isinstance(value, dict):
        return {}
    bounded = {}
    for raw_key, raw_value in list(value.items())[:32]:
        key = str(raw_key)[:128]
        if isinstance(raw_value, (bool, int, float)) or raw_value is None:
            bounded[key] = raw_value
            continue
        if not isinstance(raw_value, str):
            raw_value = json.dumps(raw_value, sort_keys=True, default=str)
        if len(raw_value) > _MAX_ATTRIBUTE_CHARS:
            raw_value = raw_value[:_MAX_ATTRIBUTE_CHARS] + "…[truncated]"
        bounded[key] = raw_value
    return bounded


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
        self._lifecycle_lock = threading.Lock()
        self._registrations: "dict[str, _Registration]" = {}
        self._clock_offsets: "dict[str, tuple[int, int]]" = {}
        self._clock_sync_estimates: "dict[str, tuple[int, int]]" = {}
        self._open_remote_spans: "dict[tuple[str, str], _RemoteSpanStart]" = {}
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
            self._registrations[token] = _Registration(ag, run_context, attributes, exact_events)

    def unregister(self, token: str) -> None:
        with self._lock:
            self._registrations.pop(token, None)
            self._clock_offsets.pop(token, None)
            self._clock_sync_estimates.pop(token, None)
            stale = [key for key in self._open_remote_spans if key[0] == token]
            for key in stale:
                self._open_remote_spans.pop(key, None)

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
                estimate = self._clock_sync_estimates.get(token)
                if estimate is None or any(
                    abs(reported - observed) > _MAX_CLOCK_OFFSET_DISAGREEMENT_NS
                    for reported, observed in zip(offsets, estimate)
                ):
                    return {"ok": False, "error": "clock offset failed host validation"}
                self._clock_offsets[token] = offsets
            return {"ok": True}
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
            start = _RemoteSpanStart(
                name,
                perf_ns,
                wall_ns,
                _bounded_metadata(event.get("metadata")),
            )
            with self._lock:
                self._open_remote_spans[key] = start
            return {"ok": True}

        with self._lock:
            start = self._open_remote_spans.pop(key, None)
        if start is None:
            return {"ok": False, "error": "span_end without matching span_start"}
        metadata = {**start.metadata, **_bounded_metadata(event.get("metadata"))}
        # Correlation data is host-owned and wins over remote attributes.
        metadata.update(registration.span_attributes)
        agprof.ingest_remote_span(
            start.name,
            start_perf_ns=start.start_perf_ns,
            end_perf_ns=max(start.start_perf_ns, perf_ns),
            start_wall_ns=start.start_wall_ns,
            end_wall_ns=max(start.start_wall_ns, wall_ns),
            metadata=metadata,
            parent_context=registration.run_context,
        )
        return {"ok": True}

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
