"""Dependency-free profiler event emission for container-resident code.

This module is loaded directly by file path inside the native harness
container.  Keep it stdlib-only: importing ``agency.profiler`` there would
first import the top-level package and pull host-only LLM dependencies into
the sandbox image.

Events use the same 8-byte-length-prefixed JSON framing as the native harness
bridge.  A bounded queue keeps telemetry off the ReAct hot path; a full or
broken queue drops telemetry instead of delaying the workload being measured.
"""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
import uuid


def _recv_exactly(sock, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("profiler ingest closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_framed(sock, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(body)) + body)


def _recv_framed(sock) -> dict:
    (length,) = struct.unpack(">Q", _recv_exactly(sock, 8))
    return json.loads(_recv_exactly(sock, length).decode("utf-8"))


class _RemoteSpan:
    def __init__(self, emitter, name: str, span_id: "str | None", metadata: "dict | None"):
        self._emitter = emitter
        self._name = name
        self._span_id = span_id or uuid.uuid4().hex
        self._metadata = dict(metadata or {})

    def __enter__(self):
        self._emitter.emit(
            "span_start", span_id=self._span_id, name=self._name, metadata=self._metadata
        )
        return self

    def annotate(self, **metadata) -> None:
        self._metadata.update(metadata)

    def __exit__(self, exc_type, exc_value, _traceback) -> None:
        metadata = dict(self._metadata)
        metadata.setdefault("outcome", "failure" if exc_type is not None else "success")
        if exc_type is not None:
            metadata.setdefault("error_type", exc_type.__name__)
            metadata.setdefault("error", str(exc_value))
        self._emitter.emit("span_end", span_id=self._span_id, name=self._name, metadata=metadata)


class RemoteProfilerEmitter:
    """Best-effort, bounded event sender for one authenticated native run."""

    def __init__(
        self,
        sock_path: "str | None",
        token: str,
        *,
        queue_size: int = 256,
        socket_timeout_s: float = 1.0,
    ) -> None:
        self._sock_path = sock_path
        self._token = token
        self._socket_timeout_s = socket_timeout_s
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._closed = False
        self.dropped_events = 0
        self._thread = None
        if sock_path:
            self._thread = threading.Thread(
                target=self._send_loop,
                daemon=True,
                name="agprof-emitter",
            )
            self._thread.start()

    def span(
        self,
        name: str,
        *,
        span_id: "str | None" = None,
        metadata: "dict | None" = None,
    ) -> _RemoteSpan:
        return _RemoteSpan(self, name, span_id, metadata)

    def emit(self, event: str, **fields) -> None:
        if self._closed or self._thread is None:
            return
        payload = {
            "token": self._token,
            "ev": event,
            "wall_ns": time.time_ns(),
            "perf_ns": time.perf_counter_ns(),
            **fields,
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self.dropped_events += 1

    def close(self, timeout_s: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        thread = self._thread
        if thread is None:
            return
        flushed = threading.Event()
        deadline = time.monotonic() + timeout_s
        try:
            self._queue.put(("flush", flushed), timeout=timeout_s)
        except queue.Full:
            self.dropped_events += 1
            return
        flushed.wait(max(0.0, deadline - time.monotonic()))
        thread.join(timeout=0.1)

    def _send_loop(self) -> None:
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self._socket_timeout_s)
            sock.connect(self._sock_path)
            self._synchronize_clocks(sock)
            while True:
                item = self._queue.get()
                if isinstance(item, tuple) and item and item[0] == "flush":
                    item[1].set()
                    return
                try:
                    _send_framed(sock, item)
                    response = _recv_framed(sock)
                    if not response.get("ok"):
                        self.dropped_events += 1
                except Exception:
                    self.dropped_events += 1
                    return
        except Exception:
            # Telemetry must remain fail-open. Count queued events as dropped
            # without retrying a broken listener on the workload's behalf.
            self.dropped_events += 1
        finally:
            if sock is not None:
                sock.close()
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, tuple) and item and item[0] == "flush":
                    item[1].set()
                else:
                    self.dropped_events += 1

    def _synchronize_clocks(self, sock) -> None:
        wall_0 = time.time_ns()
        perf_0 = time.perf_counter_ns()
        _send_framed(
            sock,
            {
                "token": self._token,
                "ev": "clock_sync",
                "wall_ns": wall_0,
                "perf_ns": perf_0,
            },
        )
        response = _recv_framed(sock)
        wall_1 = time.time_ns()
        perf_1 = time.perf_counter_ns()
        if not response.get("ok"):
            return
        _send_framed(
            sock,
            {
                "token": self._token,
                "ev": "clock_offset",
                "wall_offset_ns": int(response["host_wall_ns"] - (wall_0 + wall_1) / 2),
                "perf_offset_ns": int(response["host_perf_ns"] - (perf_0 + perf_1) / 2),
            },
        )
        _recv_framed(sock)


__all__ = ["RemoteProfilerEmitter"]
