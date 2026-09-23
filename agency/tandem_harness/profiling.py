"""Attempt-scoped telemetry for native Python, without importing the host package.

Reports spans and (optionally) automatic function-call samples through the
same canonical ``record_span``/``record_samples`` bridge routes any harness
reporting its own telemetry would use -- there is no clock-calibration
handshake to perform first: wall-clock timestamps are already comparable
across processes on one host, so the host translates a reported timestamp
into its own ``perf_counter_ns`` domain itself, using its own simultaneous
clock readings, not a negotiated offset.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path


class NativeProfiler:
    def __init__(self, bridge):
        self.bridge = bridge
        self.enabled = False
        self.automatic_settings = None
        self.dropped = 0
        self.collector = None
        self.failed = False
        self.stack: "list[str]" = []
        self.process_identity: dict = {}
        self._previous_profiler = None

    def _report_span(self, payload: dict) -> None:
        if self.failed:
            return
        try:
            payload["attributes"] = {
                **payload.get("attributes", {}),
                **self.process_identity,
                "agency.namespace_tid": threading.get_native_id(),
            }
            result = self.bridge.record_profiler_span(payload)
            if not result.get("ok", True):
                raise RuntimeError(result.get("error", "span report rejected"))
        except Exception as exc:
            self.failed = True
            print(f"[agprof] native span report failed: {exc}", file=sys.stderr)

    def _report_samples(self, samples: list) -> None:
        try:
            result = self.bridge.record_profiler_samples(samples)
            if not result.get("ok"):
                self.dropped += result.get("rejected", len(samples))
        except Exception as exc:
            self.dropped += len(samples)
            print(f"[agprof] native sample report failed: {exc}", file=sys.stderr)

    def __enter__(self):
        # A nested run_react_loop call (a worker segment launched from within
        # the supervisor's own turn) enters/exits its own NativeProfiler on
        # this same bridge, so this must nest like a stack, not a flat
        # assignment -- otherwise the worker's __exit__ would clobber the
        # supervisor's still-in-flight self.bridge._profiler back to None,
        # silently turning every later profile_span() in the outer loop into
        # a no-op nullcontext() for the rest of the run.
        self._previous_profiler = getattr(self.bridge, "_profiler", None)
        try:
            settings = self.bridge.profiler_settings()
            self.enabled = bool(settings.get("enabled"))
            if not self.enabled:
                return self
            self.bridge._profiler = self
            try:
                stat = Path("/proc/self/stat").read_text()
                self.process_identity = {
                    "agency.namespace_pid": os.getpid(),
                    "agency.process_start_ticks": int(stat[stat.rfind(") ") + 2 :].split()[19]),
                    "agency.pid_namespace": os.readlink("/proc/self/ns/pid"),
                }
            except (OSError, ValueError, IndexError):
                # A restricted /proc may prevent mapping. Spans still report
                # under the attempt identity; never guess an OS process.
                self.process_identity = {}
            self.automatic_settings = settings.get("automatic")
            if self.automatic_settings:
                # Reuse the host's own collector while avoiding agency.__init__
                # and its host-only dependencies and process-level environment
                # startup.
                path = Path(__file__).resolve().parents[1] / "observability/profiler/agprof.py"
                spec = importlib.util.spec_from_file_location("_agency_container_profiler", path)
                collector = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(collector)
                collector._enable_auto_functions(
                    collector._make_auto_settings(
                        include=[str(Path.cwd()), str(Path(__file__).resolve().parent)],
                        **self.automatic_settings,
                    )
                )
                self.collector = collector
        except Exception as exc:
            self.dropped += 1
            print(f"[agprof] native profiler unavailable: {exc}", file=sys.stderr)
        return self

    def __exit__(self, *_exc):
        try:
            if self.collector is None:
                return
            records = self.collector._disable_auto_functions()
            self.dropped += self.collector._auto_dropped
            deadline = time.monotonic() + 5.0
            for start in range(0, len(records), 128):
                if self.failed or time.monotonic() > deadline:
                    self.dropped += len(records) - start
                    break
                self._report_samples(
                    [
                        {
                            "name": r[2],
                            "filename": r[3],
                            "lineno": r[4],
                            "perf_ns": r[5],
                            "duration_ns": r[6],
                            "tid": r[1],
                            "outcome": r[7],
                            "process_identity": self.process_identity,
                        }
                        for r in records[start : start + 128]
                    ]
                )
        finally:
            self.bridge._profiler = self._previous_profiler

    @contextmanager
    def span(self, name):
        if not self.enabled or self.failed:
            yield
            return
        span_id = uuid.uuid4().hex
        parent_id = self.stack[-1] if self.stack else None
        started_ts = time.time()
        self._report_span(
            {
                "name": name,
                "span_id": span_id,
                "parent": parent_id,
                "start_ts": started_ts,
                "attributes": {},
            }
        )
        self.stack.append(span_id)
        outcome = "unknown"
        try:
            yield
        except BaseException:
            outcome = "failure"
            raise
        else:
            outcome = "success"
        finally:
            self.stack.pop()
            self._report_span(
                {
                    "name": name,
                    "span_id": span_id,
                    "parent": parent_id,
                    "start_ts": started_ts,
                    "end_ts": time.time(),
                    "attributes": {"outcome": outcome},
                }
            )


def span(bridge, name):
    profiler = getattr(bridge, "_profiler", None)
    return profiler.span(name) if profiler is not None else nullcontext()


def profile_run(function):
    @wraps(function)
    def run(*args, **kwargs):
        bridge = kwargs.get("bridge")
        context = NativeProfiler(bridge) if hasattr(bridge, "_client") else nullcontext()
        with context as profiler:
            llm = args[2] if len(args) > 2 else kwargs.get("llm")
            if hasattr(llm, "_client"):
                llm._profiler = profiler if getattr(bridge, "_profiler", None) else None
            try:
                return function(*args, **kwargs)
            finally:
                if hasattr(llm, "_client"):
                    llm._profiler = None

    return run
