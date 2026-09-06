"""Attempt-scoped telemetry for native Python, without importing the host package.

Semantic boundaries are acknowledged before the next LLM/tool request so the
host can parent that request correctly. Automatic calls are bounded in memory
and transferred in batches at teardown; a lost batch is reported, never retried
as if it were a new event.
"""

from __future__ import annotations

import importlib.util
import time
import sys
import uuid
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path


class NativeProfiler:
    def __init__(self, bridge):
        self.bridge = bridge
        self.config = {}
        self.offset = 0
        self.uncertainty = 0
        self.dropped = 0
        self.stack = []
        self.collector = None
        self.failed = False

    def request(self, path, payload):
        response = self.bridge._client.post(
            "/agprof/" + path,
            json=payload,
            headers={"Authorization": f"Bearer {self.bridge.token}"},
            timeout=2.0,
        )
        response.raise_for_status()
        return response.json()

    def send(self, events):
        try:
            result = self.request(
                "events",
                {
                    "session_id": self.config["session_id"],
                    "clock_uncertainty_ns": self.uncertainty,
                    "events": events,
                    "dropped": self.dropped,
                },
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "profile ingest rejected events"))
            self.dropped = 0
            return True
        except Exception as exc:
            self.failed = True
            self.dropped += len(events)
            print(f"[agprof] native telemetry failed: {exc}", file=sys.stderr)
            return False

    def __enter__(self):
        try:
            started = time.perf_counter_ns()
            self.config = self.request("config", {})
            ended = time.perf_counter_ns()
            if not self.config.get("enabled"):
                return self
            self.offset = self.config["host_perf_ns"] - (started + ended) // 2
            self.uncertainty = (ended - started) // 2
            self.bridge._profiler = self
            settings = self.config.get("automatic")
            if settings:
                # Reuse Tony's collector while avoiding agency.__init__ and its
                # host-only dependencies and process-level environment startup.
                path = Path(__file__).resolve().parents[1] / "observability/profiler/agprof.py"
                spec = importlib.util.spec_from_file_location("_agency_container_profiler", path)
                collector = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(collector)
                collector._enable_auto_functions(
                    collector._make_auto_settings(
                        include=[str(Path.cwd()), str(Path(__file__).resolve().parent)], **settings
                    )
                )
                self.collector = collector
        except Exception as exc:
            self.dropped += 1
            print(f"[agprof] native collector unavailable: {exc}", file=sys.stderr)
        return self

    def __exit__(self, *_exc):
        try:
            if self.collector is not None:
                records = self.collector._disable_auto_functions()
                self.dropped += self.collector._auto_dropped
                deadline = time.monotonic() + 5.0
                for start in range(0, len(records), 128):
                    if self.failed or time.monotonic() > deadline:
                        self.dropped += len(records) - start
                        break
                    self.send(
                        [
                            {
                                "kind": "automatic",
                                "name": r[2],
                                "filename": r[3],
                                "lineno": r[4],
                                "perf_ns": r[5] + self.offset,
                                "duration_ns": r[6],
                                "tid": r[1],
                                "outcome": r[7],
                            }
                            for r in records[start : start + 128]
                        ]
                    )
            if self.config.get("enabled") and self.dropped:
                self.send([])
        finally:
            self.bridge._profiler = None

    @contextmanager
    def span(self, name):
        if self.failed:
            yield
            return
        identifier = uuid.uuid4().hex
        accepted = self.send(
            [
                {
                    "kind": "start",
                    "name": name,
                    "id": identifier,
                    "parent_id": self.stack[-1] if self.stack else None,
                    "perf_ns": time.perf_counter_ns() + self.offset,
                }
            ]
        )
        if accepted:
            self.stack.append(identifier)
        outcome = "unknown"
        try:
            yield
        except BaseException:
            outcome = "failure"
            raise
        finally:
            if accepted:
                self.stack.pop()
                self.send(
                    [
                        {
                            "kind": "end",
                            "name": name,
                            "id": identifier,
                            "outcome": outcome,
                            "perf_ns": time.perf_counter_ns() + self.offset,
                        }
                    ]
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
