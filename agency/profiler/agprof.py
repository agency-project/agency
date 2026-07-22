"""agprof — profiling switch for the framework's built-in span annotations.

The framework's hot paths (skill runs, LLM calls, tool dispatch, sandbox ops,
agmap tasks) are permanently annotated with ``agprof.span(name)`` calls. With no
active session (the default) every one of them returns a shared no-op context
manager — one module-global load and an ``is None`` check; torch is never
imported. Applications opt in with one line (or one env var) and the framework
acts accordingly:

    from agency import agprof

    with agprof.session(run_dir / "tb_trace"):   # torch.profiler lifecycle
        team.run()

    # or, zero application changes:
    #   AGENCY_PROFILE=1 [AGENCY_PROFILE_DIR=path] python app.py

Today the only backend is torch.profiler (kineto): ``span()`` maps to
``torch.profiler.record_function`` and the session wraps ``profile(...)`` with
``profile_all_threads=True`` — REQUIRED, because the framework runs skills and
agmap tasks on plain ``threading.Thread`` daemons, and without a global observer
kineto silently drops every span opened on them. Traces land as
``*.pt.trace.json`` (TensorBoard torch-tb-profiler plugin / ui.perfetto.dev).
"""
from __future__ import annotations

import atexit
import itertools
import os
import threading
from contextlib import contextmanager, nullcontext

_NULL = nullcontext()

_session = None            # None = profiling off (the fast path checks only this)
_profiler = None           # the live torch.profiler.profile object, if any
_state_lock = threading.Lock()

_counters: "dict[str, itertools.count]" = {}
_counters_lock = threading.Lock()


class _TorchSession:
    """Maps span() to torch.profiler.record_function."""

    __slots__ = ("_record_function",)

    def __init__(self, record_function) -> None:
        self._record_function = record_function

    def span(self, name: str):
        return self._record_function(name)


def enabled() -> bool:
    """True while a profiling session is active."""
    return _session is not None


def span(name: str):
    """A timed, named interval on the current thread.

    No-op (a shared ``nullcontext``) unless a session is active. Nesting on the
    same thread produces parent/child spans in the trace.
    """
    s = _session
    if s is None:
        return _NULL
    return s.span(name)


_PR_SET_NAME = 15  # linux prctl option
_libc = None


def thread_name(name: str) -> None:
    """Set the OS-level thread name — what kineto records as the trace lane
    label (Python's ``Thread.name`` never reaches the pthread name). Linux
    truncates to 15 chars. No-op when profiling is off or prctl is unavailable.
    """
    if _session is None:
        return
    global _libc
    try:
        if _libc is None:
            import ctypes

            _libc = ctypes.CDLL(None, use_errno=True)
        _libc.prctl(_PR_SET_NAME, name[:15].encode(), 0, 0, 0)
    except Exception:
        pass


def next_index(key: str = "run") -> int:
    """Monotonic per-key counter for span labels (run0, run1, ...; agmap[0], ...)."""
    with _counters_lock:
        c = _counters.get(key)
        if c is None:
            c = _counters[key] = itertools.count()
    return next(c)


def start(out_dir=None, *, all_threads: bool = True, worker_name: "str | None" = None):
    """Start a profiling session. Prefer the ``session()`` context manager.

    *out_dir*: directory for the TensorBoard trace (``None`` = no trace file;
    the returned profiler object still supports ``key_averages()``).
    """
    global _session, _profiler
    # Imported lazily: the framework must not require torch unless profiling.
    from torch.profiler import (
        ProfilerActivity,
        profile,
        record_function,
        tensorboard_trace_handler,
    )
    from torch._C._profiler import _ExperimentalConfig

    with _state_lock:
        if _session is not None:
            raise RuntimeError("agprof: a profiling session is already active")
        handler = (
            tensorboard_trace_handler(str(out_dir), worker_name=worker_name)
            if out_dir is not None
            else None
        )
        prof = profile(
            activities=[ProfilerActivity.CPU],
            experimental_config=_ExperimentalConfig(profile_all_threads=all_threads),
            on_trace_ready=handler,
        )
        prof.start()
        _profiler = prof
        _session = _TorchSession(record_function)
    return prof


def stop():
    """Stop the active session (writing the trace). Returns the profiler, or
    None if no session was active."""
    global _session, _profiler
    with _state_lock:
        if _session is None:
            return None
        prof = _profiler
        _session = None
        _profiler = None
    prof.stop()
    return prof


@contextmanager
def session(out_dir=None, *, all_threads: bool = True, worker_name: "str | None" = None):
    """Profile everything inside the block; write the trace on exit.

    Yields the torch profiler object — after the block exits,
    ``prof.key_averages().table(...)`` gives the per-span summary.
    """
    prof = start(out_dir, all_threads=all_threads, worker_name=worker_name)
    try:
        yield prof
    finally:
        stop()


def _maybe_autostart() -> None:
    """AGENCY_PROFILE=1 profiles an unmodified application for its whole life."""
    if os.environ.get("AGENCY_PROFILE", "").lower() not in ("1", "true"):
        return
    out_dir = os.environ.get("AGENCY_PROFILE_DIR", "agprof_trace")
    start(out_dir)
    atexit.register(stop)


_maybe_autostart()
