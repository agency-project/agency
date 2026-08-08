"""agency.profiler — the framework's profiling subsystem.

`agprof` is the profiling switch + span API (OpenTelemetry backend; see the
module documentation for usage).

Import styles supported:

    from agency import agprof              # re-exported at package root
    from agency.profiler import agprof     # explicit
    from agency.profiler import span, session, workload, enabled  # direct API
"""

from . import agprof
from .agprof import (
    annotate,
    cancel_external_span,
    enabled,
    interrupt_external_span,
    next_index,
    profile_scope,
    session,
    spawn_traced,
    span,
    start,
    start_external_span,
    stop,
    summary_metrics,
    summary_table,
    thread_name,
    workload,
)

__all__ = [
    "agprof",
    "annotate",
    "cancel_external_span",
    "enabled",
    "interrupt_external_span",
    "next_index",
    "profile_scope",
    "session",
    "spawn_traced",
    "span",
    "start",
    "start_external_span",
    "stop",
    "summary_metrics",
    "summary_table",
    "thread_name",
    "workload",
]
