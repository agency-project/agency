"""agency.profiler — the framework's profiling subsystem.

`agprof` is the profiling switch + span API (torch.profiler backend today; see
README.md in this directory for usage and the span glossary.

Import styles supported:

    from agency import agprof              # re-exported at package root
    from agency.profiler import agprof     # explicit
    from agency.profiler import span, session, enabled   # direct API
"""

from . import agprof
from .agprof import (
    enabled,
    next_index,
    session,
    span,
    start,
    stop,
    summary_table,
    thread_name,
)

__all__ = [
    "agprof",
    "enabled",
    "next_index",
    "session",
    "span",
    "start",
    "stop",
    "summary_table",
    "thread_name",
]
