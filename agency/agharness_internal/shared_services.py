"""Process-wide lifecycle coordination for shared harness services."""

from __future__ import annotations

import threading
import time
import weakref

_lock = threading.Lock()
_services: "weakref.WeakSet[object]" = weakref.WeakSet()


def register_shared_service(service: object) -> None:
    """Include a process-wide service in ordered shutdown draining."""
    with _lock:
        _services.add(service)


def drain_shared_services(timeout_s: "float | None" = None) -> bool:
    """Wait for in-flight work without tearing down shared service instances."""
    with _lock:
        services = list(_services)
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    drained = True
    for service in services:
        drain = getattr(service, "drain", None)
        if drain is None:
            continue
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if drain(timeout_s=remaining) is False:
            drained = False
    return drained


__all__ = ["drain_shared_services", "register_shared_service"]
