"""Host-side unit tests for `native.py`'s `_NativeBackend.execute()`
add_tools/replace_tools handling -- specifically the cloudpickle fail-fast
check (Phase 0.F). Deliberately separate from both test_native_loop_fast.py
(exercises `_run_react_loop()` inside the entrypoint via `NativeLoopHarness`)
and test_native.py (real, docker-backed only) -- this file's own tests run
before any entrypoint launch or docker interaction is even attempted, so
neither of those seams fit.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from agency.agdata import agdata, agerror
from agency.agtool import agtool
from agency.agharness_internal.agharness_backends import native as native_mod
from agency.agharness_internal.agharness_backends.native import _NativeBackend


def test_unpicklable_closure_fails_fast_without_docker():
    """A closure capturing a live, inherently unpicklable object (a
    `threading.Lock`) must fail right here, host-side, before
    `_NativeBackend.execute()` does anything else -- no entrypoint launch,
    no docker interaction, no socket round-trip. This is the "unpicklable-
    closure error handling" named in this phase's own plan."""
    lock = threading.Lock()

    def bad_fn(arg):
        with lock:
            return agdata(ok=True)

    tool = agtool("bad_tool", "captures an unpicklable lock", bad_fn)

    skill = MagicMock()
    skill.replace_tools = None
    skill.add_tools = [tool]

    from agency.agconfig import agConfig

    backend = _NativeBackend(agConfig())
    result, ctx, delta = backend.execute(
        MagicMock(), MagicMock(), agdata(), None, skill=skill,
    )

    assert isinstance(result, agerror)
    assert "bad_tool" in result.error


def test_picklable_closure_is_not_rejected():
    """A self-contained, picklable closure (no captured host-only state)
    must NOT trip the fail-fast check -- proves the check is specific to
    unpicklable closures, not a blanket rejection of add_tools/replace_tools
    the way the pre-0.F code had. Patches `_ensure_entrypoint` (the next
    thing `execute()` calls after the fail-fast check) to raise a distinct
    marker exception, so seeing THAT exception -- not just "some exception"
    -- proves execution actually got past the pickling check."""
    import cloudpickle

    def good_fn(arg):
        return agdata(doubled=arg.n * 2)

    # Sanity: this closure really is picklable -- if this ever stops being
    # true (e.g. someone adds a captured host object above), the test
    # should fail loudly here, not silently pass for the wrong reason.
    cloudpickle.dumps(good_fn)

    tool = agtool("double", "doubles a number", good_fn)

    skill = MagicMock()
    skill.replace_tools = None
    skill.add_tools = [tool]

    class _PastFailFastMarker(Exception):
        pass

    from agency.agconfig import agConfig

    with patch.object(native_mod, "_ensure_entrypoint", side_effect=_PastFailFastMarker()):
        with pytest.raises(_PastFailFastMarker):
            _NativeBackend(agConfig()).execute(MagicMock(), MagicMock(), agdata(), None, skill=skill)
