"""Regression coverage for tandem_harness's NativeProfiler nesting.

The tandem harness reuses one `run_react_loop` (and therefore one
`NativeProfiler`-per-call) for both the supervisor's own turn-taking and
each worker segment nested inside it via `smart_tool`. Both levels share
the same bridge object, so `NativeProfiler.__enter__`/`__exit__` must
save/restore `bridge._profiler` like a stack rather than unconditionally
setting it -- otherwise a worker segment finishing mid-supervisor-turn
would clobber the still-in-flight supervisor profiler back to None."""

from types import SimpleNamespace

from agency.tandem_harness.profiling import NativeProfiler


def _bridge():
    return SimpleNamespace(
        profiler_settings=lambda: {"enabled": True},
        record_profiler_span=lambda payload: {"ok": True},
        record_profiler_samples=lambda samples: {"ok": True, "rejected": 0},
    )


def test_nested_profiler_restores_outer_bridge_profiler_on_exit():
    bridge = _bridge()
    with NativeProfiler(bridge) as outer:
        assert bridge._profiler is outer
        with NativeProfiler(bridge) as inner:
            assert bridge._profiler is inner
        assert bridge._profiler is outer
    assert bridge._profiler is None


def test_single_level_profiler_still_clears_bridge_profiler_on_exit():
    bridge = _bridge()
    with NativeProfiler(bridge) as native:
        assert bridge._profiler is native
    assert bridge._profiler is None
