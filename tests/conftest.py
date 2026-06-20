"""Reset mutable agent class-level and module-level state between tests."""
import sys
import pytest
import agency.agskill  # ensure registered in sys.modules
_agskill_module = sys.modules["agency.agskill"]
from agency.agent import agent, _allocated_agnames, _noun_counters, _live_agents


@pytest.fixture(autouse=True, scope="session")
def _stop_gpu_markers_for_tests():
    """Stop GPU marker processes before any test runs.

    conftest.py imports agency.agent at module level, which triggers
    agresource_pool = agResourcePool(mark_gpus=True) and starts one marker
    subprocess per GPU.  The markers call cuCtxCreate via libcuda.so.1; the
    NVIDIA container runtime then spawns helper processes inside any Docker
    container that uses --gpus all.  Those helpers appear in /proc but are not
    captured in _baseline_pids (they start after the snapshot), causing
    get_live_pids() to report phantom live PIDs and breaking PID-tracking tests.

    Stopping the markers before the first container is created prevents those
    helpers from appearing.  TestGpuMarkers creates its own short-lived pools
    (mark_gpus=True) that are stopped within each test.
    """
    agent.agresource_pool._stop_gpu_markers()
    yield


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Set stream-batch delay to zero so tests don't sleep 100 ms per LLM call."""
    monkeypatch.setattr(_agskill_module, "_BATCH_INTERVAL_S", 0.0)


@pytest.fixture(autouse=True)
def reset_agent_state():
    saved = {
        "ping_interval_s": agent.ping_interval_s,
        "poll_interval_s": agent.poll_interval_s,
        "max_outer_iters": agent.max_outer_iters,
        "log_dir":         agent.log_dir,
        "output_dir":      agent.output_dir,
    }
    yield
    agent.ping_interval_s = saved["ping_interval_s"]
    agent.poll_interval_s = saved["poll_interval_s"]
    agent.max_outer_iters = saved["max_outer_iters"]
    agent.log_dir         = saved["log_dir"]
    agent.output_dir      = saved["output_dir"]
    # Reset module-level name registry so tests don't bleed agnames into each other
    _allocated_agnames.clear()
    _noun_counters.clear()
    # WeakSet clears itself as objects die; force a GC pass to help along
    import gc
    gc.collect()
