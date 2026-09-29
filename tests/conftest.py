"""Reset mutable agent class-level and module-level state between tests."""

import os
import subprocess
import sys

import pytest
import agency.utils.agutil as _agutil_module
from agency.agent import agent
from agency.agname import agname as _agname

CONTAINER_BACKEND = os.environ.get("AGENCY_TEST_CONTAINER_BACKEND", "docker")


@pytest.fixture(scope="module")
def golden_image():
    image = os.environ.get("AGENCY_TEST_HARNESS_IMAGE", "docker.io/library/python:3.12-slim")
    problem = None
    if not sys.platform.startswith("linux"):
        problem = "golden profiling and real external harnesses require Linux"
    else:
        try:
            subprocess.run([CONTAINER_BACKEND, "info"], capture_output=True, check=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            problem = f"golden execution requires {CONTAINER_BACKEND}: {exc}"
        else:
            try:
                subprocess.run(
                    [CONTAINER_BACKEND, "image", "inspect", image],
                    capture_output=True,
                    check=True,
                    timeout=15,
                )
            except subprocess.SubprocessError:
                try:
                    subprocess.run(
                        [CONTAINER_BACKEND, "pull", image],
                        capture_output=True,
                        check=True,
                        timeout=120,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    problem = f"golden execution requires the local image {image}: {exc}"
    if problem:
        if os.environ.get("CI") or os.environ.get("AGENCY_TEST_EXTERNAL_HARNESSES") == "1":
            pytest.fail(problem)
        pytest.skip(problem)
    return image


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    """Set stream-batch delay to zero so tests don't sleep 100 ms per LLM call."""
    monkeypatch.setattr(_agutil_module, "_BATCH_INTERVAL_S", 0.0)


@pytest.fixture(autouse=True)
def reset_agent_state():
    saved = {
        "log_dir": agent.log_dir,
        "output_dir": agent.output_dir,
    }
    yield
    from agency.orchestrator.orchestrator import _reset_orchestrator_for_tests

    _reset_orchestrator_for_tests()
    agent.log_dir = saved["log_dir"]
    agent.output_dir = saved["output_dir"]
    # Reset module-level name registry so tests don't bleed agnames into each other
    _agname._allocated.clear()
    _agname._noun_counters.clear()
    # WeakSet clears itself as objects die; force a GC pass to help along
    import gc

    gc.collect()
