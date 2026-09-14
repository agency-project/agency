"""Tests for agsandbox container naming / run isolation.

Split out from the old test_agterm.py, whose agterm-specific coverage was
removed along with agterm.py itself -- these tests were never actually about
agterm, just co-located in the same file."""

from __future__ import annotations

import re
import subprocess
import sys
import pytest
from unittest.mock import MagicMock, patch


def _worker_get_run_id():
    """Top-level so ProcessPoolExecutor can pickle it."""
    from agency.sandbox.agsandbox import _RUN_ID

    return _RUN_ID


def _naming_sandbox(agname: str):
    """Construct a sandbox facade without selecting a live runtime.

    These tests inspect names only; requiring Docker or Podman would turn a
    pure identity test into an unrelated integration test.
    """
    from agency.sandbox.agsandbox import agSandbox

    with patch("agency.sandbox.base.agsandbox_backend.for_config", return_value=MagicMock()):
        return agSandbox(agname)


class TestSandboxNaming:
    def test_container_name_includes_run_id(self):
        from agency.sandbox.agsandbox import _RUN_ID

        sb = _naming_sandbox("myagent")
        assert _RUN_ID in sb._name

    def test_container_name_includes_agname(self):
        sb = _naming_sandbox("myagent")
        assert "myagent" in sb._name

    def test_container_name_format(self):
        """The agname component is deduplicated (see agsandbox.py's
        __init__) -- "myagent" becomes "sandbox_myagent_XXXX" via the
        shared agname registry, so the exact suffix isn't predictable
        (it depends on how many times this base has already been claimed
        elsewhere in this same test process), only the overall shape is."""
        from agency.sandbox.agsandbox import _RUN_ID

        sb = _naming_sandbox("myagent")
        assert re.fullmatch(rf"sandbox-{_RUN_ID}-sandbox_myagent_[0-9a-z]{{4}}", sb._name), sb._name

    def test_two_sandboxes_same_agname_get_deduplicated_names(self):
        """Every agSandbox construction claims its own unique name from the
        shared agname registry (see agsandbox.py's __init__) -- passing the
        same literal agname twice must NOT collide into the same container
        identity. Anyone who genuinely needs to reference an existing
        sandbox must keep the object/backend itself around, not re-pass its
        name string (see test_agsandbox.py's
        test_ensure_started_reuses_running_container for the supported way
        to do that)."""
        sb1 = _naming_sandbox("shared-agent")
        sb2 = _naming_sandbox("shared-agent")
        assert sb1._name != sb2._name

    def test_two_sandboxes_different_agnames_differ(self):
        sb1 = _naming_sandbox("agent-alpha")
        sb2 = _naming_sandbox("agent-beta")
        assert sb1._name != sb2._name

    def test_run_id_is_run_scoped_not_pid(self):
        """_RUN_ID must not be the current PID (we switched to UUID)."""
        import os
        from agency.sandbox.agsandbox import _RUN_ID

        assert str(os.getpid()) not in _RUN_ID

    def test_run_id_format(self):
        """_RUN_ID must be 'r' followed by 8 hex chars."""
        from agency.sandbox.agsandbox import _RUN_ID

        assert re.fullmatch(r"r[0-9a-f]{8}", _RUN_ID), f"unexpected _RUN_ID: {_RUN_ID!r}"


class TestRunIsolation:
    def test_separate_imports_produce_different_run_ids(self):
        """Two separate process invocations must never share a _RUN_ID."""
        script = "from agency.sandbox.agsandbox import _RUN_ID; print(_RUN_ID)"
        r1 = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        r2 = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        id1 = r1.stdout.strip()
        id2 = r2.stdout.strip()
        assert id1 and id2
        assert id1 != id2, (
            f"Two separate processes got the same _RUN_ID: {id1!r} — "
            "UUID generation is broken or _RUN_ID is PID-based"
        )

    def test_separate_imports_produce_different_container_names(self):
        """Container names from two separate runs must not collide."""
        script = """
from unittest.mock import MagicMock, patch
from agency.sandbox.agsandbox import agSandbox

with patch(
    'agency.sandbox.base.agsandbox_backend.for_config',
    return_value=MagicMock(),
):
    sb = agSandbox('DataGen_0000')
print(sb._name)
"""
        r1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        name1 = r1.stdout.strip()
        name2 = r2.stdout.strip()
        assert name1 and name2
        assert name1 != name2, (
            f"Same container name across two runs: {name1!r} — cross-run isolation is broken"
        )

    def test_checkpoint_image_tag_differs_across_runs(self):
        """Lifecycle image tags must be run-scoped to prevent cross-run clobber."""
        script = """
from types import SimpleNamespace
from unittest.mock import patch
from agency.sandbox.agsandbox import agSandbox

def fake_backend(*args, **kwargs):
    tag = f"{kwargs['name']}-checkpoint"
    return SimpleNamespace(_lifecycle_tag=lambda: tag)

with patch(
    'agency.sandbox.base.agsandbox_backend.for_config',
    side_effect=fake_backend,
):
    sb = agSandbox('DataGen_0000')
print(sb._backend._lifecycle_tag())
"""
        r1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        tag1 = r1.stdout.strip()
        tag2 = r2.stdout.strip()
        assert tag1 and tag2
        assert tag1 != tag2, (
            f"Same lifecycle image tag across two runs: {tag1!r} — "
            "a new run would clobber the previous run's checkpoint"
        )

    def test_forked_worker_process_inherits_run_id(self):
        """A forked worker inherits the parent process's module-level run ID."""
        import concurrent.futures
        import multiprocessing
        from agency.sandbox.agsandbox import _RUN_ID

        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("fork start method is unavailable")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            worker_id = pool.submit(_worker_get_run_id).result(timeout=30)

        assert worker_id == _RUN_ID, (
            f"Worker got _RUN_ID={worker_id!r}, main has {_RUN_ID!r} — "
            "workers must inherit the parent's run ID"
        )
