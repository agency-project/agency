"""CPU caps must also work on hosts using the unified cgroup hierarchy."""

import os
from pathlib import Path

import pytest

from agency.sandbox.container import _ContainerBackendBase


@pytest.mark.parametrize(
    "legacy_quota,controllers,expected",
    [
        (True, None, True),
        (False, "cpuset cpu io memory pids", True),
        (False, "cpuset io memory pids", False),
        (False, None, False),
    ],
)
def test_cpu_quota_support_detects_legacy_and_unified_controllers(
    monkeypatch, legacy_quota, controllers, expected
):
    def read_controllers(_path, *args, **kwargs):
        if controllers is None:
            raise FileNotFoundError
        return controllers

    with monkeypatch.context() as patch:
        patch.setattr(os.path, "exists", lambda path: legacy_quota)
        patch.setattr(Path, "read_text", read_controllers)
        assert _ContainerBackendBase._cfs_supported() is expected
