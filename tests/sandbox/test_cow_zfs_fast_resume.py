from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.sandbox import checkpoint as cp
from agency.sandbox.base import agsandbox_backend
from agency.sandbox.container import _ContainerBackendBase


def _backend():
    cfg = agconfig()
    cfg.sandbox.backend = "podman"
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_zfs_parent = "tank/agency"
    cfg.sandbox.checkpoint_fast_resume = True
    state = {"running": True}
    backend = _ContainerBackendBase.__new__(_ContainerBackendBase)
    backend._agconfig = cfg
    backend._runtime = "podman"
    backend._name = "sandbox"
    backend._gpu_count_requested = 0
    backend._checkpointer = cp.CowZfsCheckpoint()
    backend._normalized_checkpoint_manifest = Mock(return_value={"schema_version": 1})
    backend._container_running = lambda: state["running"]
    backend._stop_image = Mock(side_effect=lambda: state.update(running=False))
    backend._ensure_started_profiled = Mock(side_effect=lambda: state.update(running=True))
    backend._register_prof_container = Mock()
    backend._checkpoint_storage = SimpleNamespace(
        prepared=True,
        dataset="tank/agency/private",
        prepare=Mock(),
        capture_dataset=Mock(return_value="tank/agency/private"),
        snapshot=Mock(),
        snapshot_stats=Mock(return_value={"zfs_snapshot_used_bytes": 0}),
        rollback=Mock(),
        delete=Mock(),
    )
    return backend, state


def test_fast_resume_is_optional_and_requires_cow_zfs():
    cfg = agconfig()
    cfg.sandbox.checkpoint_fast_resume = True
    with pytest.raises(ValueError, match="requires checkpoint_backend='cow_zfs'"):
        agsandbox_backend()._validate_config(cfg)


def test_dump_and_restore_criu_stats_use_distinct_keys():
    raw = '{"podman_checkpoint_duration": 123}'

    assert cp._criu_stats(raw) == {"podman_criu_stats": {"podman_checkpoint_duration": 123}}
    assert cp._criu_stats(raw, "podman_criu_restore_stats") == {
        "podman_criu_restore_stats": {"podman_checkpoint_duration": 123}
    }


def test_successful_process_checkpoint_remains_an_acceleration_layer(monkeypatch):
    backend, state = _backend()
    artifact = {"name": "agency-test", "daemon_count": 1, "live_pty": True}

    def dump(_sandbox):
        state["running"] = False
        return artifact

    restore = Mock(side_effect=lambda _sandbox, _artifact: state.update(running=True))
    monkeypatch.setattr(backend._checkpointer, "_runtime_checkpoint", dump)
    monkeypatch.setattr(backend._checkpointer, "_runtime_restore", restore)

    handle = backend.checkpoint()
    assert handle.stats["fast_resume_available"] is True
    assert handle.stats["fast_resume_live_pty"] is True
    assert backend._checkpoint_storage.snapshot.call_count == 1

    backend._checkpointer.restore(backend, handle)
    restore.assert_called_once_with(backend, artifact)
    assert handle.stats["fast_resume_used"] is True
    assert state["running"] is True


def test_process_dump_failure_still_publishes_filesystem_checkpoint(monkeypatch):
    backend, state = _backend()
    monkeypatch.setattr(
        backend._checkpointer,
        "_runtime_checkpoint",
        Mock(side_effect=cp.CheckpointCapabilityError("CRIU unavailable")),
    )

    handle = backend.checkpoint()

    assert handle.stats["fast_resume_available"] is False
    assert handle.stats["fast_resume_checkpoint_error"] == "CheckpointCapabilityError"
    backend._stop_image.assert_called_once_with()
    assert state["running"] is False


def test_process_restore_failure_rolls_back_again_for_fresh_start(monkeypatch):
    backend, state = _backend()
    artifact = {"name": "agency-test", "daemon_count": 1, "live_pty": True}

    def dump(_sandbox):
        state["running"] = False
        return artifact

    monkeypatch.setattr(backend._checkpointer, "_runtime_checkpoint", dump)
    handle = backend.checkpoint()
    monkeypatch.setattr(
        backend._checkpointer,
        "_runtime_restore",
        Mock(side_effect=RuntimeError("restore rejected")),
    )

    backend._checkpointer.restore(backend, handle)

    assert backend._checkpoint_storage.rollback.call_count == 2
    assert handle.stats["fast_resume_used"] is False
    assert handle.stats["fast_resume_restore_error"] == "RuntimeError"
    assert backend._checkpointer.fast_artifact is None
    assert state["running"] is False
