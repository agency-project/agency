"""Unit coverage for filesystem-only ZFS checkpoint transactions."""

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agency.configs.agconfig import agconfig
from agency.sandbox import checkpoint as cp
from agency.sandbox.base import agsandbox_backend
from agency.sandbox.container import _ContainerBackendBase


@pytest.fixture
def sandbox():
    cfg = agconfig()
    cfg.sandbox.backend = "podman"
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_zfs_parent = "tank/agency"
    state = {"running": True, "slots": 1}
    events = []
    backend = _ContainerBackendBase.__new__(_ContainerBackendBase)
    backend._agconfig = cfg
    backend._runtime = "podman"
    backend._name = "test-container"
    backend._checkpointer = cp.CowZfsCheckpoint()
    backend._normalized_checkpoint_manifest = Mock(return_value={"schema_version": 1})
    backend._container_running = lambda: state["running"]
    backend._register_prof_container = Mock()

    def stop():
        events.append("stop")
        state["running"] = False
        state["slots"] -= 1

    def start():
        events.append("start")
        state["running"] = True
        state["slots"] += 1

    backend._stop_image = Mock(side_effect=stop)
    backend._ensure_started_profiled = Mock(side_effect=start)
    backend._checkpoint_storage = SimpleNamespace(
        prepared=True,
        dataset="tank/agency/private",
        prepare=Mock(),
        capture_dataset=Mock(return_value="tank/agency/private"),
        snapshot=Mock(side_effect=lambda _ref: events.append("snapshot")),
        snapshot_stats=Mock(return_value={"zfs_snapshot_used_bytes": 0}),
        rollback=Mock(side_effect=lambda _ref: events.append("rollback")),
        delete=Mock(),
    )
    return backend, state, events


def test_checkpoint_stops_all_processes_before_snapshot_and_restore_starts_fresh(sandbox):
    backend, state, events = sandbox
    handle = backend.checkpoint()
    assert handle.backend == "cow_zfs"
    assert events == ["stop", "snapshot"]
    assert state == {"running": False, "slots": 0}

    backend.stop()  # Engine's commit+stop path must not checkpoint twice.
    assert events == ["stop", "snapshot"]
    backend._ensure_started()
    assert events == ["stop", "snapshot", "rollback", "start"]
    assert state == {"running": True, "slots": 1}


def test_repeated_checkpoints_keep_only_latest_snapshot(sandbox):
    backend, _state, _events = sandbox
    first = backend.checkpoint()
    backend._ensure_started()
    second = backend.checkpoint()
    backend._checkpoint_storage.delete.assert_called_once_with(first.reference)
    assert backend._checkpointer.latest == second


def test_snapshot_failure_does_not_publish_checkpoint_or_require_criu(sandbox):
    backend, state, events = sandbox
    backend._checkpoint_storage.snapshot.side_effect = RuntimeError("ZFS failed")
    with pytest.raises(RuntimeError, match="ZFS failed"):
        backend.checkpoint()
    assert backend._checkpointer.latest is None
    assert state == {"running": False, "slots": 0}
    assert events == ["stop"]


def test_profiler_has_filesystem_only_component_spans(sandbox, monkeypatch):
    backend, _state, _events = sandbox
    spans = []

    @contextmanager
    def span(name):
        spans.append(name)
        yield

    monkeypatch.setattr(cp.agprof, "span", span)
    backend._ensure_started_profiled = Mock()
    backend._checkpointer.restore(backend, backend.checkpoint())
    assert {"checkpoint.total", "checkpoint.quiesce", "checkpoint.fs_snapshot"} <= set(spans)
    assert {"restore.total", "restore.fs"} <= set(spans)
    assert not any("criu" in name or "process" in name for name in spans)


def test_zfs_calls_are_direct_snapshot_and_rollback(monkeypatch):
    storage = cp.ZfsRuntimeStorage.__new__(cp.ZfsRuntimeStorage)
    storage.dataset = "tank/private"
    command = Mock()
    monkeypatch.setattr(cp, "_command", command)
    reference = "tank/private@agency-123"
    storage.snapshot(reference)
    storage.rollback(reference)
    storage.delete(reference)
    assert [call.args[0] for call in command.call_args_list] == [
        ["zfs", "snapshot", reference],
        ["zfs", "rollback", reference],
        ["zfs", "destroy", reference],
    ]


def test_docker_uses_its_container_owned_zfs_dataset():
    storage = cp.ZfsRuntimeStorage.__new__(cp.ZfsRuntimeStorage)
    storage.runtime = "docker"
    storage.parent = "tank/docker"
    storage.dataset = None
    backend = SimpleNamespace(
        _name="sandbox",
        _run=Mock(
            return_value=SimpleNamespace(
                stdout=json.dumps(
                    [{"GraphDriver": {"Data": {"Dataset": "tank/docker/container-layer"}}}]
                ).encode()
            )
        ),
    )
    assert storage.capture_dataset(backend) == "tank/docker/container-layer"


def test_docker_rejects_dataset_outside_configured_parent():
    storage = cp.ZfsRuntimeStorage.__new__(cp.ZfsRuntimeStorage)
    storage.runtime = "docker"
    storage.parent = "tank/docker"
    storage.dataset = None
    backend = SimpleNamespace(
        _name="sandbox",
        _run=Mock(
            return_value=SimpleNamespace(
                stdout=b'[{"GraphDriver":{"Data":{"Dataset":"tank/unrelated/layer"}}}]'
            )
        ),
    )
    with pytest.raises(cp.CheckpointCapabilityError, match="container-owned"):
        storage.capture_dataset(backend)


def test_config_requires_zfs_parent_for_cow_backend():
    cfg = agconfig()
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    with pytest.raises(ValueError, match="checkpoint_zfs_parent"):
        agsandbox_backend()._validate_config(cfg)


def test_benchmark_accepts_docker_and_podman_cow_layouts():
    from benchmarks.checkpoint_size_microbenchmark.runner import validate_checkpoint_comparison

    for runtime in ("docker", "podman"):
        validate_checkpoint_comparison(
            {
                "checkpoint_backend": "cow_zfs",
                "backend": runtime,
                "checkpoint_zfs_parent": "tank/runtime",
            }
        )


def test_docker_destroy_keeps_private_runtime_available_through_container_removal():
    backend = _ContainerBackendBase.__new__(_ContainerBackendBase)
    backend._destroyed = False
    backend._runtime = "docker"
    backend._checkpoint_storage = SimpleNamespace(
        prepared=True,
        dataset="tank/docker/writable",
        destroy=Mock(),
    )
    events = []

    def remove():
        assert backend._checkpoint_storage.prepared is True
        events.append("remove")

    backend.rm_container = Mock(side_effect=remove)
    backend.destroy()

    backend._checkpoint_storage.destroy.assert_called_once_with()
    backend.rm_container.assert_called_once_with()
    assert events == ["remove"]
    assert backend._checkpoint_storage.prepared is False
    assert backend._checkpoint_storage.dataset is None
    assert backend._destroyed is True
