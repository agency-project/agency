"""Startup cache behavior without downloading or compiling upstream in tests."""

import json
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import Mock

import pytest

from agency.observability.agwebui import build_perfetto as builder


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "DEST", tmp_path / "assets")
    monkeypatch.setattr(builder, "SOURCE", tmp_path / "source")

    def complete(**kwargs):
        builder.DEST.mkdir(exist_ok=True)
        (builder.DEST / "index.html").write_text("Perfetto")
        (builder.DEST / "engine.wasm").write_bytes(b"wasm")
        (builder.DEST / "agency-build.json").write_text(
            json.dumps(
                {
                    "fingerprint": builder._fingerprint(),
                    "files": ["index.html", "engine.wasm"],
                }
            )
        )

    build = Mock(side_effect=complete)
    monkeypatch.setattr(builder, "_build_viewer", build)
    return build, complete


def test_build_once_and_reuse_without_subprocesses(cache, monkeypatch):
    build, _ = cache
    builder.ensure_viewer(skip_deps=True)
    build.assert_called_once_with(skip_deps=True)
    monkeypatch.setattr(
        builder.subprocess, "run", Mock(side_effect=AssertionError("offline cache"))
    )
    builder.ensure_viewer()
    assert build.call_count == 1


@pytest.mark.parametrize(
    "change", ["revision", "builder", "missing_asset", "bad_manifest", "force"]
)
def test_rebuilds_stale_or_incomplete_assets(cache, monkeypatch, change):
    build, complete = cache
    complete()
    if change == "revision":
        monkeypatch.setattr(builder, "REVISION", "new-revision")
    elif change == "builder":
        monkeypatch.setattr(builder, "_fingerprint", lambda: "new-builder")
    elif change == "missing_asset":
        (builder.DEST / "engine.wasm").unlink()
    elif change == "bad_manifest":
        (builder.DEST / "agency-build.json").write_text("incomplete")
    builder.ensure_viewer(force=change == "force")
    build.assert_called_once_with(skip_deps=False)
    assert builder.viewer_is_current()


def test_failed_build_can_be_retried(cache):
    build, complete = cache
    build.side_effect = OSError("network unavailable")
    with pytest.raises(RuntimeError, match="web UI was not started"):
        builder.ensure_viewer()
    assert not builder.viewer_is_current()
    build.side_effect = complete
    builder.ensure_viewer()
    assert builder.viewer_is_current()


def test_presentation_source_invalidates_cache(cache, monkeypatch, tmp_path):
    build, complete = cache
    complete()
    source = tmp_path / "presentation.ts"
    source.write_text("new plugin")
    monkeypatch.setattr(builder, "PRESENTATION", source)
    builder.ensure_viewer()
    build.assert_called_once()


def test_concurrent_starts_share_one_build(cache):
    build, complete = cache
    entered, release = threading.Event(), threading.Event()

    def slow_build(**kwargs):
        entered.set()
        assert release.wait(5)
        complete()

    build.side_effect = slow_build
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(builder.ensure_viewer)
        assert entered.wait(5)
        second = executor.submit(builder.ensure_viewer)
        release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert build.call_count == 1


def test_build_failure_prevents_server_and_application_start(monkeypatch):
    import socket
    from unittest.mock import MagicMock
    from agency.observability.agwebui import agwebui

    connection = MagicMock()
    connection.__enter__.return_value.connect_ex.return_value = 1
    monkeypatch.setattr(socket, "socket", Mock(return_value=connection))
    monkeypatch.setattr(builder, "ensure_viewer", Mock(side_effect=RuntimeError("build failed")))
    launch = Mock()
    monkeypatch.setattr(builder.subprocess, "Popen", launch)
    application = Mock()
    with pytest.raises(RuntimeError, match="build failed"):
        agwebui.run(application, linger=False)
    launch.assert_not_called()
    application.assert_not_called()
