import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from agency import cli
from agency.configs.agconfig import agconfig, sandboxconfig
from agency.host import profile, setup


def options(**changes):
    return Namespace(
        **{
            "runtime": "podman",
            "fast_resume": False,
            "pool_size_gib": 16,
            "dry_run": False,
            "skip_install": False,
            **changes,
        }
    )


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    state = tmp_path / "state"
    units = tmp_path / "units"
    units.mkdir()
    monkeypatch.setattr(setup, "STATE", state)
    monkeypatch.setattr(setup, "MOUNT", state / "mount")
    monkeypatch.setattr(setup, "UNITS", units)
    monkeypatch.setattr(setup, "DEFAULT_PROFILE", tmp_path / "etc/host.json")
    monkeypatch.setattr(setup, "secure_path", lambda path: None)
    monkeypatch.setattr(setup, "check_host", lambda: None)
    # These tests provision mock pools under pytest's /tmp, which can be a
    # small tmpfs even when the actual host storage has plenty of capacity.
    monkeypatch.setattr(setup.shutil, "disk_usage", lambda path: SimpleNamespace(free=32 * 1024**3))
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.delenv("AGENCY_HOST_CONFIG", raising=False)
    return state


def result(stdout="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_preview_never_checks_or_changes_host(monkeypatch, capsys):
    monkeypatch.setattr(setup, "check_host", lambda: pytest.fail("not read-only"))
    monkeypatch.setattr(setup, "command", lambda *a, **k: pytest.fail("ran a command"))
    assert cli.main(["setup-host", "--fast-resume", "--dry-run"]) == 0
    assert "sparse 16 GiB" in capsys.readouterr().out


def test_refuses_existing_unowned_directory(isolated, monkeypatch):
    isolated.mkdir()
    marker = isolated / "unrelated"
    marker.write_text("keep")
    monkeypatch.setattr(setup, "command", lambda *a, **k: pytest.fail("ran a command"))
    with pytest.raises(ValueError, match="ownership manifest"):
        setup.setup_host(options())
    assert marker.read_text() == "keep"


def test_refuses_existing_unowned_pool_before_install(isolated, monkeypatch):
    monkeypatch.setattr(setup, "command", lambda *a, **k: result())
    monkeypatch.setattr(setup, "install_tools", lambda a: pytest.fail("installed packages"))
    with pytest.raises(ValueError, match="Pool name already exists"):
        setup.setup_host(options())
    assert not isolated.exists()


def test_setup_is_repeatable_and_publishes_only_after_smoke(isolated, monkeypatch):
    events = []
    monkeypatch.setattr(
        setup, "command", lambda *a, **k: result(returncode=1 if a[:2] == ("zpool", "list") else 0)
    )
    monkeypatch.setattr(setup, "install_tools", lambda a: events.append("install"))
    monkeypatch.setattr(setup, "ensure_pool", lambda m: events.append("pool"))

    def smoke(p):
        events.append("smoke")
        return {"fast_resume_used": False}

    monkeypatch.setattr(setup, "smoke_test", smoke)
    setup.setup_host(options())
    manifest = json.loads((isolated / "setup.json").read_text())
    setup.setup_host(options())
    assert json.loads((isolated / "setup.json").read_text()) == manifest
    assert json.loads(setup.DEFAULT_PROFILE.read_text())["validated"] is True
    assert events == ["install", "pool", "smoke"] * 2


def test_failed_smoke_does_not_publish_profile_and_can_retry(isolated, monkeypatch):
    monkeypatch.setattr(setup, "command", lambda *a, **k: result(returncode=1))
    monkeypatch.setattr(setup, "install_tools", lambda a: None)
    monkeypatch.setattr(setup, "ensure_pool", lambda m: None)

    def fail(p):
        raise RuntimeError("restore failed")

    monkeypatch.setattr(setup, "smoke_test", fail)
    with pytest.raises(RuntimeError, match="restore failed"):
        setup.setup_host(options())
    assert not setup.DEFAULT_PROFILE.exists()
    assert (isolated / "setup.json").exists()
    monkeypatch.setattr(setup, "smoke_test", lambda p: {})
    setup.setup_host(options())
    assert setup.DEFAULT_PROFILE.exists()


def test_rerun_cannot_resize_or_switch_runtime(isolated, monkeypatch):
    isolated.mkdir()
    setup.write_json(
        isolated / "setup.json",
        {"id": "test", "pool_size_gib": 16, "profile": setup.profile_for(options())},
    )
    monkeypatch.setattr(setup, "install_tools", lambda a: pytest.fail("installed packages"))
    with pytest.raises(ValueError, match="different options"):
        setup.setup_host(options(pool_size_gib=8))


def test_pool_uses_sparse_file_and_never_force_create(isolated, monkeypatch):
    isolated.mkdir()
    manifest = {"id": "test", "pool_size_gib": 8, "profile": setup.profile_for(options())}
    calls = []

    def command(*args, **kwargs):
        calls.append(args)
        if args[:2] in [("zpool", "list"), ("zfs", "list")]:
            return result(returncode=1)
        if args[:2] == ("zpool", "status"):
            return result(f"  {isolated}/pool.vdev ONLINE 0 0 0")
        if args[:2] == ("zfs", "get"):
            return result(
                str(isolated / "mount" / ("s" if args[-1].endswith("/sandboxes") else ""))
                if "mountpoint" in args
                else "test"
            )
        return result()

    monkeypatch.setattr(setup, "command", command)
    monkeypatch.setattr(setup.shutil, "disk_usage", lambda p: SimpleNamespace(free=30 * 1024**3))
    setup.ensure_pool(manifest)
    backing = isolated / "pool.vdev"
    assert backing.stat().st_size == 8 * 1024**3
    assert backing.stat().st_blocks * 512 < 1024 * 1024
    assert any(c[:2] == ("zpool", "create") for c in calls)
    assert any(c[:2] == ("zfs", "create") and f"mountpoint={isolated}/mount/s" in c for c in calls)
    assert all("-f" not in c and "-a" not in c for c in calls)


def test_unimportable_backing_file_is_never_truncated(isolated, monkeypatch):
    isolated.mkdir()
    backing = isolated / "pool.vdev"
    backing.write_bytes(b"unrelated content")
    monkeypatch.setattr(setup, "command", lambda *a, **k: result(returncode=1))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        setup.ensure_pool({"id": "x", "pool_size_gib": 8})
    assert backing.read_bytes() == b"unrelated content"


def test_foreign_pool_owner_is_rejected(isolated, monkeypatch):
    isolated.mkdir()
    calls = []

    def command(*args, **kwargs):
        calls.append(args)
        return result("foreign")

    monkeypatch.setattr(setup, "command", command)
    with pytest.raises(ValueError, match="not owned"):
        setup.ensure_pool({"id": "our-setup"})
    assert all(c[1] in {"list", "get", "status"} for c in calls)


def test_service_collision_is_not_overwritten(isolated):
    service = setup.UNITS / "agency-zfs.service"
    service.write_text("unrelated unit")
    with pytest.raises(ValueError, match="existing service"):
        setup.install_unit(service.name, "ours")
    assert service.read_text() == "unrelated unit"


def test_docker_daemon_is_isolated_and_does_not_manage_firewall(isolated, monkeypatch):
    isolated.mkdir()
    monkeypatch.setattr(setup, "command", lambda *a, **k: result())
    setup.setup_docker(setup.profile_for(options(runtime="docker")))
    config = json.loads((isolated / "docker.json").read_text())
    assert config["data-root"] == str(isolated / "mount/d")
    assert config["bridge"] == "none"
    assert config["iptables"] is False and config["ip-forward"] is False
    assert config["hosts"] == ["unix:///run/agency-docker/docker.sock"]
    assert "--config-file=" in (setup.UNITS / "agency-docker.service").read_text()


def test_pool_boot_service_imports_only_its_pool():
    content = setup.pool_unit({"id": "test", "profile": setup.profile_for(options())})
    assert "import -d /var/lib/agency-host agency_host" in content
    assert "import -a" not in content and "mount -a" not in content
    assert "org.agency:setup-id" in content


def test_default_config_stays_legacy(monkeypatch):
    monkeypatch.delenv("AGENCY_HOST_CONFIG", raising=False)
    assert agconfig().sandbox.checkpoint_backend == "image_commit"


def test_selected_profile_and_explicit_namespace_override(monkeypatch):
    saved = setup.profile_for(options(fast_resume=True))
    monkeypatch.setenv("AGENCY_HOST_CONFIG", "/etc/agency/host.json")
    monkeypatch.setattr(profile.os, "geteuid", lambda: 0)
    monkeypatch.setattr(profile, "read_profile", lambda path: saved)
    cfg = agconfig()
    assert cfg.sandbox.backend == "podman" and cfg.sandbox.checkpoint_fast_resume
    assert agconfig(sandboxconfig()).sandbox.checkpoint_backend == "image_commit"


def test_run_selects_profile_and_strips_remote_environment(monkeypatch):
    saved = setup.profile_for(options(runtime="docker"))
    monkeypatch.setattr(cli, "read_profile", lambda path: saved)
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)
    monkeypatch.setenv("DOCKER_CONTEXT", "unrelated-remote")
    monkeypatch.setenv("CONTAINER_HOST", "ssh://remote")
    captured = {}

    def call(argv, env):
        captured.update(argv=argv, env=env)
        return 7

    monkeypatch.setattr(cli.subprocess, "call", call)
    assert cli.main(["run", "--", "script.py", "--foo"]) == 7
    assert captured["argv"][1:] == ["script.py", "--foo"]
    assert captured["env"]["DOCKER_HOST"] == saved["docker_host"]
    assert "DOCKER_CONTEXT" not in captured["env"] and "CONTAINER_HOST" not in captured["env"]


def test_nonroot_launch_fails_before_reading_profile(monkeypatch):
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(cli, "read_profile", lambda p: pytest.fail("read profile"))
    assert cli.main(["run", "script.py"]) == 1


def test_invalid_pool_size_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["setup-host", "--pool-size-gib", "64"])


def test_profile_rejects_symlinks(tmp_path):
    target = tmp_path / "profile"
    target.write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="root-owned"):
        profile.read_profile(link)


@pytest.mark.parametrize("fast,used", [(False, False), (True, True), (True, False)])
def test_smoke_requires_requested_fast_restore_and_always_cleans_up(monkeypatch, fast, used):
    import importlib
    from contextlib import nullcontext

    destroyed = []

    class Sandbox:
        def __init__(self, *a, **k):
            self.token = None

        def exec(self, cmd):
            self.token = cmd.split()[2]
            return "ready", 0

        def checkpoint(self):
            return SimpleNamespace(stats={"fast_resume_available": fast, "fast_resume_used": used})

        def restore(self, cp):
            pass

        def read_file(self, path):
            return self.token

        def destroy(self):
            destroyed.append(True)

    client = SimpleNamespace(daemon_identity=lambda: (207, 0), is_ready=lambda: True)
    daemon = SimpleNamespace(client=lambda: nullcontext(client))
    monkeypatch.setattr(importlib.import_module("agency.sandbox.agsandbox"), "agSandbox", Sandbox)
    monkeypatch.setattr(
        importlib.import_module("agency.engine.harness_daemon_launcher"),
        "ensure_harness_daemon",
        lambda *a, **k: daemon,
    )
    selected = setup.profile_for(options(fast_resume=fast))
    if fast and not used:
        with pytest.raises(RuntimeError, match="fell back"):
            setup.smoke_test(selected)
    else:
        setup.smoke_test(selected)
    assert destroyed == [True]


def test_install_only_missing_packages_and_never_upgrade(monkeypatch):
    calls = []
    monkeypatch.setattr(
        setup.shutil, "which", lambda name: None if name == "criu" else "/usr/bin/" + name
    )
    monkeypatch.setattr(
        setup, "command", lambda *a, **k: calls.append(a) or result("Candidate: 1.0")
    )
    monkeypatch.setattr(setup.Path, "exists", lambda p: True)
    setup.install_tools(options(fast_resume=True))
    assert ("apt-get", "install", "-y", "--no-upgrade", "--no-remove", "criu") in calls
    assert all("upgrade" not in call for call in calls)


def test_skip_install_fails_before_mutation(monkeypatch):
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup, "command", lambda *a, **k: pytest.fail("changed host"))
    with pytest.raises(ValueError, match="Missing system packages"):
        setup.install_tools(options(skip_install=True))


def test_wrong_docker_endpoint_is_rejected(isolated, monkeypatch):
    monkeypatch.setattr(
        setup,
        "command",
        lambda *a, **k: result(
            json.dumps({"Driver": "overlay2", "DockerRootDir": "/var/lib/docker"})
        ),
    )
    with pytest.raises(ValueError, match="not the isolated"):
        setup.wait_for_docker()


def test_managed_docker_endpoint_is_accepted(isolated, monkeypatch):
    isolated.mkdir()
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(setup, "command", lambda *a, **k: result())
    setup.setup_docker(setup.profile_for(options(runtime="docker")))
    config = json.loads((isolated / "docker.json").read_text())
    monkeypatch.setattr(
        setup,
        "command",
        lambda *a, **k: result(json.dumps({"Driver": "zfs", "DockerRootDir": config["data-root"]})),
    )
    setup.wait_for_docker()


def test_low_disk_fails_before_packages_or_state_creation(isolated, monkeypatch):
    monkeypatch.setattr(setup, "command", lambda *a, **k: result(returncode=1))
    monkeypatch.setattr(setup.shutil, "disk_usage", lambda p: SimpleNamespace(free=4 * 1024**3))
    monkeypatch.setattr(setup, "install_tools", lambda a: pytest.fail("installed packages"))
    with pytest.raises(ValueError, match="Insufficient free disk"):
        setup.setup_host(options())
    assert not isolated.exists()


def test_missing_candidate_adds_only_owned_official_universe_source(isolated, monkeypatch):
    target = isolated.parent / "agency-universe.sources"
    monkeypatch.setattr(setup, "APT_SOURCE", target)
    monkeypatch.setattr(
        setup.platform, "freedesktop_os_release", lambda: {"VERSION_CODENAME": "resolute"}
    )
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")
    calls = []

    def command(*args, **kwargs):
        calls.append(args)
        return result("Candidate: 1.0" if target.exists() else "Candidate: (none)")

    monkeypatch.setattr(setup, "command", command)
    setup.ensure_package_candidates(["podman"])
    source = target.read_text()
    assert "Suites: resolute resolute-updates" in source
    assert "Components: universe" in source
    assert "https://archive.ubuntu.com/ubuntu" in source
    assert "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg" in source
    assert ("apt-get", "update") in calls


def test_available_packages_leave_sources_untouched(monkeypatch):
    monkeypatch.setattr(setup, "command", lambda *a, **k: result("Candidate: 5.7.0"))
    monkeypatch.setattr(setup, "secure_path", lambda p: pytest.fail("touched sources"))
    setup.ensure_package_candidates(["podman"])
