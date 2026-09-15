"""One shared, file-backed ZFS pool; never adopt or reformat existing storage."""

import json
import os
import platform
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .profile import DEFAULT_PROFILE, apply_profile

STATE = Path("/var/lib/agency-host")
POOL = "agency_host"
MOUNT = Path("/agy")
UNITS = Path("/etc/systemd/system")
OWNER = "org.agency:setup-id"
APT_SOURCE = Path("/etc/apt/sources.list.d/agency-universe.sources")
IMAGE = "docker.io/library/python:3.12-slim"


def command(*args, check=True, timeout=900):
    result = subprocess.run(
        list(args),
        text=True,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
    )
    if check and result.returncode:
        raise RuntimeError(
            f"{shlex.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def secure_path(path):
    # Refuse symlink traversal and untrusted parent directories before writing as root.
    for entry in (path, *path.parents):
        if entry.is_symlink():
            raise ValueError(f"Refusing symlink: {entry}")
        if entry.exists():
            info = entry.stat()
            if info.st_uid != 0 or info.st_mode & 0o022:
                raise ValueError(f"Expected root-owned path without group/world writes: {entry}")


def write_json(path, value):
    secure_path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def install_unit(name, content):
    path = UNITS / name
    secure_path(path)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to replace an existing service: {path}")
    else:
        with path.open("x") as stream:
            stream.write(content)


def profile_for(args):
    return {
        "schema_version": 1,
        "runtime": args.runtime,
        "fast_resume": args.fast_resume,
        "dataset": f"{POOL}/{'sandboxes' if args.runtime == 'podman' else 'docker'}",
        "docker_host": "unix:///run/agency-docker/docker.sock"
        if args.runtime == "docker"
        else None,
        "validated": False,
    }


def check_host():
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise ValueError(
            "setup-host requires root on Ubuntu Linux; use --dry-run to preview anywhere"
        )
    release = platform.freedesktop_os_release()
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") not in {
        "22.04",
        "24.04",
        "26.04",
    }:
        raise ValueError("Automatic installation supports Ubuntu 22.04, 24.04, and 26.04 only")
    if not Path("/run/systemd/system").is_dir():
        raise ValueError("A booted systemd host is required, not a container or Docker Desktop")
    if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
        raise ValueError(
            "cgroup v2 is required; setup does not change boot or kernel configuration"
        )
    for key in (
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "CONTAINER_HOST",
        "CONTAINER_CONNECTION",
        "AGENCY_HOST_CONFIG",
    ):
        if os.environ.get(key):
            raise ValueError(f"Unset {key} before provisioning this local host")


def install_tools(args):
    packages = {
        "zfs": "zfsutils-linux",
        "zpool": "zfsutils-linux",
        "runc": "runc",
        args.runtime: "podman" if args.runtime == "podman" else "docker.io",
        "modprobe": "kmod",
    }
    if args.runtime == "docker":
        packages["dockerd"] = "docker.io"
    else:
        packages["catatonit"] = "catatonit"
    if args.fast_resume:
        packages["criu"] = "criu"
    missing = sorted({package for binary, package in packages.items() if not shutil.which(binary)})
    if missing and args.skip_install:
        raise ValueError(f"Missing system packages: {', '.join(missing)}")
    if missing:
        print("Installing missing system packages: " + ", ".join(missing), flush=True)
        command("apt-get", "update")
        ensure_package_candidates(missing)
        command("apt-get", "install", "-y", "--no-upgrade", "--no-remove", *missing)
    command("modprobe", "zfs")
    if not Path("/dev/zfs").exists():
        raise ValueError("No /dev/zfs; install a compatible Ubuntu kernel/module and rerun setup")
    if args.fast_resume:
        command("criu", "check")


def ensure_package_candidates(packages):
    def unavailable():
        missing = []
        for package in packages:
            policy = command("apt-cache", "policy", package).stdout
            if "Candidate:" not in policy or "Candidate: (none)" in policy:
                missing.append(package)
        return missing

    if not unavailable():
        return
    release = platform.freedesktop_os_release()
    suite = release.get("VERSION_CODENAME")
    if suite not in {"jammy", "noble", "resolute"}:
        raise ValueError("Cannot configure official Ubuntu universe for this release")
    if platform.machine() in {"x86_64", "amd64"}:
        archive = "https://archive.ubuntu.com/ubuntu"
        security = "https://security.ubuntu.com/ubuntu"
    else:
        archive = security = "https://ports.ubuntu.com/ubuntu-ports"
    content = (
        f"Types: deb\nURIs: {archive}\nSuites: {suite} {suite}-updates\n"
        "Components: universe\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n\n"
        f"Types: deb\nURIs: {security}\nSuites: {suite}-security\n"
        "Components: universe\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
    )
    source = APT_SOURCE
    secure_path(source)
    if source.exists():
        if source.read_text() != content:
            raise ValueError(f"Refusing to replace modified package source {source}")
    else:
        with source.open("x") as stream:
            stream.write(content)
    print(
        "Enabling Ubuntu's official universe component in an Agency-owned source file", flush=True
    )
    command("apt-get", "update")
    missing = unavailable()
    if missing:
        raise ValueError(
            f"Packages unavailable from official Ubuntu repositories: {', '.join(missing)}"
        )


def dataset_mount(profile):
    return MOUNT / ("s" if profile["runtime"] == "podman" else "d")


def ensure_pool(manifest):
    backing = STATE / "pool.vdev"
    mount = MOUNT
    present = command("zpool", "list", "-H", "-o", "name", POOL, check=False).returncode == 0
    if not present and backing.exists():
        # Import only this directory's pool; no global import or force flags.
        command("zpool", "import", "-d", str(STATE), POOL, check=False)
        present = command("zpool", "list", "-H", "-o", "name", POOL, check=False).returncode == 0
    if not present:
        if backing.exists():
            raise ValueError(
                "Existing pool.vdev could not be imported; refusing to overwrite or recreate it"
            )
        if mount.exists() and any(mount.iterdir()):
            raise ValueError(f"Refusing to mount over nonempty directory {mount}")
        required = manifest["pool_size_gib"] * 1024**3
        if shutil.disk_usage(STATE).free < required + 2 * 1024**3:
            raise ValueError("Need pool size plus 2 GiB free headroom on the backing filesystem")
        with backing.open("xb") as stream:
            os.chmod(backing, 0o600)
            stream.truncate(required)  # Sparse: no fallocate or physical preallocation.
        command(
            "zpool",
            "create",
            "-o",
            f"cachefile={STATE}/zpool.cache",
            "-O",
            f"{OWNER}={manifest['id']}",
            "-O",
            f"mountpoint={mount}",
            "-O",
            "compression=lz4",
            POOL,
            str(backing),
        )
    owner = command("zfs", "get", "-H", "-o", "value", OWNER, POOL).stdout.strip()
    status = command("zpool", "status", "-P", POOL).stdout
    devices = [line.split()[0] for line in status.splitlines() if line.strip().startswith("/")]
    if owner != manifest["id"] or devices != [str(backing)]:
        raise ValueError("Existing pool is not owned by this setup; refusing to modify it")
    if backing.stat().st_size != manifest["pool_size_gib"] * 1024**3:
        raise ValueError("Pool backing file size differs from setup manifest")
    expected_mount = command("zfs", "get", "-H", "-o", "value", "mountpoint", POOL).stdout.strip()
    if expected_mount != str(mount):
        raise ValueError("Pool mountpoint differs from setup manifest")
    dataset = manifest["profile"]["dataset"]
    if command("zfs", "list", "-H", dataset, check=False).returncode:
        command(
            "zfs",
            "create",
            "-o",
            f"{OWNER}={manifest['id']}",
            "-o",
            f"mountpoint={dataset_mount(manifest['profile'])}",
            dataset,
        )
    if command("zfs", "get", "-H", "-o", "value", OWNER, dataset).stdout.strip() != manifest["id"]:
        raise ValueError("Existing dataset belongs to another setup")
    actual_dataset_mount = command(
        "zfs", "get", "-H", "-o", "value", "mountpoint", dataset
    ).stdout.strip()
    if actual_dataset_mount != str(dataset_mount(manifest["profile"])):
        raise ValueError("Dataset mountpoint differs from setup manifest")


def pool_unit(manifest):
    dataset = manifest["profile"]["dataset"]
    commands = [
        f"/usr/sbin/zpool list -H {POOL} >/dev/null 2>&1 || /usr/sbin/zpool import -d {STATE} {POOL}",
        f'test "$(/usr/sbin/zfs get -H -o value {OWNER} {POOL})" = "{manifest["id"]}"',
    ]
    for name in (POOL, dataset):
        commands.append(
            f'test "$(/usr/sbin/zfs get -H -o value mounted {name})" = yes || /usr/sbin/zfs mount {name}'
        )
    # systemd treats $ specially even inside shell quotes.
    script = "; ".join(commands).replace("$", "$$")
    return f"""[Unit]
Description=Agency isolated ZFS pool
After=local-fs.target
RequiresMountsFor={STATE}
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=/usr/sbin/modprobe zfs
ExecStart=/bin/sh -ec '{script}'
[Install]
WantedBy=multi-user.target
"""


def docker_unit():
    return f"""[Unit]
Description=Agency isolated Docker ZFS daemon
Requires=agency-zfs.service
After=agency-zfs.service network-online.target
[Service]
Type=simple
RuntimeDirectory=agency-docker
RuntimeDirectoryMode=0700
ExecStart={shutil.which("dockerd")} --config-file={STATE}/docker.json
Restart=on-failure
[Install]
WantedBy=multi-user.target
"""


def setup_docker(profile):
    config = {
        "hosts": [profile["docker_host"]],
        "data-root": str(MOUNT / "d"),
        "exec-root": "/run/agency-docker/exec",
        "pidfile": "/run/agency-docker/docker.pid",
        "storage-driver": "zfs",
        "features": {"containerd-snapshotter": False},
        "containerd-namespace": "agency-host",
        "containerd-plugins-namespace": "agency-host-plugins",
        "experimental": True,
        "exec-opts": ["native.cgroupdriver=systemd"],
        # All managed Docker containers use host networking. No bridge or firewall edits.
        "bridge": "none",
        "iptables": False,
        "ip6tables": False,
        "ip-forward": False,
        "ip-masq": False,
    }
    target = STATE / "docker.json"
    if target.exists() and json.loads(target.read_text()) != config:
        raise ValueError("Refusing to replace modified Agency Docker configuration")
    write_json(target, config)
    command(shutil.which("dockerd"), "--validate", "--config-file", str(target))
    install_unit("agency-docker.service", docker_unit())


def smoke_test(profile):
    """Exercise Agency's real checkpoint backend without any model credentials."""
    from agency.configs.agconfig import agconfig
    from agency.sandbox.agsandbox import agSandbox

    cfg = agconfig()
    apply_profile(cfg.sandbox, profile)
    cfg.sandbox.base_image = IMAGE
    sandbox = agSandbox("setup-host-smoke-" + uuid.uuid4().hex[:8], agconfig=cfg)
    try:
        token = uuid.uuid4().hex
        output, code = sandbox.exec(f"printf %s {token} >/workspace/setup-marker; echo ready")
        if code or "ready" not in output:
            raise RuntimeError("Smoke container could not start")
        # Require an actual retained daemon as well as filesystem recovery.
        # A silent CRIU fallback must not pass setup's fast-resume check.
        if profile["fast_resume"]:
            from agency.engine.harness_daemon_launcher import ensure_harness_daemon
            from agency.utils.agutil import new_uds_path

            host_socket = new_uds_path("setup-host")
            daemon = ensure_harness_daemon(
                sandbox, host_socket, "setup-host", "native", agconfig=cfg
            )
            with daemon.client() as client:
                before = client.daemon_identity()[0]
        checkpoint = sandbox.checkpoint()
        sandbox.restore(checkpoint)
        if sandbox.read_file("/workspace/setup-marker").strip() != token:
            raise RuntimeError("Smoke checkpoint lost filesystem state")
        if profile["fast_resume"]:
            if not checkpoint.stats.get("fast_resume_available") or not checkpoint.stats.get(
                "fast_resume_used"
            ):
                raise RuntimeError(
                    "CRIU smoke test fell back to a cold restart; profile was not enabled"
                )
            restored = ensure_harness_daemon(
                sandbox, host_socket, "setup-host", "native", agconfig=cfg
            )
            with restored.client() as client:
                if client.daemon_identity()[0] != before or not client.is_ready():
                    raise RuntimeError("CRIU smoke test did not retain the harness daemon")
        return dict(checkpoint.stats)
    finally:
        sandbox.destroy()


def setup_host(args):
    print(
        f"Plan: {args.runtime}, shared sparse {args.pool_size_gib} GiB pool {POOL} at {STATE}, "
        f"CRIU {'required' if args.fast_resume else 'off'}. Rootful Ubuntu 22.04/24.04/26.04 only."
    )
    print(
        "Install missing packages; create only Agency-owned storage/services; smoke-test before saving defaults."
    )
    if args.runtime == "docker":
        print(
            "Managed Docker uses an isolated daemon and host networking (no separate network namespace)."
        )
    if args.dry_run:
        return
    check_host()
    import fcntl

    for path in (
        STATE,
        MOUNT,
        DEFAULT_PROFILE,
        UNITS / "agency-zfs.service",
        UNITS / "agency-docker.service",
    ):
        secure_path(path)
    manifest_path = STATE / "setup.json"
    secure_path(manifest_path)
    if STATE.exists() and not manifest_path.exists():
        raise ValueError(f"{STATE} exists without an ownership manifest; refusing to adopt it")
    if not manifest_path.exists():
        if MOUNT.exists():
            raise ValueError(f"Mount path already exists: {MOUNT}")
        if DEFAULT_PROFILE.exists() or any(
            (UNITS / name).exists() for name in ("agency-zfs.service", "agency-docker.service")
        ):
            raise ValueError("Existing Agency profile/service without ownership manifest")
        if (
            shutil.which("zpool")
            and command("zpool", "list", "-H", POOL, check=False).returncode == 0
        ):
            raise ValueError("Pool name already exists; no existing pool will be adopted")
    if not (STATE / "pool.vdev").exists():
        required = (args.pool_size_gib + 2) * 1024**3
        if shutil.disk_usage(STATE.parent).free < required:
            raise ValueError(
                "Insufficient free disk: need pool size plus 2 GiB before installing packages"
            )
    STATE.mkdir(mode=0o700, exist_ok=True)
    with (STATE / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        profile = profile_for(args)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["profile"] != profile or manifest["pool_size_gib"] != args.pool_size_gib:
                raise ValueError(
                    "Existing setup has different options; rerun with the original options"
                )
            if DEFAULT_PROFILE.exists() and json.loads(DEFAULT_PROFILE.read_text()) != {
                **profile,
                "validated": True,
            }:
                raise ValueError("Refusing to overwrite a modified host profile")
        else:
            manifest = {
                "id": uuid.uuid4().hex,
                "pool_size_gib": args.pool_size_gib,
                "profile": profile,
            }
            write_json(manifest_path, manifest)
        install_tools(args)
        ensure_pool(manifest)
        install_unit("agency-zfs.service", pool_unit(manifest))
        if args.runtime == "docker":
            setup_docker(profile)
        command("systemctl", "daemon-reload")
        command("systemctl", "enable", "--now", "agency-zfs.service")
        if args.runtime == "docker":
            command("systemctl", "enable", "--now", "agency-docker.service")
            os.environ["DOCKER_HOST"] = profile["docker_host"]
            wait_for_docker()
        command(args.runtime, "pull", IMAGE)
        print("Running Agency filesystem/harness checkpoint and restore smoke test...", flush=True)
        stats = smoke_test(profile)
        write_json(STATE / "smoke-test.json", stats)
        DEFAULT_PROFILE.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        write_json(DEFAULT_PROFILE, {**profile, "validated": True})
        print(
            f"Setup verified. Saved {DEFAULT_PROFILE}. Run: sudo .venv/bin/agency run your_script.py"
        )


def wait_for_docker():
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        info = command("docker", "info", "--format", "{{json .}}", check=False, timeout=10)
        if not info.returncode:
            value = json.loads(info.stdout)
            if value.get("Driver") != "zfs" or value.get("DockerRootDir") != str(MOUNT / "d"):
                raise ValueError("Docker endpoint is not the isolated Agency ZFS daemon")
            return
        time.sleep(0.5)
    raise RuntimeError("Agency Docker did not become ready; inspect journalctl -u agency-docker")
