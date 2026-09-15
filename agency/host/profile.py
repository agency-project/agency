"""Load only the host profile explicitly selected by the launcher."""

import json
import os
from pathlib import Path

DEFAULT_PROFILE = Path("/etc/agency/host.json")


def read_profile(path):
    path = Path(path)
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_uid != 0 or info.st_mode & 0o022:
        raise ValueError(
            f"Host profile must be a root-owned file without group/world writes: {path}"
        )
    profile = json.loads(path.read_text())
    if profile.get("schema_version") != 1 or profile.get("validated") is not True:
        raise ValueError("Host setup has not passed its checkpoint/restore smoke test")
    runtime = profile.get("runtime")
    if runtime not in {"podman", "docker"} or type(profile.get("fast_resume")) is not bool:
        raise ValueError("Invalid host runtime profile")
    if profile.get("dataset") != f"agency_host/{'sandboxes' if runtime == 'podman' else 'docker'}":
        raise ValueError("Unexpected host dataset")
    if profile.get("docker_host") != (
        "unix:///run/agency-docker/docker.sock" if runtime == "docker" else None
    ):
        raise ValueError("Unexpected Docker endpoint")
    return profile


def apply_profile(sandbox, profile):
    sandbox.backend = profile["runtime"]
    sandbox.checkpoint_backend = "cow_zfs"
    sandbox.checkpoint_zfs_parent = profile["dataset"]
    sandbox.checkpoint_fast_resume = profile["fast_resume"]
    if profile["runtime"] == "docker":
        sandbox.flags = [*sandbox.flags, "--network=host"]


def apply_selected_profile(sandbox):
    path = os.environ.get("AGENCY_HOST_CONFIG")
    if not path:
        return
    if os.geteuid() != 0:
        raise ValueError(
            "The managed ZFS host profile requires root; use sudo .venv/bin/agency run"
        )
    profile = read_profile(path)
    if profile["runtime"] == "docker" and os.environ.get("DOCKER_HOST") != profile["docker_host"]:
        raise ValueError("Docker endpoint does not match the host profile; use agency run")
    apply_profile(sandbox, profile)
