"""Thin, engine-agnostic glue shared by every `agharness_backends/*`
concrete backend.

Deliberately small -- per-harness config-file format and CLI argv
construction stay in each concrete backend, not here. This module only
holds what's genuinely shared: an isolated per-launch config-home
directory (so concurrent harness-driven agents never see each other's
token/base_url, and a run leaves no trace in the user's own `~/.claude`/
`~/.codex`/`~/.config/opencode`).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils.agutil import agency_config_homes_dir

if TYPE_CHECKING:
    from ..agent import agent


def _runtime_name(owner: "agent | str") -> str:
    return owner if isinstance(owner, str) else owner.agname


def materialize_config_home(ag: "agent | str", token: str, base_url: str) -> Path:
    """Create a fresh, isolated config-home directory for one harness
    launch; concrete backends write their own harness-specific files into
    it. Nested under this run's config_homes/ dir so it's cleaned up with
    the rest of the run's ephemeral state."""
    return Path(
        tempfile.mkdtemp(
            prefix=f"agharness-{_runtime_name(ag)}-", dir=str(agency_config_homes_dir())
        )
    )


def cleanup_config_home(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def is_container_backed(sandbox) -> bool:
    """True for a docker/podman-backed sandbox; chroot's jail is already a
    real host directory and needs neither ptrace bridge nor this."""
    return sandbox is not None and getattr(sandbox._backend, "IMAGE_KIND", "") == "container"


def materialize_config_home_in_container(ag: "agent | str", sandbox, token: str) -> str:
    """In-container counterpart to `materialize_config_home`: creates the
    directory inside *sandbox*'s own filesystem, since a host tempdir is
    invisible to a process in the container's mount namespace. Paired
    with `cleanup_config_home_in_container` in the caller's `finally`."""
    import shlex

    path = f"/tmp/agharness-{_runtime_name(ag)}-{token}"
    sandbox.exec(f"mkdir -p {shlex.quote(path)}", workdir="/")
    return path


def cleanup_config_home_in_container(sandbox, path: str) -> None:
    import shlex

    sandbox.exec(f"rm -rf {shlex.quote(path)}", workdir="/")


def mcp_config_for(
    harness_base_url: str, token: str, *, has_sandbox_mcp_tools: bool = False
) -> dict:
    """Host tools plus the separate attempt-local sandbox server, when present."""
    config = {
        "mcpServers": {
            "agency": {
                "type": "http",
                "url": f"{harness_base_url}/mcp",
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    }
    if has_sandbox_mcp_tools:
        config["mcpServers"]["agency-sandbox"] = {
            "type": "http",
            "url": f"{harness_base_url}/sandbox/mcp",
            "headers": {"Authorization": f"Bearer {token}"},
        }
    return config


__all__ = [
    "materialize_config_home",
    "cleanup_config_home",
    "is_container_backed",
    "materialize_config_home_in_container",
    "cleanup_config_home_in_container",
    "mcp_config_for",
]
