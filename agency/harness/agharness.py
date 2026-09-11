"""Shared isolated configuration directories and MCP endpoint configuration.

CLI dialects and session formats belong to the concrete adapters. Skill
prompt construction belongs to agskill.
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


def materialize_config_home(ag: "agent | str") -> Path:
    """Create a fresh, isolated directory for one harness launch's config
    home. Concrete backends write their own harness-specific gateway config
    into this directory -- what to write is backend-specific, only the
    "give me an isolated directory" part is shared. Nested under this run's
    own config_homes/ directory (agutil.agency_config_homes_dir()) rather
    than a bare OS tempdir, so it's cleaned up with the rest of the run's
    ephemeral state; still one uniquely-named directory per launch."""
    return Path(
        tempfile.mkdtemp(
            prefix=f"agharness-{_runtime_name(ag)}-", dir=str(agency_config_homes_dir())
        )
    )


def cleanup_config_home(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def materialize_config_home_in_container(ag: "agent | str", sandbox, token: str) -> str:
    """In-container counterpart to `materialize_config_home` -- creates a
    fresh, isolated directory INSIDE *sandbox*'s own container filesystem
    instead of a host tempdir. Required once the harness process itself
    runs inside the container: a host tempdir is invisible to a process in
    the container's own mount namespace, so `cwd`/`CLAUDE_CONFIG_DIR` (or
    each other harness's equivalent) must point somewhere the harness can
    actually see. Returns the in-container path; paired with
    `cleanup_config_home_in_container` in the caller's `finally`, mirroring
    `materialize_config_home`/`cleanup_config_home`'s own pairing."""
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
    "materialize_config_home_in_container",
    "cleanup_config_home_in_container",
    "mcp_config_for",
]
