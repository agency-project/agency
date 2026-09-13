"""Host-side discovery and read-only installation mounts for external CLIs.

Mounts are chosen before container creation. Keeping installation directories at
their original absolute paths preserves package-relative lookups and symlinks;
the daemon receives a path, never responsibility for copying host files.
System interpreters and shared libraries belong in the sandbox image.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
from pathlib import Path

from .adapters.agharness_backend import agharness_backend


HARNESS_PATH = "/usr/local/bin:/usr/bin:/bin"


def external_binary(harness, config) -> str | None:
    adapter = agharness_backend.for_config(harness, config)
    default = adapter._DEFAULT_BINARY
    return (config.harness_adapter.binary_path or default) if default else None


def _installation_root(executable: Path) -> Path:
    # npm can hoist platform packages beside the CLI package. Include the
    # outer node_modules tree, not just the entry script's immediate package.
    node_modules = [parent for parent in executable.parents if parent.name == "node_modules"]
    if node_modules:
        return node_modules[-1]
    for parent in executable.parents:
        if (parent / "package.json").is_file():
            return parent
    # A conventional app/bin layout may have sibling lib/data directories.
    root = executable.parent
    if root.name == "bin":
        root = root.parent
    return root


def harness_installation_mounts(config) -> dict[str, tuple[str, str, str]]:
    """Discover the selected harness without starting or modifying a container."""
    binary = external_binary(config.agent.harness, config)
    if binary is None:
        return {}
    host_path = shutil.which(binary)
    if host_path is None:
        # An image-provided executable needs no host installation.
        return {}
    root = _installation_root(Path(host_path).resolve(strict=True))
    roots = [root]
    mounts = {}
    while roots:
        root = roots.pop()
        if root in {
            Path("/"),
            Path("/usr"),
            Path("/usr/local"),
            Path("/opt"),
            Path("/home"),
            Path.home(),
            Path.home() / ".local",
        }:
            raise ValueError(
                f"Cannot mount broad host directory {root} as a harness installation; "
                "install the CLI in the sandbox image or use a dedicated installation directory"
            )
        if str(root) in {mount[0] for mount in mounts.values()}:
            continue
        name = "_harness_install_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        mounts[name] = (str(root), str(root), "ro")
        # Linked workspace packages can live outside node_modules. Mount their
        # complete directories too, at the original paths the links reference.
        for directory, dirs, files in os.walk(root, followlinks=False):
            for entry in dirs + files:
                path = Path(directory) / entry
                if not path.is_symlink():
                    continue
                target = path.resolve()
                if target.is_relative_to(root) or not target.exists():
                    continue
                linked_root = target if target.is_dir() else _installation_root(target)
                if linked_root not in roots:
                    roots.append(linked_root)
    return mounts


def prepare_harness_executable(sandbox, harness, config) -> str | None:
    """Return a sandbox executable after testing its real startup dependencies."""
    binary = external_binary(harness, config)
    if binary is None:
        return None

    out, rc = sandbox.exec(f"command -v -- {shlex.quote(binary)}", workdir="/")
    if rc == 0 and out.strip():
        resolved = str(Path("/") / out.strip())
    else:
        host_path = shutil.which(binary)
        if host_path is None:
            raise FileNotFoundError(
                f"{harness} executable {binary!r} is absent from the sandbox and host PATH"
            )
        resolved = str(Path(host_path).resolve(strict=True))

    path = shlex.quote(resolved)
    out, rc = sandbox.exec(f"test -f {path} && test -x {path}", workdir="/")
    if rc != 0:
        raise FileNotFoundError(
            f"{harness} executable {resolved!r} is not executable inside the sandbox. "
            "Create the sandbox with this harness and binary_path configured so its "
            "installation directory is mounted before container creation."
        )
    # Use the same PATH as adapter launches. Existence alone misses an absent
    # Node interpreter, npm platform package, or ELF shared library.
    out, rc = sandbox.exec(f"PATH={HARNESS_PATH} {path} --version", workdir="/", timeout=30)
    if rc != 0:
        raise RuntimeError(
            f"{harness} executable {resolved!r} cannot start inside the sandbox "
            f"(exit {rc}). Its complete installation must be mounted and the image "
            f"must provide its system interpreter and shared libraries. Output: {out[-2000:]}"
        )
    if harness == "codex" and "0.147.0" not in out:
        # 0.154.0 hangs waiting for native prompt acknowledgment on large
        # (>1000 char) bracketed pastes -- the PTY driver's normal submission
        # path -- because codex routes those through a placeholder/expand-on-
        # submit pipeline that was heavily refactored between the two
        # versions. Only 0.147.0 is verified working; fail closed instead of
        # a confusing mid-run submit timeout.
        raise RuntimeError(
            f"codex executable {resolved!r} reports version {out.strip()!r}, expected "
            "0.147.0. Untested/newer codex builds have hung waiting for native prompt "
            "acknowledgment on real (large) task prompts -- install/pin 0.147.0."
        )
    return resolved
