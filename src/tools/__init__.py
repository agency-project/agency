from __future__ import annotations

from typing import TYPE_CHECKING

from .bash import bash, make_bash
from .read import read, make_read
from .write import write, make_write
from .edit import edit, make_edit
from .glob import glob, make_glob
from .grep import grep, make_grep
from .webfetch import webfetch
from .todowrite import todowrite
from .resource import make_gpu_acquire, make_gpu_release, make_cpu_acquire, make_cpu_release, make_daemon_release
from ..agtool import agtool as _tool_cls

if TYPE_CHECKING:
    from ..agsandbox import agSandbox
    from ..agresources import agResourcePool

default_tools: list[_tool_cls] = [bash, read, write, edit, glob, grep, webfetch, todowrite]


def make_sandboxed_tools(
    sandbox: "agSandbox",
    pool: "agResourcePool | None" = None,
) -> list[_tool_cls]:
    """Build the tool list for a sandboxed agent.

    All filesystem tools route through *sandbox*'s container (docker or podman).
    ``webfetch`` and ``todowrite`` remain host-side.
    GPU/CPU resource tools are added when *pool* is provided.
    """
    tools: list[_tool_cls] = [
        make_bash(sandbox),
        make_read(sandbox),
        make_write(sandbox),
        make_edit(sandbox),
        make_glob(sandbox),
        make_grep(sandbox),
        webfetch,
        todowrite,
    ]
    tools += [make_daemon_release(sandbox)]
    if pool is not None:
        tools += [
            make_gpu_acquire(sandbox, pool),
            make_gpu_release(sandbox, pool),
            make_cpu_acquire(sandbox),
            make_cpu_release(sandbox, pool),
        ]
    return tools


__all__ = [
    "bash", "read", "write", "edit", "glob", "grep", "webfetch", "todowrite",
    "default_tools",
    "make_bash", "make_read", "make_write", "make_edit", "make_glob", "make_grep",
    "make_gpu_acquire", "make_gpu_release", "make_cpu_acquire", "make_cpu_release", "make_daemon_release",
    "make_sandboxed_tools",
]
