from __future__ import annotations

from typing import TYPE_CHECKING

from .bash import make_bash
from .read import make_read
from .write import make_write
from .edit import make_edit
from .glob import make_glob
from .grep import make_grep
from .webfetch import webfetch
from .todowrite import todowrite
from .resource import (
    make_gpu_reserve,
    make_cpu_reserve,
    make_cpu_release,
    make_daemon_release,
)
from .human import make_ask_human
from ..agtool import agtool as _tool_cls

if TYPE_CHECKING:
    from ..agsandbox import agSandbox
    from ..agresources import agResourcePool


def make_sandboxed_tools(
    sandbox: "agSandbox",
    pool: "agResourcePool | None" = None,
) -> list[_tool_cls]:
    """Build the tool list for a sandboxed agent.

    All filesystem tools route through *sandbox*'s container.
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
        make_daemon_release(sandbox),
        make_ask_human(sandbox._agname),
    ]
    if pool is not None:
        tools += [
            make_gpu_reserve(sandbox, pool),
            make_cpu_reserve(sandbox, pool),
            make_cpu_release(sandbox, pool),
        ]
    return tools


__all__ = [
    "webfetch",
    "todowrite",
    "make_bash",
    "make_read",
    "make_write",
    "make_edit",
    "make_glob",
    "make_grep",
    "make_gpu_reserve",
    "make_cpu_reserve",
    "make_cpu_release",
    "make_daemon_release",
    "make_ask_human",
    "make_sandboxed_tools",
]
