"""New, per-agent host-side manager -- see `agmanager_host.py`'s module
docstring for the design this replaces and why it isn't wired into any
backend yet."""

from __future__ import annotations

from .agmanager_host import agHostAgentManager
from .launch_state import LaunchHandle

__all__ = ["agHostAgentManager", "LaunchHandle"]
