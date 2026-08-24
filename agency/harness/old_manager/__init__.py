"""New, per-agent container-side manager -- see `agmanager_harness.py`'s
module docstring for the design this is part of and why it isn't wired into
any backend yet.

Unlike `agmanager_host` (a plain importable class, since it runs in the same
host process as the rest of agency), this package's main module is meant to
be launched as a STANDALONE SCRIPT inside a sandbox container, with `agency`
itself on its `PYTHONPATH` (it does import a couple of this package's own
pure, leaf utility modules -- see its module docstring) -- see that module's
own `main()` and `launcher.py`'s host-side launch helper for how it actually
gets started."""

from __future__ import annotations

from .launcher import ensure_launched, launch_in_container

__all__ = ["ensure_launched", "launch_in_container"]
