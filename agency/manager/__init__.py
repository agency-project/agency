"""Per-agent host/harness managers -- `agmanager_host` (host-side, one
instance per agent) and `agmanager_harness` (container-side, or in-process
on the host for a bare-host/chroot launch). See `harness/agharness.py`'s
`get_or_create_host_manager()`/`ensure_harness_bridge()` for the intended
entry points; nothing in here is part of the public API.
"""
