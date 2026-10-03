# host — machine setup and runtime profiles

`agency/host/` prepares the optional managed Linux environment used by ZFS checkpoints and fast resume. It separates machine provisioning from ordinary agent construction.

## Explicit setup

[setup.py](../../agency/host/setup.py) implements host checks, system-tool installation, a dedicated file-backed ZFS pool, runtime services and a checkpoint/restore smoke test. Setup refuses to adopt or reformat existing storage. The smoke test exercises the real sandbox checkpoint backend without requiring model credentials.

[The root CLI](../../agency/cli.py) exposes this as `agency setup-host`. Provisioning is an explicit command; importing Agency does not provision the host.

## Validated profile to runtime configuration

Setup produces a host profile after validation. [profile.py](../../agency/host/profile.py) checks its schema, validation marker, ownership, write permissions, runtime, dataset and Docker endpoint. `agency run` reads that profile and launches Python with the selected local runtime environment.

During `agconfig` construction, `apply_selected_profile()` reads the profile explicitly selected through `AGENCY_HOST_CONFIG` and applies sandbox defaults: runtime, ZFS parent and fast-resume choice. Managed Docker additionally uses its dedicated socket and host networking. Explicit sandbox namespace replacement can supply a different configuration.

This package prepares and selects machine infrastructure. Per-agent runtime operations remain in [sandbox](sandbox.md), scheduling in [orchestrator](orchestrator.md), and harness launch in [engine](engine.md). Applications using ordinary runtime/image checkpoints do not need this managed profile path. The [host setup guide](../guides/host-setup.md) covers operating requirements and commands.
