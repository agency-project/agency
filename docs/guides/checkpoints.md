# Configure local COW checkpoints

The default checkpoint backend is image_commit. Optional cow_zfs uses a private latest-only filesystem snapshot for the same sandbox; optional checkpoint_fast_resume adds a best-effort CRIU process cache. The [checkpoint architecture](../architecture/sandbox.md) defines capture/restore ordering and fallback. The [previous implementation/experiment guide](../archive/stage3-docs-fast-checkpoint.md) is preserved as historical context, with its dated EC2 layout and measurements.

## Requirements and configuration

Use explicit [host setup](host-setup.md) or separately managed compatible Linux rootful Docker/Podman and ZFS storage. Do not run provisioning merely to read/build documentation. Normal native onboarding does not require this optional backend.

```python
from agency.configs.agconfig import agconfig, sandboxconfig

cfg = agconfig(sandboxconfig(
    backend="podman",
    checkpoint_backend="cow_zfs",
    checkpoint_zfs_parent="YOUR_EXISTING_POOL/sandboxes",
    checkpoint_fast_resume=False,
))
```

For Docker, the parent must match the native ZFS graphdriver beneath DockerRootDir; the containerd overlay snapshotter is not that driver. For Podman, use the parent under which Agency may create private seeds and per-sandbox datasets. Explicit launcher-selected host profiles can supply defaults, while explicit namespace arguments override them.

Enable checkpoint_fast_resume only on a checkpoint-capable host. GPU sandboxes reject it. Docker additionally requires host networking, experimental daemon/runtime checkpoint support and compatible mounts/cgroups. A successful process restore reestablishes tracing before retained tasks resume; failed capture/restore falls back to stopped filesystem state and fresh processes. It is not guaranteed reuse on every run.

## Guarantees and exclusions

- A handle is local to one sandbox and only its latest generation. Restore requires stopped/hibernated state; the backend rejects running rollback and mismatched identity.
- External bind mounts/volumes remain external; tmpfs/process state is not durable filesystem content.
- Portable Agent.save/export and sandbox/agent fork are unsupported for cow_zfs. Use compatible image-based checkpoints for those operations; credentials and external mount data must be supplied again.
- Destroy owns per-sandbox snapshots/datasets; shared image seeds/layers remain for host-managed garbage collection. Old-snapshot cleanup is best-effort and reported.

See [sandbox API](../api/sandboxes.md#checkpoint-backends) for exact calls and [historical measurement scope](../archive/measurements.md). Specialized integration commands remain in [host setup](host-setup.md); this documentation stage did not provision or test a ZFS/CRIU host.
