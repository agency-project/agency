# COW hibernation with optional fast resume

Agency's `cow_zfs` checkpoint backend always persists the mutable container
filesystem at a completed `run()` boundary. When `checkpoint_fast_resume` is
enabled, Agency also asks Podman or Docker to capture the stopped process tree
with CRIU. The process image is a local cache; the ZFS snapshot remains the
authoritative checkpoint.

```text
run() -> completion -> detach ptrace -> CRIU checkpoint -> ZFS snapshot
      -> hibernate with CLI/PTY stopped but retained

next run() -> ZFS rollback -> CRIU restore -> reattach ptrace
           -> continue the same CLI/PTY session

CRIU failure -> ZFS rollback -> start a fresh harness/CLI/PTY
```

The older `image_commit` backend remains available for portable OCI image
materialization and hosts without ZFS. `cow_zfs` never calls `docker commit`,
`podman commit`, `tar`, `rsync`, or a recursive file copy.

## Storage ownership and layout

ZFS remains host-managed. Containers receive neither `/dev/zfs` nor ZFS
capabilities. The host OS, Agency checkout, runtime installation, host logs,
and global non-COW runtime data remain on the host's normal filesystem.

The EC2 experiment uses one sparse 16 GiB file-backed pool:

```text
agency_fast_checkpoint_20260913                         /agz
├── sandboxes                                          /agz/s
│   ├── agency-base-<image-id>@seed-v1     (Podman shared base)
│   └── agency-<sandbox-id>@agency-<id>     (Podman sandbox)
└── docker                                             /agz/docker
    ├── <Docker image-layer datasets>      (shared bases)
    └── <Docker writable-layer>@agency-<id> (sandbox state)
```

Podman uses a private graphroot/runroot/tmpdir in a ZFS clone of a shared,
immutable image seed. Native OverlayFS stores the complete image and writable
upper layer in that dataset. Docker uses a dedicated host daemon configured
with its native `zfs` storage driver; Docker creates a dataset for every image
and writable container layer. Agency verifies the dataset reported by Docker
is beneath the configured parent before snapshotting it.

The Docker daemon must use the classic image store because Docker's native ZFS
graphdriver is not the containerd overlayfs snapshotter. A representative
experiment daemon is:

```sh
sudo zfs create -o mountpoint=/agz/docker agency_fast_checkpoint_20260913/docker
sudo dockerd \
  --host=unix:///run/agency-docker-zfs.sock \
  --data-root=/agz/docker \
  --exec-root=/run/agency-docker-zfs \
  --pidfile=/run/agency-docker-zfs.pid \
  --storage-driver=zfs \
  --feature containerd-snapshotter=false \
  --containerd-namespace=agency-zfs \
  --containerd-plugins-namespace=agency-zfs-plugins \
  --experimental
```

Use a dedicated bridge/subnet for that daemon in a multi-daemon deployment.
The experiment configuration is isolated from the normal Docker daemon.

## Captured state

The snapshot covers the container writable root filesystem: repository edits,
created/deleted/renamed files, package installs, virtual environments, `/etc`
and home-directory changes, permissions, ownership, links, xattrs, ACLs, and
package-manager metadata. Immutable base layers remain shared.

External bind mounts and volumes are not part of the ZFS rootfs snapshot. Their
mount specifications remain in the runtime container configuration and are
reattached on start. tmpfs contents are recreated empty. Agency repairs
runtime-owned `/etc/hosts`, `/etc/hostname`, and `/etc/resolv.conf` files during
CRIU restore; Docker also retains private backup copies for the durable fallback.

On a successful fast restore, process memory, the harness, CLI, PTY, file
descriptors, background processes, working directories, environment variables,
and in-memory caches survive. Agency detaches its ptrace profiler immediately
before CRIU checkpoint and reattaches it before releasing retained CLI tasks.
If CRIU cannot dump or restore the tree, these process-local properties are
discarded and Agency starts a fresh harness from the ZFS state.

## Runtime metadata and identity

Agency's existing `sandboxconfig` remains the normalized launch manifest for
the base image, flags, environment, workdir, user, mounts, networking,
resources, devices, and security options that Agency supports. The COW handle
adds a backend, runtime, logical container name, ZFS snapshot reference, and
timing statistics. It is intentionally a local, latest-only handle and cannot
be exported through the agent image API.

The implementation retains the stopped runtime container object. A successful
CRIU restore preserves the container process identities. A durable fallback
starts new processes in that object. Agency's public identity remains the
logical sandbox name.

## Checkpoint transaction and cleanup

At completion the persistent harness keeps its CLI and PTY alive. Agency asks
the daemon to release ptrace ownership, validates the process tree, and invokes
the runtime checkpoint command. Only after the runtime confirms that the
container stopped does Agency issue the ZFS snapshot. Restore rolls back ZFS,
invokes the runtime restore command, seizes the retained tracees, and then
releases them from a temporary cgroup freezer. Any failure takes the durable
fallback path.

Each sandbox retains only its latest checkpoint. Publishing a new snapshot
deletes the superseded snapshot. Destroying a sandbox deletes its Agency-owned
snapshots and dataset; shared immutable Podman seeds and Docker image layers are
left for host-level image GC.

Profiler spans include `checkpoint.total`, `checkpoint.quiesce`,
`checkpoint.fs_snapshot`, `restore.total`, `restore.fs`,
`runtime.container_create`, and `runtime.container_start`. Pool provisioning and
base-image import happen in `checkpoint.setup`, outside checkpoint latency.

## Configuration

To provision a new supported Ubuntu host, use the explicit
[`agency setup-host` command](setup-host.md). The configuration below remains
available for separately managed storage.

```python
cfg.sandbox.backend = "podman"  # or "docker" with a ZFS-backed daemon
cfg.sandbox.checkpoint_backend = "cow_zfs"
cfg.sandbox.checkpoint_zfs_parent = "agency_fast_checkpoint_20260913/sandboxes"
cfg.sandbox.checkpoint_fast_resume = True
```

For Docker, set `checkpoint_zfs_parent` to the dataset mounted exactly at the
daemon's `DockerRootDir`. For Podman, set it to the parent under which Agency
may create base and per-sandbox datasets.

Docker's experimental CRIU restore currently fails with its managed network
namespace on the validated host. Agency therefore enables Docker fast resume
only when `"--network=host"` is present in `sandbox.flags`; otherwise it records
the capability failure and uses the ZFS checkpoint. Podman does not have this
restriction. Docker also requires daemon experimental mode, cgroup v2 with the
systemd driver, CRIU, runc checkpoint support, and OCI annotation support.

Native integration tests use:

```sh
sudo env AGENCY_TEST_ZFS_PARENT=agency_fast_checkpoint_20260913/sandboxes \
  AGENCY_TEST_COW_RUNTIMES=podman \
  python -m pytest -q tests/sandbox/test_cow_zfs_linux.py
```

Docker tests run against the dedicated daemon socket with
`DOCKER_HOST=unix:///run/agency-docker-zfs.sock` and its Docker parent dataset.
Benchmark commands and measured results are indexed in
`benchmarks/checkpoint_size_microbenchmark/results/fast-checkpoint-cow-only-20260913/README.md`.
