# sandbox — runtimes and persistent state

`agency/sandbox/` owns the execution environment and its filesystem lifecycle. Harnesses use this environment; the [engine](engine.md) determines when working state is accepted or discarded.

## Facade and backends

[agsandbox.py](../../agency/sandbox/agsandbox.py) exposes `agSandbox`: execution, file access, limits, process tracking and lifecycle operations. It wraps one backend selected through [base.py](../../agency/sandbox/base.py). [container.py](../../agency/sandbox/container.py) shares Docker/Podman mechanics, with runtime-specific implementations in [docker.py](../../agency/sandbox/docker.py) and [podman.py](../../agency/sandbox/podman.py). [chroot.py](../../agency/sandbox/chroot.py) implements the chroot path and its own snapshot format.

This facade keeps skills and engines from embedding container command details. Backend selection does not imply identical checkpoint capabilities or isolation guarantees. The facade also supplies a lock so engines sharing it cannot interleave attempts and teardown.

A sandbox establishes mounts and runtime identity during construction and starts processes as needed. Shared mounts expose the host-service socket, logs and Agency source to the container. Resource-pool hooks connect runtime limits and GPU ownership to [orchestrator](orchestrator.md).

## Three kinds of persistent state

| State | Owner and mechanism |
| --- | --- |
| Conversation and harness session | Agent/context modules; accepted session updates are staged by the engine. |
| Sandbox filesystem | Runtime image commit by default, or an optional local ZFS snapshot. |
| Live process memory and descriptors | Optional CRIU process cache alongside ZFS checkpoints. |

[checkpoint.py](../../agency/sandbox/checkpoint.py) implements the container checkpoint strategies. `image_commit` materializes an OCI image through the runtime. `cow_zfs` records a private filesystem snapshot for the same sandbox; only its latest handle may restore, and rollback requires a stopped runtime. The ZFS backend does not support image export, agent save or fork.

Optional fast resume retains process state to avoid fresh harness startup. The filesystem snapshot remains authoritative. If process restoration fails, the code clears retained daemon handles, stops partial processes and rolls the filesystem back again before allowing fresh startup. Tracing is reattached while restored tasks are frozen, before they resume. Supporting state and mount repair live in [checkpoint_state.py](../../agency/sandbox/checkpoint_state.py) and [_criu_restore_mounts.py](../../agency/sandbox/_criu_restore_mounts.py).

## Boundary with the application

Agent save/load and fork coordinate settled conversation with supported filesystem snapshots. A checkpoint does not capture host Python closures, provider credentials or arbitrary external effects. Bind mounts remain external live state, and filesystem snapshots do not by themselves retain process memory.

The [checkpoint guide](../guides/checkpoints.md) covers configuration; [sandbox APIs](../api/sandboxes.md) describe capabilities. [Restore tests](../../tests/sandbox/test_cow_zfs_fast_resume.py) exercise the filesystem fallback ordering.
