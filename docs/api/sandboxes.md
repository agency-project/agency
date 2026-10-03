# Sandboxes and checkpoints

```python
from agency import agSandbox, get_container_runtime
from agency.sandbox.checkpoint import CheckpointHandle, CheckpointCapabilityError
```

`agSandbox(name, checkpoint_image=None, agconfig=None)` constructs the facade and clones configuration. It resolves image/mount/backend choices at construction; command/file access starts the actual sandbox lazily. Explicitly destroy sandboxes you own, including forks. Agent-created sandboxes are managed by agent execution/cleanup; do not destroy one while its request is running.

`get_container_runtime() -> str` probes local Docker/Podman executables and reachable engines; prefers usable Podman and caches the result process-wide. Raises `RuntimeError` if neither is usable. Runtime selection does not establish support for native execution on remote engines or Docker Desktop. Follow [host prerequisites](../getting-started.md).

## Commands, files and lifetime

Facade methods delegate with `*args, **kwargs`; these are the useful backend forms. Generated facade signatures below preserve that distinction.

| Form | Return, blocking, side effects |
| --- | --- |
| `exec(cmd: str, workdir="/workspace", timeout=120)` | Blocks, returns `(combined_output: str, exit_code: int)`; nonzero command exit is represented by the code. Runtime/subprocess/timeout errors can propagate. Acquires requested GPU lease if needed and tracks discovered background PIDs. |
| `exec_detached(cmd: str, workdir="/workspace")` | Starts detached command and returns `None` without tracking output/exit status. Container-backed backends only. |
| `read_file(path: str)` / `read_file_bytes(path: str)` | Blocking text str / bytes read from sandbox. File/runtime errors propagate. |
| `write_file(path: str, content: str)` / `write_file_bytes(path: str, data: bytes)` | Blocking write, creates parent directories; returns `None`. |
| `update_limits(*, cpus=None, memory=None)` | Applies sandbox CPU/memory limits, returns `None`. None leaves that limit unchanged; returns without action if the container is not running. Direct limit updates do not apply resource-pool minimum floors. |
| `stop()` | Stops working runtime/processes and releases attached resources, retaining checkpoint state; returns `None`. Subsequent operation can recreate working state. |
| `rm_container()` | Removes working container without deleting checkpoint images; returns `None`, infrastructure operation. |
| `destroy()` | Idempotent best-effort final cleanup of working sandbox and owned checkpoint resources; returns `None`. No sandbox context-manager protocol. |
| `get_config_copy()` / `change_config(config)` | Deep copy / clone-and-replace; change affects settings read going forward, not image/mounts already resolved at construction. |
| `image_kind` | `container` or `chroot` (not necessarily a specific Docker/Podman engine). |

```python
from agency import agSandbox
from agency.configs.agconfig import agconfig, sandboxconfig

sandbox = agSandbox("direct-example", agconfig=agconfig(sandboxconfig(gpu_passthrough=False)))
try:
    sandbox.write_file("/workspace/note.txt", "hello")
    output, code = sandbox.exec("cat /workspace/note.txt", timeout=10)
    assert code == 0 and "hello" in output
finally:
    sandbox.destroy()
```

## Checkpoint backends

| API | Contract |
| --- | --- |
| `commit(tag=None)` | Blocks, returns bool success. Named image tag is supported with image_commit; don't pass tags to cow_zfs. Captures sandbox filesystem, not external mounted host data. |
| `checkpoint()` | Blocks, returns `CheckpointHandle` or `None` if unavailable/failed. Facade accepts **no tag argument**. This handle API is implemented by container backends; chroot does not implement it. |
| `restore(tag_or_handle)` | Blocks and discards current working state in favor of checkpoint state; returns `None`. Image backend accepts image tag or handle. cow_zfs requires its handle and matching sandbox identity. |
| `delete_checkpoint(handle)` | Returns `None`; removes backend checkpoint resources. Container handle API; preserve a handle as long as you need to restore it. |
| `fork(new_name, agconfig=None)` | Returns another sandbox starting from the last checkpoint, not an automatic snapshot of current uncommitted writes. Clones supplied/current config. cow_zfs raises `ValueError`: same-sandbox rollback only. Caller destroys fork. |

`CheckpointHandle` is a frozen descriptor with required backend/runtime/container/reference strings and a fresh stats dict (stats excluded from equality). Keep it with its owning sandbox/backend. `CheckpointCapabilityError` signals a backend capability limitation, not ordinary execution cancellation.

Default `image_commit` uses runtime images. `cow_zfs` requires a local Linux rootful Docker/Podman host, accessible ZFS parent dataset and the documented host setup. Handles support same-sandbox filesystem rollback; export as image, portable Agent.save and fork are unsupported. Optional `checkpoint_fast_resume` adds best-effort CRIU acceleration; durable ZFS state remains authoritative and failed acceleration falls back to a fresh process tree. Chroot has its separate image commit/restore path, not the container checkpoint-handle protocol. [Checkpoint setup](../guides/checkpoints.md) explains capability provisioning.

## Advanced helpers

`get_live_pids()` returns tracked live PIDs; `pid_status_summary()` returns human-readable state. `release_daemon(pid)` untracks an intentional daemon so completion won't wait for it; it does not kill it. `wait_for_processes(skill_name, agname="", ping_interval_s=300, poll_interval_s=5, state_fn=None)` blocks up to the ping boundary and returns `None` if already clean or a completion/still-running message. PID tracking is best-effort, not universal process enumeration.

`current_gpu_ids()` returns leased IDs or `None`; `ensure_gpu_acquired(agname, *, is_cancelled=None)` coordinates the attached resource pool, returns None and can block or be cancelled; pauses/resumes the named harness best-effort while waiting. `release_resources(pool=None)` releases held leases/limits; coordinate with the owner. `remove_files(paths)` removes pipeline cleanup paths. These return `None` unless stated otherwise.

Static `backend_for_image_kind(kind)` returns the implementation class. `tag_image(source, dest)`, `delete_image(tag, force=False)` and `import_image(image_bytes, timeout)` perform runtime operations and return `None`; `export_image(tag, timeout)` returns archive bytes. Use only compatible image-based backends; runtime errors propagate and image operations can be slow. These are infrastructure primitives, not cross-host portability promises.

## Source signatures

[Source: agsandbox.py](../../agency/sandbox/agsandbox.py)

::: agency.sandbox.agsandbox.agSandbox
    options:
      members: ["__init__", "exec", "exec_detached", "read_file", "read_file_bytes", "write_file", "write_file_bytes", "update_limits", "commit", "checkpoint", "delete_checkpoint", "stop", "rm_container", "restore", "release_daemon", "get_live_pids", "pid_status_summary", "release_resources", "remove_files", "destroy", "fork", "image_kind", "change_config", "get_config_copy", "current_gpu_ids", "ensure_gpu_acquired", "backend_for_image_kind", "tag_image", "delete_image", "export_image", "import_image", "wait_for_processes"]

[Source: container.py](../../agency/sandbox/container.py)

::: agency.sandbox.container.get_container_runtime

[Source: checkpoint.py](../../agency/sandbox/checkpoint.py)

::: agency.sandbox.checkpoint.CheckpointHandle
    options:
      members: []

[Source: checkpoint.py](../../agency/sandbox/checkpoint.py)

::: agency.sandbox.checkpoint.CheckpointCapabilityError
    options:
      members: []
