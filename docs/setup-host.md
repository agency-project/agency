# One-time ZFS host setup

`agency setup-host` provisions an isolated, shared ZFS pool for Agency sandbox
state. Ordinary imports and `uv sync` do not change the host. The default
`agconfig()` remains on `image_commit` unless launched with the saved profile.

The installer currently supports **rootful Ubuntu 22.04/24.04/26.04 with systemd and
cgroup v2**. It needs network access to Ubuntu package repositories and the
container registry. It does not change kernels, reboot, add third-party package
repositories, or convert the host filesystem. If the kernel cannot load ZFS, or
the installed runtime lacks checkpoint support, setup fails with the failing
check; fix that prerequisite and rerun.

## Install and run

From the Agency checkout:

```sh
uv sync
uv run agency setup-host --runtime podman --fast-resume --dry-run
sudo .venv/bin/agency setup-host --runtime podman --fast-resume
sudo .venv/bin/agency run your_script.py
# Python module entry points also work:
sudo .venv/bin/agency run -- -m your_package
```

The first command installs Agency's Python dependencies and command entry point.
The explicit privileged setup command installs missing system packages, loads
ZFS, creates storage, enables an Agency-specific boot service, pulls the small
Python smoke-test image, and runs the real Agency checkpoint/restore path.
If needed, setup enables Ubuntu’s official `universe` component in its own
`/etc/apt/sources.list.d/agency-universe.sources`, preserving existing sources.
`--fast-resume` requires CRIU restore and retention of the native harness daemon;
a silent fallback fails setup. The smoke test makes no model API calls and
needs no model credentials. This is a host capability test; real CLI/PTY
integration is covered by the separate EC2 harness test matrix.

Successful setup writes `/etc/agency/host.json`. `agency run` launches the same
virtualenv's Python with that validated profile, so a script's ordinary
`agconfig()` uses the configured runtime, ZFS, and optional CRIU. Explicit
`sandboxconfig(...)` arguments still override those defaults. The launcher
requires root because the current ZFS implementation is rootful; setup does
not grant non-root users privileged runtime or dataset access. Pass any model
credentials your script needs through your usual secure runtime configuration;
setup does not save them.

Use `--skip-install` to require already installed system tools. Omit
`--fast-resume` for filesystem-only COW with a fresh harness on each invocation.
A CRIU failure during ordinary use still falls back to the filesystem snapshot.

## Storage and reruns

The fixed, owned layout is:

```text
/var/lib/agency-host/setup.json        ownership and immutable setup options
/var/lib/agency-host/pool.vdev         sparse backing file (default 16 GiB)
/var/lib/agency-host/zpool.cache       private ZFS cache
/agy/                                 agency_host pool mount
  s/                                  shared Podman base + per-sandbox clones
  d/                                  Docker layers (Docker mode only)
/var/lib/agency-host/smoke-test.json   most recent successful smoke-test stats
/etc/agency/host.json                 published only after validation
/etc/systemd/system/agency-zfs.service
```

The short `/agy/s` mountpoint keeps Podman runtime socket paths within its limit.

Select `--pool-size-gib 8` through `16` on the first setup. This uses file
truncation to establish logical capacity, **not physical preallocation**. It
requires capacity plus 2 GiB free headroom, because writes can eventually
consume the full logical size. It is a bounded single-host starter setup; it
is not an automatically growing production storage service. The OS, repo,
installed tools, and host logs remain on their existing filesystem.

The boot service imports only `agency_host` from its backing-file directory,
including after an explicit pool export that clears the cache.

Rerun the identical command to verify/reuse the owned resources and repeat the
smoke test. It never truncates an existing backing file, force-imports a pool,
imports all pools, resizes storage, or changes existing setup options. Conflicting
pool names, non-owned paths, and modified service/config files cause an error.
There is no automatic teardown: a failed setup retains its owned resources for
inspection and retry, and never publishes a new usable profile. If pool creation
fails leaving a file without an importable pool, manual inspection is required;
the installer deliberately refuses to recreate that file.

## Docker mode

```sh
sudo .venv/bin/agency setup-host --runtime docker
sudo .venv/bin/agency run your_script.py
```

Choose one runtime per managed setup. Docker mode installs Docker only if it is
missing, then runs a separate `agency-docker.service` with its own data root,
PID/runtime directories, containerd namespaces, configuration, and root-only
socket `/run/agency-docker/docker.sock`. It does not reconfigure or restart an
existing Docker daemon. Installing a missing distribution Docker package may
also start that package's default service.

The managed daemon uses the classic ZFS storage driver. To avoid conflicts with
an existing daemon's bridge/firewall rules, it creates no default bridge and
disables its firewall/forwarding management. **Managed Docker sandboxes use host
networking**, sharing the host network namespace. Filesystem-only restart is the
default; `--fast-resume` additionally requests experimental Docker CRIU support
and must pass the smoke test. The launcher selects the isolated Docker socket
and clears inherited remote Docker/Podman endpoint settings.

The design follows [OpenZFS's Ubuntu installation guidance](https://openzfs.github.io/openzfs-docs/Getting%20Started/Ubuntu/index.html)
and [Docker's multiple-daemon guidance](https://docs.docker.com/reference/cli/dockerd/#run-multiple-daemons).
See [checkpoint semantics](fast-checkpoint.md) for snapshot scope and limitations.

## Live six-harness validation

`tests/host/test_live_harnesses_linux.py` is an opt-in paid E2E test for Native,
Claude Code, Codex, Grok, OpenCode, and Kimi. It uses actual `gpt-5.6-luna`
responses, the public `Agent.run()` API, three invocations per harness, cumulative
filesystem state, and profiler artifacts. Fast-resume profiles require retained
CLI/PTY identities and no fallback. It is skipped in ordinary pytest runs.

A CLI image recipe is provided for this test:

```sh
sudo podman build -t localhost/agency-host-harnesses \
  -f tests/host/fixtures/Dockerfile.harnesses .
```

The recipe pins all five CLI versions; the native harness is supplied by Agency.
Set `AGENCY_TEST_HARNESS_IMAGE=localhost/agency-host-harnesses`. The installed
binary paths are `/usr/local/bin/claude`, `codex`, `grok`, `opencode`, and `kimi`
(with the same `/usr/local/bin/` prefix for each).

After provisioning, supply `AGENCY_HOST_LUNA_E2E=1`,
`AGENCY_TEST_API_KEY_FILE` (a private file), `AGENCY_TEST_HARNESS_IMAGE`, and
`AGENCY_TEST_{CLAUDE_CODE,CODEX,GROK,OPENCODE,KIMI}_BINARY` paths for installed
CLIs. Set `AGENCY_HOST_E2E_RESULTS` to the artifact directory, then launch:

```sh
# Supply the variables above to this privileged process using your normal
# secure environment mechanism. The API key itself is never a command argument.
sudo .venv/bin/agency run -- -m pytest tests/host/test_live_harnesses_linux.py -v
```

The installer provisions container/checkpoint infrastructure, not external
model credentials or third-party CLI installations.
