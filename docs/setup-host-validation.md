# Host setup validation

Implementation: working tree on `eric/kimi-code` at `0c0765f`, including the
fast-checkpoint merge and uncommitted setup command, 2026-09-14/15. The merge
commit has the same tracked file tree as `1cab572`.

## Existing EC2 host

Validated on the authorized Ubuntu 24.04 host `3.145.66.64`:

- Provisioned the separate `agency_host` sparse 16 GiB pool from an empty setup.
- Installed/enabled `agency-zfs.service` and published the profile only after a
  real Agency filesystem + native daemon CRIU checkpoint/restore passed.
- Repeated the same setup command successfully without recreating storage.
- Confirmed `agency run` applies the saved Podman/ZFS/fast-resume defaults.
- Stopped the Agency boot service, exported only its idle pool, and started the
  service again. The pool was imported and mounted with its original GUID.
- Verified the prior `agency_fast_checkpoint_20260913` pool's GUID stayed the
  same. It was neither exported nor modified by these tests.
- Backing file: 17,179,869,184 logical bytes; 202,997,760 allocated bytes after
  the measured smoke run. It was not preallocated.

The first smoke test exposed Podman's runroot path limit. The installer now
uses `/agy/s`; the corrected version was tested from an empty setup.
These are infrastructure/CRIU tests without model calls, not six-harness Luna
E2E results. The host already had its system dependencies installed, so this
run did not exercise installation of missing packages.

Credential-free raw results are retained under
`artifacts/host-setup-validation/` (ignored generated artifacts).

## Regression checks

- Final local setup/config/checkpoint/liveness regression suite: 93 passed.
- Host setup tests within that suite: 28 passed.
- Live Luna tests are skipped without explicit credentials/setup.
- Ruff and whitespace checks passed.
- `uv sync --extra dev` successfully installed the `agency` console entry point.

## Initial undersized host

The first supplied fresh host, `18.220.229.106`, had only 4.6 GiB free on an
8 GiB EBS disk. The installer refused insufficient capacity before installing
system packages or creating `/var/lib/agency-host`. The user then supplied the
larger host below; the small-host run is superseded.

## Fresh installation: 18.116.204.176

The user supplied the expanded host with a 100 GiB EBS disk. Its entire
`/home/ubuntu/agency-host-setup` development directory was deleted, including
root-owned virtualenv files. Every tracked and nonignored working-tree file was
copied again. uv, managed Python 3.12, its cache, and the virtualenv were rebuilt
inside that directory. No previous ZFS/CRIU/Podman setup was present.

The actual `agency setup-host --runtime podman --fast-resume` command installed
missing catatonit, CRIU, Podman, runc, and ZFS utilities, provisioned the sparse
16 GiB pool, and passed its filesystem/native-daemon CRIU smoke test on its first
attempt. Versions: Ubuntu 26.04, kernel 7.0.0-1006-aws, Podman 5.7.0, CRIU 4.2,
runc 1.4.0, ZFS userspace 2.4.1-1ubuntu5.1 and kernel module 2.4.1-1ubuntu5.

A fresh CLI image was built from the checked-in recipe
`tests/host/fixtures/Dockerfile.harnesses`. Image ID:
`0e1a21a5019eb146d6e6ef461908a0a1995c9cfb3e5cfe88dbc19383eb2fdc0c`.
All pinned CLI version checks passed. The six-harness synthetic-reply matrix
passed 6/6 with real containers, CLIs, PTYs, and CRIU (252.96 seconds). The
80-test setup/config/checkpoint regression suite also passed on this host.
Three initial mock-setup tests exposed `/tmp` being a small tmpfs; the fixture
now controls its mock free-disk capacity instead of depending on host `/tmp`.

The user explicitly approved a private mode-0600 Luna key file on this host.
Live calls use `gpt-5.6-luna`, with no replacement model responses. The test makes
three public `Agent.run()` calls per harness and requires cumulative file state,
three successful CRIU restores, and unchanged daemon/CLI identities. Native
retains its daemon and has no CLI PTY.

The fixture uses the sandbox MCP tool for Native, Claude Code, and Codex, and
built-in terminal tools for Grok, OpenCode, and Kimi. Grok/OpenCode adapters do
not configure the sandbox MCP server; their initial MCP-only test failures were
fixture errors before checkpointing. Claude initially reported an unavailable
tool, then passed its diagnostic rerun with the tool present. Earlier fixture
attempts also exposed Native's different tool interface and pytest assertion
rewriting in a function serialized into the container. These failed attempts
are not counted as successful E2E runs.

All six harnesses passed across targeted runs after the fixture corrections:

| Harness | Live invocations | Checkpoints / CRIU restores | Same daemon / CLI PID |
| --- | ---: | ---: | --- |
| Native | 3 | 3 / 3 | 98 / no CLI |
| Claude Code | 3 | 3 / 3 | 122 / 131 |
| Codex | 3 | 3 / 3 | 133 / 142 |
| Grok | 3 | 3 / 3 | 129 / 137 |
| OpenCode | 3 | 3 / 3 | 128 / 136 |
| Kimi | 3 | 3 / 3 | 129 / 137 |

PIDs are container-namespace identities and may repeat between sandboxes.
**18/18 checkpoints restored through CRIU without fallback.** Each test checked
that the final file contained its unique token and all three cumulative turn
markers. The last restore is triggered by a final file read; the first two
restores are followed by another live model invocation through the same PTY.

The final regression suite passed
**93 tests on both macOS and fresh EC2**, including 28 setup tests. Review also
caught and fixed Docker's stale data-root validation path; its regression test
failed before the fix and passes afterward.


Repeating the final setup command passed again and retained the original pool
GUID. The boot service was active. After the CLI test image and live tests, the
sparse backing file had 17,179,869,184 logical bytes and 4,452,556,800 physically
allocated bytes (about 4.15 GiB), rather than preallocating its capacity.

The 72 collected artifact files were checked for the supplied API key before
transfer. Credential-free logs, JSON results, and profiler traces are retained in
`artifacts/host-setup-validation/fresh-20260915/`, including a SHA-256 manifest.
No full pytest suite was rerun; the final 93-test targeted suite covers setup,
configuration, COW checkpoint/fast resume, and hibernation liveness.

## Docker COW with cold harness restart

The same `18.116.204.176` host was then tested with Docker 29.1.3 from Ubuntu's
repository. The existing environment and Podman datasets were retained. A
separate `agency_host/docker` dataset was added to the same shared pool and
mounted at `/agy/d`. An isolated Agency Docker daemon reported the ZFS storage
driver and `/agy/d` data root. Its profile explicitly set `fast_resume: false`.

The real Agency smoke test passed with a 0.143-second COW checkpoint. It reported
`fast_resume_available: false` and did not invoke CRIU. A Docker image containing
the same pinned harness CLIs was built with host networking, matching the
managed Docker sandbox configuration.

All six harnesses then passed with live `gpt-5.6-luna` responses and exactly two
public `Agent.run()` calls/checkpoints apiece:

| Harness | Invocations | COW checkpoints | CRIU available / used |
| --- | ---: | ---: | ---: |
| Native | 2 | 2 | 0 / 0 |
| Claude Code | 2 | 2 | 0 / 0 |
| Codex | 2 | 2 | 0 / 0 |
| Grok | 2 | 2 | 0 / 0 |
| OpenCode | 2 | 2 | 0 / 0 |
| Kimi | 2 | 2 | 0 / 0 |

The 12 checkpoint times ranged from 0.181 to 0.380 seconds, with a median of
0.199 seconds. Each second invocation ran through a newly started harness/PTY
against the restored filesystem and verified the first invocation's marker.
The final read restored the second checkpoint and verified both markers.

The profiler traces separate filesystem restore from interactive readiness.
Docker filesystem restore had a 0.031-second median (0.029–0.039 seconds), while
starting a fresh harness daemon had a 5.110-second median (4.623–6.121 seconds).
For comparison, the earlier Podman+CRIU traces had a 1.315-second checkpoint
median, a 0.833-second restore median, and a 0.982-second median for
`sandbox:ensure_daemon`. COW-only makes persistence much cheaper, while CRIU
keeps the interactive harness warm after restore.

Before this Docker run, the prior Podman profiler artifacts were copied again to
`artifacts/host-setup-validation/previous-podman-run/`; all 30 files matched the
earlier verified hashes. The 33 credential-free Docker logs, results, and
profiler files are in `artifacts/host-setup-validation/docker-cow-20260915/`
with their SHA-256 manifest. The temporary EC2 API-key file was removed after
collection.
