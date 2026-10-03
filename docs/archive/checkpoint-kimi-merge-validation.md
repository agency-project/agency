# Kimi/PTY and checkpoint merge validation

## Reconciliation

Merged `eric/kimi-code` (`529d4a6`) into `eric/fast-checkpoint` (starting at
`18c7835`). The shared `PtyExecution` runner, driver subclasses, Kimi transcript
completion, workspace trust, and portable session fixes come from Kimi Code.
The COW storage, checkpoint manifests, optional CRIU restore, and durable
filesystem fallback remain from fast-checkpoint.

The unified runner retains a successful CLI only when fast resume is enabled.
Its configuration is prepared once, and subsequent attempts reuse the same PTY.
Fresh-process COW and legacy image-commit paths still retire the CLI per attempt.

EC2 validation exposed and fixed three integration issues:

- Fast-resume MCP endpoints must explicitly decline optional GET event streams.
  Stateless JSON responses alone still allowed Claude to hold a stream open,
  preventing the host attempt from draining. Both tool surfaces now return 405
  for this optional stream while continuing to serve ordinary MCP requests.
  This follows the [MCP optional GET stream contract](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports).
- The sandbox MCP server now receives the manager's retained-session setting.
- A syscall callback racing with token retirement must not terminate the tracer.
  New admissions fail closed, and late completion telemetry is discarded when
  the attempt is already inactive. Errors during active attempts still propagate.

Test doubles now include the checkpoint backend handle map/configuration. Linux
process/profiler checks declare their platform requirements; the legacy Docker
fast-layer-accumulator test requires its supported overlay storage driver.

## Environment

Dedicated EC2 host: `3.145.66.64`, accessed using the existing SSH instructions.
Experiment checkout: `/home/eric/agency-merged-checkpoint-20260914/source`.

- Shared sparse 16 GiB ZFS test pool: `agency_fast_checkpoint_20260913`.
- Podman clone parent: `agency_fast_checkpoint_20260913/sandboxes`.
- Isolated Docker ZFS daemon: `/run/agency-docker-zfs.sock`.
- Image: `docker.io/library/agency-merge-harnesses:20260914`.
- CLIs: Claude Code 2.1.251, Codex 0.147.0, Grok Build 1.0.0,
  OpenCode 1.18.15, Kimi Code 0.42.0.
- Agency source, tools, and test logs remain on the host's normal filesystem.

Kimi uses the image's Node 24 runtime. Its experiment-only npm installation had
an unused node-gyp build-time Python symlink removed so installation discovery
would not attempt to mount the host's `/usr` directory.

## Checkpoint matrix

All six harnesses passed three turns, three checkpoints, and two restores for
each requested runtime mode:

| Harness | Podman COW + fast resume | Docker COW + fresh restart |
| --- | --- | --- |
| Native | Pass | Pass |
| Claude Code | Pass | Pass |
| Codex | Pass | Pass |
| Grok | Pass | Pass |
| OpenCode | Pass | Pass |
| Kimi Code | Pass | Pass |

The test verifies filesystem contents and prior conversation context after
restore. Podman requires fast resume to be available and used, with unchanged
daemon and CLI namespace PIDs. Docker explicitly disables fast resume and
requires a fresh daemon each turn. Native retains its daemon but has no PTY.

After fixing the token-retirement race, Codex and Kimi each passed ten turns:
20 checkpoints and 18 restores total, with no fast-resume fallback and the same
CLI identity throughout each case. The Podman CRIU regression suite also passed
8 tests, including three PTY handoffs, ptrace reattachment, and forced-restore
failure fallback; its paid-model case was not enabled.

The matrix uses real CLIs, containers, ptrace, Agency model routing, PTYs, ZFS,
and CRIU. Only upstream model replies are synthetic. This validates lifecycle
and state continuity, not external model-provider availability or performance.
These concurrent correctness runs are not controlled latency benchmarks.

Raw checkpoint JSON is collected locally under
`benchmarks/checkpoint_size_microbenchmark/results/merged-harnesses-20260914/`,
including the longer runs in `stress/`. That generated directory remains ignored
by Git, as do other benchmark results.

## Pytest

- Full local suite: **2,180 passed, 268 skipped** (platform and opt-in checks).
- Full EC2 suite: **2,391 passed, 57 skipped** (2,448 tests total).
  Engine/harness: 652 passed, 28 skipped; sandbox: 357 passed, 28 skipped;
  remaining tests: 1,382 passed, 1 skipped. All five public-agent golden
  lifecycle cases passed with their real CLIs and replayed model responses.
- Ruff and whitespace checks: pass.

The full EC2 suite is partitioned without overlap into `tests/engine` plus
`tests/harness`, `tests/sandbox`, and all remaining tests. It uses the complete
tracked test assets and standard system PATH, including `/usr/sbin` for Podman's
networking helpers. The original broad run exposed incomplete experiment assets
and an overly narrow PATH; those setup issues were corrected before the final run.
