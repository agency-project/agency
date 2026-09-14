# EC2 benchmark preparation — no benchmark run

**Status:** environment staged; live validation explicitly withheld at the user's
request. No N=1/N=2 invocation, overlap test, failure experiment, live affinity
probe, or sweep was executed. No concurrency level is demonstrated by this audit.

Audit: 2026-09-06 UTC. Connection followed `AGENT_INSTRUCTIONS/SSH.md` exactly:
`ssh -i ~/.ssh/id_ed25519_personal eric@3.145.66.64`.

## Machine

| Item | Observed state |
|---|---|
| EC2 identity | `i-0d3e0d1fa85c691df`, `g6.8xlarge`, `us-east-2a`; confirmed by IMDSv2 identity document |
| CPU | AMD EPYC 7R13; 32 online vCPUs, 1 socket, 16 exposed cores, 2 SMT threads/core |
| SMT topology | CPU `i` and `i+16` share a core, for `i=0..15` |
| NUMA | One node, node 0, CPUs 0–31 |
| RAM | 130,090,360,832 bytes (121.2 GiB); initially 3.7 GiB used, 117.4 GiB available; no swap |
| GPU | NVIDIA **L4**, 23,034 MiB VRAM; not the L40 named in SSH.md |
| GPU software | Driver 595.71.05; CUDA driver compatibility 13.2; installed nvcc 13.2.51 |
| GPU activity | 0% utilization, 0 MiB used, no compute processes at audit |
| OS/kernel | Ubuntu 24.04.4 LTS, Linux 6.17.0-1019-aws, x86_64 |
| Runtime | Docker 29.6.2, containerd 2.2.6, runc 1.3.6; systemd cgroups v2; Podman absent |
| Python | Isolated benchmark environment uses CPython 3.12.13 |
| Governor/frequency | No per-CPU cpufreq governor/current-frequency files exposed; no governor changed |
| Root storage | 2 TiB EBS; `/` about 138 GiB used, 1.9 TiB available; Docker root `/var/lib/docker` |
| Local storage | Two 419.1 GiB instance NVMe devices combined into an 838.2 GiB LVM volume, mounted `/opt/dlami/nvme`; about 710 GiB filesystem space available |
| Network | `enp39s0` private interface, `docker0` bridge and two container veth interfaces |
| Clock | UTC, synchronized chrony against `169.254.169.123`; same-instance Linux host/container wall clock |

GPU compute is absent from this replay/sleep workload. The sandbox backend can
expose GPU devices, but the benchmark does not request inference or launch CUDA.
GPU state should still be recorded for the later run.

## Initial machine state and potential interference

Initial load averages were 0.10/0.07/0.02. Two interval samples showed 99.94% CPU
idle, no reported I/O wait or steal, and no material disk activity. These are
short observations, not evidence that a future run will be uncontended.

Other users have live sessions:

- `ubuntu`: Claude PIDs 1167798 and 21349, approximately 457/399 MiB RSS.
- `jerry`: Claude PID 2262264, approximately 342 MiB RSS; editor and tmux sessions.
- Three-second `pidstat` samples showed PID 1167798 averaging 2.33% of one CPU
  (a one-second sample reached 5%); PID 2262264 averaged 0.33%. The `ps` CPU
  column is a lifetime average and must not be mistaken for interval utilization.

Two containers have been up about eight days:
`test-eager-f66767eb` and `test-eager-d93e9c79`. Both contain only `tail`, show
0.00% CPU and about 436 KiB RAM, and have no Agency owner-PID label. They look
like idle test leftovers, but ownership was not proven. **Neither was stopped.**

Final read-only snapshot at 05:16:43 UTC: load 0.48/0.31/0.10, about 117.4 GiB
available RAM, GPU still idle, and exactly the same two pre-existing containers.
This snapshot follows package installation and preparation tests; it is not a
benchmark baseline.

Normal services include Docker/containerd, EC2 SSM, chrony, NVIDIA host monitoring,
sshd, systemd, journald and irqbalance. Scheduled apt/fwupd/sysstat/logrotate jobs
remain enabled and are recorded in the inventory. None were killed or repinned.
No active LLM server or GPU job was found in the captured process/container data.

Existing `/home/eric/agency` has extensive unrelated local modifications, and
other candidate directories are snapshots without Git metadata. They were not
updated, cleaned, or overwritten.

## Changes made

Tooling commit: `df96762a6a9017d7e3518405e789f15f21d14956`, based on `e558734`.

- Added read-only inventory and topology-derived placement tools.
- Added `sandboxconfig.cpuset_cpus` and `cpuset_mems`; applied at container
  creation, constraining the complete harness/tool subtree. Also fixed the
  checkpoint-image recreation path to retain CPU and memory limits, which it
  previously omitted. Hibernation retains the existing container configuration.
- Benchmark pins its host driver before Agency threads are created; all host
  scheduler/engine/RPC/profiler threads inherit the control CPU set. Containers
  receive independent CPU sets through Docker, not through parent-process pinning.
- Added one separate warmup plus three measured repetitions by default, reviewed
  layout/topology checks, child affinity/cgroup receipts, post-run container
  configuration capture, cleanup verification, and an explicit `--execute` gate.
- Kept scheduler semantics unchanged and reused Agency events and profiling.
- Converted evidence checks to preparation-only pytest tests: generated child
  source is compiled, not executed; runtime commands are mocked.

Staged an archive of that exact tooling commit in the **new** directory:
`/home/eric/agency-stress-prep-df96762`. Its `SOURCE_REVISION` records the commit.
Created only that directory's `.venv`, installed runtime/profiler/dev dependencies,
and saved exact installed versions. Source and CPU-layout SHA-256 hashes matched
between local and remote copies. No existing environment or system package was changed.

Cleanup performed: none of the pre-existing workloads. No benchmark containers
were created. The installed environment and source files are intentionally retained.

## Final proposed benchmark layout

All memory-node assignments are node `0`.

| Role / N | CPU assignment |
|---|---|
| OS/container-runtime headroom | `0,1,16,17` — not exclusive; existing services remain unmodified |
| Agency driver, host engine/RPC threads, monitoring | `2,3,18,19` — two complete SMT cores |
| N=1 | Agent 0 → `4` |
| N=2 | Agents 0–1 → `4,5`, one CPU each |
| N=4 | Agents 0–3 → `4,5,6,7`, one CPU each |
| N=8 | Agents 0–7 → `4,5,6,7,8,9,10,11`, one CPU each |
| N=12 | Agents 0–11 → `4–15`, one CPU each |
| N=16/32 | Rejected by this separate-core policy; would require a different, explicitly shared-core experiment |

SMT siblings `20–31` are unused by the benchmark. Maps are identical across
warmups/repetitions for a given N, and smaller N uses a prefix of larger N.
Cross-blocking, ordering and rollback scenarios use worker slots 0 and 1.

These are **proposed and implemented constraints, not a claim of live enforcement**.
The OS, interrupts, runtime, or another user's unrestricted process can still run
on a worker CPU. No exclusive cpuset partition or administrator-level isolation
has been applied. The two-core host control budget is also an experimental
constraint; poor scaling under it is not an intrinsic maximum of Agency.

Planned collection: per-invocation epoch intervals, child elapsed time/affinity,
creation/submission/total durations, successful throughput, success/failure counts,
engine and child peaks, scheduler running counts and lifecycle events, host CPU
seconds/utilization and peak RSS, CPU maps, image/configuration metadata, cleanup
status, JSON/CSV and a trace timeline. The existing optional Linux profiler adds
container/daemon/host resource samples and startup/commit/discard spans. Warmup
rows are labeled and excluded from demonstrated-N selection.

## Validation and remaining threats

- Local: 11 preparation tests passed; lint, format and pre-commit hooks passed.
- EC2: 96 installed packages checked compatible; the same 11 preparation tests
  passed in 6.54 seconds. No test starts a real container or child workload.
- Exact source/layout hash agreement recorded in `remote_validation.txt`.
- Live affinity enforcement: **not run**.
- N>1 overlap, ordering, rollback/cleanup and profiling overhead: **not run**.
- Profiler dependencies are installed; systemd launch permissions and actual
  sampler coverage remain unvalidated. `perf_event_paranoid=4` is recorded;
  optional perf-based collection must not be assumed available.

The hardware/runtime are appropriate for the planned experiment. **The host is
not yet validated for final defensible performance claims.** Reserve a quiet
window with the other users, re-audit, and obtain authorization for the small
live checks before a sweep. Short idle samples do not establish exclusive use.
Keep image ID, dependency pins, code revision, tracing configuration and host CPU
budget fixed. Container startup/checkpoint costs, logging contention, cache state,
VM scheduling/steal and scheduled OS jobs remain possible confounders.

## Reproduction commands — future execution only

The source and environment are already staged. Re-run read-only preparation:

```sh
ssh -i ~/.ssh/id_ed25519_personal eric@3.145.66.64
cd /home/eric/agency-stress-prep-df96762
python3 benchmarks/agent_stress/inspect_host.py > inventory-new.json
python3 benchmarks/agent_stress/placement.py --inventory inventory-new.json --out cpu-layout-new.json
```

The following commands **have not been run**. Run only after authorizing workloads:

```sh
# Smallest case, then N=2 with correctness phases.
.venv/bin/python benchmarks/agent_stress/run.py --execute --layout cpu_layout.json \
  --ns 1 --warmups 0 --repeats 1 --out runs/authorized-n1
.venv/bin/python benchmarks/agent_stress/run.py --execute --layout cpu_layout.json \
  --ns 1 2 --warmups 1 --repeats 3 --scenarios --rollback --out runs/authorized-small

# Only after the small live checks pass and profiler permissions are verified.
AGENCY_PROFILE=1 AGENCY_PROFILE_DIR=./runs/authorized-profile \
  .venv/bin/python benchmarks/agent_stress/run.py --execute --layout cpu_layout.json \
  --ns 1 2 4 8 12 --warmups 1 --repeats 3 --seconds 5 --slow-seconds 30 \
  --scenarios --rollback --out runs/authorized-sweep
```

All use the preinstalled `agency-sandbox:latest`, whose audited ID is
`sha256:1958879d1dd78eb7f5ae5fc411ff9213ec15f874a0d609b459903e786e0dac99`.
Verify the tag still resolves to that ID before execution, or pass that exact ID
through `--image`. No image was pulled, rebuilt, or started during preparation.

For another isolated checkout of the tooling commit, create a Python 3.12.13
virtualenv, synchronize `requirements-pinned.txt` using `uv pip sync`, then
install that checkout with `uv pip install --no-deps -e .`. The original editable
path is retained in `dependency-freeze.txt` for provenance; the portable pinned
file excludes only that editable Agency entry. These are version pins, not wheel
content hashes. No Git push or modification of another user's checkout occurred.
