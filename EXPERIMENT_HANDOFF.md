# Agency CPU oversubscription experiment handoff

## Implementation and validation status

The new `benchmarks.cpu_oversubscription` CLI runs **one explicitly selected condition** on a dedicated Ubuntu host. DCB gives six sequential workers one physical core each; LSB puts all six on the first of those same six cores. The remaining two cores run the OS, Docker daemon, controller, watchdog, and telemetry. Each worker receives five independent SWE-bench Verified tasks in a fixed order.

**Every condition has its own 1,800-second wall-clock budget.** It starts when the controller releases the six-worker batch barrier, after preparation and readiness checks. Preparation never calls a model. The budget is not multiplied by agents or tasks, and it is never replenished between tasks. `condition_timeout_seconds` is configurable; both example configurations explicitly use `1800`. There is no CLI command to run both conditions automatically.

Implemented on macOS using source inspection, static Python/JSON validation, and lint/format checks only. **No agents, SWE-bench tasks, model requests, Docker workloads, cgroup changes, or Ubuntu integration tests were run.** The Linux execution path remains unverified on the target hardware. Pure policy tests have been supplied for future execution; they were not executed during implementation.

Example configuration files deliberately contain a placeholder binary hash, and the version-controlled manifest is an empty template. They fail closed until the deployment engineer prepares real images, selects actual dataset rows, and freezes their configuration. No fabricated task IDs are provided.

## Files and architecture

| File under `benchmarks/cpu_oversubscription/` | Responsibility |
|---|---|
| `__main__.py` | Explicit plan, preflight, selection, run, evaluate, recover, compare commands |
| `config.py` | Configuration validation, manifest checks, assignment rows, comparison identity |
| `workload.py` | Fetch a pinned Verified dataset; select real IDs; validate snapshot membership |
| `linux.py` | Topology, preflight, host lock, cgroups, CPU isolation script, process-tree termination |
| `backend.py` | Private Docker workspaces, six worker processes, existing Agency Codex gateway |
| `runner.py` | Preparation, barrier, independent watchdog, journals, telemetry, recovery |
| `metrics.py` | Cutoff reconciliation, censored observations, JSON/CSV metric calculations |
| `report.py` | Untimed official SWE-bench evaluation and paired trial comparisons |
| `configs/dcb.json`, `configs/lsb.json` | Matching baseline configurations |
| `manifest.template.json` | Six empty assignment groups; intentionally not runnable |

`tests/test_cpu_oversubscription.py` contains pure tests for deadline boundaries, observed-duration means, denominator preservation, completed-attempt throughput, independent budgets, topology, manifests, and configuration validation.

The backend reuses `benchmarks.tool_annotation_effect.codex.CodexGateway` and `codex_files`, Agency's `CodexAdapter` Responses translation, and `SweBenchAdapter` patch extraction, prediction export, official evaluation command, and report importer. It does not launch Agency's native agent loop or implement a second model client. Docker lifecycle is benchmark-specific because the existing sandbox lifecycle does not enforce this experiment's shared deadline/cgroup contract. Agency production components are unchanged.

Execution flow:

1. Acquire a host-wide lock, validate a clean Git checkout, settings, dataset, binary, CPU topology, isolation, and local images.
2. Create 30 independent writable container layers from immutable prepared images. Mount only the reviewed Codex binary, read-only. Verify container cpusets, image repository commits, and binary executability. `/workspace` points to that container's own `/testbed`.
3. Create six host worker cgroups, start the watchdog and six workers, transfer provider credentials through anonymous stdin pipes, and wait for readiness. Workers join their assigned cgroup before importing the model stack.
4. Durably record one monotonic start/deadline and release the barrier. Each worker uses fresh Codex state and a fresh Agency gateway per task, sequentially. The host gateway and its Docker CLI subprocesses occupy the worker core; container tools occupy the same physical core.
5. Record starts/completions in fsynced per-worker journals. Stream redacted Codex output and gateway events to private host files. Extract each completed patch and stop that task's container before moving on.
6. At completion, failure, interruption, or deadline, terminate remaining processes and seal the observed results. Offline grading is a separate command and cannot overlap a run through this CLI.

## Ubuntu prerequisites

Use the intended AMD Ryzen 7 9800X3D machine, with eight online physical cores, one socket/NUMA node, and **SMT disabled** in firmware. The topology code groups `(socket, physical core)` identities from `lscpu`; logical CPU numbers are never treated as physical-core numbers. The first two physical cores in sorted topology order are reserved. SMT enabled, missing cores, multiple NUMA nodes, or eight apparent CPUs that include sibling threads all cause failure. The code permits another eight-core host for development but records its model/topology; do not mix hardware in a comparison.

Requirements:

- Ubuntu with unified cgroup v2, writable `cpuset`, `cpu`, and `cgroup.kill`; a recent Ubuntu 24.04 kernel is the intended target.
- Local rootful Docker Engine with the systemd cgroup driver and default Unix socket. Remote contexts, Docker Desktop, and rootless Docker are unsupported.
- Python 3.12+, this repository's dependencies, and compatible `datasets` and official `swebench` packages.
- A reviewed Linux x86-64 Codex binary. The existing Agency experiment configuration uses version `0.147.0`; the Mac's inspected installation was `0.140.0` for ARM macOS and is unsuitable. Freeze the actual Linux version and SHA256 in both configs. No unsupported flags are assumed: Linux preflight checks `codex exec --help` and `--version`, without launching an agent.
- Sufficient storage for all selected instance images and 30 writable layers, plus memory for six agents and tests. No CPU quotas or benchmark memory limits are imposed. Avoid swapping and record available resources before trials.

Example installation on Ubuntu, after repository checkout and Docker installation:

```bash
cd /opt/agency
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install datasets swebench
.venv/bin/pip freeze > /srv/agency-data/python-environment.txt
```

Review and pin the compatible dataset/harness release before trials. SWE-bench's current and older releases differ in dataset schema and log layout. This runner uses the official module interface and refuses to manufacture a passing verdict when an installed release cannot grade a row. Preflight records installed versions; comparisons require them to match. Follow the [official SWE-bench harness documentation](https://www.swebench.com/SWE-bench/reference/harness/) for image preparation and package-specific setup. All image downloads/builds happen before the timed runs.

## CPU isolation: review, apply, verify

`isolation-script` only **prints** proposed host changes, using the current topology. It does not apply them. Have the host owner review/approve this script before running it as root:

```bash
.venv/bin/python -m benchmarks.cpu_oversubscription isolation-script > /tmp/agency-isolation.sh
cat /tmp/agency-isolation.sh
```

Before applying it, save existing `AllowedCPUs` properties for `system.slice`, `user.slice`, and `init.scope`, and inspect any custom top-level slices. After approval:

```bash
sudo sh /tmp/agency-isolation.sh
```

It creates `/run/systemd/system/agencycpu.slice`, enables CPU/cpuset accounting below it, and applies **runtime-only** restrictions to the background slices. No quota, scheduler algorithm, CPU weight, firmware setting, or boot parameter is changed. Start the controller from a shell in a reserved background slice. Preflight verifies the shell's allowed CPUs and nice/policy rather than silently changing them.

Host workers enter private child cgroups under `agencycpu.slice`. Docker uses `--cgroup-parent agencycpu.slice` and explicit `--cpuset-cpus`; preflight/run preparation inspect actual host container PIDs, cgroup membership, effective cpusets, and ancestor quotas. New descendants remain constrained even if they detach from their parent. Agents have no Docker socket or writable host cgroup mount, run without Linux capabilities, and cannot widen their container cpuset.

DCB worker/container pairs occupy six distinct physical cores. In LSB all worker/container pairs use one of those cores. The other five are unused by experiment workloads. All 30 prepared containers initially have only sleeping init processes; only six task attempts can be executing concurrently.

Read-only verification on Ubuntu:

```bash
lscpu -p=CPU,CORE,SOCKET,NODE,ONLINE
cat /sys/fs/cgroup/agencycpu.slice/cpuset.cpus.effective
cat /sys/fs/cgroup/system.slice/cpuset.cpus.effective
cat /sys/fs/cgroup/user.slice/cpuset.cpus.effective
systemd-cgls /agencycpu.slice
```

Full preflight also checks populated custom background slices and userspace processes left in the root cgroup. During execution, the controller rechecks cpusets and scheduler policy and records CPU statistics/pressure approximately once per second. This is **best-effort userspace isolation**, not complete kernel/interrupt isolation. IRQs, kernel workers, shared caches, frequency scaling, thermal limits, NUMA/memory bandwidth, and privileged external administrators remain potential interference sources. The kernel's [cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html) describes the underlying cpuset and group-kill behavior.

## LiteLLM and Codex configuration

Set `LITELLM_API_KEY` and `LITELLM_BASE_URL` using your existing secure environment provisioning. The runner does not load or display `.env`, persist keys, probe the model endpoint, or make inference calls during preflight. If using sudo, preserve only those required variables:

```bash
sudo --preserve-env=LITELLM_API_KEY,LITELLM_BASE_URL \
  /opt/agency/.venv/bin/python -m benchmarks.cpu_oversubscription preflight \
  --config benchmarks/cpu_oversubscription/configs/dcb.json
```

Both configs use `gpt-6-luna`. A LiteLLM deployment alias may replace `model`, but the operator must verify that it routes to GPT-6 Luna; preflight deliberately cannot prove upstream routing without contacting the service. The model/provider identifier, context limit, request settings, token cap, gateway steps, and all timeout settings must match between conditions. `max_completion_tokens` is a per-model-response cap, not a total per-task token budget. `max_steps` limits gateway model exchanges. Provider-side retries remain bounded by the condition deadline and watchdog.

Codex sends Responses requests to Agency's authenticated gateway; Agency translates these to LiteLLM's OpenAI-compatible chat-completions interface. `model_provider`, `base_url`, `env_key`, and `wire_api = "responses"` follow the [official Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference). The requested model has an [official GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna); this does not establish availability on your LiteLLM deployment.

Only a per-task random gateway token reaches the container. The real LiteLLM credential stays in host worker memory/environment, transferred through an anonymous pipe and scrubbed from stored traces/errors. The base URL is recorded only as a hash for comparison. Run directories are mode 0700 with restrictive file creation permissions. Gateway servers bind on the host so bridge containers can reach them through `host.docker.internal`; firewall access to the Docker bridge/trusted host as appropriate. This implementation inherits that binding behavior from the existing Agency gateway. Do not expose the benchmark host as a public gateway.

## Freeze the workload and images

Fetch an explicitly pinned commit of the official Verified dataset on Ubuntu. Substitute a real 40-character dataset-repository SHA; this step downloads data but does not execute tasks or call models:

```bash
.venv/bin/python -m benchmarks.cpu_oversubscription fetch \
  --revision "$VERIFIED_DATASET_COMMIT" --out /srv/agency-data/verified
```

This writes the Hugging Face snapshot, provenance/hash, and `rows.json`. The latter contains evaluator gold and must remain outside containers. Both configs' `dataset_snapshot` must point to it. Hashes and selected task fields are checked again before execution and grading. Official grading uses this local JSON snapshot, supported by the [official dataset loader](https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/utils.py), to avoid evaluating a moving dataset revision.

Inspect counts by `(repo, version)` in the downloaded data and choose a group with at least 30 tasks. This is a conservative homogeneous workload profile, not a claim that all tasks have identical difficulty. Select before observing agent results. Keep repo, version, and seed identical in these two invocations:

```bash
.venv/bin/python -m benchmarks.cpu_oversubscription select \
  --dataset-dir /srv/agency-data/verified --repo "$TASK_REPO" --version "$TASK_VERSION" \
  --seed 1729 --out /srv/agency-data/image-map.json
```

Without `--images`, selection writes an image-map template containing **real sampled instance IDs**. Prepare official instance images for those IDs with the installed SWE-bench image tooling. Check that each has a pristine `/testbed` at its dataset base commit, `/opt/miniconda3/envs/testbed/bin/python`, `/bin/sh`, Git, and a compatible runtime for the reviewed Codex ELF binary. If a derived image is necessary for binary dependencies, make only that dependency addition, save its Dockerfile, and use the identical resulting image for both conditions. The runner verifies these constraints before starting the batch and never downloads a missing image itself.

Replace each image-map value with its immutable local ID from `docker image inspect --format '{{.Id}}' IMAGE`. Then freeze the manifest:

```bash
.venv/bin/python -m benchmarks.cpu_oversubscription select \
  --dataset-dir /srv/agency-data/verified --repo "$TASK_REPO" --version "$TASK_VERSION" \
  --seed 1729 --images /srv/agency-data/image-map.json \
  --out benchmarks/cpu_oversubscription/manifest.json
```

By default this selects 30 distinct tasks and splits the deterministic sample into six groups of five. Optional `--replicated` selects five tasks and repeats the same ordered group for all six workers; use the flag during both image-map and manifest selection. Replicated attempts still use 30 separate containers and have a denominator of 30. Do not compare replicated and distinct modes.

Populate the Linux binary SHA256/version in both configs, review the generated manifest, and version-control them with the implementation before running. The runner requires a clean Git checkout so a recorded commit identifies executable code. No commit was made as part of this implementation. Keep run directories and evaluator data outside the checkout or in ignored paths.

## Commands for deployment and trials

Run commands from the repository root, using the same virtual environment throughout. Dry planning is safe on macOS **after** valid frozen configuration/manifest files exist:

```bash
.venv/bin/python -m benchmarks.cpu_oversubscription plan \
  --config benchmarks/cpu_oversubscription/configs/dcb.json
```

Add `--topology /path/to/topology.json` for offline CPU assignment inspection. That JSON is an array of integer `cpu`, `core`, `socket`, and `node` records copied from a target-host inventory, not assumed Mac CPU IDs. On Ubuntu, `preflight` prints the actual assignment and full non-secret provenance without launching agents:

```bash
sudo --preserve-env=LITELLM_API_KEY,LITELLM_BASE_URL .venv/bin/python \
  -m benchmarks.cpu_oversubscription preflight --config benchmarks/cpu_oversubscription/configs/dcb.json
sudo --preserve-env=LITELLM_API_KEY,LITELLM_BASE_URL .venv/bin/python \
  -m benchmarks.cpu_oversubscription preflight --config benchmarks/cpu_oversubscription/configs/lsb.json
```

Explicit DCB invocation:

```bash
sudo --preserve-env=LITELLM_API_KEY,LITELLM_BASE_URL .venv/bin/python \
  -m benchmarks.cpu_oversubscription run --config benchmarks/cpu_oversubscription/configs/dcb.json \
  --trial-id trial-001 --out /srv/agency-runs/trial-001-dcb
```

Explicit LSB invocation, with a new independent 1,800-second budget:

```bash
sudo --preserve-env=LITELLM_API_KEY,LITELLM_BASE_URL .venv/bin/python \
  -m benchmarks.cpu_oversubscription run --config benchmarks/cpu_oversubscription/configs/lsb.json \
  --trial-id trial-001 --out /srv/agency-runs/trial-001-lsb
```

Do not grade while another condition is active. After both timed runs, invoke official grading and comparison:

```bash
sudo .venv/bin/python -m benchmarks.cpu_oversubscription evaluate /srv/agency-runs/trial-001-dcb
sudo .venv/bin/python -m benchmarks.cpu_oversubscription evaluate /srv/agency-runs/trial-001-lsb
sudo .venv/bin/python -m benchmarks.cpu_oversubscription compare \
  /srv/agency-runs/trial-001-dcb /srv/agency-runs/trial-001-lsb \
  --out /srv/agency-runs/comparison-001
```

For independent repeats, use new trial IDs/output directories. Counterbalance condition order across trials, record idle/cooldown procedures, and keep the configuration, manifest, code commit, dependencies, binary, kernel, hardware, and endpoint fixed. `compare` accepts multiple complete DCB/LSB pairs and emits per-trial metrics/differences and throughput ratios. It rejects mismatched identities, duplicate/missing conditions, pending official grades, infrastructure failures, and unknown-cutoff recovery results. It intentionally does not pool heterogeneous runs or claim statistical significance.

The comparison JSON also aggregates compatible trials by condition: pass rate uses `30 × trial_count`, average duration pools observed task durations, and throughput divides total completions by summed batch elapsed time. It records how many trials reached their deadlines.

## Deadline, results, and three primary metrics

The watchdog is a separate process on the reserved cores and inherits the host lock. It survives loss of the controller, detects that loss, and terminates the experiment. At the monotonic deadline it first persists the cutoff, then sends SIGTERM to all experiment-cgroup descendants, including detached subprocesses and Docker tools. After the configurable `termination_grace_seconds` (default five), `cgroup.kill` forces termination of anything remaining. Grace is **shutdown time only**. No completion at or after the deadline is credited; task durations and throughput never include shutdown grace. Deadline enforcement has normal OS scheduling latency, recorded as `signal_started - cutoff`; this is not a hard-real-time kernel guarantee.

Rows are allocated for all 30 assignments before work begins. A start journal entry is written immediately before launching the Codex task. This conservative boundary includes Docker-exec launch overhead. A finish is written only after patch export and container termination; task duration includes this small finalization cost as well as all tools and inference waits. Gateway creation/configuration between tasks counts toward the batch budget but precedes each task's start. Dataset/image preparation and official grading are excluded.

| Task state | Representation |
|---|---|
| Finished before cutoff, regardless of process exit code | `status=completed`; graded officially |
| Started but interrupted by deadline | `status=timed_out`, `reason=deadline`, `censored=true` |
| Never started before deadline | `status=unattempted`, `reason=deadline`, null duration |
| Started but operator/infrastructure interruption | `status=interrupted`; run invalid for baseline comparison |

1. **Pass rate:** successful completed attempts / **30 assigned tasks**. All timeouts, failures, and never-started assignments remain in the denominator. Timeout/interrupted rows cannot earn a success from a late patch. Missing official verdicts remain `null`/pending rather than silently becoming failures or successes.
2. **Average task wall-clock time:** arithmetic mean of the durations of started tasks. For a deadline-interrupted task, observed duration is `deadline - task_start` and `censored=true`. Never-started tasks have no observation and are excluded from this mean, not assigned zero or 1,800 seconds. Report the observed/censored sample counts alongside the mean; this is a descriptive censored mean, not an estimate of eventual uncensored completion time.
3. **Throughput:** completed task attempts / (`min(actual_batch_elapsed, condition_timeout_seconds) / 3600`). A nonzero-exit completed attempt still counts; interrupted and never-started tasks do not. When all 30 finish, this reduces to `30 / elapsed_hours`. Successful tasks/hour is an auxiliary derivation using the same elapsed time.

Example: one completed success, one task interrupted after 1,770 seconds, and 28 never started in a 1,800-second batch yields pass rate `1/30` and completed throughput `2/hour`. The mean uses the completed duration and the one censored duration, not all 30 assignments. The comparison table explicitly highlights **YES** for each condition that reached its deadline.

## Artifacts and telemetry

Each output directory contains:

- `plan.json`, `assigned.json`: frozen assignments, configuration, manifest hash, code commit, software versions, topology, scheduling settings, experiment resource names, and comparison identity.
- `start.json`, `end.json`: monotonic deadline, boot ID, UTC start, cutoff reason, signal/cleanup timestamps and outcome.
- `worker-*/journal.jsonl`: durable task starts/completions, tolerant of a truncated final line after forced termination.
- `aN-tM/codex.log`, `events.jsonl`: combined CLI stdout/stderr and flushed Agency gateway events, including partial prefixes for interrupted tasks.
- `aN-tM/prediction.patch`, `prediction.jsonl`, `gateway-result.json`: completed-task artifacts. Interrupted tasks retain logs; exporting interrupted filesystem patches is not guaranteed.
- `aN-tM/grade.json`, evaluation stdout/stderr and official reports: untimed grading evidence, kept separately per attempt even in replicated mode.
- `telemetry.jsonl`: `/proc/stat` CPU counters and host/cgroup CPU pressure, usage/throttling counters, quotas, and effective cpusets. Derive utilization from counter deltas; telemetry is auxiliary, not a fourth primary metric.
- `results.json`, `tasks.csv`, `metrics.csv`, `cleanup.json`: reconciled rows, measurements, deadline status, and resource cleanup errors.

## Interruption, recovery, and cleanup

Output directories are never reused by `run`. A half-completed condition cannot resume under a fresh budget. Recovery only seals existing journals, preserves grades, and removes owned resources:

```bash
sudo .venv/bin/python -m benchmarks.cpu_oversubscription recover /srv/agency-runs/trial-001-dcb
```

The host lock covers run, watchdog, evaluation, and recovery. If it is busy, wait for the owning process/watchdog or investigate it; do not remove the lock file to bypass mutual exclusion. Controller SIGINT/SIGTERM requests cancellation. A pre-start failure cancels preparation; it is not a timed baseline. If a host reboot or watchdog failure left no durable cutoff, recovery uses only the last provable journal timestamp, marks timing unknown, and excludes that trial from comparisons. Start a new independent trial to replace it. `evaluate` can resume missing grades without rerunning agents.

Only containers carrying the matching run label/name and recorded worker cgroups are removed. No image, volume, unrelated container, or host-wide Docker prune is performed. If kernel tasks cannot be killed (for example uninterruptible I/O), cleanup reports failure and the run is not accepted. Inspect `systemd-cgls /agencycpu.slice`, `cleanup.json`, and the recorded resource map. Do not manually kill arbitrary host processes to force a clean preflight.

Runtime background restrictions and `agencycpu.slice` remain until explicitly restored or rebooted. After all runs/evaluation are finished and the experiment slice is empty, restore each saved `AllowedCPUs` value using `systemctl set-property --runtime UNIT AllowedCPUs=VALUE`, stop the experiment slice, remove only `/run/systemd/system/agencycpu.slice`, and run `systemctl daemon-reload`. An originally unrestricted value can be restored with `AllowedCPUs=`. Do not clear pre-existing administrator restrictions. A reboot clears the runtime configuration; firmware SMT changes require their own restoration.

## Limitations, confounders, and deployment checks

The following are unverified until the Ubuntu host is available: Docker/systemd cgroup parenting and delegation permissions, effective cpuset propagation, host-gateway routing/firewall, Linux Codex compatibility in each image, exact LiteLLM model alias and supported generation settings, official dataset/harness schema compatibility, and graceful/forced teardown behavior. The supplied unit tests exercise pure logic only; future Ubuntu validation must explicitly test detached subprocesses, cancellation, controller loss, deadline-boundary events, failed workers, and cgroup cleanup before collecting research results.

Preflight intentionally makes no model availability probe. Provider failures, rate limits, gateway/API latency, token/cache behavior, and remote model nondeterminism can dominate CPU effects. The host gateway CPU cost is included on the worker's assigned core, while Docker-daemon work is on reserved cores. Default Linux cgroup scheduling/fairness applies separately to host worker and container cgroups. This is part of this baseline, and a future policy must preserve or explicitly account for it.

Other confounders include filesystem/page caches and image preparation order, provider caches across conditions, thermal/turbo state, memory pressure/swap, the shared last-level cache, background interrupts, unequal task difficulty, gateway startup and patch-export overhead, sequential-task ordering under censoring, and grading/image drift. Thirty sleeping prepared containers add small resident/init overhead equally in both conditions. Record and counterbalance order; do not drop timeouts to make results look better. Stronger IRQ/kernel isolation and workload stratification require a separately reviewed experiment protocol.

Troubleshooting: SMT/topology failures require fixing firmware/core availability; cpuset/background failures require reviewing the runtime isolation script and custom slices; hash/image failures require freezing the intended artifacts; readiness failures leave `_worker-error-*` files; endpoint errors appear in redacted gateway traces; pending grades require checking official evaluator logs and package compatibility. A completed coding process is not proof of task success.

For a future agent-aware scheduler, add a placement/policy implementation around `layout_from_rows`, `worker_cgroup`, and container placement while retaining the same backend, manifest, barrier/deadline, official grader, and metric definitions. Record the new policy and decisions in provenance/telemetry and extend comparison validation deliberately. Do not change model budgets or censoring rules as part of scheduler integration.
