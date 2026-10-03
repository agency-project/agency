# Measurement evidence and limits

This page distinguishes historical experiments from verification executed for the documentation. No provider, container benchmark, specialized checkpoint host or external evaluator was rerun during Stage 3. The [inspection snapshot](measurement-evidence.json) records selected configuration/source identities, artifact hashes and recomputed summaries; it is not a replacement for the raw runs.

## Historical checkpoint cohorts: September 13, 2026

Local artifacts are present under `benchmarks/checkpoint_size_microbenchmark/results/fast-checkpoint-cow-only-20260913/`: cohort manifests, per-condition run.json/records.json and validation/report files. This results directory is Git-ignored and is **not shipped in a clean checkout or the documentation site**. The historical README describes Linux EC2 experiments with a sparse 16 GiB ZFS pool, dedicated Docker ZFS graphdriver and Podman private ZFS-backed storage. Its COW-only path used fresh harness processes, without optional CRIU reuse.

The inspected manifests specify 4 sandbox CPUs, 8g memory, OpenAI/gpt-5.6-luna, 5 Hz sampling and GPU sampling disabled. The [runner](../../benchmarks/checkpoint_size_microbenchmark/runner.py) fixes an instrumented Requests SWE-bench image, payload generation and randomized order. Mutation sizes are 0 B, 64 KiB, 128 MiB and 1 GiB; three initial replicates per size, with two extra 1 GiB replicates after a prespecified spread rule. The recorded source is a dirty snapshot with executing-source hash `581526015367ca5fe03f4a3daa2a3cdb3d894fefbeb81bdcb95ac152294b8039`, not the current checkout or simply the manifest's older base Git revision.

The following medians were **recomputed from existing records**, summing checkpoint.total durations within each completed run and taking the median across its size cohort:

| Historical runtime/backend | 0 B, n=3 | 64 KiB, n=3 | 128 MiB, n=3 | 1 GiB, n=5 |
| --- | ---: | ---: | ---: | ---: |
| Docker cow_zfs | 0.322947 s | 0.315347 s | 0.329864 s | 0.366492 s |
| Podman cow_zfs | 0.581923 s | 0.584925 s | 0.542403 s | 0.574548 s |

These are checkpoint envelopes for those historical configurations. Restore.fs/restore.total, container start, fresh daemon/PTY startup and complete invocation time have separate scopes; checkpoint latency is not end-to-end latency. Nested spans must not be summed as independent costs. The [comparison script](../../benchmarks/checkpoint_size_microbenchmark/compare_backends.py) checks completed cohorts, source identity and selected matching config, but those checks do not remove host/storage or old/new implementation confounds. Retained OCI/CRIU prototype comparisons do not establish present-backend superiority. The cohort's bounded dirty-size observations do not prove constant time for every workload/image/storage system.

## Historical trajectory evidence

The saved E1A.1 pair_metrics.csv contains 15 matched pairs across five tasks and three repetitions. Raw trajectory index/event streams, integrity report and campaign archive are present locally under ignored `artifacts/e1a1/`; those raw files are also absent from a clean checkout/site. [Figure methodology](../assets/figures/e1a1/README.md) explains longest-common-subsequence similarity over coarse action categories; it ignores exact commands, model text, tool results and timing. [Figure code](../assets/figures/e1a1/make_trajectory_figure.py) checks source counts and patch evidence.

Reading the existing pair CSV confirms 15/15 contended runs were slower, median wall-time ratio 1.747021, median similarity difference −0.003289, 15/15 matching final-patch file sets and 0/15 byte-identical patches. These are descriptive saved-data results, not a new experiment or proof that contention preserves trajectories. There are five task clusters, no established equivalence margin, and category matching can hide meaningful command/code differences. The original environment/campaign integrity evidence must travel with any broader performance claim.

## Failed stress execution is evidence of a limit

The [retained local N=1 result](../../benchmarks/agent_stress/results/local_n1/summary.md) reports blocked execution on macOS ARM64/Python 3.12.13 because the Docker socket was unavailable. It has no successful phases and proves no concurrency ceiling or scaling curve. EC2 preparation records are setup evidence, not successful stress throughput. The [current stress README](../../benchmarks/agent_stress/README.md) explicitly distinguishes removed historical runner APIs from current acceptance tests.

## A narrow hot-path measurement

[Persistent MCP state initialization](../../benchmarks/runtime_hotpaths.md) reports a historical single-host macOS ARM64/Python 3.12.13 comparison: 1,000 production callbacks initialize an expensive persistent factory once instead of 1,000 times. The [script](../../benchmarks/runtime_hotpaths.py) excludes models, containers and remote services. Its deterministic factory-call count supports the initialization choice; its 454.540 ms versus 5.048 ms single measurements are not an end-to-end speedup, fresh Stage 3 reproduction or universal overhead estimate. Current host/sandbox MCP code uses a lock around initialization, with tool bodies outside it.

## Evidence needed for a new claim

Record the executing source/configuration, host/runtime/storage/image identities, cold versus subsequent invocation costs, workload correctness, failures, replicate/order policy and timing scope. Preserve raw artifacts with a portable access path; an ignored local directory cannot support a public reproducibility claim by itself. Separate preparation, validation, exploratory analysis and measurements; keep evaluator success independent of harness completion. [Observability limits](../architecture/observability.md#measurement-limits) apply to every profiler-based number above.
