# agprof summary

- Duration: **4.871 s**
- Runs: **0/0 completed**, 0 succeeded, 0 failed, 0 interrupted
- Completed throughput: **0.000 runs/s**
- LLM: **0 calls**, 0 succeeded, 0 failed, 0 interrupted, 0 retries, 0.000 s total wait
- Tools: **0/0 completed**, 0 failed, 0 interrupted
- Raw resource samples: **1159** at 9.731 Hz effective (10 Hz configured)
- GPU sampling: **available** (requested)

## Run, LLM, and tool metrics

| Metric | Value |
|---|---:|
| Run latency p50 / p95 | n/a / n/a ms |
| LLM latency p50 / p95 | n/a / n/a ms |
| LLM TTFT p50 / p95 | n/a / n/a ms |
| LLM input / output tokens | 0 / 0 |
| LLM output throughput | n/a tokens/s |
| LLM attempts | 0 total, 0 succeeded, 0 failed, 0 interrupted |
| Tool latency p50 / p95 | n/a / n/a ms |

### Tool outcomes

_No tool spans were recorded._

## Workload aggregate

| CPU avg | CPU peak | CPU time | Memory avg | Memory peak | Disk read | Disk write |
|---:|---:|---:|---:|---:|---:|---:|
| 50.946% | 125.241% | 2.408 s | 195.468 MB | 257.902 MB | 0.000000 MB | 0.011719 MB |

## Per-process metrics

| Process | PID | Sandbox | Samples | CPU avg | CPU peak | CPU time | RSS avg | RSS peak | VMS avg | VMS peak | Disk read | Disk write |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| docker | 307641 |  | 1 | n/a% | n/a% | n/a s | 24.395 MB | 24.395 MB | 2271.438 MB | 2271.438 MB | n/a MB | n/a MB |
| python | 307555 |  | 47 | 2.963% | 19.459% | 0.140 s | 93.242 MB | 93.297 MB | 410.045 MB | 410.055 MB | 0.000000 MB | 0.000000 MB |
| docker | 307666 |  | 3 | 4.899% | 9.799% | 0.010 s | 27.802 MB | 27.969 MB | 2272.008 MB | 2272.008 MB | 0.000000 MB | 0.000000 MB |
| runc:[2:INIT] | 307714 | ec2_m1_probe | 19 | 2.709% | 48.764% | 0.050 s | 1.758 MB | 14.285 MB | 166.387 MB | 1572.766 MB | n/a MB | n/a MB |
| docker | 307841 |  | 1 | n/a% | n/a% | n/a s | 27.973 MB | 27.973 MB | 2200.191 MB | 2200.191 MB | n/a MB | n/a MB |
| tail | 307820 | ec2_m1_probe | 17 | 0.000% | 0.000% | 0.000 s | 0.801 MB | 0.801 MB | 1.602 MB | 1.602 MB | n/a MB | n/a MB |
| runc:[2:INIT] | 307867 |  | 1 | n/a% | n/a% | n/a s | 11.043 MB | 11.043 MB | 1571.688 MB | 1571.688 MB | n/a MB | n/a MB |
| docker | 307908 |  | 1 | n/a% | n/a% | n/a s | 24.477 MB | 24.477 MB | 2271.688 MB | 2271.688 MB | n/a MB | n/a MB |
| dd | 307974 | ec2_m1_probe | 1 | n/a% | n/a% | n/a s | 1.816 MB | 1.816 MB | 2.598 MB | 2.598 MB | n/a MB | n/a MB |
| sh | 307968 | ec2_m1_probe | 15 | 100.132% | 107.401% | 1.440 s | 1.065 MB | 1.082 MB | 1.613 MB | 1.613 MB | n/a MB | n/a MB |
| docker | 307992 |  | 4 | 0.000% | 0.000% | 0.000 s | 27.695 MB | 27.695 MB | 2272.195 MB | 2272.195 MB | 0.000000 MB | 0.000000 MB |
| docker | 308087 |  | 2 | 0.000% | 0.000% | 0.000 s | 27.195 MB | 27.195 MB | 2271.945 MB | 2271.945 MB | 0.000000 MB | 0.000000 MB |
| runc:[2:INIT] | 308134 | ec2_m1_probe | 17 | 0.000% | 0.000% | 0.000 s | 1.311 MB | 13.473 MB | 93.460 MB | 1572.203 MB | n/a MB | n/a MB |
| docker | 308233 |  | 1 | n/a% | n/a% | n/a s | 5.113 MB | 5.113 MB | 65.527 MB | 65.527 MB | n/a MB | n/a MB |
| tail | 308197 | ec2_m1_probe | 16 | 0.000% | 0.000% | 0.000 s | 0.832 MB | 0.832 MB | 1.602 MB | 1.602 MB | n/a MB | n/a MB |
| sleep | 308292 | ec2_m1_probe | 15 | 0.000% | 0.000% | 0.000 s | 0.812 MB | 0.812 MB | 1.598 MB | 1.598 MB | n/a MB | n/a MB |
| docker | 308332 |  | 3 | 0.000% | 0.000% | 0.000 s | 26.445 MB | 26.445 MB | 2344.949 MB | 2344.949 MB | 0.000000 MB | 0.000000 MB |
| docker | 308396 |  | 1 | n/a% | n/a% | n/a s | 9.715 MB | 9.715 MB | 1477.824 MB | 1477.824 MB | n/a MB | n/a MB |

## GPU metrics

| GPU | Util avg | Util peak | VRAM avg | VRAM peak | Power avg | Power peak | Energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000% | 0.000% | 468.875 MB | 468.875 MB | 27.980 W | 28.111 W | 132.271 J |

## Sandbox metrics

| Sandbox | CPU avg | CPU peak | CPU time | Memory avg | Memory peak | Disk read | Disk write | Net receive | Net transmit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ec2_m1_probe | 49.982% | 100.572% | 1.438 s | 86.154 MB | 102.582 MB | 0.000000 MB | 0.000000 MB | 0.000683 MB | 0.000080 MB |

## Incomplete spans

_No spans were still open when profiling stopped._

## Span metrics

| Label | Completed/started | Failed | Interrupted | Wall (s) | CPU (s) | Blocked (s) | Mean (ms) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sandbox:start | 4/4 | 0 | 0 | 0.817 | 0.004 | 0.812 | 204.166 | 143.732 | 478.449 |
| run:detect | 1/1 | 0 | 0 | 0.045 | 0.001 | 0.044 | 44.543 | 44.543 | 44.543 |
| sync:container | 20/20 | 0 | 0 | 0.002 | 0.002 | 0.000 | 0.107 | 0.079 | 0.199 |

## Resource metrics

| Metric | Unit | Samples | Mean | Min | Max | Last | Total | Energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dockerd CPU | percent | 40 | 7.865 | 0.000 | 65.297 | 25.595 | 0.322458 CPU seconds | n/a |
| GPU 0 memory | MB | 47 | 468.875 | 468.875 | 468.875 | 468.875 | n/a | n/a |
| GPU 0 power | W | 47 | 27.980 | 27.903 | 28.111 | 27.966 | n/a | 132.271 J |
| GPU 0 utilization | percent | 47 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| python (PID 307555) CPU | percent | 46 | 2.963 | 0.000 | 19.459 | 9.749 | 0.140000 CPU seconds | n/a |
| python (PID 307555) io read MB/s | MB/s | 46 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| python (PID 307555) io write MB/s | MB/s | 46 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| python (PID 307555) rss_mb | MB | 47 | 93.242 | 92.738 | 93.297 | 93.297 | n/a | n/a |
| python (PID 307555) vms_mb | MB | 47 | 410.045 | 410.039 | 410.055 | 410.055 | n/a | n/a |
| docker (PID 307641) rss_mb | MB | 1 | 24.395 | 24.395 | 24.395 | 24.395 | n/a | n/a |
| docker (PID 307641) vms_mb | MB | 1 | 2271.438 | 2271.438 | 2271.438 | 2271.438 | n/a | n/a |
| docker (PID 307666) CPU | percent | 2 | 4.899 | 0.000 | 9.799 | 0.000 | 0.010000 CPU seconds | n/a |
| docker (PID 307666) io read MB/s | MB/s | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 307666) io write MB/s | MB/s | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 307666) rss_mb | MB | 3 | 27.802 | 27.469 | 27.969 | 27.969 | n/a | n/a |
| docker (PID 307666) vms_mb | MB | 3 | 2272.008 | 2272.008 | 2272.008 | 2272.008 | n/a | n/a |
| docker-init [ec2_m1_probe] (PID 307714) CPU | percent | 18 | 2.709 | 0.000 | 48.764 | 0.000 | 0.050000 CPU seconds | n/a |
| docker-init [ec2_m1_probe] (PID 307714) rss_mb | MB | 19 | 1.758 | 0.551 | 14.285 | 0.551 | n/a | n/a |
| docker-init [ec2_m1_probe] (PID 307714) vms_mb | MB | 19 | 166.387 | 1.039 | 1572.766 | 1.039 | n/a | n/a |
| tail [ec2_m1_probe] (PID 307820) CPU | percent | 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| tail [ec2_m1_probe] (PID 307820) rss_mb | MB | 17 | 0.801 | 0.801 | 0.801 | 0.801 | n/a | n/a |
| tail [ec2_m1_probe] (PID 307820) vms_mb | MB | 17 | 1.602 | 1.602 | 1.602 | 1.602 | n/a | n/a |
| docker (PID 307841) rss_mb | MB | 1 | 27.973 | 27.973 | 27.973 | 27.973 | n/a | n/a |
| docker (PID 307841) vms_mb | MB | 1 | 2200.191 | 2200.191 | 2200.191 | 2200.191 | n/a | n/a |
| runc:[2:INIT] (PID 307867) rss_mb | MB | 1 | 11.043 | 11.043 | 11.043 | 11.043 | n/a | n/a |
| runc:[2:INIT] (PID 307867) vms_mb | MB | 1 | 1571.688 | 1571.688 | 1571.688 | 1571.688 | n/a | n/a |
| docker (PID 307908) rss_mb | MB | 1 | 24.477 | 24.477 | 24.477 | 24.477 | n/a | n/a |
| docker (PID 307908) vms_mb | MB | 1 | 2271.688 | 2271.688 | 2271.688 | 2271.688 | n/a | n/a |
| sh [ec2_m1_probe] (PID 307968) CPU | percent | 14 | 100.132 | 97.000 | 107.401 | 107.401 | 1.440000 CPU seconds | n/a |
| sh [ec2_m1_probe] (PID 307968) rss_mb | MB | 15 | 1.065 | 0.824 | 1.082 | 1.082 | n/a | n/a |
| sh [ec2_m1_probe] (PID 307968) vms_mb | MB | 15 | 1.613 | 1.613 | 1.613 | 1.613 | n/a | n/a |
| dd [ec2_m1_probe] (PID 307974) rss_mb | MB | 1 | 1.816 | 1.816 | 1.816 | 1.816 | n/a | n/a |
| dd [ec2_m1_probe] (PID 307974) vms_mb | MB | 1 | 2.598 | 2.598 | 2.598 | 2.598 | n/a | n/a |
| docker (PID 307992) CPU | percent | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| docker (PID 307992) io read MB/s | MB/s | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 307992) io write MB/s | MB/s | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 307992) rss_mb | MB | 4 | 27.695 | 27.695 | 27.695 | 27.695 | n/a | n/a |
| docker (PID 307992) vms_mb | MB | 4 | 2272.195 | 2272.195 | 2272.195 | 2272.195 | n/a | n/a |
| docker (PID 308087) CPU | percent | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| docker (PID 308087) io read MB/s | MB/s | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 308087) io write MB/s | MB/s | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 308087) rss_mb | MB | 2 | 27.195 | 27.195 | 27.195 | 27.195 | n/a | n/a |
| docker (PID 308087) vms_mb | MB | 2 | 2271.945 | 2271.945 | 2271.945 | 2271.945 | n/a | n/a |
| docker-init [ec2_m1_probe] (PID 308134) CPU | percent | 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| docker-init [ec2_m1_probe] (PID 308134) rss_mb | MB | 17 | 1.311 | 0.551 | 13.473 | 0.551 | n/a | n/a |
| docker-init [ec2_m1_probe] (PID 308134) vms_mb | MB | 17 | 93.460 | 1.039 | 1572.203 | 1.039 | n/a | n/a |
| tail [ec2_m1_probe] (PID 308197) CPU | percent | 15 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| tail [ec2_m1_probe] (PID 308197) rss_mb | MB | 16 | 0.832 | 0.832 | 0.832 | 0.832 | n/a | n/a |
| tail [ec2_m1_probe] (PID 308197) vms_mb | MB | 16 | 1.602 | 1.602 | 1.602 | 1.602 | n/a | n/a |
| docker (PID 308233) rss_mb | MB | 1 | 5.113 | 5.113 | 5.113 | 5.113 | n/a | n/a |
| docker (PID 308233) vms_mb | MB | 1 | 65.527 | 65.527 | 65.527 | 65.527 | n/a | n/a |
| sleep [ec2_m1_probe] (PID 308292) CPU | percent | 14 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| sleep [ec2_m1_probe] (PID 308292) rss_mb | MB | 15 | 0.812 | 0.812 | 0.812 | 0.812 | n/a | n/a |
| sleep [ec2_m1_probe] (PID 308292) vms_mb | MB | 15 | 1.598 | 1.598 | 1.598 | 1.598 | n/a | n/a |
| docker (PID 308332) CPU | percent | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| docker (PID 308332) io read MB/s | MB/s | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 308332) io write MB/s | MB/s | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 308332) rss_mb | MB | 3 | 26.445 | 26.445 | 26.445 | 26.445 | n/a | n/a |
| docker (PID 308332) vms_mb | MB | 3 | 2344.949 | 2344.949 | 2344.949 | 2344.949 | n/a | n/a |
| docker (PID 308396) rss_mb | MB | 1 | 9.715 | 9.715 | 9.715 | 9.715 | n/a | n/a |
| docker (PID 308396) vms_mb | MB | 1 | 1477.824 | 1477.824 | 1477.824 | 1477.824 | n/a | n/a |
| sandbox ec2_m1_probe CPU | percent | 28 | 49.982 | 0.000 | 100.572 | 0.000 | 1.437601 CPU seconds | n/a |
| sandbox ec2_m1_probe io read MB/s | MB/s | 29 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| sandbox ec2_m1_probe io write MB/s | MB/s | 29 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| sandbox ec2_m1_probe memory | MB | 30 | 86.154 | 69.918 | 102.582 | 69.918 | n/a | n/a |
| sandbox ec2_m1_probe net rx MB/s | MB/s | 28 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000683 MB | n/a |
| sandbox ec2_m1_probe net tx MB/s | MB/s | 28 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000080 MB | n/a |
| workload total CPU | percent | 46 | 50.946 | 1.975 | 125.241 | 10.611 | 2.407538 CPU seconds | n/a |
| workload total io read MB/s | MB/s | 39 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| workload total io write MB/s | MB/s | 39 | 0.003 | 0.000 | 0.114 | 0.000 | 0.011719 MB | n/a |
| workload total memory | MB | 47 | 195.468 | 83.539 | 257.902 | 83.539 | n/a | n/a |

## GPU lease metrics

_No GPU leases were recorded._
