# agprof summary

- Duration: **4.884 s**
- Runs: **0/0 completed**, 0 succeeded, 0 failed, 0 interrupted
- Completed throughput: **0.000 runs/s**
- LLM: **0 calls**, 0 succeeded, 0 failed, 0 interrupted, 0 retries, 0.000 s total wait
- Tools: **0/0 completed**, 0 failed, 0 interrupted
- Raw resource samples: **1175** at 9.737 Hz effective (10 Hz configured)
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
| 50.736% | 102.899% | 2.395 s | 199.619 MB | 253.230 MB | 0.000000 MB | 0.023438 MB |

## Per-process metrics

| Process | PID | Sandbox | Samples | CPU avg | CPU peak | CPU time | RSS avg | RSS peak | VMS avg | VMS peak | Disk read | Disk write |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| python | 304713 |  | 47 | 2.949% | 18.894% | 0.140 s | 93.365 MB | 93.414 MB | 410.045 MB | 410.055 MB | 0.000000 MB | 0.000000 MB |
| docker | 304800 |  | 1 | n/a% | n/a% | n/a s | 24.777 MB | 24.777 MB | 2200.684 MB | 2200.684 MB | n/a MB | n/a MB |
| docker | 304824 |  | 3 | 0.000% | 0.000% | 0.000 s | 26.648 MB | 26.648 MB | 2336.516 MB | 2336.516 MB | 0.000000 MB | 0.000000 MB |
| runc:[2:INIT] | 304872 | ec2_m1_probe | 19 | 1.628% | 29.312% | 0.030 s | 1.833 MB | 14.527 MB | 166.400 MB | 1572.766 MB | n/a MB | n/a MB |
| docker | 305001 |  | 1 | n/a% | n/a% | n/a s | 27.387 MB | 27.387 MB | 2200.449 MB | 2200.449 MB | n/a MB | n/a MB |
| tail | 304979 | ec2_m1_probe | 17 | 0.000% | 0.000% | 0.000 s | 0.789 MB | 0.789 MB | 1.602 MB | 1.602 MB | n/a MB | n/a MB |
| runc:[2:INIT] | 305027 |  | 1 | n/a% | n/a% | n/a s | 11.152 MB | 11.152 MB | 1572.203 MB | 1572.203 MB | n/a MB | n/a MB |
| docker | 305099 |  | 1 | n/a% | n/a% | n/a s | 4.863 MB | 4.863 MB | 65.527 MB | 65.527 MB | n/a MB | n/a MB |
| sh | 305126 | ec2_m1_probe | 15 | 99.612% | 107.601% | 1.430 s | 0.816 MB | 0.816 MB | 1.613 MB | 1.613 MB | n/a MB | n/a MB |
| docker | 305149 |  | 3 | 0.000% | 0.000% | 0.000 s | 27.449 MB | 27.449 MB | 2336.266 MB | 2336.266 MB | 0.000000 MB | 0.000000 MB |
| docker | 305230 |  | 1 | n/a% | n/a% | n/a s | 1.801 MB | 1.801 MB | 65.527 MB | 65.527 MB | n/a MB | n/a MB |
| docker | 305246 |  | 1 | n/a% | n/a% | n/a s | 26.891 MB | 26.891 MB | 2272.461 MB | 2272.461 MB | n/a MB | n/a MB |
| runc:[2:INIT] | 305293 | ec2_m1_probe | 18 | 1.145% | 19.465% | 0.020 s | 1.204 MB | 12.316 MB | 88.269 MB | 1571.172 MB | n/a MB | n/a MB |
| nvidia-cdi-hook | 305300 |  | 1 | n/a% | n/a% | n/a s | 8.023 MB | 8.023 MB | 1661.883 MB | 1661.883 MB | n/a MB | n/a MB |
| tail | 305354 | ec2_m1_probe | 17 | 0.000% | 0.000% | 0.000 s | 0.816 MB | 0.816 MB | 1.602 MB | 1.602 MB | n/a MB | n/a MB |
| docker | 305423 |  | 1 | n/a% | n/a% | n/a s | 23.191 MB | 23.191 MB | 1622.582 MB | 1622.582 MB | n/a MB | n/a MB |
| sleep | 305449 | ec2_m1_probe | 15 | 0.000% | 0.000% | 0.000 s | 0.844 MB | 0.844 MB | 1.598 MB | 1.598 MB | n/a MB | n/a MB |
| docker | 305489 |  | 4 | 0.000% | 0.000% | 0.000 s | 27.496 MB | 27.496 MB | 2200.191 MB | 2200.191 MB | 0.000000 MB | 0.000000 MB |

## GPU metrics

| GPU | Util avg | Util peak | VRAM avg | VRAM peak | Power avg | Power peak | Energy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000% | 0.000% | 468.875 MB | 468.875 MB | 28.013 W | 28.093 W | 132.349 J |

## Sandbox metrics

| Sandbox | CPU avg | CPU peak | CPU time | Memory avg | Memory peak | Disk read | Disk write | Net receive | Net transmit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ec2_m1_probe | 50.258% | 100.702% | 1.546 s | 80.822 MB | 103.133 MB | 0.000000 MB | 0.000000 MB | 0.000698 MB | 0.000080 MB |

## Incomplete spans

_No spans were still open when profiling stopped._

## Span metrics

| Label | Completed/started | Failed | Interrupted | Wall (s) | CPU (s) | Blocked (s) | Mean (ms) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sandbox:start | 4/4 | 0 | 0 | 0.795 | 0.004 | 0.790 | 198.677 | 143.331 | 460.240 |
| run:detect | 1/1 | 0 | 0 | 0.046 | 0.001 | 0.045 | 45.653 | 45.653 | 45.653 |
| sync:container | 20/20 | 0 | 0 | 0.002 | 0.002 | 0.000 | 0.101 | 0.082 | 0.163 |

## Resource metrics

| Metric | Unit | Samples | Mean | Min | Max | Last | Total | Energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dockerd CPU | percent | 41 | 8.114 | 0.000 | 44.888 | 18.037 | 0.341274 CPU seconds | n/a |
| GPU 0 memory | MB | 47 | 468.875 | 468.875 | 468.875 | 468.875 | n/a | n/a |
| GPU 0 power | W | 47 | 28.013 | 27.957 | 28.093 | 28.039 | n/a | 132.349 J |
| GPU 0 utilization | percent | 47 | 0.000 | 0.000 | 0.000 | 0.000 | n/a | n/a |
| python (PID 304713) CPU | percent | 46 | 2.949 | 0.000 | 18.894 | 0.000 | 0.140000 CPU seconds | n/a |
| python (PID 304713) io read MB/s | MB/s | 46 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| python (PID 304713) io write MB/s | MB/s | 46 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| python (PID 304713) rss_mb | MB | 47 | 93.365 | 92.855 | 93.414 | 93.414 | n/a | n/a |
| python (PID 304713) vms_mb | MB | 47 | 410.045 | 410.039 | 410.055 | 410.055 | n/a | n/a |
| docker (PID 304800) rss_mb | MB | 1 | 24.777 | 24.777 | 24.777 | 24.777 | n/a | n/a |
| docker (PID 304800) vms_mb | MB | 1 | 2200.684 | 2200.684 | 2200.684 | 2200.684 | n/a | n/a |
| docker (PID 304824) CPU | percent | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| docker (PID 304824) io read MB/s | MB/s | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 304824) io write MB/s | MB/s | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 304824) rss_mb | MB | 3 | 26.648 | 26.648 | 26.648 | 26.648 | n/a | n/a |
| docker (PID 304824) vms_mb | MB | 3 | 2336.516 | 2336.516 | 2336.516 | 2336.516 | n/a | n/a |
| docker-init [ec2_m1_probe] (PID 304872) CPU | percent | 18 | 1.628 | 0.000 | 29.312 | 0.000 | 0.030000 CPU seconds | n/a |
| docker-init [ec2_m1_probe] (PID 304872) rss_mb | MB | 19 | 1.833 | 0.551 | 14.527 | 0.551 | n/a | n/a |
| docker-init [ec2_m1_probe] (PID 304872) vms_mb | MB | 19 | 166.400 | 1.039 | 1572.766 | 1.039 | n/a | n/a |
| tail [ec2_m1_probe] (PID 304979) CPU | percent | 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| tail [ec2_m1_probe] (PID 304979) rss_mb | MB | 17 | 0.789 | 0.789 | 0.789 | 0.789 | n/a | n/a |
| tail [ec2_m1_probe] (PID 304979) vms_mb | MB | 17 | 1.602 | 1.602 | 1.602 | 1.602 | n/a | n/a |
| docker (PID 305001) rss_mb | MB | 1 | 27.387 | 27.387 | 27.387 | 27.387 | n/a | n/a |
| docker (PID 305001) vms_mb | MB | 1 | 2200.449 | 2200.449 | 2200.449 | 2200.449 | n/a | n/a |
| runc:[2:INIT] (PID 305027) rss_mb | MB | 1 | 11.152 | 11.152 | 11.152 | 11.152 | n/a | n/a |
| runc:[2:INIT] (PID 305027) vms_mb | MB | 1 | 1572.203 | 1572.203 | 1572.203 | 1572.203 | n/a | n/a |
| docker (PID 305099) rss_mb | MB | 1 | 4.863 | 4.863 | 4.863 | 4.863 | n/a | n/a |
| docker (PID 305099) vms_mb | MB | 1 | 65.527 | 65.527 | 65.527 | 65.527 | n/a | n/a |
| sh [ec2_m1_probe] (PID 305126) CPU | percent | 14 | 99.612 | 97.268 | 107.601 | 97.276 | 1.430000 CPU seconds | n/a |
| sh [ec2_m1_probe] (PID 305126) rss_mb | MB | 15 | 0.816 | 0.816 | 0.816 | 0.816 | n/a | n/a |
| sh [ec2_m1_probe] (PID 305126) vms_mb | MB | 15 | 1.613 | 1.613 | 1.613 | 1.613 | n/a | n/a |
| docker (PID 305149) CPU | percent | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| docker (PID 305149) io read MB/s | MB/s | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 305149) io write MB/s | MB/s | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 305149) rss_mb | MB | 3 | 27.449 | 27.449 | 27.449 | 27.449 | n/a | n/a |
| docker (PID 305149) vms_mb | MB | 3 | 2336.266 | 2336.266 | 2336.266 | 2336.266 | n/a | n/a |
| docker (PID 305230) rss_mb | MB | 1 | 1.801 | 1.801 | 1.801 | 1.801 | n/a | n/a |
| docker (PID 305230) vms_mb | MB | 1 | 65.527 | 65.527 | 65.527 | 65.527 | n/a | n/a |
| docker (PID 305246) rss_mb | MB | 1 | 26.891 | 26.891 | 26.891 | 26.891 | n/a | n/a |
| docker (PID 305246) vms_mb | MB | 1 | 2272.461 | 2272.461 | 2272.461 | 2272.461 | n/a | n/a |
| docker-init [ec2_m1_probe] (PID 305293) CPU | percent | 17 | 1.145 | 0.000 | 19.465 | 0.000 | 0.020000 CPU seconds | n/a |
| docker-init [ec2_m1_probe] (PID 305293) rss_mb | MB | 18 | 1.204 | 0.551 | 12.316 | 0.551 | n/a | n/a |
| docker-init [ec2_m1_probe] (PID 305293) vms_mb | MB | 18 | 88.269 | 1.039 | 1571.172 | 1.039 | n/a | n/a |
| nvidia-cdi-hook (PID 305300) rss_mb | MB | 1 | 8.023 | 8.023 | 8.023 | 8.023 | n/a | n/a |
| nvidia-cdi-hook (PID 305300) vms_mb | MB | 1 | 1661.883 | 1661.883 | 1661.883 | 1661.883 | n/a | n/a |
| tail [ec2_m1_probe] (PID 305354) CPU | percent | 16 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| tail [ec2_m1_probe] (PID 305354) rss_mb | MB | 17 | 0.816 | 0.816 | 0.816 | 0.816 | n/a | n/a |
| tail [ec2_m1_probe] (PID 305354) vms_mb | MB | 17 | 1.602 | 1.602 | 1.602 | 1.602 | n/a | n/a |
| docker (PID 305423) rss_mb | MB | 1 | 23.191 | 23.191 | 23.191 | 23.191 | n/a | n/a |
| docker (PID 305423) vms_mb | MB | 1 | 1622.582 | 1622.582 | 1622.582 | 1622.582 | n/a | n/a |
| sleep [ec2_m1_probe] (PID 305449) CPU | percent | 14 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| sleep [ec2_m1_probe] (PID 305449) rss_mb | MB | 15 | 0.844 | 0.844 | 0.844 | 0.844 | n/a | n/a |
| sleep [ec2_m1_probe] (PID 305449) vms_mb | MB | 15 | 1.598 | 1.598 | 1.598 | 1.598 | n/a | n/a |
| docker (PID 305489) CPU | percent | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 CPU seconds | n/a |
| docker (PID 305489) io read MB/s | MB/s | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 305489) io write MB/s | MB/s | 3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| docker (PID 305489) rss_mb | MB | 4 | 27.496 | 27.496 | 27.496 | 27.496 | n/a | n/a |
| docker (PID 305489) vms_mb | MB | 4 | 2200.191 | 2200.191 | 2200.191 | 2200.191 | n/a | n/a |
| sandbox ec2_m1_probe CPU | percent | 30 | 50.258 | 0.000 | 100.702 | 0.000 | 1.546250 CPU seconds | n/a |
| sandbox ec2_m1_probe io read MB/s | MB/s | 31 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| sandbox ec2_m1_probe io write MB/s | MB/s | 31 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| sandbox ec2_m1_probe memory | MB | 32 | 80.822 | 2.738 | 103.133 | 68.879 | n/a | n/a |
| sandbox ec2_m1_probe net rx MB/s | MB/s | 30 | 0.000 | 0.000 | 0.002 | 0.000 | 0.000698 MB | n/a |
| sandbox ec2_m1_probe net tx MB/s | MB/s | 30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000080 MB | n/a |
| workload total CPU | percent | 46 | 50.736 | 2.032 | 102.899 | 2.032 | 2.395097 CPU seconds | n/a |
| workload total io read MB/s | MB/s | 40 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 MB | n/a |
| workload total io write MB/s | MB/s | 39 | 0.006 | 0.000 | 0.114 | 0.000 | 0.023438 MB | n/a |
| workload total memory | MB | 47 | 199.619 | 88.406 | 253.230 | 252.758 | n/a | n/a |

## GPU lease metrics

_No GPU leases were recorded._
