# Agency stress experiment

Status: blocked
Largest N with all children overlapping and all requests successful: 0.

No unrun concurrency level is supported by this result.

| Phase | N | Submit s | Total s | requests/s | Engine peak | Child peak | Pass |
|---|---:|---:|---:|---:|---:|---:|---|

Blocking error: RuntimeError('Sandbox backend unavailable: failed to connect to the docker API at unix:///Users/ericzhou/.docker/run/docker.sock; check if the path is correct and if the daemon is running: dial unix /Users/ericzhou/.docker/run/docker.sock: connect: no such file or directory\n')

Engine intervals are scheduler admission-to-terminal events, including startup/teardown.
Child intervals come from real sandbox Python receipts in existing tool-result logs.
Cold and subsequent invocations both include sandbox lifecycle costs; subsequent calls are not guaranteed hot.
No LLM service is used. Without profiler resource data, poor scaling cannot be attributed to Agency.
See README.md for the later sweep and bottleneck attribution protocol.
