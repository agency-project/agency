# Interactive CLI runtime

`agency codex` and `agency claude` default to the user's existing terminal.
The unmodified CLI inherits stdin, stdout, stderr, and the controlling
terminal. Agency does not allocate another PTY, read the keyboard, parse the
screen to submit prompts, or inject terminal input in native mode.

```text
user terminal <== original stdio ==> real Claude Code / Codex
                                          |
                            tools and local child processes
                                          |
                         systemd workload scope / cgroup

Agency stays outside that workload scope:
  shared ptrace supervisor -> interaction service / policy
  hooks + telemetry ------> shared SQLite logger
  /proc + cgroup sampling -> shared profiler / dashboard
  optional native HTTP ---> capture archive + exchange/context records
```

## Run locally

```sh
uv run agency codex --cwd /path/to/project
uv run agency claude --cwd /path/to/project

# Agency arguments precede --; native CLI arguments follow it.
uv run agency codex --cwd /path/to/project -- --sandbox workspace-write
```

Native mode requires a terminal on stdin/stdout. It starts immediately and
exits when the CLI exits. The dashboard shows records, profiling, context,
and external process controls; it is not a second keyboard for this mode.
Use the native CLI's own Escape/Ctrl-C behavior locally. Shell Ctrl-Z/fg/bg
is preserved; suspending the wrapper also suspends its collectors and relay
until the shell continues it. Agency cannot send an Escape keystroke into
an inherited terminal from the browser.

The browser terminal remains available explicitly:

```sh
uv run agency codex --terminal browser
```

Browser mode keeps its PTY, terminal resize/reconnect/screen snapshots, and
optional UI linger after CLI exit. It does not enable ptrace. `--trace`
with browser mode is rejected rather than claiming syscall supervision.

## Tracing and policy

Tracing defaults on in native mode. `--no-trace` disables it while retaining
tool hooks, model telemetry/capture, cgroup profiling, and process controls.
`--trace-file-access` additionally records the existing tracer's optional
file-access metadata. This is partial metadata coverage, not a filesystem
content recorder or a deny-by-default sandbox.

```sh
uv run agency codex --no-trace
uv run agency codex --trace-file-access --policy ./session_policy.py
```

A policy file is local Python configuration exporting an existing Agency
`agpolicy` instance named `policy`. For example:

```python
from agency.agpolicy import agpolicy

def check_exec(event):
    denied = (event.path or "").endswith("/blocked-tool")
    return not denied, "blocked-tool is disabled for this session"

policy = agpolicy(syscall_hooks={"execve": check_exec, "execveat": check_exec})
```

Tool hook names and input dictionaries are those reported by the native CLI.
An Agency tool denial is returned through that CLI's hook protocol. An
Agency allow returns no forced permission decision: the CLI's own approval
and sandbox rules still apply. Syscall policy uses the same
`HostInteractionServer` admission/completion logic as managed execution.
Syscall rules and `default_to_deny` require tracing; the CLI rejects those
configurations with `--no-trace`.

The shared supervisor follows local descendants and records process events.
The workload scope excludes Agency's collector/control process. Pause/resume
uses the cgroup freezer when available, then tracer/process signals as a
fallback; stop ends the local workload. A paused CLI cannot read a resume
keystroke, so resume it externally. These controls do not freeze the remote
model provider, an external MCP service, or already-issued remote effects.
Tracing can affect timings and conflict with another debugger/tracer.

## Model capture and configuration

Capture is independent of terminal mode and tracing:

```sh
# Explicitly export the existing router key as LITE_LLM_KEY first.
uv run agency codex --capture --capture-model gpt-6-luna
```

Capture forwards native bodies to the configured LiteLLM route, preserves
original request/stream bytes, and projects them into the shared context UI.
Model OTel records are suppressed in capture mode to avoid duplicates; tool
and lifecycle events remain active. See [capture details](interactive-capture.md)
for opaque state, compaction, cancellation, and server-history limits.

With capture off, Agency does not replace the native provider or login.
Subscription-preserving exact request capture is separate future work.
Capture on selects the LiteLLM API-key route; it does not preserve native
subscription billing. Native Codex runs with `--no-daemon` so its actual
execution remains in this session's tracing/control scope. Native approval
and sandbox flags are not replaced with managed-harness bypass flags.

## Three-way comparison

All three paths run the real Codex implementation. They can complete the
same task, but they deliberately have different owners and execution contracts.

| Area | Interactive Codex with Agency | Managed base Agency with Codex | Pure Codex |
| --- | --- | --- | --- |
| Input and UI | User types directly into original terminal | Skill/driver submits prompts through a managed PTY | User types into original terminal |
| Lifetime | Long-lived attended session; exits with CLI | Scheduled request/attempt; engine validates and settles result | Native session lifetime |
| Filesystem | Host working directory | Sandbox `/workspace`, runtime/checkpoint lifecycle | Host working directory |
| Settings/state | Normal user configuration plus Agency hooks/exporter; local server via `--no-daemon` | Generated isolated config/state and Agency MCP setup | Native user config; may use shared daemon |
| Permissions | Native approvals/sandbox retained; optional Agency policy adds gates | Managed adapter bypasses native approvals/sandbox; Agency policy governs | Native approvals and sandbox |
| Model transport | Native provider with capture off; native passthrough with capture on | Matching `openai_responses` preserves native items/fields through the shared runtime; cross-provider routes translate | Native provider directly |
| Model options/context | CLI owns conversation and compaction; capture records supplied context | CLI conversation plus Agency retained history, explicit backend overrides, and attempt restoration | CLI owns conversation and compaction |
| Tools | Native tools/plugins/MCP plus Agency observation/policy | Native tools plus Agency host/sandbox MCP tools | Native tools/plugins/MCP |
| Profiling/logging | Shared collectors, optional ptrace, resources, optional wire archives | Shared collectors, tracing, scheduler/engine spans and native-preserving exchange records | Native session logs/telemetry; no Agency dashboard |
| Pause/stop | External cgroup/tracer controls | Harness control service and tracer | Native interruption or external OS controls |
| Structured output/repair | No skill output contract | Optional skill schema validation/repair | Native CLI options |
| Browser terminal | Explicit alternate mode | Managed execution is driven by Agency | Not supplied by Agency |

For a fair behavior comparison, pin one Codex version, the same model,
reasoning settings, task, and fixture files. Compare actual tool results and
filesystem contents; exact prose, token usage, and timing need not match.
Managed mode also adds instructions and tool schemas, so identical model
payloads are not a valid expectation. See the [managed harness architecture](../architecture/harness.md)
and [engine lifecycle](../architecture/engine.md) for those contracts.

The managed path has also been checked through the public `Agent` and
`agskill` API with Codex 0.162.1, Docker `python:3.12-slim`, and LiteLLM
`gpt-6-luna`: two sequential tasks wrote and reread `parity.txt`, then ran a
second shell command. Actual tool outputs, the sandbox file, reconstructed
SQLite exchanges, and profiler data were verified before removing that test
sandbox. This confirms the task behavior; managed instructions, isolated
configuration, backend processing, and checkpoint overhead still differ.

With the matching Responses provider, the managed adapter preserves tool schemas,
unknown request fields, and native response events. Explicit Agency LLM options
still override fields such as reasoning effort and output cap. This preserves
protocol content through the shared message store; the interactive capture
gateway additionally archives exact HTTP request and response bytes.
