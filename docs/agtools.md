# Tools

Tools are the functions an LLM can call during a ReAct loop. Each tool is an `agtool` instance: a name, a description, an OpenAI-compatible JSON Schema for parameters, and a Python callable that receives and returns `agdata`.

## Built-in tools

| Tool | Host or sandbox | `run_in_subprocess` | Description |
|---|---|---|---|
| `bash` | sandbox | `False` | Run a shell command; all spawned processes tracked automatically |
| `read` | sandbox | `False` | Read a file with line-range pagination, or list a directory |
| `write` | sandbox | `False` | Write a file, creating parent directories as needed |
| `edit` | sandbox | `False` | Fuzzy in-place string replacement |
| `glob` | sandbox | `False` | Find files matching a glob pattern (`rg --files` or `find` fallback) |
| `grep` | sandbox | `False` | Search file contents by regex (`rg --json` or Python fallback) |
| `webfetch` | host | `False` | Fetch a URL and convert HTML to Markdown |
| `todowrite` | host | `False` | Persist a structured todo list to disk |
| `ask_human` | host | `False` | Ask the user a question; blocks until a reply arrives (from UI or stdin) |
| `daemon_release` | sandbox | `False` | Release a PID from monitoring so a long-lived service doesn't block skill completion |
| `reserve_gpu` | sandbox | `False` | Reserve GPU access (virtual); physical GPU assigned lazily when bash runs, held for the rest of the sandbox's lifetime |
| `reserve_cpu` | sandbox | `False` | Boost container CPU/memory limits for compute-intensive work |
| `cpu_release` | sandbox | `False` | Reset CPU/memory limits back to idle defaults |

All filesystem tools (bash, read, write, edit, glob, grep) have two variants: a host-side singleton and a sandboxed factory function (`make_<tool>(sandbox)`) that routes all I/O through `docker/podman exec`.

**There is no `gpu_release` tool.** A GPU, once actually acquired via `reserve_gpu`, is released only by `sandbox.rm_container()`/`sandbox.destroy()` — never by an explicit mid-skill call, and never by a mere hibernate (see [agresources.md](agresources.md#agent-callable-resource-tools) and [container.md](agsandbox_backends/container.md)'s "GPU device access").

**`run_in_subprocess` no longer gates whether `stop()` runs after a tool call.** Every built-in tool above sets it `False` for an unrelated reason (they need to run synchronously against the same persistent sandbox object, not a disposable cloudpickled worker copy), but `agtool.py`'s `dispatch_tools()` now calls `sandbox.stop()` (hibernate only — no checkpoint or removal) after *every* tool call regardless of this flag — deferred only when `sandbox._has_pending_background_work()` is true, not based on this column at all.

## Sandboxed tool construction

`make_sandboxed_tools(sandbox, pool=None)` in `agency/tools/__init__.py` builds the full tool list for an agent:

```python
tools = [
    make_bash(sandbox),
    make_read(sandbox),
    make_write(sandbox),
    make_edit(sandbox),
    make_glob(sandbox),
    make_grep(sandbox),
    webfetch,                         # host-side singleton
    todowrite,                        # host-side singleton
    make_ask_human(sandbox._agname),  # host-side, routes to agwebui or stdin
    make_daemon_release(sandbox),
]
if pool is not None:
    tools += [
        make_gpu_reserve(sandbox, pool),
        make_cpu_reserve(sandbox, pool),
        make_cpu_release(sandbox, pool),
    ]
```

## Tool logging

Every tool call is logged automatically after it returns. `agskill._build_toolkit()` calls `tool.attach_logger(ag.terminal, ag.log)` on each tool when building the toolkit for a skill run, wiring up both terminal and file logging.

| Tool | Terminal line |
|---|---|
| `bash` | `bash  rc=0  (42ms)  $ ls /workspace` |
| `read` | `read  file  /workspace/train.py  (120 lines)  (8ms)` |
| `write` | `write  /workspace/out.txt  (1024 bytes)  (5ms)` |
| `edit` | `edit  /workspace/main.py  ✓  (12ms)` |
| `glob` | `glob  '*.py'  → 14 files  (30ms)` |
| `grep` | `grep  'def train'  → 3 matches  (25ms)` |
| others | `<name>  in=[...]  out=[...]  (Nms)` |

Every call also appends a `type="tool"` entry to the agent's JSONL log file. See [aglog.md](aglog.md).

To override logging for a custom tool, pass `log_fn` to the constructor:

```python
def my_log(tool, arg, result, elapsed_ms):
    tool._term.log("TOOL ✓   ", f"my_tool  key={arg.key}  ({elapsed_ms}ms)")
    if tool._aglog:
        tool._aglog._tool_call(tool.name, arg.to_dict(), result.to_dict(), elapsed_ms)

my_tool = agtool(name="my_tool", description="...", fn=my_fn, log_fn=my_log)
```

## Process offloading

Every `agtool.__call__` runs the tool function in a **separate worker process** via a shared `ProcessPoolExecutor(max_workers=256)`. Workers are created lazily on demand and scale up to 256 concurrent calls.

**Why a separate process?** Python's GIL serialises bytecode execution across threads, so a CPU-bound tool running in the same process would block every other agent thread for its duration. A subprocess gets its own GIL — tools can burn CPU freely without affecting LLM streaming or other agents.

`cloudpickle` serialises the tool function for cross-process transport, so lambdas and closures work at runtime. Nevertheless, **prefer module-level functions or bound methods** in your own code — they are debuggable, importable, and can be pickled by standard `pickle` if needed.

The pool uses the default `"fork"` start method on Linux for low-overhead worker creation.

### Making your tool function serialisable

Tool functions are sent to a worker process via `cloudpickle`, so lambdas, closures, and locally-defined functions all work. Despite that flexibility, the **recommended pattern** is a module-level function or a bound method on a picklable class — it is easier to test, import, and reason about:

```python
# Recommended: bound method on the skill class
class MySkill(agskill):
    def __init__(self):
        tool = agtool(name="compute", description="...", fn=self._compute, params={...})
        super().__init__(name="my_skill", system_prompt="...", tools=[tool])

    def _compute(self, arg: agdata) -> agdata:
        return agdata(result=heavy_computation(arg.value))

# Also fine: module-level function
def _my_compute(arg: agdata) -> agdata:
    return agdata(result=heavy_computation(arg.value))
```

> **Warning:** Even though `cloudpickle` handles lambdas and closures, avoid using them for tool functions that capture large objects (e.g. model weights, database connections) — those objects will be serialised and sent to each worker process on every call.

`agtool.__getstate__` excludes the logger references (`_term`, `_aglog`) from serialisation — they hold threading locks and file handles that cannot safely cross process boundaries. They are restored to `None` in the worker and re-attached via `attach_logger` when needed.

### Testing tools with mocked I/O

Tool functions run in a worker subprocess, so `unittest.mock.patch` applied in the test process is invisible to the worker. To test tool logic with mocked I/O, call the underlying function directly:

```python
from agency.tools.webfetch import webfetch

def test_html_to_markdown():
    with patch("httpx.get", return_value=mock_response):
        result = webfetch.fn(agdata(url="https://example.com"))  # .fn(), not webfetch()
    assert "Hello" in result.output
```

This tests the fetch/convert logic in-process. The process-pool dispatch mechanism is tested separately via `test_agtool.py::test_process_pool_runs_in_different_pid`.

## Defining a custom tool

```python
from agency.agtool import agtool
from agency.agdata import agdata

def _add(arg: agdata) -> agdata:
    return agdata(result=arg.a + arg.b)

my_tool = agtool(
    name="add",
    description="Add two integers and return their sum.",
    fn=_add,
    params={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
    # run_in_subprocess defaults to True for custom tools — set False if the tool
    # runs entirely on the host (HTTP calls, file reads from the host, etc.)
    run_in_subprocess=False,
)
```

### `run_in_subprocess` flag

Every `agtool` has a `run_in_subprocess` flag (default `True`). It controls **two** things simultaneously:

| | `run_in_subprocess=True` | `run_in_subprocess=False` |
|---|---|---|
| **Execution context** | Worker subprocess via `ProcessPoolExecutor` | Calling thread, in-process |
| **Sandbox container** | Started on first call if not yet running | Never touched |
| **Process isolation** | Full — own GIL, own memory space | None — shares caller's state |
| **Timeout enforced** | Yes (`TOOL_TIMEOUT_S`, default 30 s) | No — caller controls blocking |

**Default `True`** is the safe default for custom tools. The subprocess isolation prevents a CPU-heavy or crashing tool from blocking LLM streaming or corrupting agent state.

**Set `False`** for any tool that must access host-process state — module-level singletons, UI handles, queues, or anything that lives only in the main process and would be `None` or missing in a subprocess worker. In particular, all sandbox-backed tools (bash, read, write, edit, glob, grep, daemon_release, and the GPU/CPU tools) use `run_in_subprocess=False` because their functions close over the sandbox object directly — serialising a live sandbox connection across process boundaries does not work.

```python
# Wrong: _agwebui._active is a singleton in the main process;
# it is None in every subprocess worker — the tool silently fails.
my_tool = agtool(name="notify_ui", ..., fn=_notify_fn)          # run_in_subprocess=True default

# Correct:
my_tool = agtool(name="notify_ui", ..., fn=_notify_fn, run_in_subprocess=False)
```

Common cases that require `run_in_subprocess=False`:

- **Sandbox-backed tools** — their `fn` closes over the sandbox object (docker/podman exec handle); the sandbox cannot be serialised across process boundaries.
- **`ask_human` and any human-interaction tool** — they read `_agwebui._active` to route questions to the live UI, then block-poll for a reply. In a subprocess, that singleton is `None` and stdin is an EOF pipe, so the tool either hangs or returns a timeout reply immediately.
- **Tools that write to shared in-process state** — progress queues, event emitters, result caches.
- **Tools that perform outbound I/O only** — HTTP requests, host file reads — where no container is needed and running in-process is simpler.

> **Rule of thumb:** if the tool's function body imports or reads anything from `agency` (agents, UI handles, queues, sandbox handles) rather than just transforming its input, set `run_in_subprocess=False`.

Exceptions thrown by `run_in_subprocess=False` tools are caught by `agtool.__call__` and returned as `agdata(error=...)`, exactly like subprocess tools. The LLM sees the error and can decide how to proceed.

Tools belong to skills, not agents. Pass custom tools when defining the skill:

```python
# Add a custom tool on top of the default sandboxed set:
skill = agskill("research", "Research the topic.", add_tools=[my_tool])

# Replace the full tool list with only your tool:
skill = agskill("custom", "Use only my tool.", replace_tools=[my_tool])

# No tools — pure LLM reasoning:
skill = agskill("classify", "Classify this text.", replace_tools=[])
```

`add_tools` extends the defaults; `replace_tools` overrides them entirely.

## Tool hibernation, skill-level revert, and timeout

### No per-tool checkpoint or revert

A tool call's own success or failure has no effect on the sandbox's checkpoint state anymore. Every tool call ends with `sandbox.stop()` — a hibernate only (`docker/podman stop`, container kept, never committed or removed) — regardless of whether the tool succeeded or failed, unless it left background work still running in the sandbox (in which case `stop()` is deferred until a later call finds nothing pending). See [container.md](agsandbox_backends/container.md)'s "Container lifecycle" for the full mechanics.

Checkpointing and revert both happen once per *skill* call instead, at `agskill.py`'s teardown: `sandbox.commit()` on success, `sandbox.rm_container()` (discarding everything since the last successful skill) on failure, with a revert notice delivered to the agent via its `inbox` rather than inlined into any one tool's result — see [agskill.md](agskill.md#tool-call-hibernation-and-skill-level-revert) for the full mechanics and why the notice can't live in the failed skill's own result.

### Agent-controlled timeout

The default tool watchdog deadline is `TOOL_TIMEOUT_S = 30` seconds. An agent can override this per-call by passing a `"timeout"` integer (seconds) in the tool arguments:

```
LLM calls: bash({"command": "python train.py", "timeout": 600})
```

The framework strips the key before passing args to the tool function and forwards it to `agtool.__call__(timeout=600)`. Non-integer or absent values fall back to the default.

## Tool output

Tool functions receive an `agdata` and must return an `agdata`. The return value is serialized to JSON and injected into the LLM's message history as a `tool` role message. Errors should be returned as `agdata(error="...")` rather than raised — the LLM will see the error and can decide how to proceed.

### Large output offloading

If the serialized result exceeds the offload threshold and a sandbox is available, the framework automatically writes the content to `/workspace/long_tool_call_outputs/<tool_name>_<call_id>.txt` and replaces the tool message with a short note pointing to that path. The agent reads the file using its `read` tool. This prevents a single large tool result (e.g. a raw PDF or a lengthy webpage) from consuming the entire context window. See [agskill.md — Tool output offloading](agskill.md#tool-output-offloading) for details.

## bash process tracking

The sandboxed `bash` tool uses a `/proc` diff inside the container to detect all processes spawned by a command — regardless of whether they were started with `&`, via `subprocess.Popen`, or through a double-fork. The before-snapshot is taken immediately before the command runs; the after-scan runs immediately after. Any new PID not in the before-snapshot is added to `sandbox._watched_pids` and monitored by `wait_for_processes` inside `agskill.execute_react()`. See [agsandbox.md](agsandbox.md) for exec wrapper details and [execution_process_control.md](execution_process_control.md) for per-scenario process monitoring traces.

## `daemon_release` tool

Use this when a process is intentionally long-lived (a server, monitor, or background service) and should not block skill completion:

```
LLM calls: daemon_release({"pid": 1234})
```

This moves PID 1234 (and all its future descendants) from `_watched_pids` to `_daemon_pids`. `wait_for_processes` no longer sees it as a live process, and the skill resolves normally. The process keeps running in the container until the container is destroyed.

## `ask_human` tool

The agent calls `ask_human` when it needs information it cannot determine on its own:

```
LLM calls: ask_human({"question": "Which dataset should I use?"})
```

When `agwebui` is active, the question is emitted as an `ask_human` event and the tool blocks until the browser UI delivers a reply (or the timeout elapses). Without `agwebui`, the question is printed to stdout and the agent reads from stdin. The reply is returned as `agdata(reply="...")` and injected into the ReAct loop. The agent's state is set to `"human"` while waiting.
