# Tools

Tools are the functions an LLM can call during a ReAct loop. Each tool is an `agtool` instance: a name, a description, an OpenAI-compatible JSON Schema for parameters, and a Python callable that receives and returns `agdata`.

## Built-in tools

| Tool | Host or sandbox | `need_sandbox` | Description |
|---|---|---|---|
| `bash` | sandbox | `True` | Run a shell command; all spawned processes tracked automatically |
| `read` | sandbox | `True` | Read a file with line-range pagination, or list a directory |
| `write` | sandbox | `True` | Write a file, creating parent directories as needed |
| `edit` | sandbox | `True` | Fuzzy in-place string replacement |
| `glob` | sandbox | `True` | Find files matching a glob pattern (`rg --files` or `find` fallback) |
| `grep` | sandbox | `True` | Search file contents by regex (`rg --json` or Python fallback) |
| `webfetch` | host | `False` | Fetch a URL and convert HTML to Markdown |
| `todowrite` | host | `False` | Persist a structured todo list to disk |
| `ask_human` | host | `False` | Ask the user a question; blocks until a reply arrives (from UI or stdin) |
| `daemon_release` | sandbox | `True` | Release a PID from monitoring so a long-lived service doesn't block skill completion |
| `gpu_acquire` | sandbox | `True` | Acquire exclusive GPU access from the resource pool |
| `gpu_release` | sandbox | `True` | Return the GPU to the pool |
| `cpu_acquire` | sandbox | `True` | Boost container CPU/memory limits for compute-intensive work |
| `cpu_release` | sandbox | `True` | Reset CPU/memory limits back to idle defaults |

All filesystem tools (bash, read, write, edit, glob, grep) have two variants: a host-side singleton and a sandboxed factory function (`make_<tool>(sandbox)`) that routes all I/O through `docker/podman exec`.

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
    make_ask_human(sandbox._agname),  # host-side, routes to agUI or stdin
    make_daemon_release(sandbox),
]
if pool is not None:
    tools += [
        make_gpu_acquire(sandbox, pool),
        make_gpu_release(sandbox, pool),
        make_cpu_acquire(sandbox),
        make_cpu_release(sandbox, pool),
    ]
```

## Tool logging

Every tool call is logged automatically after it returns. `agent.__init__` calls `tool.attach_logger(term, log)` on each tool, wiring up both terminal and file logging.

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
    # need_sandbox defaults to True for custom tools — set False if the tool
    # runs entirely on the host (HTTP calls, file reads from the host, etc.)
    need_sandbox=False,
)
```

### `need_sandbox` flag

Every `agtool` has a `need_sandbox` flag (default `True`). When an agent calls a tool with `need_sandbox=True`, the sandbox container is started on that first call if it has not been started yet. Tools with `need_sandbox=False` run without ever touching the container.

- **Default `True`** — the safe default for custom tools; guarantees a container is available before the tool runs.
- **Set `False`** explicitly for tools that are entirely host-side: HTTP requests, reading host files, spawning sub-agents, etc.

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

## Tool output

Tool functions receive an `agdata` and must return an `agdata`. The return value is serialized to JSON and injected into the LLM's message history as a `tool` role message. Errors should be returned as `agdata(error="...")` rather than raised — the LLM will see the error and can decide how to proceed.

## bash process tracking

The sandboxed `bash` tool uses a `/proc` diff inside the container to detect all processes spawned by a command — regardless of whether they were started with `&`, via `subprocess.Popen`, or through a double-fork. The before-snapshot is taken immediately before the command runs; the after-scan runs immediately after. Any new PID not in the before-snapshot is added to `sandbox._watched_pids` and monitored by the outer loop. See [agsandbox.md](agsandbox.md) for exec wrapper details and [execution_loop.md](execution_loop.md) for the outer monitoring loop.

## `daemon_release` tool

Use this when a process is intentionally long-lived (a server, monitor, or background service) and should not block skill completion:

```
LLM calls: daemon_release({"pid": 1234})
```

This moves PID 1234 (and all its future descendants) from `_watched_pids` to `_daemon_pids`. The outer loop no longer waits for it, and the skill resolves normally. The process keeps running in the container until the container is destroyed.

## `ask_human` tool

The agent calls `ask_human` when it needs information it cannot determine on its own:

```
LLM calls: ask_human({"question": "Which dataset should I use?"})
```

When `agUI` is active, the question is displayed in the interaction pane and the UI blocks until the user types a reply. Without `agUI`, the question is printed to stdout and the agent reads from stdin. The reply is returned as `agdata(reply="...")` and injected into the ReAct loop. The agent's state is set to `"human"` while waiting.
