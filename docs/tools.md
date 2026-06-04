# Tools

Tools are the functions an LLM can call during a ReAct loop. Each tool is an `agtool` instance: a name, a description, an OpenAI-compatible JSON Schema for parameters, and a Python callable that receives and returns `agdata`.

## Built-in tools

| Tool | Host or sandbox | Description |
|---|---|---|
| `bash` | sandbox | Run a shell command; all spawned processes tracked automatically |
| `read` | sandbox | Read a file with line-range pagination, or list a directory |
| `write` | sandbox | Write a file, creating parent directories as needed |
| `edit` | sandbox | Fuzzy in-place string replacement |
| `glob` | sandbox | Find files matching a glob pattern (`rg --files` or `find` fallback) |
| `grep` | sandbox | Search file contents by regex (`rg --json` or Python fallback) |
| `webfetch` | host | Fetch a URL and convert HTML to Markdown |
| `todowrite` | host | Persist a structured todo list to disk |
| `ask_human` | host | Ask the user a question; blocks until a reply arrives (from UI or stdin) |
| `daemon_release` | sandbox | Release a PID from monitoring so a long-lived service doesn't block skill completion |
| `gpu_acquire` | sandbox | Acquire exclusive GPU access from the resource pool |
| `gpu_release` | sandbox | Return the GPU to the pool |
| `cpu_acquire` | sandbox | Boost container CPU/memory limits for compute-intensive work |
| `cpu_release` | sandbox | Reset CPU/memory limits back to idle defaults |

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

Every call also appends a `type="tool"` entry to the agent's JSONL log file. See [logging.md](logging.md).

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

The `cloudpickle` library serialises tool functions for cross-process transport, which handles bound methods and closures that Python's built-in pickle cannot.

The pool uses the `"spawn"` start method, avoiding `fork`-in-multithreaded-process deadlocks on Linux.

### Making your tool picklable

Define the tool function as a top-level function or a method on a picklable class:

```python
class MySkill(agskill):
    def __init__(self):
        tool = agtool(name="compute", description="...", fn=self._compute, params={...})
        super().__init__(name="my_skill", system_prompt="...", tools=[tool])

    def _compute(self, arg: agdata) -> agdata:   # bound method — picklable
        return agdata(result=heavy_computation(arg.value))
```

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

my_tool = agtool(
    name="add",
    description="Add two integers and return their sum.",
    fn=lambda arg: agdata(result=arg.a + arg.b),
    params={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
)
```

Pass custom tools to a skill or directly to an agent:

```python
skill = agskill("math", "You are a calculator.", tools=[my_tool])
```

If `agskill.tools` is `None`, the skill inherits the agent's full sandboxed tool list. Setting `tools=[]` gives the skill no tools (pure reasoning).

## Tool output

Tool functions receive an `agdata` and must return an `agdata`. The return value is serialized to JSON and injected into the LLM's message history as a `tool` role message. Errors should be returned as `agdata(error="...")` rather than raised — the LLM will see the error and can decide how to proceed.

## bash process tracking

The sandboxed `bash` tool uses a `/proc` diff inside the container to detect all processes spawned by a command — regardless of whether they were started with `&`, via `subprocess.Popen`, or through a double-fork. The before-snapshot is taken immediately before the command runs; the after-scan runs immediately after. Any new PID not in the before-snapshot is added to `sandbox._watched_pids` and monitored by the outer loop. See [container.md](container.md) for exec wrapper details and [execution_loop.md](execution_loop.md) for the outer monitoring loop.

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
