# Skills

An `agskill` is a named, self-contained ReAct loop with its own system prompt, optional input/output schemas, and an optional tool extension. Skills are the unit of work submitted to an agent via `agent.run(skill, input)`.

## Defining a skill

```python
from agency import agskill, agdata

skill = agskill(
    name="summarize",
    system_prompt="You are a concise summarizer. Return only valid JSON.",
    input_schema=agdata(text=str),
    output_schema=agdata(summary=str, word_count=int),
)
```

Both schemas are serialized and appended to the system prompt so the LLM knows the expected format.

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Descriptive identifier for the skill |
| `system_prompt` | `str` | System message prepended to every LLM call |
| `add_tools` | `list[agtool] \| None` | Extra tools appended to the default sandboxed tool list |
| `replace_tools` | `list[agtool] \| None` | Replaces the tool list entirely; use `replace_tools=[]` for no tools |
| `input_schema` | `agdata \| None` | Required input fields and their types |
| `output_schema` | `agdata \| None` | Required output fields; enforced with retries |
| `output_validator` | `Callable \| None` | Custom validation function, called after schema check |
| `max_output_schema_retries` | `int` | Times to retry on output schema failure (default `10`) |

## Schema field types

Schema fields in `agdata` are plain Python type objects:

| Type | JSON hint sent to LLM | Notes |
|---|---|---|
| `str` | `"str"` | Plain text |
| `int` | `"int"` | |
| `float` | `"float"` | |
| `bool` | `"bool"` | |
| `list` | `"list"` | Untyped list — no item validation |
| `dict` | `"dict"` | |
| `agfile` | `"file"` | File-backed field — see [Typed field values](#typed-field-values-agtype-and-agfile) |

Any `agtype` subclass is also valid; its `schema_type()` classmethod provides the JSON hint.

```python
# All built-in Python types work directly:
output_schema=agdata(summary=str, word_count=int, passed=bool)

# agfile for large text outputs:
output_schema=agdata(report=agfile)
```

### Typed list fields

To express a list whose items have a known structure, pass a one-element list containing a plain `dict` that maps field names to types:

```python
output_schema = agdata(
    papers=[{"title": str, "url": str, "abstract": str}],
    count=int,
)
```

The serialized hint shown to the LLM is:

```json
{"papers": [{"title": "str", "url": "str", "abstract": "str"}], "count": "int"}
```

The framework validates every element against the template during output schema checking: each item must be a `dict` containing the declared keys with the declared Python types. A failure triggers an automatic retry with a message identifying the exact index and key that failed. Use this form instead of bare `list` whenever item structure matters.

## ReAct loop

Each call to `agskill.run()` executes a standard ReAct loop:

1. Offload oversized input fields to files in the agent's sandbox (see below)
2. Build messages: `[system] + history + [user: input.to_json()]`
3. Drain user inbox (injected mid-conversation messages from `agUI` or `agent._inbox`)
4. Pre-call compaction — check character-based token estimate; compact if over threshold (see [compaction.md](compaction.md))
5. Call the LLM with `stream=True`; accumulate tokens via `_iter_batched()` (see below); retry on connection failure with exponential backoff
6. Post-call compaction — check actual `prompt_tokens` from API usage; compact again if needed
7. If response contains tool calls → for each tool:
   a. Coerce malformed JSON arguments to `"{}"` so history replay never crashes
   b. If `need_sandbox=True`, commit the sandbox to a pre-call checkpoint image
   c. Execute the tool (offloaded to a worker process)
   d. If the result contains `"error"`, restore the sandbox from the checkpoint and append `workspace_reverted` to the error message (see [Tool failure and checkpoint revert](#tool-failure-and-checkpoint-revert))
   e. If the result is large, offload to a file (see [Tool output offloading](#tool-output-offloading))
   f. Append the tool result message and go to 3
8. If response is a final answer → parse JSON, validate against `output_schema`
9. If validation fails and retries remain → inject correction message, go to 3
10. If sandbox has live background processes → `_wait_for_processes()` polls until they exit or `ping_interval_s` elapses; inject status message and go to 3
11. Delete offloaded input files, return `(result, updated_history, history_delta, token_counts)`

The loop exits early when `max_steps` (default `AGSKILL_REACT_MAX_STEPS = 4096`) is exceeded.

### LLM timeout and exponential backoff

Each LLM call is protected by an idle watchdog that doubles its deadline on every retry, following the sequence `[60, 120, 240, 480, 960]` seconds (1 → 2 → 4 → 8 → 16 minutes), for a maximum total wait of 31 minutes across 5 attempts.

The watchdog runs in the main thread: it polls a queue fed by the streaming drain thread and raises `_LLMIdleTimeout` if no chunk arrives within the deadline. Unlike `httpx.ReadTimeout` (which is a per-chunk idle timer enforced inside httpcore), this approach works even when the underlying `ssl.read()` is blocked indefinitely — for example when the server closes the TCP connection without sending an SSL `close_notify` (CLOSE-WAIT state). On timeout, `client.close()` is called best-effort to unblock the drain thread.

`ssl.SSLError` and `OSError` raised during streaming (e.g. `[SSL] record layer failure` when vLLM drops the TCP connection mid-stream) are caught by the same retry block and follow the same exponential backoff sequence.

A connection failure on attempt *N* logs a `LLM ✗` line and retries with the next deadline. If all 5 attempts fail, the skill returns `agdata(error="LLM connection error after 5 attempts: ...")` without raising.

The connect, write, and pool timeouts are fixed at 30 s, 180 s, and 30 s respectively.

### Concurrency semaphore

A process-wide semaphore (`_llm_call_semaphore`, size 128) limits how many skills can be in an active LLM call simultaneously. The semaphore is acquired just before the OpenAI client is constructed and released immediately after the streaming call finishes — whether it succeeds, times out, or retries. Skills waiting for input validation, tool execution, or output validation do not hold a slot.

This prevents runaway parallelism from exhausting vLLM server connections when hundreds of agents are spawned concurrently.

### Streaming and GIL pressure

The LLM call uses `stream=True`. Without batching, each SSE token chunk would acquire and release the Python GIL once, creating O(tokens) context switches that slow down all concurrent agent threads.

`_iter_batched()` reduces this to O(tokens / batch_size):
- A background thread drains the SSE stream, doing one `queue.put` per chunk (minimal GIL hold)
- The main thread sleeps for `_BATCH_INTERVAL_S` (100 ms default) — GIL fully released
- On wake, the main thread drains everything buffered during the sleep in one burst

This means threads running other agents get 100 ms of uncontested GIL time for every batch of streamed tokens, dramatically improving throughput under concurrent load.

### Generation parameters

Standard OpenAI generation parameters set in `llm_config` are forwarded directly to every API call. vLLM-specific parameters that are not part of the OpenAI spec are merged into `extra_body` automatically.

**Standard OpenAI parameters** (forwarded directly):

| Key | Example |
|---|---|
| `temperature` | `0.6` |
| `max_tokens` | `16000` |
| `top_p` | `0.95` |
| `frequency_penalty` | `0.1` |
| `presence_penalty` | `0.3` |
| `n`, `stop`, `logprobs`, `seed` | — |

**vLLM-specific parameters** (merged into `extra_body`):

| Key | Example |
|---|---|
| `top_k` | `50` |
| `repetition_penalty` | `1.1` |
| `min_p`, `min_tokens` | — |
| `guided_json`, `guided_regex` | — |

Any value already in `llm_config["extra_body"]` is preserved; vLLM-specific keys are added on top.

```python
LLM_CONFIG = {
    "base_url": "http://localhost:8000/v1",
    "model": "Qwen/Qwen3-30B",
    "temperature": 0.6,
    "max_tokens": 16000,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
}
```

### Tool output offloading

Tool results longer than `_TOOL_OUTPUT_OFFLOAD_CHARS` (default 20 000 characters) are automatically saved to a file inside the agent's sandbox instead of being inlined into the message history. The tool message is replaced with a short JSON note:

```json
{"note": "Output was too large and has been saved to /workspace/long_tool_call_outputs/webfetch_abc123.txt. Use the read tool to access it."}
```

Files are written to `/workspace/long_tool_call_outputs/<tool_name>_<call_id>.txt`. The agent can read the content at any point using its `read` tool.

This guard prevents a single oversized tool result (e.g. a raw PDF fetched via `webfetch`) from filling the entire context window. If no sandbox is available the result is kept inline unchanged.

### Tool failure and checkpoint revert

Before every `need_sandbox=True` tool call, the framework commits the sandbox container to a lightweight checkpoint image:

```
agency/pretool-<container_name>-<call_id[:8]>
```

If the tool returns an `agdata(error=...)`, the framework automatically:

1. Calls `sandbox.restore(checkpoint_tag)` — stops the running container and restarts it from the checkpoint image, so any partial filesystem changes made by the tool are rolled back.
2. Appends `"workspace_reverted": "The workspace has been reverted to the state before this tool call."` to the tool result JSON.

The LLM sees both the error and the revert notice, so it knows the filesystem is clean and can try a different approach.

```json
{
  "error": "...",
  "workspace_reverted": "The workspace has been reverted to the state before this tool call."
}
```

**When revert does NOT happen:**

- `need_sandbox=False` — no checkpoint is taken, so no revert is possible.
- `sandbox` is `None` — no container exists.
- `sandbox.commit()` raised — checkpoint tag is discarded; the error is still forwarded to the LLM unchanged.
- The tool succeeded — restore is never called on success.

Checkpoint images are named with the container name, so they are scoped to a single sandbox lifetime. All `agency/pretool-<container_name>-*` images are deleted when `sandbox.destroy()` is called.

#### Agent-controlled timeout

Every tool call accepts an optional `"timeout"` key in its arguments (an integer, in seconds). The framework extracts it before calling the tool and passes it to `agtool.__call__(timeout=...)`, which uses it as the `ProcessPoolExecutor.result()` deadline instead of the default `TOOL_TIMEOUT_S` (30 s). Use this when a tool is expected to take longer than the default:

```
LLM calls: bash({"command": "python train.py", "timeout": 600})
```

Non-integer or missing `"timeout"` values are silently ignored and the default applies.

## Typed field values (`agtype` and `agfile`)

Schema field types can be `agtype` subclasses as well as plain type-name strings.  The built-in subclass is `agfile`.  See [agtype.md](agtype.md) for the full interface and instructions for writing custom field types.

Declare a schema field with `agfile` as its type to make the framework handle file I/O transparently for that field.

```python
from agency import agskill, agdata, agfile

design_skill = agskill(
    name="design",
    system_prompt="Create a story design document.",
    input_schema=agdata(theme=str, background=agfile),
    output_schema=agdata(design_doc=agfile),
)
```

### Input `agfile` fields

Before the ReAct loop starts, each input field declared as `agfile` is written to `/workspace/inputs/<skill_name>_<field>.txt` inside the agent's sandbox. The field value in the JSON sent to the LLM is replaced with the file path:

```json
{"theme": "space opera", "background": "/workspace/inputs/design_background.txt"}
```

The system prompt automatically gains an instruction telling the agent to use the `read` tool to access the file, with a warning that the file is temporary and will be deleted after the task ends.

From Python, the caller always passes and receives plain string content — the file path is an internal detail invisible outside the framework.

### Output `agfile` fields

For each output field declared as `agfile`, the system prompt instructs the agent to write the content to a file (e.g. `/workspace/outputs/<skill_name>_<field>.txt`) and return the path as the field value. After the skill completes, the framework reads the file content from the sandbox and stores it as a plain string in the result agdata. The caller receives content, not a path.

### Cleanup

All `agfile` files — both input and output — are deleted from the sandbox in the `finally` block after the skill ends, whether it succeeded or raised. They never persist between skill invocations on the same agent.

### Schema display

In the JSON format sections appended to the system prompt, `agfile` fields are shown with the type hint `"file"` (from `agfile.schema_type()`):

```json
{"background": "file", "theme": "string"}
```

---

## Automatic input offloading

After `agfile` (and other `agtype`) fields have been prepared, the framework checks every remaining top-level string field. Any value that still exceeds `INPUT_OFFLOAD_CHARS` (default `2000`) is automatically written to a temporary file and the field value is replaced with a short reference:

```
(content saved to /workspace/inputs/design_doc.txt — use the read tool to access it)
```

Because `agfile` fields are already converted to short file paths before this check runs, they are never double-offloaded.

When auto-offloading occurs, the system prompt receives an extra note listing the affected fields:

```
Note: The following input fields contain large content that has been automatically
saved to temporary files in your sandbox: `design_doc`, `previous_chapter`. The
file paths are shown in the input JSON. Use the read tool to access the full content.
WARNING: these files are temporary and will be automatically deleted after this task ends.
```

The threshold can be adjusted at the module level:

```python
import agency.agent as _ag
_ag.INPUT_OFFLOAD_CHARS = 4000
```

Only top-level string fields are auto-offloaded. Non-string values (integers, booleans, lists, nested agdata) are always inlined. Auto-offloaded files are cleaned up in the same `finally` block as `agfile` files.

## Input validation

Input is validated against `input_schema` before the loop starts. Validation checks that all required fields are present and have the correct Python type. If validation fails, the skill returns immediately with an `agdata(error=...)` without calling the LLM.

Input validation is **skipped** when `_is_continuation=True` so that process-status messages injected by `_wait_for_processes` can flow through without matching the skill's declared input schema.

## Output validation and retries

After a non-tool-call LLM response:

1. Parse the response content as JSON (markdown code fences are stripped)
2. Check all fields in `output_schema` are present with the correct type
3. Run `output_validator(result)` if provided — returns a list of error strings
4. If errors exist and `retries_left > 0`: inject a correction message and loop
5. If errors persist after all retries: return `agdata(error="output schema error after retries: ...")`

Output validation is skipped when the LLM response is answering a mid-conversation user message injected via the inbox (`had_inbox=True`), because the LLM is engaged in dialogue rather than producing a final structured answer.

## Process monitoring

After output validation passes (and the result is not an error), the loop checks for live sandbox background processes before returning:

```
if sandbox has live processes:
    _wait_for_processes() polls every poll_interval_s for up to ping_interval_s
    → returns "Background processes have completed." or "still running: ..."
    → message appended as user turn; loop continues
else:
    return result
```

The LLM receives the status message, can read log files or call more tools, then produces another final answer — which triggers another `_wait_for_processes` check. The skill only exits when `_wait_for_processes` returns `None` (no live watched PIDs).

`ping_interval_s` and `poll_interval_s` are class-level attributes on `agent` (defaults 300 s and 5 s) passed through to `agskill.run()` at call time. The total number of monitoring continuations is bounded by `max_steps`.

See [execution_process_control.md](execution_process_control.md) for per-scenario traces.

## History

The history passed to `agskill.run()` is the agent's shared conversation context. The skill appends its full message exchange to this history and returns the updated version. The system prompt is re-injected fresh on every call and is not persisted in the stored history.

## Skill tools

Tools live on the skill, not the agent. Each skill run builds its tool list fresh from the sandbox:

| Parameter | Behaviour |
|---|---|
| Neither set (default) | Full sandboxed tool list (`bash`, `read`, `write`, `edit`, `glob`, `grep`, `webfetch`, …) |
| `add_tools=[t]` | Default sandboxed tools **plus** `t` |
| `replace_tools=[t]` | Only `t` — no sandboxed tools |
| `replace_tools=[]` | No tools — pure LLM reasoning |

```python
search_tool = agtool("search", "Search the web.", fn=my_search_fn, params={...})

# Add a custom tool on top of the defaults:
skill = agskill(name="research", system_prompt="Research the topic.", add_tools=[search_tool])

# Pure reasoning, no tools:
skill = agskill(name="classify", system_prompt="Classify this text.", replace_tools=[])
```

## Common skills (`agency.common_skills`)

`agency.common_skills` provides ready-made skill classes for common tasks. Each is a thin subclass of `agskill` with a fixed name, system prompt, and schemas. All accept `**kwargs` forwarded to `agskill.__init__` (e.g. `max_output_schema_retries=1`).

### `WriterSkill`

Writes content to a file path inside the sandbox container. Uses the default tool set (includes the sandbox `write` tool).

```python
from agency.common_skills import WriterSkill
skill = WriterSkill()
# input:  agdata(file_path=str, content=str)
# output: agdata(path=str, status=str)
```

### `SummariserSkill`

Summarises a piece of text in one sentence. Has no tools (`replace_tools=[]`) — pure reasoning.

```python
from agency.common_skills import SummariserSkill
skill = SummariserSkill()
# input:  agdata(text=str)
# output: agdata(summary=str)
```

### `FindPapersSkill`

Searches Hugging Face Papers for papers on a topic using a bundled `search_papers` tool. The output schema uses a typed list field so the LLM sees the exact item structure and each element is validated on every response:

```python
from agency.common_skills import FindPapersSkill
skill = FindPapersSkill(max_papers=8)   # default 10
# input:  agdata(topic=str)
# output: agdata(papers=[{"title": str, "url": str, "abstract": str}], count=int)
```

If the LLM returns an empty list or items that are not dicts with a `title` key, the skill retries automatically. The bundled tool is also accessible as `skill.search_papers` if you need to reuse it elsewhere.

### `SummarisePaperSkill`

Fetches the full text of an arxiv paper and writes a technical summary covering contribution, method, results, limitations, and conclusions. Uses a bundled `fetch_paper` tool that converts any arxiv URL form (`/abs/`, `/pdf/`, `/html/`) to the HTML version and extracts plain text via `html2text` (capped at 32 000 characters).

```python
from agency.common_skills import SummarisePaperSkill
skill = SummarisePaperSkill()
# input:  agdata(title=str, url=str, abstract=str)
# output: agdata(summary=str)
```

The system prompt requires the agent to call `fetch_paper` before responding — it will not summarise from the abstract alone.

### `CompileReportSkill`

Writes a structured markdown research report to a path inside the sandbox. Requires the agent to have the sandbox `write` tool available.

```python
from agency.common_skills import CompileReportSkill
skill = CompileReportSkill()
# input:  agdata(topic=str, summaries=list, output_path=str)
# output: agdata(report_path=str, paper_count=int)
```

### Using common skills in an `agteam`

```python
from agency import agteam, agdata
from agency.common_skills import FindPapersSkill, SummarisePaperSkill, CompileReportSkill

class PaperCrawlerTeam(agteam):
    llm_config = LLM_CONFIG

    def setup(self):
        self.find_papers     = FindPapersSkill(max_papers=10)
        self.summarise_paper = SummarisePaperSkill()
        self.compile_report  = CompileReportSkill()
        self.agent           = agent()
```

## Example with output validator

```python
def validate_score(result: agdata) -> list[str]:
    if not (0 <= result.score <= 100):
        return [f"score must be 0–100, got {result.score}"]
    return []

skill = agskill(
    name="grade",
    system_prompt="Grade the submission from 0 to 100.",
    output_schema=agdata(score=int, feedback=str),
    output_validator=validate_score,
)
```
