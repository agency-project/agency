# Skills

An `agskill` is a named, self-contained ReAct loop with its own system prompt, optional input/output schemas, and an optional tool extension. Skills are the unit of work submitted to an agent via `agent.run(skill, input)`.

## Defining a skill

```python
from agency import agskill, agdata

skill = agskill(
    name="summarize",
    system_prompt="You are a concise summarizer.",
    input_schema=agdata(text=str),
    output_schema=agdata(summary=str, word_count=int),
)
```

The input schema is serialized and appended to the system prompt. The output schema is used to generate one typed tool per output field (see [Output collection via tools](#output-collection-via-tools)).

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Descriptive identifier for the skill |
| `system_prompt` | `str` | System message prepended to every LLM call |
| `add_tools` | `list[agtool] \| None` | Extra tools appended to the default sandboxed tool list |
| `replace_tools` | `list[agtool] \| None` | Replaces the tool list entirely; use `replace_tools=[]` for no tools |
| `input_schema` | `agdata \| None` | Required input fields and their types |
| `output_schema` | `agdata \| None` | Required output fields; enforced with retries |
| `max_output_schema_retries` | `int` | Times to retry on output schema failure (default `10`) |

## Schema field types

Schema fields in `agdata` are plain Python type objects:

| Type | Tool `value` type | Validation |
|---|---|---|
| `str` | `string` | `isinstance(v, str)` |
| `int` | `integer` | `isinstance(v, int)` |
| `float` | `number` | `isinstance(v, float)` |
| `bool` | `boolean` | `isinstance(v, bool)` |
| `list[T]` | `array` | each element validated against `T` recursively |
| `dict[K, V]` | `object` | each value validated against `V` recursively |
| `tuple[T1, T2, ...]` | `array` | each position validated against its declared type |
| `[{"key": type, ...}]` | `array` | each item dict validated against template |
| `agfile` | `string` | must be str (file path); framework reads UTF-8 content after skill ends |
| `agbinary` | `string` | must be str (file path); framework reads raw bytes after skill ends; caller receives `bytes` |
| `agpath` | `string` | must be str and look like a path (`_looks_like_path`); passed through unchanged — never read or written by the framework |
| `agimage` | `string` | must be str (URL or data URL); injected as a multimodal image in the user message |
| `agrawstring` | `string` | bypasses JSON formatting entirely; must be the only field in its schema |

Any `agtype` subclass is also valid; its `schema_type()` classmethod provides the display hint.

```python
# All built-in Python types work directly:
output_schema=agdata(summary=str, word_count=int, passed=bool)

# agfile for large text outputs:
output_schema=agdata(report=agfile)

# agbinary for raw binary outputs (audio, images, compiled artifacts):
output_schema=agdata(trimmed=agbinary)

# agpath when the value itself must stay a path (not content read from/written to it):
output_schema=agdata(path=agpath, content=str)

# agimage as input (multimodal):
input_schema=agdata(question=str, photo=agimage)

# Typed list:
output_schema=agdata(tags=list[str])

# Dict output:
output_schema=agdata(scores=dict[str, float])

# Tuple output:
output_schema=agdata(bounds=tuple[float, float, float, float])
```

### Container nesting

`list`, `dict`, and `tuple` containers can be nested at any depth, and `agtype` subclasses can appear at any leaf position.  The framework recursively prepares inputs and recovers outputs at every agtype leaf; validation descends into containers to report the exact failing path.

```python
# nested list of images (input)
input_schema=agdata(batches=list[list[agimage]])

# dict of agfile outputs
output_schema=agdata(reports=dict[str, agfile])

# tuple with mixed leaf types
output_schema=agdata(result=tuple[agfile, int])

# deeply nested
output_schema=agdata(matrix=list[list[float]])
```

### Typed list-of-dicts fields

To express a list whose items have a known structure, pass a one-element list containing a plain `dict` that maps field names to types:

```python
output_schema = agdata(
    papers=[{"title": str, "url": str, "abstract": str}],
    count=int,
)
```

The hint shown to the LLM in the system prompt is:

```json
{"papers": [{"title": "str", "url": "str", "abstract": "str"}], "count": "int"}
```

The framework validates every element against the template: each item must be a `dict` containing the declared keys with the declared Python types. A type mismatch is caught immediately when the model calls the field's tool, which lets it correct only that field without restarting. Use this form instead of bare `list` whenever item structure matters.

## ReAct loop

`agskill.run()` is a non-blocking scheduling wrapper: it captures `prev_ctx`, spawns a daemon thread that calls `execute_react()`, and immediately returns a pending `agdata` (backed by a `Future`). The synchronous ReAct loop itself lives in `agskill.execute_react()`.

Each call to `agskill.execute_react()` runs the following steps:

1. Offload oversized input fields to files in the agent's sandbox (see below)
2. Build messages: `[system] + ctx.messages + [user: input.to_json()]`
3. Drain user inbox (`agent._drain_inbox()` — injects mid-conversation messages from `agent.inbox`)
4. Pre-call compaction — check character-based token estimate; compact if over threshold (see [compaction.md](compaction.md))
5. Call the LLM with `stream=True`; accumulate tokens via `_iter_batched()` (see below); retry on connection failure with exponential backoff
6. Post-call compaction — check actual `prompt_tokens` from API usage; compact again if needed
7. If response contains tool calls → for each tool:
   a. Coerce malformed JSON arguments to `"{}"` so history replay never crashes
   b. If the tool is a `return_<field>` output tool → validate and store the value (see [Output collection via tools](#output-collection-via-tools)); skip normal dispatch
   c. If `run_in_subprocess=True`, commit the sandbox to a pre-call checkpoint image
   d. Execute the tool (in the calling thread for sandbox tools, or offloaded to a worker process)
   e. If the result contains `"error"`, restore the sandbox from the checkpoint and append `workspace_reverted` to the error message (see [Tool failure and checkpoint revert](#tool-failure-and-checkpoint-revert))
   f. If the result is large, offload to a file (see [Tool output offloading](#tool-output-offloading))
   g. Append the tool result message and go to 3
8. If response has no tool calls → check if all required output fields have been registered
9. If fields are missing and retries remain → inject reprompt message listing missing fields, go to 3
10. If sandbox has live background processes → `agSandbox.wait_for_processes()` polls until they exit or `ping_interval_s` elapses; inject status message and go to 3
12. Delete offloaded input files, return `(result, updated_ctx, delta)`

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

Standard OpenAI generation parameters set on the agent's `agconfig` (`cfg.agllm_backend.<field>`) are forwarded directly to every API call. vLLM-specific parameters that are not part of the OpenAI spec are merged into `extra_body` automatically.

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

Any value already set on `cfg.agllm_backend.extra_body` is preserved; vLLM-specific keys are added on top.

```python
cfg = agConfig(agLLMBackendConfig(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen3-30B",
    temperature=0.6,
    max_tokens=16000,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
))
```

### Tool output offloading

Tool results longer than the offload threshold are automatically saved to a file inside the agent's sandbox instead of being inlined into the message history. The threshold is `max(40 000, context_limit × 0.1 × 4)` characters — 10 % of the model context window expressed in characters (4 chars/token), with a 40 000-character floor. The tool message is replaced with a short JSON note:

```json
{"note": "Output was too large and has been saved to /workspace/long_tool_call_outputs/webfetch_abc123.txt. Use the read tool to access it."}
```

Files are written to `/workspace/long_tool_call_outputs/<tool_name>_<call_id>.txt`. The agent can read the content at any point using its `read` tool.

This guard prevents a single oversized tool result (e.g. a raw PDF fetched via `webfetch`) from filling the entire context window. If no sandbox is available the result is kept inline unchanged.

### Tool failure and checkpoint revert

Before every `run_in_subprocess=True` tool call, the framework commits the sandbox container to a lightweight checkpoint image:

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

- `run_in_subprocess=False` — no checkpoint is taken, so no revert is possible.
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

Schema field types can be `agtype` subclasses as well as plain type-name strings.  Built-in subclasses are `agfile`, `agbinary`, `agpath`, `agimage`, and `agrawstring`.  See [agtype.md](agtype.md) for the full interface and instructions for writing custom field types.

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

For each output field declared as `agfile`, the system prompt instructs the agent to write the content to a file (e.g. `/workspace/outputs/<skill_name>_<field>.txt`) and return the path as the field value. The `return_<field>` tool description also explicitly tells the agent to pass a file path, not raw content.

**Live validation during the tool call** — when the agent calls `return_<field>` for an `agfile` field, the framework immediately reads the file from the sandbox while the agent is still alive. This lets it catch problems early and reprompt the agent rather than silently failing. The following checks are applied in order:

| Condition | Error returned to agent |
|---|---|
| Path is a directory (`IsADirectoryError`) | `"'<path>' is a directory, not a file. Pass the path to a specific output file (e.g. <path>/<field>.txt)."` |
| File is not UTF-8 text (`UnicodeDecodeError`) | `"file at '<path>' contains binary data and cannot be read as text. Write a UTF-8 text file instead."` |
| Path not found (`FileNotFoundError`) | `"no file found at path '<path>'. Write your output to a file first, then call this tool with that file's path."` |
| File is empty | `"file at '<path>' is empty. Write the actual content to the file before registering the path."` |
| File contains only another path | `"file at '<path>' contains only a path reference ('<inner>'), not real content. Write the actual content to a file and return that file's path."` |

If the sandbox raises an `IsADirectoryError`, the framework checks whether the sandbox path is a directory using `test -d` in the container rather than relying on file-extension heuristics. If it raises a `UnicodeDecodeError`, it means the file's raw bytes could not be decoded as strict UTF-8 — the agent must write a text file. Any other exception from `read_file` is treated as a missing-file error.

On a successful validation the path is stored in the collected outputs. After the skill completes, `agfile.recover()` re-reads the file content from the sandbox and stores it as a plain string in the result agdata. The caller always receives content, not a path.

### Cleanup

All `agfile` files — both input and output — are deleted from the sandbox in the `finally` block after the skill ends, whether it succeeded or raised. They never persist between skill invocations on the same agent.

### Output `agbinary` fields

`agbinary` follows the same file-path contract as `agfile`, but for raw binary data (audio, images, compiled artifacts, etc.) that must not be decoded as text.

**Live validation** uses lightweight shell tests — no content read — since binary files may be large:

| Check | Shell command | Error sent to agent |
|---|---|---|
| Is a directory | `test -d <path>` | Points to a specific binary output file |
| Does not exist | `test -e <path>` fails | Write the file first |
| Exists but empty | `test -s <path>` fails, `test -e` passes | Write actual binary content |

After the skill completes, `agbinary.recover()` reads the raw bytes via `sandbox.read_file_bytes()` (no UTF-8 decode) and the caller receives `bytes`.

### Output `agpath` fields

`agpath` is for fields whose value **is** a path — not content the framework should read from or write to a file. Unlike `agfile`/`agbinary`, `agpath` never touches the sandbox filesystem: the value passes through `prepare()`/`recover()` unchanged.

**Live validation** only checks the *shape* of the value — does it look like a path (`_looks_like_path`)? If not, the tool call returns an error and the agent gets to correct it:

```json
{"error": "field_name 'moved_to': 'here is the content you asked for' does not look like a path. Pass the path string itself, not file content."}
```

This matters because a plain `str` output field has a convenience fallback: if its value looks like a path, the framework assumes the real content lives in that file and silently reads it back (see the "str field with a sandbox path value" bullet below). That fallback is exactly wrong for a field whose value is meant to stay a path — e.g. `output_schema=agdata(path=str, content=str)`, where a correct `return_path("/data/note.txt")` call gets silently overwritten with the note's own content because `/data/note.txt` "looks like a path" to the same str heuristic meant for `content`. Declare such a field as `agpath` instead of `str` to opt out of that fallback entirely.

**Input validation** works the same way: if the caller passes a value that doesn't look like a path for an `agpath` input field, `validate_input()` rejects it before the skill runs (see [Input validation](#input-validation)).

### Schema display

In the JSON format sections appended to the system prompt, `agfile` fields are shown as `"file"`, `agbinary` fields as `"binary_file"`, and `agpath` fields as `"path"`:

```json
{"background": "file", "audio": "binary_file", "moved_to": "path", "theme": "string"}
```

---

## Automatic input offloading

After `agfile` (and other `agtype`) fields have been prepared, the framework checks every remaining top-level string field. Any value that still exceeds the offload threshold is automatically written to a temporary file and the field value is replaced with a short reference:

```
(content saved to /workspace/inputs/design_doc.txt — use the read tool to access it)
```

The threshold is `max(40 000, context_limit × 0.1 × 4)` characters — 10 % of the model context window expressed in characters (4 chars/token), with a 40 000-character floor. `INPUT_OFFLOAD_CHARS` is the constant floor; the effective value is computed dynamically at runtime and passed to `_offload_large_fields`.

Because `agfile` fields are already converted to short file paths before this check runs, they are never double-offloaded.

When auto-offloading occurs, the system prompt receives an extra note listing the affected fields:

```
Note: The following input fields contain large content that has been automatically
saved to temporary files in your sandbox: `design_doc`, `previous_chapter`. The
file paths are shown in the input JSON. Use the read tool to access the full content.
WARNING: these files are temporary and will be automatically deleted after this task ends.
```

Only top-level string fields are auto-offloaded. Non-string values (integers, booleans, lists, nested agdata) are always inlined. Auto-offloaded files are cleaned up in the same `finally` block as `agfile` files.

## Input validation

Input is validated against `input_schema` before the loop starts. Validation checks that all required fields are present and have the correct Python type. For `agtype` fields, the check delegates to that class's `validate_input_value()` — the default (used by `agfile`, `agbinary`, `agimage`) just requires a `str`; `agpath` additionally requires the string to look like a path. If validation fails, the skill returns immediately with an `agdata(error=...)` without calling the LLM.


## Output collection via tools

When an `output_schema` is declared (and it is not an `agrawstring` schema), the framework generates one typed tool per output field and adds them to the tool list at the start of the skill run:

```
return_summary(summary: string)
return_word_count(word_count: integer)
return_passed(passed: boolean)
```

Each tool has exactly one parameter named after the field (not `"value"`). This allows models to match the tool name to its parameter by name, improving tool call reliability across LLM vendors.

The system prompt instructs the model to call each `return_<field>` tool once it has the final value for that field. The model may interleave these calls freely with other tool use (bash, read, write, etc.) — it is not required to call them all at once or last.

### Per-field validation

Each `return_<field>` call is validated immediately against the schema hint. The result of each call is logged to `ag.terminal` and `ag.log`:

- **Type mismatch** → the tool returns `{"error": "field 'X': expected bool, got str"}` inline, and **`TOOL ✗`** is logged to `ag.terminal` and `ag.log` with the raw tool call args and the error message. The model sees the error in the same response turn and can retry just that field without losing any other already-registered outputs.
- **`agfile` field** → file is read from the sandbox immediately; see [Output `agfile` fields](#output-agfile-fields) for the full set of checks and error messages.
- **`agpath` field** → the value is checked against `_looks_like_path` only — no sandbox file is read or written. If it doesn't look like a path, the tool returns an error and the agent retries; see [Output `agpath` fields](#output-agpath-fields).
- **`str` field with a sandbox path value** → if the value looks like a sandbox path (starts with `/`, only word characters, dots, and hyphens per segment), the framework reads the file at that path and substitutes its content, printing a `[agschema] WARNING` to stderr each time this fires. If the file is unreadable or its content is itself a path, the original value is kept. This handles the common case where the agent writes a `str` output to a file and returns the path instead of the content — but it also means a plain `str` field can never reliably hold a path-shaped value; use `agpath` for fields whose value must stay a path.
- **Success** → the tool returns `{"result": "✓ 'X' registered. Still needed: [...]"}` (or `"All required fields complete."` on the last one), and **`TOOL ✓`** is logged to `ag.terminal` and `ag.log` with the tool call args.

### Completeness check and reprompt

When the model produces a response with no tool calls at all, the framework checks whether all required fields have been registered:

- **All fields present** → assemble `agdata(**collected)`, then return (or inject background-process status if needed).
- **Fields missing** → inject a reprompt: `"You have not yet provided all required output fields. Still missing: ['X', 'Y']. Call return_<field> for each missing field."` and continue the loop. This consumes one retry from `max_output_schema_retries`.
- **No retries left** → return `agerror("output schema error: missing fields after retries: ...")`.

Output collection is skipped when the LLM response is answering a mid-conversation user message injected via the inbox (`had_inbox=True`), because the LLM is engaged in dialogue rather than producing a final structured answer.

### No-schema output

When `output_schema=None` (or an `agrawstring` schema is used), no `return_<field>` tools are generated. Instead, when the model produces a response with no tool calls the framework takes the raw text content of the assistant message and returns it as `agdata(result=content)` — it is **not** JSON-parsed. For `agrawstring` schemas the field key is taken from the schema's raw key; for `output_schema=None` the key is always `"result"`.

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

The LLM receives the status message, can read log files or call more tools, then produces another final answer — which triggers another `agSandbox.wait_for_processes` check. The skill only exits when `agSandbox.wait_for_processes` returns `None` (no live watched PIDs).

`ping_interval_s` and `poll_interval_s` are class-level attributes on `agent` (defaults 300 s and 5 s) passed through to `agskill.run()` at call time. The total number of monitoring continuations is bounded by `max_steps`.

See [execution_process_control.md](execution_process_control.md) for per-scenario traces.

## Context (`agcontext`)

The `prev_ctx` (an `agcontext`) passed to `agskill.run()` is the agent's shared conversation context. `agskill.run()` captures it before spawning the thread; `execute_react()` appends the full message exchange and returns the updated `ctx`. The system prompt is re-injected fresh on every call and is not persisted in the stored context.

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
    agconfig = cfg

    def setup(self):
        self.find_papers     = FindPapersSkill(max_papers=10)
        self.summarise_paper = SummarisePaperSkill()
        self.compile_report  = CompileReportSkill()
        self.agent           = agent()
```
