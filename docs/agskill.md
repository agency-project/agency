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
| `max_retries` | `int` | Times to retry on output schema failure (default `3`) |

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
4. Call the LLM with `stream=True`; accumulate tokens via `_iter_batched()` (see below)
5. Check token usage — compact context if over threshold (see [compaction.md](compaction.md))
6. If response contains tool calls → execute each tool (offloaded to a worker process), append results, go to 3
7. If response is a final answer → parse JSON, validate against `output_schema`
8. If validation fails and retries remain → inject correction message, go to 3
9. Delete offloaded input files, return `(result, updated_history, history_delta)`

The loop exits early on `max_steps` (default `10`) exceeded.

### Streaming and GIL pressure

The LLM call uses `stream=True`. Without batching, each SSE token chunk would acquire and release the Python GIL once, creating O(tokens) context switches that slow down all concurrent agent threads.

`_iter_batched()` reduces this to O(tokens / batch_size):
- A background thread drains the SSE stream, doing one `queue.put` per chunk (minimal GIL hold)
- The main thread sleeps for `_BATCH_INTERVAL_S` (100 ms default) — GIL fully released
- On wake, the main thread drains everything buffered during the sleep in one burst

This means threads running other agents get 100 ms of uncontested GIL time for every batch of streamed tokens, dramatically improving throughput under concurrent load.

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

Input validation is **skipped** on outer-loop re-entries (`_is_continuation=True`) so that process-status ping messages can flow through without matching the skill's declared input schema.

## Output validation and retries

After a non-tool-call LLM response:

1. Parse the response content as JSON (markdown code fences are stripped)
2. Check all fields in `output_schema` are present with the correct type
3. Run `output_validator(result)` if provided — returns a list of error strings
4. If errors exist and `retries_left > 0`: inject a correction message and loop
5. If errors persist after all retries: return `agdata(error="output schema error after retries: ...")`

Output validation is skipped when the LLM response is answering a mid-conversation user message injected via the inbox (`had_inbox=True`), because the LLM is engaged in dialogue rather than producing a final structured answer.

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

`agency.common_skills` provides ready-made skill classes for common tasks. Each is a thin subclass of `agskill` with a fixed name, system prompt, and schemas. All accept `**kwargs` forwarded to `agskill.__init__` (e.g. `max_retries=1`).

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
