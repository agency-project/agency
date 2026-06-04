# Skills

An `agskill` is a named, self-contained ReAct loop with its own system prompt, optional input/output schemas, and an optional tool override. Skills are the unit of work submitted to an agent via `agent.run("skill_name", input)`.

## Defining a skill

```python
from agency import agskill, agdata

skill = agskill(
    name="summarize",
    system_prompt="You are a concise summarizer. Return only valid JSON.",
    input_schema=agdata(text="str"),
    output_schema=agdata(summary="str", word_count="int"),
)
```

Both schemas are serialized and appended to the system prompt so the LLM knows the expected format.

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Identifier used in `agent.run("name", ...)` |
| `system_prompt` | `str` | System message prepended to every LLM call |
| `tools` | `list[agtool] \| None` | Tool override; `None` inherits the agent's full tool list |
| `input_schema` | `agdata \| None` | Required input fields and their types |
| `output_schema` | `agdata \| None` | Required output fields; enforced with retries |
| `output_validator` | `Callable \| None` | Custom validation function, called after schema check |
| `max_retries` | `int` | Times to retry on output schema failure (default `3`) |

## ReAct loop

Each call to `agskill.run()` executes a standard ReAct loop:

1. Build messages: `[system] + history + [user: input.to_json()]`
2. Drain user inbox (injected mid-conversation messages from `agUI` or `agent._inbox`)
3. Call the LLM
4. Check token usage — compact context if over threshold (see [compaction.md](compaction.md))
5. If response contains tool calls → execute each tool, append results, go to 2
6. If response is a final answer → parse JSON, validate against `output_schema`
7. If validation fails and retries remain → inject correction message, go to 2
8. Return `(result, updated_history, history_delta)`

The loop exits early on `max_steps` (default `10`) exceeded.

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

## Skill-level tool override

```python
search_tool = agtool("search", "Search the web.", fn=my_search_fn, params={...})

skill = agskill(
    name="research",
    system_prompt="Research the topic using web search only.",
    tools=[search_tool],   # agent's sandbox tools are not available in this skill
)
```

Setting `tools=[]` gives the skill no tools at all — pure reasoning only.

## Common skills (`agency.common_skills`)

`agency.common_skills` provides ready-made skill classes for common tasks. Each is a thin subclass of `agskill` with a fixed name, system prompt, and schemas. All accept `**kwargs` forwarded to `agskill.__init__` (e.g. `max_retries=1`).

### `WriterSkill`

Writes content to a file path inside the sandbox container. Requires the agent to have the sandbox `write` tool available (inherited via `tools=None`).

```python
from agency.common_skills import WriterSkill
skill = WriterSkill()
# input:  agdata(file_path="str", content="str")
# output: agdata(path="str", status="str")
```

### `SummariserSkill`

Summarises a piece of text in one sentence. Has no tools (`tools=[]`) — pure reasoning.

```python
from agency.common_skills import SummariserSkill
skill = SummariserSkill()
# input:  agdata(text="str")
# output: agdata(summary="str")
```

### `FindPapersSkill`

Searches arxiv for papers on a topic using a bundled `search_arxiv` tool. Includes an output validator that rejects empty paper lists and forces a retry.

```python
from agency.common_skills import FindPapersSkill
skill = FindPapersSkill(max_papers=8)   # default 16
# input:  agdata(topic="str")
# output: agdata(papers="list", count="int")
```

The bundled tool is also accessible as `skill.search_arxiv` if you need to reuse it elsewhere.

### `SummarisePaperSkill`

Fetches the full text of an arxiv paper and writes a technical summary covering contribution, method, results, limitations, and conclusions. Uses a bundled `fetch_paper` tool that converts any arxiv URL form (`/abs/`, `/pdf/`, `/html/`) to the HTML version and extracts plain text via `html2text` (capped at 32 000 characters).

```python
from agency.common_skills import SummarisePaperSkill
skill = SummarisePaperSkill()
# input:  agdata(title="str", url="str", abstract="str")
# output: agdata(summary="str")
```

The system prompt requires the agent to call `fetch_paper` before responding — it will not summarise from the abstract alone.

### `CompileReportSkill`

Writes a structured markdown research report to a path inside the sandbox. Requires the agent to have the sandbox `write` tool available.

```python
from agency.common_skills import CompileReportSkill
skill = CompileReportSkill()
# input:  agdata(topic="str", summaries="list", output_path="str")
# output: agdata(report_path="str", paper_count="int")
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
        self.agent           = self.make_agent(
            [self.find_papers, self.summarise_paper, self.compile_report]
        )
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
    output_schema=agdata(score="int", feedback="str"),
    output_validator=validate_score,
)
```
