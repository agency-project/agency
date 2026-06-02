# Skills

An `agskill` is a named, self-contained ReAct loop with its own system prompt, optional input/output schemas, and an optional tool override. Skills are the unit of work submitted to an agent via `agent.run("skill_name", input)`.

## Defining a skill

```python
from src.agskill import agskill
from src.agdata import agdata

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
2. Call the LLM
3. If the response contains tool calls → execute each tool, append results, go to 2
4. If the response is a final answer → parse JSON, validate against `output_schema`
5. If validation fails and retries remain → inject correction message, go to 2
6. Return `(result, updated_history, history_delta)`

The loop exits early on `max_steps` (default `10`) exceeded.

## Input validation

Input is validated against `input_schema` before the loop starts. Validation checks that all required fields are present and have the correct Python type. If validation fails, the skill returns immediately with an `agdata(error=...)` without calling the LLM.

Input validation is **skipped** on outer-loop re-entries (`_is_continuation=True`) so that `process_completed` and `process_update` ping messages can flow through without matching the skill's declared input schema.

## Output validation and retries

After a non-tool-call LLM response:

1. Parse the response content as JSON (markdown code fences are stripped)
2. Check all fields in `output_schema` are present with the correct type
3. Run `output_validator(result)` if provided — returns a list of error strings
4. If errors exist and `retries_left > 0`: inject a correction message and loop
5. If errors persist after all retries: return `agdata(error="output schema error after retries: ...")`

## History

The history passed to `agskill.run()` is the agent's shared conversation context. The skill appends its full message exchange (user → tool calls → tool results → assistant) to this history and returns the updated version. The system prompt is included in `history_delta` but not persisted in the stored history, so it is re-injected fresh on every call.

## Skill-level tool override

```python
search_tool = agtool("search", "Search the web.", fn=my_search_fn, params={...})

skill = agskill(
    name="research",
    system_prompt="Research the topic using web search only.",
    tools=[search_tool],   # agent's sandbox tools are not available in this skill
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
