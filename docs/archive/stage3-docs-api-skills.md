# Skills, tools and policies

```python
from agency import agskill, agtool, agdata
from agency.agpolicy import agpolicy
```

## Skills

`agskill(name, prompt, add_host_mcp_tools=None, add_sandbox_mcp_tools=None, input_schema=None, output_schema=None, max_output_schema_retries=10, policy=None)` defines a reusable task. Name/prompt describe the model instruction. Schema arguments are agdata templates (internally wrapped as agschema); absent schemas omit that contract. Extra tools are lists of agtool. Policy is shared, not cloned.

Construction does not call a model or launch a sandbox. `skill.run(ag, skill_input, max_steps=None)` submits and returns pending agdata; `await skill.asyncio_run(...)` waits. Prefer `ag.run(skill, input)` for an application. The max-step and dependency contracts match [Agent](../api/agents.md).

**Execution does not automatically call input schema validation.** Schemas guide prompts and input preparation. If input validity is required, first resolve dependencies and call `skill.input_schema.validate_input(payload)` explicitly (if a schema exists). It returns `None` or an error string; it does not raise merely because validation failed.

```python
from agency import agdata, agskill

skill = agskill("title", "Give the text a title.",
                input_schema=agdata(text=str), output_schema=agdata(title=str))
payload = agdata(text="Documentation is an application interface.")
payload.resolve_input_dependencies()
error = skill.input_schema.validate_input(payload)
if error is not None:
    raise ValueError(error)
```

Output fields are validated during submission/final-answer handling. Invalid final output triggers correction attempts up to the configured retry count; exhausted execution returns an error payload. `build_user_content(skill_input)` returns JSON text, a raw single-field string, or multimodal content blocks; may resolve/serialize input and access host image files. The exact schema/type semantics are in [schemas and types](../api/types.md).

Default host tools provide resource reserve/release/inspection, daemon release, `submit_output` and `submitted_output`. Extra tools extend these. Structured output normally submits named fields; single-field `agrawstring` is a raw-text contract.

## Tools and execution location

`agtool(name, description, fn, params=None, log_fn=None, persistent_vars=None)` stores an executable callable and OpenAI-style JSON parameter schema. `params=None` uses an empty object schema. The callable receives one agdata argument plus explicitly named context parameters (selected from its signature). It must return agdata; direct calls do not enforce/coerce this return type or validate arguments against `params`.

| API/placement | Return, effects and ownership |
| --- | --- |
| `tool(arg, timeout=None, **context)` | Synchronous in the calling thread/process, returns `fn`'s result. Catches callable exceptions into a direct agerror with traceback. **The accepted timeout is not enforced.** Implement an actual deadline inside long-running tools. |
| `add_host_mcp_tools=[tool]` | Executes the real host callable/closure, with access to captured host objects. MCP transport waits for the call; tool configuration is not a general hard kill deadline. |
| `add_sandbox_mcp_tools=[tool]` | Cloudpickles/reconstructs tools in the sandbox. Captured values must be serializable and dependencies installed there. Host-only live objects are not magically available. |
| `persistent_vars` | Dict of variable names to factories for persistent MCP-server state on the selected host/sandbox side; callable execution context receives requested named state. Factories must work in the executing environment. |
| `log(arg, result, elapsed_ms)` | Calls `log_fn(tool, arg, result, elapsed_ms)` when supplied, returns `None`. Logging is outside the callable exception wrapper; callback exceptions propagate. |
| `to_openai_tool()` | Returns the named function schema dict. Does not execute or validate input. |

```python
from agency import agdata, agtool

def double(arg: agdata) -> agdata:
    return agdata(value=arg.value * 2)

tool = agtool("double", "Double an integer", double,
              params={"type": "object", "properties": {"value": {"type": "integer"}},
                      "required": ["value"]})
assert tool(agdata(value=3)).value == 6
```

## Policies

`agpolicy(tool_hooks=None, syscall_hooks=None, default_to_deny=False)` is a mutable dataclass supplied to a skill. Hook maps are keyed by tool/syscall name. Tool hooks receive a dict of arguments; syscall hooks receive an `agsyscallevent`. Hooks return bool or `(bool, reason)`. Without a matching hook, default is allow unless `default_to_deny=True`. Hook exceptions deny with an explanatory reason. Policy applies through the harness interaction paths; a direct Python `tool(...)` call bypasses it.

`agsyscallevent` is the callback payload from `agency.harness._syscall_event` (despite that module's internal-looking name). Required fields are syscall, pid, tid, argv, envp, path and timestamp. Optional tool_name/tool_args/program/address/port default to `None`; ptrace has kernel-resolved arguments, native hooks have semantic tool fields. Do not assume every harness populates every field, or that these controls form a universal security boundary.

```python
from agency.agpolicy import agpolicy

policy = agpolicy(tool_hooks={"double": lambda args: (args["value"] >= 0, "nonnegative only")})
```

## Source signatures

[Source: agskill.py](../../agency/agskill.py)

::: agency.agskill.agskill
    options:
      members: ["__init__", "build_user_content", "run", "asyncio_run"]

[Source: agtool.py](../../agency/agtool.py)

::: agency.agtool.agtool
    options:
      members: ["__init__", "__call__", "log", "to_openai_tool"]

[Source: agpolicy.py](../../agency/agpolicy.py)

::: agency.agpolicy.agpolicy
    options:
      members: []

[Source: _syscall_event.py](../../agency/harness/_syscall_event.py)

::: agency.harness._syscall_event.agsyscallevent
    options:
      members: []
