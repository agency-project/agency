# Context tools

`context_tools()` gives a skill host tools with which the model edits what its own
next LLM calls see. The edits are patches in the run's `message_patch_lookup`, the
table `agent.patch_history` writes (see [Message_Patches.md](Message_Patches.md)):
the harness keeps sending its own history and `LlmHandlerServer` swaps the edited
blocks in on the way to the LLM backend.

```python
from agency import agskill, context_tools

skill = agskill(name="research", prompt="...", add_host_mcp_tools=context_tools())
```

A runnable walkthrough is `examples/12_self_context_editing.py`. The tools carry no
instructions on when to edit; the skill prompt or input has to say.

## The tools

| Tool | Arguments | What it does |
|---|---|---|
| `list_context` | — | One line per block of the latest request: `[id] role/type, size: preview`, marked `(edited)`, `(deleted)` or `(protected: why)` |
| `read_context_block` | `id`, `offset`?, `max_chars`? (4000) | One block in full, in pages |
| `edit_context` | `id`, and `text` or `delete: true` | Replaces the block's text, or removes the block |
| `revert_context_edit` | `id` | Withdraws the model's edit of that block |

An `id` is the first 8 characters of the block's hash; a block keeps it from request
to request. An edit applies from the model's next LLM call.

`context_tools(*, max_edits=50, allow_self_edit=False, persist=False)`:

* `max_edits` caps `edit_context` and `revert_context_edit` calls per skill run
  (`None`: no cap). Each edit restarts the provider's prompt cache from that block.
* `persist=True` keeps the edits in the agent's `agcontext` for later skill runs.
  Otherwise they are withdrawn when the run's host services stop, on success or
  failure (`HostServerManager.stop` -> `ContextView.close`).
* `allow_self_edit=True` lets the model edit these tools' own results, which are
  otherwise kept as a record of what it edited.

## Protected blocks

The host refuses to edit the system prompt, the `tool_schema` message (not listed),
`thinking` and `redacted_thinking` (signed), `tool_use`, a `tool_result` that is not
plain text, and any block the operator patched.

A `tool_result` keeps its block and only loses its text, so it stays paired with its
`tool_use`; deleting it, or the only block of a message, leaves
`[removed by context edit]`. Replacing a non-text block puts a text block in its
place.

Each patch records its origin, `operator` or `model`, in
`MessageStore.message_patch_origin` (saved in `state.json`) and in the SQLite
`message_patch_lookup.origin` column.

## Wiring

`HostServerManager` makes one `ContextView` (`agency/context_tools.py`) per skill run
and `HostMcpServer` passes it as `context_view=` to the host tools whose function
takes it. The view reads `LlmHandlerServer.latest_request()`, the harness's latest
request that is not `agency_internal_kind`, with its hashes and the patches applied
to it, and writes into the run's `message_store` with `origin="model"`. It recognises
its tools' calls under every harness's MCP prefix (`mcp__agency__edit_context`,
`agency.edit_context`, ...).

## Limits

* When the harness compacts or rewrites a block, the edit of it matches nothing.
  Once a later request did not use an edit, `list_context` and `edit_context`
  return a `warning` naming it.
* With several conversations in one skill run (subagents), "the latest request" is
  whichever called the LLM last.
* The harness counts tokens and schedules compaction on its own unedited copy.
* Edits replace or remove whole blocks; there is no free-text edit across blocks.
* On Claude models with preserved thinking, an edit invalidates the signed thinking
  blocks after it; the API drops them (see the Limits of
  [Message_Patches.md](Message_Patches.md)).
* Models use the tools rarely unless the task tells them to.

## Tests

```sh
pytest tests/test_context_tools.py
```

Covers listing, paging, every edit and protection rule, the edit cap, stale-edit
warnings, withdrawal at the end of a run, and a model edit through a real
`HostServerManager` over the host UDS with the SQLite record checked.
