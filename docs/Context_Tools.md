# Context tools: the model edits its own context

`context_tools()` lets the model, from inside a skill run, change what its next LLM
call sees. It drives the same `message_patch_lookup` as `agent.patch_history` (see
[Message_Patches.md](Message_Patches.md)), so nothing in the harness changes: Claude
Code, Codex, the native harness and the rest keep sending their own history, and
`LlmHandlerServer` swaps the edited blocks in on the way to the LLM backend. This is
the "context as something the model edits" idea of Context Language Models, on any
harness.

```python
from agency import agskill, context_tools

skill = agskill(
    name="research",
    prompt="...",
    add_host_mcp_tools=context_tools(),          # opt in, per skill
)
```

A runnable walkthrough is `examples/12_self_context_editing.py`.

## The tools

| Tool | Arguments | What it does |
|---|---|---|
| `list_context` | — | One line per block of the latest request: `[id] role/type, size: preview`, with `(edited)`, `(deleted)` or `(protected: why)` |
| `read_context_block` | `id`, `offset`?, `max_chars`? (4000) | One block in full, in pages |
| `edit_context` | `id`, and `text` or `delete: true` | Replaces the block's text, or removes the block |
| `revert_context_edit` | `id` | Withdraws the model's edit of that block |

An `id` is the first 8 characters of the block's hash in the latest request the
harness sent. The same block keeps its id from call to call, so the model can list
once and edit later.

An edit takes effect from the model's next LLM call: the model calls the tool, the
harness waits for the result, and its next request goes through the patch table.
No `pause()` is needed.

## Options

`context_tools(*, max_edits=50, allow_self_edit=False, persist=False)`

* `max_edits` caps `edit_context` and `revert_context_edit` calls per skill run. Each
  edit makes the provider re-read the prompt from that block on (the prompt cache
  is invalidated from there), so unbounded editing is expensive. `None` is no cap.
* `persist=True` keeps the model's edits when the skill run ends. They stay in the
  agent's `agcontext`, so later skill runs (and `agent.history`) see them. By
  default the edits are withdrawn when the skill run's host services stop
  (`HostServerManager.stop` -> `ContextView.close`), whether the run succeeded or
  failed.
* `allow_self_edit=True` lets the model edit the results of these tools themselves.
  By default they stay in context as a record of what it edited.

## What the model can never edit

Enforced on the host, whatever the model asks:

* the system prompt (`system` blocks) and the `tool_schema` message (tool
  definitions, `tool_choice`, request fields) -- the latter is not even listed;
* signed reasoning (`thinking`, `redacted_thinking`): an edit breaks the provider's
  signature check;
* tool calls (`tool_use`): a changed or missing call orphans its result;
* a block the operator patched with `agent.patch_history` (operator patches can
  only be changed by the operator);
* a `tool_result` that is not plain text.

Editing keeps a request valid for the provider:

* a `tool_result` is never removed, only its `text` replaced (`delete` leaves
  `[removed by context edit]`), so it stays paired with its `tool_use`;
* deleting the only block of a message leaves `[removed by context edit]` instead of
  dropping the message, which could break the provider's turn order;
* replacing a non-text block (an image, say) puts a text block in its place.

## Who made a patch

Every patch has an origin: `operator` (`agent.patch_history`, the default) or
`model` (the context tools). It is kept in `MessageStore.message_patch_origin`,
saved in `state.json`, and recorded per exchange in the SQLite
`message_patch_lookup.origin` column (`reconstruct_llm_patches` returns it), so
operator and model edits can always be told apart afterwards.

## When an edit stops matching

A patch matches the block the harness sends, by hash or, failing that, by hash
without `index` (so a block the harness inserts in front, like a Claude Code
`<system-reminder>`, does not break it). If the harness compacts its history or
changes the block, the edit matches nothing. The context tools notice: once a
request made after an edit did not use it, `list_context` and `edit_context` return
a `warning` naming the edits that no longer have any effect.

## How it is wired

* `ContextView` (`agency/context_tools.py`) is made per skill run by
  `HostServerManager` from the run's `LlmHandlerServer`. It reads
  `LlmHandlerServer.latest_request()` -- the latest request the harness itself made
  (not an `agency_internal_kind` one), with the hashes of what the harness sent and
  the keys of the patches applied to it -- and writes edits into the run's
  `message_store` with `origin="model"`.
* `HostMcpServer` passes `context_view=` to every host tool; `agtool` hands it only
  to tools whose function asks for it, so no other tool sees it.
* Harnesses prefix MCP tool names differently (`mcp__agency__edit_context`,
  `agency.edit_context`, ...); the view recognises the context tools' own calls
  under any of these.

## Limits

* With several conversations in one skill run (a harness's subagents), "the latest
  request" is whichever conversation called the LLM last.
* The harness still counts tokens and decides when to compact on its own,
  unedited copy. Large deletions make the harness think its context is fuller than
  the LLM sees; large insertions the opposite.
* Edits are per block. Free-text edits across blocks (regex over the whole
  context) are not supported.
* Models that were not trained to manage their context tend to use these tools
  sparsely; measure before relying on them.

## Tests

```sh
pytest tests/test_context_tools.py
```

It covers listing, paging, stable ids, every edit and protection rule, the edit
limit, stale-edit warnings, withdrawal at the end of a run, the tools through
`HostMcpServer`, and a model edit through a real `HostServerManager` over the host
UDS (request, tool call, request) with the SQLite record checked.
