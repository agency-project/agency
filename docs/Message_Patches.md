# Message hash lists and persistent message patches

`agent.patch_history()` changes what the LLM backend is sent, in the skill run that
is active now and in every later one. Combined with `agent.pause()` it lands at an
exact LLM call.

```python
from agency import MessageBlock

ag.pause()                                   # returns once the harness is stopped
key = ag.patch_history(
    MessageBlock("user", {"type": "text", "index": 0, "text": "secret"}),        # as sent
    [MessageBlock("user", {"type": "text", "index": 0, "text": "[redacted]"})],  # replaces it
)
ag.resume()                                  # the next LLM call carries the patch
ag.remove_history_patch(key)                 # or ag.clear_history_patches()
```

A runnable walkthrough is `examples/13_context_manipulation.py`.

## The API

* `MessageBlock(role, fields)` is one content block (`fields` is a mapping in the
  agency block format, for example `{"type": "text", "index": 0, "text": "..."}`)
  together with the role of the message that carries it. It is immutable and copies
  `fields`. `.hash` is the SHA-256 of the whole payload, which is the key of its row
  in `message_block_lookup`; `.role`, `.type`, `.to_dict()` (the block without its
  role) and `.payload` (the stored row) read it back. Two `MessageBlock`s are equal
  when their hashes are. A `tool_use` block has `id`, `name` and `arguments`; a
  `tool_result` block has `tool_call_id` and `text`. `agent.history` still returns
  plain message dicts; build the block to patch from one with
  `MessageBlock(message["role"], block)`.
* `patch_history(old_block, new_blocks) -> key`
  * `old_block` is a `MessageBlock`: the block exactly as the harness sends it.
    `new_blocks` is a list of `MessageBlock`. A raw dict is a `TypeError`. A patch
    that leaves a `tool_result` without its `tool_use` is not rejected.
  * `new_blocks` is the list of replacement blocks, in order. They are spliced into
    the message as separate, consecutive blocks, not joined. An empty list removes
    the block, and a message left with no blocks is dropped. `index` fields in a
    patched message are renumbered to match position.
  * `key` is `old_block.hash`; pass it to `remove_history_patch`.
* `remove_history_patch(key)` removes one patch. `clear_history_patches()` removes
  all of them.

Calling `patch_history` again with the same `old_block` replaces the earlier link.
A patch matches the block exactly as the harness sends it, and replaces every
identical block (a tool call or result is unique by id). Patches are not applied
recursively.

## When a patch lands

`patch_history`, `remove_history_patch` and `clear_history_patches` do two things:

1. They are published in order, like `queue_message`: skill runs submitted after
   the call start from an `agcontext` that has the change.
2. If a skill run of this agent is active, they are also applied at once to that
   skill run's own `agcontext` (a skill run edits its `agcontext` in place).
   `LlmHandlerServer` reads `message_patch_lookup` on every LLM call, so the patch
   lands at that skill run's next call. A skill run that fails keeps the patch.
   Applying a patch a second time, when its ordered turn comes, changes nothing.

A request that reached the host server before the patch was made is not patched.

## The hash lists and lookups

`agency/llm/message_store.py` is shared by the SQLite data collector and `agcontext`:

* a **block** is `{"role", **block}`, keyed by the SHA-256 of its complete canonical
  JSON. Nothing is left out of the key (tool-call arguments, tool-result content,
  `citations`, `index`, ...), so blocks that differ in any field are different rows
  and a stored payload is exactly what was seen.
* a **message** is `{"role", "blocks": [block hashes], "extra"?}`, keyed the same
  way. This keeps message boundaries. A message that is not in the agency shape is
  stored whole as `{"opaque": message}`.

`agcontext` holds:

* `message_hash_list`: the ordered message hashes of the most recent skill run;
* `message_store`: a `MessageStore` with three lookups, all keyed by content hash:
  * `message_structure_lookup`: message hash to its record;
  * `message_block_lookup`: block hash to block payload;
  * `message_patch_lookup`: original block hash to the ordered hashes of the blocks
    that replace it. Both sides are rows of `message_block_lookup`; the replacement
    blocks are written there when the patch is added.

The store is written with `add_block(MessageBlock) -> hash`, `add_message(message) -> hash`
and `add_messages(messages) -> hashes`; each stores a row once, however often it is called.
`add_patch(old_hash, new_hashes) -> old_hash` is hash-based: it links a block hash to the
hashes of the replacement blocks, which must already be rows of `message_block_lookup`
(`add_block` stores a new one first; `old_hash` need not be a row yet). `patch_history`
does both steps for the skill runs it reaches.

`MessageStore.materialize(hashes)` rebuilds messages: for each message hash it reads
the record, and for each block hash it either uses the block from
`message_block_lookup` or, if the hash is a key in `message_patch_lookup`, the
linked replacement blocks. `agent.history` and `agcontext.get_resolved_transcript()`
call it, so a rebuilt history always shows the patches; mutating the result changes
nothing. `agent.history = ...` replaces `message_hash_list` with the hashes of the
messages given. `agent.save()` writes `message_hash_list`, `message_structure_lookup`,
`message_block_lookup` and `message_patch_lookup` into `state.json`, and
`agent.load()` still reads the old `history` message list.

Access to a `MessageStore` is exclusive: every method holds one reentrant lock, so a
patch added, withdrawn or cleared from another thread (for example `patch_history`
while an LLM call is being staged) is never seen halfway through a request or a
rebuild. A request that has already been sent is not changed; the patch applies at
the next one. Code that reads the lookups directly, or needs several calls to see one
state (`agDataLogger.record_llm_exchange`, `agcontext.message_state`), does so inside
`with store:`.

Nothing is pruned: the lookups only gain rows. A skill run edits its `agcontext` in
place and hashes into its `message_store`. The engine stages the new
`message_hash_list` and writes it only after the skill run's sandbox commit
succeeds, so a failed or cancelled skill run leaves `message_hash_list` as it was
(patches survive it).

### Tools and other request fields are messages

A message is any data exchanged between a harness and an LLM backend, so a request's
tool definitions, its `tool_choice` and its other fields are messages too. The
harness adapter writes them into the request's `messages` with
`with_tool_schema(messages, tools, tool_choice, extra)` (in `message_store.py`): one
message with role `tool_schema`, placed first, with one block per item and every key
kept:

* `{"type": "tool_definition", "index": i, "tool": <the whole tool dict>}` per tool;
* `{"type": "tool_choice", "index": i, "value": <tool_choice>}`;
* `{"type": "request_extra", "index": i, "value": {<every other field>}}`
  (the native harness's `agency_internal_kind` is here).

`LlmHandlerServer` and everything on the host see only this message list, so these
blocks are hashed, patched (`patch_history`), recorded and written to `agent.history`
like any other. The LLM backend calls `split_tool_schema(messages)` just before it
builds its own request, getting `(chat messages, tools, tool_choice, extra)`; it
forwards `tools` and `tool_choice` natively and ignores `extra`. `stream` only
chooses the transport, so the `/dispatch` route removes it and it is not recorded.

An exchange that fails or is cancelled is recorded as one `assistant` response message
whose single block is `{"type": "error", "index": 0, "error": "<Type: message>"}` or
`{"type": "cancelled", "index": 0}`.

### When hashes are made

1. A request arrives, already translated to agency format by the adapter.
   `LlmHandlerServer._stage_request` hashes its messages with `add_messages`:
   `harness_message_hashes`, what the harness sent.
2. `materialize(harness_message_hashes)` rebuilds the messages with every block that is
   a key in `message_patch_lookup` replaced. Those are the `patched_messages`; they
   replace the request's messages and are hashed again as `patched_message_hashes`.
3. The patched messages go to the LLM backend. When the reply arrives,
   `_finalize_success` hashes it and hands `agDataLogger.record_llm_exchange` the
   `message_store` and the three hash lists (`harness_message_hashes`,
   `patched_message_hashes`, `response_message_hashes`); the data collector reads
   the records, blocks and patches from the `message_store` lookups itself.
4. When the skill run ends, `get_main_history` takes the call with the most
   harness messages (for a stateful harness, the last, preferring calls that contain
   the skill's task text), drops system messages and appends the reply. The engine
   sets that list as `message_hash_list`. It holds the harness's hashes, so
   withdrawing a patch also restores the history.

## Pause makes it synchronous

`agent.pause()` blocks until the harness is stopped, not merely until SIGSTOP has
been sent:

* **Tracer.** `TracerLoop.pause()` holds every known pid and marks the loop as
  pausing; `wait_paused()` waits on a condition until every known pid is parked.
  A process forked after `pause()` took its snapshot is held from its first stop; a
  multi-threaded process has every thread stopped with a thread-directed `tgkill`;
  and a child's own initial SIGSTOP, which coalesces with the one `pause()` sends,
  is parked when held.
* **Daemon.** `/control/pause` issues the pause under the control lock, then waits
  outside it, so `cancel` and `resume` are never blocked behind the wait. The reply
  carries `parked`.
* **Agent.** `pause(timeout_s=10.0)` raises `TimeoutError` if the harness is not
  parked in time. The pause stays requested, so `resume()` still works. Pausing while
  no harness process exists returns immediately; the sticky flag holds the next
  process paused from spawn. An unreachable daemon is still a warning. The web UI's
  `pause_all` isolates each agent. `pause(async_=True)` only requests the stop and
  returns at once, without waiting to learn whether the harness actually parked (so
  it never raises `TimeoutError`, and `timeout_s` is unused); use it when the caller
  does not need the pause to have landed before the next line runs.

## SQLite

The collector keeps the same hashes in tables of the same names:

* `message_block_lookup` (formerly `blocks`, keyed by the lossless hash);
* `message_structure_lookup`;
* `per_exchange_messages`: the ordered message hashes of each exchange's prompt and
  reply;
* `message_patch_lookup`: the patches an exchange's request went through, as rows
  `(call_label, original_hash, seq, replacement_hash)`. One original hash links, in
  `seq` order, to one or more replacement hashes; a removed block is a single row with
  a NULL `replacement_hash`. Both ends are rows of `message_block_lookup`.

`reconstruct_llm_messages(connection, call_label, kind)` rebuilds one request or reply
exactly; `reconstruct_llm_patches(connection, call_label)` reads the patches back.
`reconstruct_llm_exchanges` and the mock replay backend read the flat block chains
(`exchange_chain`), which `record_llm_exchange` derives from the blocks of the prompt's
and the reply's messages, in order.

`LlmUsageTracker` matches branches on the harness's own chat messages, before any patch
and without the `tool_schema` message (the tools of one conversation can change).

## Limits

* A patch matches the block exactly as the harness sends it. If the harness rewrites
  it (compaction, a changed tool result) the patch matches nothing, silently.
* A block that arrived as the model's own reply is stored in its richer response shape
  in the skill run that produced it, so a patch keyed by the harness's flattened resend
  only starts matching the history from the next skill run.
* A patch invalidates the provider's prompt cache once; after that the prompt is stable.
* `patch_history` does not validate the result against the provider's rules.

## Tests

```sh
pytest tests/llm/test_message_store.py tests/engine/host_servers_tests/test_llm_history_tables.py \
       tests/test_agent_history_tables.py tests/test_agent_live_patch.py \
       tests/harness/test_pause_ack.py
pytest tests/harness/test_agproxy_ptrace.py -k "pause or ack"   # ptrace tests need ptrace
```

`tests/test_agent_live_patch.py` includes a test that sends a request through a real
`HostServerManager` over the host UDS, and tests that patch an agent while its skill run
is active.
