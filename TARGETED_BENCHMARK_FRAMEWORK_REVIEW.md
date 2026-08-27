# Targeted benchmark support: Agency review handoff

## Scope

This covers the remaining unstaged changes in `/home/jerry/agency`. The patch now contains only generic framework fixes or features used by the targeted benchmarks. Dataset workflows, benchmark tool lists, replay schemas, and run layouts remain in `agency-applications`.

## 1. Existing framework bugs

These changes are required for the affected benchmark conditions to run correctly.

| Change | Existing failure | Files |
| --- | --- | --- |
| Structured output prompt | The prompt named retired `return_<field>` tools, while the engine exposes `submit_output(field, value)`. Structured workflows could not reliably finish. | `agency/agskill.py` |
| Harness control propagation | `suppress_builtin_tools` existed in the protocol but was dropped at the sandbox server boundary. A requested replacement tool surface was therefore ignored. | `agency/harness/servers/sandbox_interaction_server.py` |
| MCP policy-name normalization | External harnesses report names such as `mcp__agency__read`, while host policy is defined against `read`. Valid Claude Code tool calls were denied. | `agency/harness/_harness_permission_hook.py` |
| Codex provider and MCP configuration | Codex was launched while ignoring the isolated config that defines Agency's model proxy and MCP server. It could not use the intended model/tool path. | `agency/harness/adapters/codex.py` |
| Responses namespace translation | Codex sends MCP tools as Responses API namespaces. The chat-completions bridge discarded that structure and could not route returned calls back to MCP. | `agency/harness/agproxy_llm_adapters.py`, `llm_router.py` |
| Per-user runtime root | A shared `/tmp/agency` can be owned by another user, causing socket creation to fail. `AGENCY_RUNTIME_ROOT` provides a short absolute per-user root. | `agency/agutil.py` |
| Profiler re-exec environment | systemd profiler re-execution replaced `PATH`, so external harness binaries could disappear during profiled runs. | `agency/profiler/agprof.py` |

## 2. Benchmark-control features

The benchmark can start without these changes, but its results are not controlled or comparable.

| Feature | Why the benchmark needs it | Files |
| --- | --- | --- |
| External-harness LLM budget | `max_steps` now limits real model requests made inside Claude Code or Codex loops, rather than only the outer Agency attempt. | `agency/harness/daemon.py`, `llm_router.py` |
| Closed Codex execution surface | Web search, shell, file mutation, subagents, and unmetered execution are disabled; unexpected built-in tool events invalidate the attempt. | `agency/harness/adapters/codex.py` |

Agency does not hard-code a benchmark tool allowlist. Codex receives the tools exposed by the current Agency MCP server, and host policy remains authoritative.

## 3. Fine-grained results and replay features

These are framework feature requests rather than execution fixes.

| Feature | Result enabled | Files |
| --- | --- | --- |
| Public raw LLM transcripts | Callers can audit full requests/responses and build deterministic replay tapes. The returned data is deep-copied. | `agency/engine/engine.py`, `host_server_manager.py`, `llm_handler_server.py` |
| Codex token accounting | Completed-turn usage is returned through the common harness result. | `agency/harness/adapters/codex.py` |

Agency exposes raw transcripts only. The targeted benchmark owns `schema_version`, round numbering, derived tool names, and `llm_tape.jsonl`.

## Profiler artifact boundary

The Perfetto artifact is benchmark-owned:

```text
<condition>/profile/agprof.trace.json
```

`agency-applications` sets `AGENCY_PROFILE_DIR` to that directory for `--profile` runs. No `ui_events.db` change is needed: that database is agwebui's event store, not the Agency profiler trace.

## Intentionally removed

- The bug-localization-specific Codex tool allowlist.
- Benchmark replay-record formatting from the Agency engine.
- Formatting-only and unrelated comment rewrites.

## Suggested review units

1. Structured-output and harness-boundary bug fixes.
2. Codex/Responses/MCP compatibility.
3. Runtime-root and profiler re-exec fixes.
4. External-harness request budgeting.
5. Transcript and token-usage observability.

## Validation

- 224 focused Agency tests pass.
- Agency Ruff lint and format checks pass.
- 41 targeted benchmark tests pass after moving replay formatting to `agency-applications`.
