# Harness engine glue (`harness/agharness.py`, `harness/agharness_backends/`)

The `engine` seam on `agent` (see [agent.md](agent.md)) lets `agskill.run()` dispatch to an
off-the-shelf coding-agent CLI instead of the native ReAct loop. `harness/agharness.py` holds what's
genuinely shared across every concrete backend; `harness/agharness_backends/` holds one file per harness,
selected via `agharness_backend.for_config(engine, agconfig)` — the exact same shape as
`llm`/`sandbox` (see [agconfig.md](agconfig.md)).

## The engine seam

```python
ag = agent(agconfig=cfg, engine="claude_code")   # "native" (default), "opencode", "claude_code", "codex", "grok"
result = ag.run(my_skill, agdata(task=...))       # completely unchanged call site
```

`agskill.run()`'s `_task()` branches on `ag.engine`: `"native"` calls `execute_react()` (untouched);
anything else calls `agskill.execute_harness()`, which validates input the same way `execute_react`
does, then dispatches to `agharness_backend.for_config(ag.engine, ag.agconfig).execute(...)`. Both
paths return the same `(result, ctx, delta)` contract — `ctx` is the *same* `prev_ctx` object,
mutated in place; `delta` is `[system_prompt_message] + every message appended this call`.

`agent.engine` round-trips through `fork()`, `save()`/`load()` (defaults to `"native"` if a
checkpoint predates this field) exactly like `agent.llm` does — see `agent.py`'s `__init__`/`fork`/
`save`/`load` for the four exact insertion points.

## What a concrete backend's `execute()` does (same shape in all four)

1. Resolve the harness binary (`shutil.which`); missing binary → `agerror`, no launch attempted.
2. Build the prompt via `agharness.build_user_turn_prompt(skill, skill_input)` (delegates to
   `agskill._build_user_content` — the *exact* JSON-input convention the native loop's first user
   message uses) plus, for a structured `output_schema`, a plain-text instruction from
   `agharness.build_output_format_instruction(skill)` — never injected as the harness's own system
   prompt or as a tool (see [Design_harness_integration.md](Design_harness_integration.md)).
3. Materialize an isolated config home (`agharness.materialize_config_home`) so concurrent agents
   never share a harness's own config/credentials directory.
4. Launch via `agProxyPtrace(ag.agconfig).launch(argv, envp, cwd=..., policy=agharness.default_policy(ag), ag=ag)`
   — real syscall-level tracing (see [agproxy_ptrace.md](harness/agproxy_ptrace.md)), not a plain
   `subprocess.run`. `agharness.default_policy(ag)` allows everything but logs every intercepted
   syscall through `ag.log`, so a harness-driven agent's execution is observable in the
   webui/logs exactly like a native one's, even with no real security policy wired up yet
   (see [agpolicy.md](agpolicy.md) for the eventual retrofit).
5. `agproxy_ptrace.wire_to_sandbox(handle, ag.sandbox)` if a sandbox is attached, so
   `ag.sandbox.get_live_pids()`/`.wait_for_processes()` reflect the harness's process tree.
6. `handle.wait(timeout=...)` — blocks for the harness to finish; non-zero exit → `agerror`.
7. Parse the harness's own final-answer text out of its headless output (backend-specific: a
   single JSON `"result"`/`"text"` field for Claude Code/Grok Build, best-effort NDJSON scanning
   for opencode/Codex).
8. Recover output via `skill.output_schema.validate_and_recover(text, ag.sandbox)`
   (see [agschema.md](agschema.md)) for a structured schema, or wrap the raw text in `agdata` for a
   raw/no-schema skill — mirrors `execute_react()`'s own raw-text fallback path.
9. Mutate `prev_ctx.messages`/token totals in place; return `(result, prev_ctx, delta)`.

## Per-backend status

| Backend | LLM routing | Tested against |
|---|---|---|
| `opencode.py` | `agproxy_llm` `/v1/chat/completions` passthrough (matches wire format) | Mocked only — no `opencode` binary installable without Node/Bun in the environment this was built in |
| `claude_code.py` | `agproxy_llm` `/v1/messages` translate (Anthropic Messages API <-> chat-completions, see [agproxy_llm.md](harness/agproxy_llm.md)) — `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` point at the gateway; the host's own real credentials (API key, OAuth login, Bedrock env) are never forwarded | **Real CLI** (v2.1.212) — raw-text and structured-`output_schema` paths, both routed genuinely through the gateway to a real backend (Bedrock), verified end-to-end |
| `codex.py` | `agproxy_llm` `/v1/responses` translate (OpenAI Responses API <-> chat-completions) — mandatory here since Codex dropped `wire_api="chat"` upstream; `CODEX_HOME/config.toml` gets a `[model_providers.agency-proxy]` block pointing `base_url` at the gateway | Mocked only — no `codex` binary available; the Responses-API adapter itself is unverified against a live run |
| `grok.py` | `agproxy_llm` `/v1/chat/completions` passthrough — Grok Build's `[model.*]` config supports `api_backend = "chat_completions"` per xAI's published docs, matching `agproxy_llm`'s existing route with zero translation, same as opencode | Mocked only — no `grok` binary installed (installing it means running xAI's `curl \| bash` script, deliberately not done without being asked first) |

Every backend now genuinely routes its LLM traffic through `agproxy_llm` rather than leaving any
harness free to use its own host credentials/endpoint — two are exact wire-format matches
(`gateway_mode="passthrough"`), two require reshaping (`gateway_mode="translate"`, implemented in
`harness/agproxy_llm_adapters.py`). See that module's docstring for the translation
fidelity cost (extended thinking, prompt-cache breakpoints, and image content blocks have no
chat-completions equivalent and are dropped, not errored on).

`grok.py` is the second backend (after opencode) that actually routes its LLM traffic through
`agproxy_llm` rather than leaving the harness's own endpoint untouched — it writes a
`config.toml` under its isolated `GROK_HOME` with a `[model.agency-proxy]` block pointing
`base_url`/`api_key` at the gateway. Config isolation uses `GROK_HOME` (redirects the entire
config directory: `config.toml`, `auth.json`, `sessions/`) rather than a `--setting-sources`/
`--ignore-user-config`-style flag — xAI's docs don't expose one, so full directory redirection
(closer to Codex's `CODEX_HOME` than Claude Code's flag-based approach) is the documented
isolation mechanism. `sessionId` from a successful run is stashed on `session_resume_id`
(an existing `AgHarnessFields` field) but not yet threaded through to a resumed second call —
multi-turn resume is future work, not wired up.

## A real bug worth knowing about: don't override `HOME`

`_ClaudeCodeBackend` initially set `envp["HOME"]` to the isolated config-home directory, on the
assumption that mirrors opencode's `OPENCODE_CONFIG` file-based isolation. This broke Claude Code's
own OAuth login (`~/.claude/.credentials.json` lives under the real `$HOME`), forcing
"Not logged in" on every run. Fixed by leaving `HOME` untouched and relying on
`--setting-sources ""` for the actual "don't inherit the caller's CLAUDE.md/settings" isolation —
see `claude_code.py`'s `execute()` for the exact reasoning, and
`test_execute_does_not_override_home` in `tests/harness/agharness_backends/test_claude_code.py` for the
regression test.
