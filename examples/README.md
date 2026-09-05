# Examples

Simple feature examples showing how to use the agency framework.

## LLM configuration

For vLLM endpoint, set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODE` enviroment variables, and launch the example scripts.

```bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_MODEL="YOUR_SERVED_MODEL"
export LLM_API_KEY="YOUR_API_KEY" # Empty ("") if unused
python examples/base_example.py
```

Please refer to the main README.md at project root for other LLM APIs, such as OpenAI or Anthropic APIs.

---

## redirect_claude_code_luna.py

**What it shows:** The smallest deterministic `Invocation.redirect()` test using
the Claude Code harness with OpenAI's `gpt-5.6-luna` model. The agent is suspended
before submission so the redirect is queued before the first model request, then
resumed to produce the redirected answer (`blue` instead of `red`).

Requires `OPENAI_API_KEY` and an installed, authenticated `claude` CLI.

```bash
uv run python examples/redirect_claude_code_luna.py
```

---

## base_example.py

**What it shows:** The simplest complete agent — one agent, two skills, shared history.

- `file_manager` skill writes a file to `/workspace/note.txt` inside the sandbox container using the `write` and `read` tools, then confirms the content.
- `qa` skill answers a follow-up question using the conversation history accumulated from the first skill run, demonstrating that history is shared across skill runs on the same agent.

```bash
python examples/base_example.py
```

---

## parallel_exec.py

**What it shows:** The two natural parallelism patterns the framework enables.

**Pattern 1 — Sequential chain:** Two `ag.run()` calls on the same agent. The second call automatically waits for the first because they chain through the history future. The agent sees both turns in order.

**Pattern 2 — Fork fan-out:** `agent.fork(parent)` deep-copies the context and copies the parent's checkpoint image; each fork's `run()` fires immediately and returns an `Invocation`. All three forks run concurrently in separate containers. Accessing `.summary` on each invocation blocks until that fork is done.

```bash
python examples/parallel_exec.py
```

---

## custom_tools.py

**What it shows:** A multi-step, multi-agent research pipeline combining a custom host-side tool, parallel summarisation forks, and the shared output directory.

1. **`find_papers`** — calls a custom `search_papers` tool (HTTP request to the arXiv API, runs on the host) and returns a list of papers.
2. **Parallel summarisation** — one `agent.fork(main_agent)` per paper; all `run(summarise_paper, ...)` calls fire concurrently. Each fork runs in its own sandbox container.
3. **`compile_report`** — waits for all pending summaries (resolved automatically when passed as input), then uses the sandboxed `write` tool to save a markdown report to `/agent_output/<agname>/report.md`.

The report appears on the host at `runs/<timestamp>_custom_tools/agent_output/<agname>/report.md`.

```bash
python examples/custom_tools.py
python examples/custom_tools.py "speculative decoding"
MAX_PAPERS=6 python examples/custom_tools.py "flash attention"
```

---

## image_processing.py

**What it shows:** `agimage` — the multimodal image input field type — across three input forms, run concurrently as separate teams.

1. **`SingleImageTeam`** — describes one local image file (`agdata(photo=agimage)`); the path is base64-encoded and injected into the message content automatically.
2. **`MultiImageTeam`** — compares two local images side by side via `agdata(frames=list[agimage])`.
3. **`UrlImageTeam`** — analyses an image passed as a public URL; no local encoding needed.

Requires a vision-capable model.

```bash
LLM_MODEL=google/gemma-4-E2B-it python examples/image_processing.py photo.jpg
LLM_MODEL=google/gemma-4-E2B-it python examples/image_processing.py before.jpg after.jpg
```

---

## sandbox_handoff.py

**What it shows:** Reading and driving an agent's `agSandbox` directly from the host, and handing one sandbox off between two agents — the sandbox is a plain `agent.sandbox` attribute, not something you have to go through a skill to touch.

1. `agent_a` runs a skill that writes `hello.py` inside its sandbox.
2. The harness reads `agent_a.sandbox` directly and calls `sandbox.exec(...)` to run the file from Python, outside of any skill.
3. The harness patches the file with `sed` via the same `sandbox.exec(...)`, introducing a syntax error.
4. `agent_b` is pointed at the same sandbox (`agent_b.sandbox = sandbox`) and runs a skill that fixes the bug.
5. The harness runs the file again to confirm the fix — `agskill` stops+commits the container after `agent_b`'s skill the same way it would for a sandbox it provisioned itself, and the harness's next `sandbox.exec()` call transparently restarts the container from that checkpoint.

```bash
python examples/sandbox_handoff.py
```

---

## dynamic_config_example.py

**What it shows:** Building one flat `agconfig` covering both LLM and sandbox fields in a single call, and updating a field on a fresh `agconfig` between two skill calls on the same agent via `ag.change_config()` — no sandbox teardown, no new agent.

1. `agconfig(provider="vllm", ..., max_completion_tokens=32)` plus `cfg.add_mount("data", ...)` builds one flat config (including a deliberately too-small `max_completion_tokens=32`) with an `agSandbox` "data" mount.
2. **Call 1** runs `write_note` with the tiny token budget; the vLLM server truncates the tool-call JSON mid-argument, so the skill can't complete within a few ReAct steps and the run fails as expected.
3. A new `agconfig(..., max_completion_tokens=4096)` is built and pushed via `ag.change_config(new_cfg)` — read fresh on every LLM call, so the bump takes effect on the very next call with no clone/teardown needed.
4. **Call 2** runs the identical skill again; with the higher budget it completes and the note is written and confirmed.

```bash
python examples/dynamic_config_example.py
```
