# Examples

Simple feature examples showing how to use the agency framework.

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

**Pattern 2 — Fork fan-out:** `agent(parent)` deep-copies the history and copies the parent's checkpoint image via `docker tag`; each fork's `run()` fires immediately and returns a pending `agdata`. All three forks run concurrently in separate containers. Accessing `.summary` on each result blocks until that fork is done.

```bash
python examples/parallel_exec.py
```

---

## custom_tools.py

**What it shows:** A multi-step, multi-agent research pipeline combining a custom host-side tool, parallel summarisation forks, and the shared output directory.

1. **`find_papers`** — calls a custom `search_papers` tool (HTTP request to the arXiv API, runs on the host) and returns a list of papers.
2. **Parallel summarisation** — one `agent(main_agent)` fork per paper; all `run(summarise_paper, ...)` calls fire concurrently. Each fork runs in its own sandbox container.
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
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct python examples/image_processing.py photo.jpg
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct python examples/image_processing.py before.jpg after.jpg
```

---

## human_in_the_loop.py

**What it shows:** Driving approval loops entirely from Python so `ask_human` is *guaranteed* to be called — the LLM never decides on its own whether to stop and ask.

1. Python asks the human what scene to write next (`ask_human`, no timeout).
2. A planner skill drafts a paragraph-by-paragraph scene plan.
3. Python shows the plan and asks for approval; on rejection it loops back into the planner with the human's feedback until approved.
4. A writer skill generates the full scene prose from the approved plan.
5. Python shows the prose and asks for approval; on rejection it loops (re-plan → re-write) until approved.
6. Approved output is appended to `plans.md` / `story.txt` in the run directory, and the loop advances to the next scene.

```bash
python examples/human_in_the_loop.py
VLLM_BASE_URL=http://... VLLM_MODEL=... python examples/human_in_the_loop.py
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
