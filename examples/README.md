# Examples

End-to-end examples showing how to use the agency framework.

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

## paper_crawler.py

**What it shows:** A multi-step, multi-agent research pipeline combining a custom host-side tool, parallel summarisation forks, and the shared output directory.

1. **`find_papers`** — calls a custom `search_arxiv` tool (HTTP request to the arXiv API, runs on the host) and returns a list of papers.
2. **Parallel summarisation** — one `agent(main_agent)` fork per paper; all `run(summarise_paper, ...)` calls fire concurrently. Each fork runs in its own sandbox container.
3. **`compile_report`** — waits for all pending summaries (resolved automatically when passed as input), then uses the sandboxed `write` tool to save a markdown report to `/agent_output/<agname>/report.md`.

The report appears on the host at `runs/<timestamp>_paper_crawler/agent_output/<agname>/report.md`.

```bash
python examples/paper_crawler.py
python examples/paper_crawler.py "speculative decoding"
MAX_PAPERS=6 python examples/paper_crawler.py "flash attention"
```

---

## openai-agent-examples/

Ports of the [openai/openai-agents-python](https://github.com/openai/openai-agents-python/tree/main/examples) examples, demonstrating feature parity and framework gaps. See [`openai-agent-examples/README.md`](openai-agent-examples/README.md) for the full concept mapping and gap analysis.

| Subdir | Contents |
|---|---|
| [`basic/`](openai-agent-examples/basic/) | `hello_world.py`, `tools.py` — minimal agent and function tools |
| [`agent_patterns/`](openai-agent-examples/agent_patterns/) | `agents_as_tools`, `routing`, `parallelization`, `deterministic`, `llm_as_a_judge`, `input_guardrails`, `output_guardrails` |
