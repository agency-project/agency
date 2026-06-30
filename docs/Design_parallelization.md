# Parallelization

Agency runs multiple agents concurrently by mixing Python threads, OS processes, and cooperative I/O scheduling. This document explains which objects are parallelized, against what, using which mechanism, and where the current limits lie.

## The GIL problem

CPython's Global Interpreter Lock (GIL) means only one thread executes Python bytecode at any moment. However, the GIL is **released during I/O waits** — blocking syscalls like `recv()` (HTTP), `read()`, `write()`, `subprocess.wait()` — so thread-based concurrency is effective whenever threads spend most of their time waiting on I/O.

LLM calls are almost pure I/O: a thread sends a request and then blocks on SSE streaming for seconds to minutes. Tool calls that involve subprocess execution, file I/O, or HTTP are similarly I/O-bound. This makes threads the right primitive for agent and skill concurrency.

Where CPU is genuinely needed — executing a Python tool function synchronously — the GIL becomes a bottleneck. We solve that with process offloading (see [Tools](#tools)).

## Parallelization matrix

| Object | Parallelized against | Mechanism | Where defined |
|---|---|---|---|
| `agteam` tasks | Each other | One daemon thread per `run()` call | `agteam._wrap_run` |
| `agent` runs | Each other (forked agents) | One daemon thread per `run()` call | `agskill.run()` |
| `agskill` steps | Other agents' steps | Same thread as the owning agent task | `agskill.execute_react()` (synchronous ReAct loop) |
| `agtool` calls | Other threads / agents | `ProcessPoolExecutor(max_workers=256)` | `agtool._pool` (module-level) |
| LLM SSE stream | Other agent threads | Background drain thread + batch queue | `agskill._iter_batched()` |

## agteam — team-level parallelism

`agteam` is the entry point for multi-agent coordination. Its `run()` method implements the workflow. Each call to `run()` spawns a single daemon thread; inside that thread each step calls `agent.run(skill, input)` on whatever agents the subclass wires up.

```python
class ResearchTeam(agteam):
    def setup(self):
        self.find_papers = agskill(name="find_papers", ...)
        self.summarise   = agskill(name="summarise", ...)
        self.agent       = agent()

    def run(self) -> agdata:
        papers = self.agent.run(self.find_papers, agdata(topic=self.topic)).papers
        return self.agent.run(self.summarise, agdata(papers=papers))
```

Using one thread per team rather than a shared pool means recursive team spawning (a team creating child teams inside its own `run()`) is safe by construction — there is no finite pool of slots to exhaust.

## agent — per-agent task serialization

Each `agent` instance serializes its own runs: `agskill.run()` (the scheduling wrapper) spawns a daemon thread that first waits for the previous task on this agent to complete (`prev_ctx.resolve_prev_dependencies()`). This ensures context is updated in order even when multiple callers call `run()` on the same agent concurrently.

**Fork for parallelism.** When you need several independent runs from the same starting state, fork the agent:

```python
results = [agent.fork(ag).run(summarise_skill, agdata(text=t)) for t in texts]
# All three forks run concurrently, each in its own thread
```

Each `agent.fork(ag)` deep-copies the history at that instant and starts its work in an independent thread. The parent agent's history is never touched.

## agskill — intra-agent ReAct steps

Within one agent, the ReAct loop is sequential: LLM call → tool call(s) → next LLM call. Steps within a single skill do not run in parallel because each step's output is the input of the next.

Across agents, skills run concurrently because each agent owns its own thread.

## agtool — process offloading

Tool functions execute in a separate OS process via a module-level `ProcessPoolExecutor(max_workers=256)`. The pool is created lazily and reuses workers across calls.

**Why a process?** Even though LLM streaming is I/O-bound, a CPU-intensive tool running in the same process holds the GIL for its entire duration, blocking all other agent threads from making forward progress. A subprocess has its own GIL — the worker runs at full CPU while the parent threads stay unblocked.

**Transport.** Tool functions are serialised with `cloudpickle` (handles bound methods and closures) and deserialised in the worker. Arguments and results travel as pickled bytes. The round-trip cost is typically 1–5 ms for small payloads; for tools that do significant work the overhead is negligible.

**Fork context.** The pool uses the default `fork` start method. Workers are short-lived and the fork happens before any heavy multithreading, so lock-inheritance issues are avoided in practice. `fork` avoids the ~100–300 ms interpreter-startup cost of `spawn` and allows workers to reuse already-imported modules.

**Logger exclusion.** `agtool.__getstate__` strips `_term` and `_aglog` before serialisation — these hold threading locks and open file handles that cannot cross process boundaries. They are re-attached via `attach_logger` on the main-process side; workers do not log.

## LLM streaming — stream batching

`agskill.execute_react()` calls the LLM with `stream=True`. Without batching, every SSE token chunk triggers a GIL acquire/release cycle (the background SSE reader thread calls `queue.put`, the main thread calls `queue.get`), creating O(tokens) context switches per LLM call.

`_iter_batched(stream)` reduces this to O(tokens / batch_size):

1. A daemon thread drains the SSE iterator and puts each chunk into a `queue.SimpleQueue` (one GIL acquire per chunk — minimal).
2. The calling thread blocks on `q.get()` for the first item, then **sleeps for 100 ms** (`_BATCH_INTERVAL_S`). During this sleep the GIL is fully released — other agent threads run unimpeded.
3. On wake, the calling thread drains everything buffered in the queue in a tight loop (one batch burst, not one chunk at a time).

The result: at 60 tokens/s a 100 ms window buffers ~6 tokens per batch, reducing GIL acquisitions by ~6×. At higher token rates or longer sleep intervals the benefit scales proportionally.

## Limitations

### Speed

| Bottleneck | Current approach | Remaining gap |
|---|---|---|
| GIL during tool execution | Process offload via `ProcessPoolExecutor` | First call pays fork overhead (~5–20 ms); subsequent calls reuse workers |
| GIL during LLM streaming | `_iter_batched` with 100 ms sleep | Adds 100 ms latency to each LLM response (final token → result). Acceptable for interactive use; tunable via `_BATCH_INTERVAL_S` |
| Process pool serialisation | `cloudpickle` bytes round-trip | ~1–5 ms overhead per tool call for small payloads |

### Scaling

- **Agent / team threads:** Each `agskill.run()` (invoked by `agent.run()`) and each `agteam.run()` spawns one daemon thread. Thread creation costs ~100 µs — negligible compared to any LLM call. OS thread limits (typically 10 000+) are the only ceiling; in practice memory is the binding constraint (~1–8 MB stack per thread).
- **Process pool (256 workers):** Workers are OS processes created lazily on first call using `fork`. Fork overhead is ~5–20 ms. At extreme concurrency (hundreds of simultaneous tool calls), new workers are created on demand; subsequent calls reuse the warm pool.
- **Memory:** Each process worker loads the full Python interpreter and all imported modules (~30–50 MB RSS typical). 256 workers = up to ~10 GB RSS if all are active. In practice, workers are created on demand and the OS reclaims pages from idle workers.

### Resources

- The process pool (`agtool._pool`) is module-level and shared across all agents. There is no per-agent tool concurrency limit — a single agent can saturate all 256 worker slots.
- No backpressure mechanism exists: if more than 256 tool calls are submitted simultaneously, they queue in the `ProcessPoolExecutor` internal queue (unbounded).

### UI/UX

- **Live token streaming** is throttled to once per 100 characters of new content (or per batch flush) to avoid repainting the terminal on every token. Sub-100-char increments are buffered.
- **Tool call state** is displayed as `"tool"` in the agent status panel for the duration of the tool call. Since tools run in a subprocess, the UI shows the agent as busy but cannot introspect the worker's progress.
- **Sandbox tool calls** (bash, read, write, etc.) run inside the container via `podman/docker exec`. These are I/O-bound (subprocess + network), so they hold the calling thread but release the GIL — the process pool is not used for sandbox tools, which run inline.
- **Process pool is unavailable in tests.** `conftest.py` sets `_use_process_pool = False` so that `unittest.mock.patch` patches propagate into tool functions (impossible across process boundaries). Tests run tools in the same process.

## Summary diagram

```
agteam.run() ──► daemon thread
                  ├─ agent A.run() [delegator]
                  │       └─► agskill.run() ──► daemon thread
                  │                                 └─► agskill.execute_react() ──► LLM (I/O, GIL released)
                  │                                                               └─► agtool.__call__()
                  │                                                                    └─► ProcessPoolExecutor (256 procs)
                  │                                                                         └─► worker: fn(arg) [own GIL]
                  ├─ agent B.run() [delegator]
                  │       └─► agskill.run() ──► daemon thread (concurrent with A)
                  │                                 └─► agskill.execute_react() ──► LLM (I/O, GIL released)
                  └─ agent C.run() [delegator]
                          └─► agskill.run() ──► daemon thread ──► ...
```

LLM calls from agents A, B, C all block on I/O simultaneously — GIL is irrelevant there. Tool calls from any agent are offloaded to the process pool and run in parallel with each other and with all LLM calls.
