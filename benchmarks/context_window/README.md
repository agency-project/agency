# Context-window benchmark

`run.py` checks, against a real model, where a harness compacts. The agent
follows a chain of files whose first and last lines name the next one, so its
context grows by one file per call; Agency's log of every LLM call then shows
the prompt size per call and where it dropped. `--limit` sets `context_limit`,
which every adapter configures its harness with (docs/Context_Window.md), so
thresholds are reachable without filling a 1M window.

```sh
sudo bash -c 'set -a; . ./.env; set +a; .venv/bin/python benchmarks/context_window/run.py \
  --harness opencode --model claude-haiku-5-5 --provider anthropic --limit 120000 \
  --files 18 --kb 12 --out runs/context_window/results.jsonl'
```

Each run appends one line to `--out`: `compacted_at_share` is the largest prompt
before the first compaction as a share of the limit (about 0.9 expected; 0.85
for Grok; Codex caps the window at 272K), and `max_prompt_share` above 1 means
the harness let the prompt outgrow the limit. It needs a Linux host set up with
`agency setup-host` and the harness image from
`tests/host/fixtures/Dockerfile.harnesses`. `.env` needs `ANTHROPIC_API_KEY`
for `--provider anthropic`, or `LITELLM_API_KEY` and `LITELLM_BASE_URL`.

Codex keeps at most ~10KB of a command's output, so its context grows more
slowly per file; give it more `--files`.

## Results

Runs from 2026-10-16 on an m7i.4xlarge with the pinned harnesses, Claude Haiku
5.5 (Anthropic API, and through LiteLLM), GPT-6 Luna and Qwen3.8-27B (LiteLLM).
"Compacts at" is the largest prompt before the first compaction, as a share of
`--limit`; "never" means the prompt outgrew the limit (peak in brackets).
Before is master at cc8ab7a.

| Harness | Model (via) | Limit | Before | After |
|---|---|---|---|---|
| native | Haiku 5.5 (Anthropic) | 120K | never (154%) | 83% |
| native | GPT-6 Luna | 120K | never (104%) | 86% |
| Claude Code | Haiku 5.5 (Anthropic) | 400K | 43% | 88% |
| Claude Code | GPT-6 Luna | 120K | 60% | 87% |
| Codex | Haiku 5.5 (Anthropic) | 120K | never (182%) | 90% |
| Codex | GPT-6 Luna | 300K | 82% | 92% |
| Grok | Haiku 5.5 (Anthropic) | 120K | never (155%) | 93% |
| OpenCode | Haiku 5.5 (Anthropic) | 120K | never (158%) | 92% |
| OpenCode | GPT-6 Luna | 120K | 74% | 91% |
| Kimi | Haiku 5.5 (Anthropic) | 120K | never (173%) | 92% |
| Kimi | GPT-6 Luna | 120K | 66% | 94% |

After the fix, every harness compacted at 83-94% with every model tried (Haiku
5.5 both ways, GPT-6 Luna, Qwen3.8-27B; 27 combinations). Prompts past 90% are
one file (~10K tokens) over a threshold a harness checks before each call.

Claude Code 2.1.251 cannot use Claude 5.5 models through a Bedrock-backed
LiteLLM ("version 2.1.280 or newer is required"); the harness image now pins
2.1.296. With it, Claude Code + Haiku 5.5 through LiteLLM:

| Limit | Before | After |
|---|---|---|
| 120K | failed at 62% | 88% (peak 96%) |
| 400K | 75% | 92% |

Repeated after the fix (two runs each, Haiku 5.5 through the Anthropic API and
GPT-6 Luna, 120K): Claude Code 88-89% and 92%, Codex 89-90%, Grok 93%, Kimi
92-94%, native 83-87%, OpenCode 91-92% - the same points as the first runs.

Steps larger than the margin (`--limit 60000 --files 8 --kb 30`: ~23K tokens a
file, Haiku 5.5 through the Anthropic API). Before, each harness compacted a
step too late or not at all and sent prompts several times the limit; after,
Agency's context fit (docs/Context_Window.md) clears old tool output from any
request over 95% of the limit. Two runs each after:

| Harness | Before: peak | After: peaks | Requests fitted |
|---|---|---|---|
| native | 245% | 85%, 53% | 0, 11 |
| Claude Code | 367% | 85%, 85% | 7, 6 |
| Grok | 340% | 71%, 70% | 14, 12 |
| OpenCode | 346% | 64%, 69% | 8, 22 |
| Kimi | 376% | 95%, 98% | 27, 35 |

OpenCode's second compaction, which re-sends the whole session, peaked at 159%
of a 120K limit before the context fit and 92% after.

A backend that rejects a request for its size: Qwen3.8-27B (262K window) with
`--limit 500000`, so that the estimate lets requests past the real window.
Before, Grok ended the turn on the first rejection. After, Agency cut and
resent each rejected request: Grok recovered 38 times and finished all 24
files; Kimi 5 times and OpenCode 4. With `--max-completion-tokens 32000` (one
earlier run ended on a reply cut off at 8,192), two more Grok runs recovered
16 times each and finished.

OpenCode returned an empty answer whenever it had compacted: Agency's plugin
looked for the final reply under the turn's prompt, and OpenCode continues a
compacted turn under a new one. Fixed, OpenCode returns all 18 codes after
compacting at 91-94% (Haiku 5.5, GPT-6 Luna, Qwen3.8-27B).

On master, Claude Code with Haiku 5.5 (Anthropic API, 120K) compacts at 68%:
Agency's 90% less Claude Code's own 33K reserve.
