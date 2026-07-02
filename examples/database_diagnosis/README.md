# Database Diagnosis

This example uses `agency` to diagnose SQLite index problems, iteratively evaluate candidate SQL fixes, and write a concise markdown report.

## Run

```bash
uv run python examples/database_diagnosis/main.py
uv run python examples/database_diagnosis/main.py --database-path data/european_soccer/database.sqlite --webui
uv run python examples/database_diagnosis/main.py --database-path runs/my_database.sqlite --write
```

Useful options:

- `--database-path`: diagnose an existing SQLite database.
- `--write`: commit the selected fix. Without it, candidate writes are tested and rolled back.
- `--webui`: show the live `agency` run graph and agent traces.
- `--rows` and `--seed`: control the generated demo database when no path is supplied.

The run writes `database_diagnosis_report.md` to `runs/<timestamp>_database_diagnosis/`.

The first command is self-contained and creates a default demo database under the run directory. The `data/european_soccer/database.sqlite` path is only available after downloading the optional Kaggle dataset shown below.

The reviewer/evaluator loop is bounded by `DB_BOT_MAX_FIX_ATTEMPTS` (default `3`). A candidate is accepted when its measured median benchmark speedup is at least `DB_BOT_MIN_FIX_SPEEDUP` (default `1.05`). If a candidate does not clear that threshold, the evaluator feedback is fed back to the reviewer so it can revise the hypothesis, choose another candidate index, or decline to recommend a fix.

## Scope and Limitations

This is an advanced `agency` example, not a production database tuning tool. The workload is inferred from schema metadata, foreign keys, column names, and generated probe queries rather than real query logs. 

Benchmarks are lightweight and are intended to validate a small number of candidate indexes for the demo path.

The goal is to demonstrate host side tools, rollback safe candidate evaluation, parallel specialist agents, and report generation.

## How Agency Is Used

`DatabaseDiagnosisTeam` is an `agteam` that wires a coordinator agent, specialist skills, host-side SQLite tools, a bounded reviewer/evaluator loop, and a report writer.

- `SchemaExplorerSkill`: inspects tables, indexes, and foreign-key shape.
- `QueryPlanSkill`: studies generated query plans and benchmark signals.
- `IndexAdvisorSkill`: proposes a concrete index candidate.
- `ReviewerSkill`: compares specialist hypotheses and chooses or revises the safest SQL fix.
- `FixEvaluatorSkill`: reviews before/after plan and benchmark evidence from the candidate index evaluation, then decides whether the recommendation is supported.
- `ReportSkill`: turns the evidence and decisions into the final markdown report.

The three specialist skills run in parallel by forking from the coordinator with `agent.fork(self.coordinator)`. Each fork receives shared preanalysis evidence but can still call SQL tools for verification. The reviewer and evaluator then run in a bounded loop: after each candidate index is tested, the measured result is fed back into the next reviewer pass if the benchmark threshold is not met. The report writer runs after the loop and includes the attempted fixes.

## Design Decisions

- One path: generated demo databases and user-supplied databases use the same team flow.
- Deterministic first pass: `preanalysis.py` gathers schema metadata, candidate query plans, and small benchmarks before the LLM agents reason over the evidence.
- Host-side tools: SQLite access is exposed as local `agtool` calls because the target database is a local file.
- Rollback-first safety: read-only mode still lets agents test DDL/DML, but `sqlite_access.py` wraps every tool call in a transaction that rolls back unless `--write` is enabled.
- Smallest useful fix: the reviewer is prompted to prefer a targeted SQL fix over broad indexing unless evidence supports more.
- Evidence-backed retry: failed or weak candidate evaluations are summarized in `attempt_history` so the reviewer can avoid repeating the same index.

## Endpoint Configuration

The example uses an OpenAI-compatible chat completions endpoint with tool calling.

```bash
VLLM_BASE_URL=http://127.0.0.1:18000/v1 \
VLLM_API_KEY=EMPTY \
VLLM_MODEL=Qwen/Qwen3.5-4B \
uv run python examples/database_diagnosis/main.py
```

For the OpenAI API:

```bash
VLLM_BASE_URL=https://api.openai.com/v1 \
OPENAI_API_KEY=... \
OPENAI_MODEL=gpt-5.5 \
uv run python examples/database_diagnosis/main.py --database-path data/european_soccer/database.sqlite
```

## OpenAI API Time and Cost

The exact cost depends on the model, database schema size, and number of revision attempts. As a rough planning estimate for `OPENAI_MODEL=gpt-5.5`, a run that accepts the first candidate usually makes six model calls: three specialists in parallel, one reviewer, one evaluator, and one reporter. Expect roughly 40k-80k input tokens, 6k-15k output tokens, and about 2-5 minutes wall time on the European Soccer demo path.

Using the OpenAI API standard short-context pricing listed on July 2, 2026 for `gpt-5.5` (`$5.00` per 1M input tokens and `$30.00` per 1M output tokens; see the [OpenAI pricing page](https://developers.openai.com/api/docs/pricing)), that first-attempt run is roughly `$0.38-$0.85`. Each extra reviewer/evaluator revision attempt typically adds about 10k-30k input tokens and 2k-6k output tokens, or around `$0.11-$0.33`; with the default three-attempt cap, budget roughly `$0.60-$1.50`. Prompt caching may lower the effective input-token cost.

## Example Result

Command:

```bash
uv run python examples/database_diagnosis/main.py \
  --database-path data/european_soccer/database.sqlite \
  --webui
```

Observed result:

- Database: `data/european_soccer/database.sqlite`
- Write enabled: `false`
- Main issue: missing child-side indexes on SQLite foreign-key/filter columns.
- Strongest finding: `Match.away_player_11` caused a full scan of `Match`.
- Recommended SQL:

```sql
CREATE INDEX IF NOT EXISTS "idx_dbbot_match_away_player_11"
ON "Match" ("away_player_11");
```

Evaluation:

- Before: `SCAN Match`, about `10.616 ms`.
- After candidate index: `SEARCH Match USING INDEX idx_dbbot_match_away_player_11 (away_player_11=?)`, about `0.026 ms`.
- The index was not persisted because the run was read-only, so the candidate was rolled back.

The full report for that run is:

```text
runs/2026-06-25_16-39-42-698411_database_diagnosis/database_diagnosis_report.md
```
