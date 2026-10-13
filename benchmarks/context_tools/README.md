# Context-tools benchmarks

`oolong_pilot.py` runs Oolong-synth questions under a capped context, a fresh agent
per question and run, in three arms: **A** the harness alone, **B** with
`context_tools()` (docs/Context_Tools.md), **C** with `context_tools()` and a note in
the task that the context is small and how to edit it. `analyze.py` breaks a
`results.jsonl` down by arm with bootstrap intervals.

```sh
sudo .venv/bin/agency setup-host --runtime podman          # docs/setup-host.md
sudo podman build -t localhost/agency-bench:py312 benchmarks/context_tools
uv run --no-project --with datasets python - <<'PY'        # the 128K-token slice
from datasets import load_dataset; import json
rows = [r for r in load_dataset("oolongbench/oolong-synth", split="validation", streaming=True)
        if r["context_len"] == 131072]
with open("oolong_val_131072.jsonl", "w") as f:
    for r in rows:
        r.pop("context_window_text_with_labels"); f.write(json.dumps(r) + "\n")
PY
sudo bash -c 'set -a; . ./.env; set +a; .venv/bin/agency run \
  benchmarks/context_tools/oolong_pilot.py --data oolong_val_131072.jsonl \
  --n 50 --arms A,B,C --out runs/oolong_pilot --image localhost/agency-bench:py312'
python benchmarks/context_tools/analyze.py runs/oolong_pilot/results.jsonl
```

`.env` needs `ANTHROPIC_API_KEY` (and `ANTHROPIC_WORKSPACE_ID` for a key not scoped
to a workspace).

The Oolong messages come from public labelled datasets; with network access the
model downloads the labels instead of classifying. Sandboxes therefore run with
`--network=none` (the image preinstalls the harness dependencies) and the
`webfetch`/`websearch` tools are denied.

Scoring follows Oolong: exact match, or `0.75 ** |error|` for counts;
`relative_error` is reported for counts too.

## Results

`results/oolong_pilot_2026-10-13.jsonl`: native harness, Claude Sonnet 5.5,
131K-token documents, `context_limit` 32768, 9 questions x 3 repeats per arm. The
4 questions that count labels over the whole document fill the context; the 5 on
subsets are solved with `grep`/`awk` and show no difference between arms.

Whole-document questions (A lost one run to an API error):

| | A | B | C |
|---|---|---|---|
| runs | 11 | 12 | 12 |
| API cost per run | $0.76 | $0.59 | $0.35 |
| harness compactions per run | 1.91 | 1.33 | 0.25 |
| peak prompt tokens | 55.9K | 56.9K | 38.0K |
| mean / median relative error | 11.1% / 2.4% | 6.3% / 2.0% | 1.9% / 2.0% |
| runs that edited | — | 0 | 7 |

C is lower than A and B on cost, compactions and peak prompt (95% intervals exclude
zero); error differences are not significant. B never edited. C's note both warns
that the context is small and explains editing, so this does not separate the two;
one model, one harness, one budget.

The native harness compacts at 90% of `context_limit` on its own estimate of its
unedited history, so edits lower what the LLM sees, not the compaction trigger.
