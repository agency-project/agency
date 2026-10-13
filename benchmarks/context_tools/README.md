# Context-tools benchmarks

Does letting the model edit its own context (`context_tools()`, docs/Context_Tools.md)
help under a tight context budget? `oolong_pilot.py` runs Oolong-synth questions with a
fresh agent per question and three arms: **A** harness alone, **B** + `context_tools()`,
**C** + `context_tools()` and a usage note in the task.

## Setup (Linux host)

```sh
sudo .venv/bin/agency setup-host --runtime podman          # docs/setup-host.md
sudo podman build -t localhost/agency-bench:py312 benchmarks/context_tools
uv run --no-project --with datasets python - <<'PY'        # e.g. the 128K-token slice
from datasets import load_dataset; import json
rows = [r for r in load_dataset("oolongbench/oolong-synth", split="validation", streaming=True)
        if r["context_len"] == 131072]
with open("oolong_val_131072.jsonl", "w") as f:
    for r in rows:
        r.pop("context_window_text_with_labels"); f.write(json.dumps(r) + "\n")
PY
```

## Run

```sh
sudo bash -c 'set -a; . ./.env; set +a; .venv/bin/agency run \
  benchmarks/context_tools/oolong_pilot.py --data oolong_val_131072.jsonl \
  --n 50 --arms A,B,C --out runs/oolong_pilot --image localhost/agency-bench:py312'
```

`.env` needs `ANTHROPIC_API_KEY` (and `ANTHROPIC_WORKSPACE_ID` for a key not scoped to a
workspace). Results: `<out>/results.jsonl` (one line per run) and `<out>/summary.md`.

## Keep the sandbox offline

The Oolong messages come from public labelled datasets (SMS Spam Collection, TREC).
With network access a model will download the labels and match messages instead of
classifying them -- Sonnet 5.5 did exactly that in a first calibration run. The runner
therefore starts sandboxes with `--network=none` (which is why the image preinstalls
the harness dependencies) and denies the `webfetch`/`websearch` tools.

## Notes

* Scoring follows Oolong: exact match for labels/users/dates/comparisons,
  `0.75 ** |error|` for counts; `relative_error` is also reported for counts.
* The native harness compacts at 90% of `context_limit`, using its own estimate of its
  own (unedited) messages, so model edits lower the prompt the LLM sees but not the
  harness's compaction trigger.
* Many Oolong questions (single users, small subsets) are solvable with `grep`/`awk`
  and never fill the context; whole-document label counts are the ones that do.
