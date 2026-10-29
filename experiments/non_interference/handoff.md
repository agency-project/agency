# Handoff: run the non-interference benchmark on another machine

## Scope and Git boundary

The requested branch is `eric/temp`. This handoff transfers the benchmark infrastructure, not previous experiment data. The original frozen task manifest is unchanged and its freeze commit remains in branch history. No new measured executions were launched while preparing this handoff.

Committed inputs are limited to source code, documentation, synthetic tests, `host.example.json`, immutable image references, and the task/freeze metadata. The manifest contains task IDs, base commits and hashes; it contains no problem statements, gold patches or results. All actual datasets, task contents, transcripts, provider responses, grades, prices, generated configuration, runtime metadata, figures, reports, binaries and archives are ignored. Existing local data has been preserved, not deleted.

Use a fresh checkout on the target host. Do not copy `configuration.json`, `preparation_complete.json`, `diagnostic_gate.json`, `attempts/`, `diagnostics/`, `runs.jsonl`, SQLite state, HOME directories or checkpoints from a previous cohort. Those would either resume old work or mix machines in one result set. A rerun on another host is a separate cohort; combining cohorts requires an explicit analysis design.

## Protocol pins

- Five tasks, in the frozen order: `matplotlib__matplotlib-25960`, `pydata__xarray-3305`, `django__django-15563`, `sphinx-doc__sphinx-7748`, `sympy__sympy-13615`.
- Ten measured repetitions per task/arm, 100 executions total.
- Claude Code 2.1.299; model `claude-haiku-5-5` through LiteLLM in both arms.
- Native binary SHA-256: `2d894a4b1bfaaa9aab72aeff7803e3c73bc2510f279a171f6b8f59f72d17d077`.
- Agency source: `a8bc606ac4a449407b43919551fbc878a8bd4b09`. New infrastructure code runs the frozen Agency source staged under `src/`.
- Manifest SHA-256: `4d9158fd7d62019d9e2a74c2a29ea82a3d78de7f0b5a661cbe737fbcd67fb76d`; freeze commit `8f1939b6b02df0f41afa1a5e602d82150d9ed95d`.
- Immutable task images are in `image_pins.json`; pull those digests, not mutable `latest` tags.
- Same model settings, benchmark prompt, native tools and permissions across arms; Agency context/tool optimizations, session-title suppression and approximate token counting disabled.

## Target prerequisites and parallelism

Use a remote Linux x86_64 machine with rootful Docker, Python 3.12, sufficient disk for the five task images and checkpoints, at least eight CPUs available to the experiment plus host capacity, and preferably 32 GiB RAM with 20 GiB free at launch. The configured host identity is checked before preparation, worker execution and grading. EC2 uses IMDSv2 and the specified instance ID. For an explicitly chosen non-EC2 Linux server, use `host_kind: "linux"` and its exact `expected_hostname`. There is no automatic fallback from EC2 to an arbitrary host, and macOS execution is rejected.

The scheduler already parallelizes with continuously replenished queues: two standalone workers and two Agency workers. Each container is limited to two CPUs / 4 GiB; graders also have bounded concurrency and resources. Measured execution can reduce to one worker per arm for documented memory headroom. Diagnostics require full four-worker headroom. The current protocol intentionally caps execution parallelism at four; a larger host does not silently increase it.

The API limiter is **host-local**, shared across this cohort's processes: 60 requests/minute, 3,000,000 conservative reserved tokens/minute, four active streams. It does **not** coordinate multiple machines or separate study directories. For concurrent cohorts sharing the same provider quota, first partition upstream quotas or implement a shared admission service; independent SQLite files do not form a global fleet limiter. Run only one cohort per Docker daemon at a time because container names are deterministic.

## External runtime artifacts (never commit)

The runner mounts two existing portable runtime trees at fixed container paths:

- `agency-python` -> `/opt/agency-python`
- `agency-harness-venv` -> `/opt/agency-harness-venv`

Their venv and executable paths reference `/opt`, so preserve these container destinations. Do not create a host-native venv with arbitrary absolute paths and assume it will work inside the old SWE-bench images.

Known read-only source artifacts on the original host (`eric@3.145.20.133`, using the operator's existing SSH key/config):

- Runtime parent: `/home/eric/agency-fidelity-expanded-20261023/runtime/` (copy only `agency-python` and `agency-harness-venv`).
- Native binary: `/home/eric/.local/share/claude/versions/2.1.299`.
- Evaluator source: `/home/eric/agency-fidelity-expanded-20261023/swe50-v6/private/evaluator/` (the `swebench` Python source tree).

Transfer these as separate, ignored runtime artifacts, via SCP through local storage if needed. Never copy the old account `.env`, study output tree or Docker checkpoints. Provision a LiteLLM credential separately on the target. Supply the evaluator's Python dependencies in the host environment; its source has no Git revision, so preparation records hashes of the actual evaluator files used. Preserve the same source for strict comparisons. Runtime/binary availability and real Docker execution on the destination still require target-host validation.

## Fresh-host setup

Clone the pushed branch on the target and verify it before making changes:

```sh
git clone --branch eric/temp git@github.com:agency-project/agency.git /srv/agency
cd /srv/agency
git branch --show-current
uv sync --extra dev
uv pip install --python .venv/bin/python -r experiments/non_interference/requirements-analysis.txt docker==7.1.0 anthropic==1.11.0 httpx==0.28.1

NI_STUDY=/srv/agency/experiments/non_interference
mkdir -p "$NI_STUDY/src"
git archive a8bc606ac4a449407b43919551fbc878a8bd4b09 agency | tar -x -C "$NI_STUDY/src"
cd "$NI_STUDY"
cp host.example.json host.local.json
```

Edit ignored `host.local.json`: set the target identity, installed runtime parent, evaluator source directory and Claude binary path. The example `/srv/...` paths are placeholders to provision on the target. Keep the API key in a private external file containing `LITE_LLM_KEY=...`, with restrictive file permissions. Set only that file's path in host settings. Alternatively supply `LITE_LLM_KEY` / `LITELLM_API_KEY` through the process environment and set `credentials_file` to null. `NI_HOST_CONFIG` can select an alternate settings file; preserve it if using sudo. No actual key belongs in JSON, command history, Git or handoff text.

With Docker access (or prefix the target commands with `sudo -n` and preserve only necessary environment settings):

```sh
/srv/agency/.venv/bin/python fetch_dataset.py
/srv/agency/.venv/bin/python prepare_ec2.py --pull-images
/srv/agency/.venv/bin/python -m pytest -q test_measurement.py test_portability.py
/srv/agency/.venv/bin/python -u pipeline.py --diagnostics-only --cohort diagnostics-target-initial
```

`fetch_dataset.py` downloads only the immutable Hugging Face revision and verifies its hash. Preparation verifies row hashes, images and binary, records the new host, discovers its Docker gateway, obtains current account pricing and refuses to overwrite an existing prepared configuration. It provisions no cloud machines. The target must be able to reach the same LiteLLM model and the token-count endpoint; no model substitution is permitted.

Inspect diagnostics and gate output. A pass requires valid usage, matching initial settings, native request/response controls, title calls, baseline project imports, grading functionality and CPU/memory limits. It never requires solving the task or a favorable cost result. Failed diagnostics remain excluded from measurement and retained for troubleshooting.

## Run, monitor and resume

After the gate passes, run the pipeline from the study directory:

```sh
nohup /srv/agency/.venv/bin/python -u pipeline.py > pipeline.log 2>&1 < /dev/null &
/srv/agency/.venv/bin/python status.py
```

The pipeline reuses a valid gate when its recorded source hashes still match, then executes the 100 runs, grades patches, checks all grading outcomes, computes statistics and figures, audits traces, exports credential-scanned evidence and validates the export. Re-running `pipeline.py` skips completed measured executions. Use a new `--cohort diagnostics-...` after fixing diagnostic infrastructure; do not reuse old diagnostic records to certify changed code. Never rerun an unfavorable completed measured execution. Interrupted attempts without a final summary require an audit. Only one additional worker attempt is permitted for a verified zero-inference setup failure.

The deterministic schedule is generated from the recorded seed. Every run starts from a fresh official image/base commit and independent session/state. Do not use `git clean -fdx` inside task images: it removes required compiled dependencies. Initial HTTP 400s rejecting `messages.1.output_config` are a known route compatibility case: Claude Code owns recovery, identically in both arms. The recorder adds no provider retries. A forwarded 5xx or truncated response without authoritative usage makes full cost unknown; never treat it as zero. The common admission limiter serializes SQLite transactions and closes connections to avoid the known pre-provider lock race.

For grader infrastructure failures, preserve the original report/log before any audited regrade. Never regenerate model patches to improve success rates. Do not bypass the diagnostic gate or change the frozen configuration to force progress.

## Outputs and transfer back

All generated outputs remain ignored in the study directory. `results.md` describes the primary cache-neutral endpoint, secondary cache-aware estimates, missingness, fixed-task bootstrap intervals and future-study planning. `run_integrity.json` checks the cohort and export, and `analysis_runtime.json` records the local analysis environment. Exact invoices require separate ledger access.

`results-evidence.tar.gz` includes raw requests/responses, summaries, grades, sources, generated metrics/report/figures and an export manifest. Its credential scanner needs the target credential to identify forbidden values. Copy this archive separately via SCP when results are needed; do not add it to Git. Validate after extraction with the archived `validate_results.py` and pinned analysis requirements. Analysis may run locally; the model workload and official grading stay on the configured remote host.

Before any future infrastructure commit, inspect `git diff --cached --name-only` and `git diff --cached --stat`. The allowlist should keep raw data and credentials out even when using `git add experiments/non_interference`; never override it with `-f`. The only JSON files intended for Git are the host template, immutable image references and frozen task/freeze metadata.

## Additional repetition cohorts

For 40 additional repetitions per task/arm after an existing 10-run cohort, prepare a fresh study with `prepare_ec2.py --repetitions 40 --repetition-start 11 --schedule-seed 2026102901`. This freezes 400 new executions, numbered 11–50, without reusing the original run IDs or changing the five tasks. Sample size is read from the frozen configuration by scheduling, analysis and validation. Keep the original cohort separate; report the extension independently and label any pooled 50-run comparison as an exploratory post-pilot extension.

Use `pipeline.py --skip-analysis` when the execution host lacks the pinned scientific plotting packages. It still performs diagnostics, measured execution, official grading and a scanned evidence export. Wait for `pipeline_complete.json` with `evidence_archive_ready: true` before transferring the archive, then run the analysis and validation commands locally using the pinned requirements. Do not alter the shared host environment just to install plotting dependencies.
