# Claude Code / Agency non-interference benchmark infrastructure

Start with [handoff.md](handoff.md) to run a fresh cohort on another machine. The runner uses LiteLLM for both arms and maintains two asynchronous workers per arm: four isolated executions, two CPUs and 4 GiB each, eight execution CPUs total. A process-shared admission limiter covers all model and auxiliary requests on that host.

The protocol fixes five SWE-bench Verified tasks, ten repetitions per arm, Claude Code 2.1.299, `claude-haiku-5-5`, and Agency source commit recorded in `freeze_metadata.json`. `task_manifest.json` contains identifiers, revisions and hashes, without task text or patches. `image_pins.json` contains immutable image references. Do not edit the frozen manifest or silently substitute a model, task, image, or binary.

Git contains only infrastructure, synthetic tests, documentation, a host configuration template and protocol metadata. The directory's `.gitignore` is an explicit allowlist. Datasets, task contents, generated host settings, keys, model transcripts, run results, grading output, charts, reports, checkpoints, binaries and archives remain local and ignored. Do not use `git add -f` to bypass this boundary.

`host.example.json` describes the target machine. Copy it to ignored `host.local.json`; set the expected EC2 identity (or explicitly choose a remote Linux host and its expected hostname), runtime/evaluator/binary paths and external credential-file path. The runner refuses macOS and wrong-machine execution. Host subprocesses use the selected Python interpreter; container paths remain fixed under `/opt` so the same portable runtime can be mounted on a new host.

Preparation downloads and verifies the pinned dataset, resolves immutable Docker images, verifies the native binary hash, records host details and freezes the account-specific pricing and execution configuration. Diagnostics check initial prompts, native request preservation, response controls, title calls, token usage and resource limits. Task success is never a diagnostic launch criterion.

`pipeline.py` runs diagnostics, measured work, official grading, analysis and the credential-scanned export. It skips completed measured attempts when resuming. A fresh diagnostic cohort is required after infrastructure changes. An unfinished attempt or evaluator failure requires an audit; unfavorable agent outcomes are never rerun. Each new machine should use a fresh checkout/study directory and its own cohort rather than reusing another host's generated configuration or state.

The primary analysis uses cache-neutral reference cost, with cache categories counted once. Cache-aware estimates are secondary and are not audited invoices. Missing forwarded-request usage stays unknown; a verified failure before any upstream request consumes zero provider tokens. The recorder and normalizer both reject a usage-less server error as a complete cost measurement. Request/response audits never drop unfavorable outcomes.

For existing results, preserve the original ignored evidence archive and analyze it in a separate directory using its archived scripts. The portable runner is a later infrastructure revision and does not alter the original measurements.
