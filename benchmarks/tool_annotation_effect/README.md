# Tool annotation effect

This runner tests whether asking the executing model to state a tool call's immediate purpose changes correctness, trajectories, or overhead. Labels are **agent-stated purposes**, not verified internal reasoning. Annotation utility is a separate human-review outcome.

The four arms are `baseline`, `schema_only`, `purpose`, and `purpose_workstreams`. `_agency` is an optional object in active schemas. Purpose-only schemas omit `workstream_ids`. Purpose and workstream membership are required by the treatment instruction, never by a corrective retry. Missing or malformed metadata is recorded and removed before admission/execution; valid executable arguments still run. Schema-only calls may annotate spontaneously.

The exact instructions are:

> For each tool call, state its immediate purpose in one short sentence.

> Assign stable workstream IDs for the subtasks this call advances. Reuse IDs when continuing a subtask. A call may advance multiple workstreams.

The experiment manifest saves the planned schemas and instructions. Each attempt also saves an immutable `treatment-manifest.json` containing the actual discovered schemas, instructions, role, and initial messages. This is authoritative for MCP servers whose discovery signatures can differ from their original `agtool.params`. Baseline preserves original schemas/prompts. Active schemas are deep copies. Original annotated assistant messages stay in conversation history, including subsequent turns; the ordinary declared native compaction policy still applies to all arms.

## Commands

Run from the repository root with its Python environment. Importing this benchmark creates no environments, calls no providers, and downloads nothing. `plan`, `analyze`, `review-export`, and `power` are offline operations. `prepare`, `run`, and `evaluate` are explicit side-effecting operations.

```sh
python -m benchmarks.tool_annotation_effect plan \
  --config benchmarks/tool_annotation_effect/configs/local-pilot.json \
  --output artifacts/tool-annotation/local-pilot

python -m benchmarks.tool_annotation_effect prepare \
  --suite swebench --output /absolute/prepared/swebench
# Add --lite for SWE-bench Lite. Verified is the primary suite.
python -m benchmarks.tool_annotation_effect prepare \
  --suite terminalbench --output /absolute/prepared/terminalbench

# Set the tasks_file paths in pilot.json after preparation, then create a NEW plan.
python -m benchmarks.tool_annotation_effect plan \
  --config benchmarks/tool_annotation_effect/configs/pilot.json \
  --output artifacts/tool-annotation/pilot

# Configure environment variables named by the model configuration.
# Keep credentials outside configuration/manifests. The provided model uses
# gpt-6-luna with reasoning_effort=none, as in the current Agency examples.
export OPENAI_BASE_URL=https://api.openai.com/v1
python -m benchmarks.tool_annotation_effect run --plan artifacts/tool-annotation/pilot
python -m benchmarks.tool_annotation_effect evaluate --plan artifacts/tool-annotation/pilot
python -m benchmarks.tool_annotation_effect analyze --plan artifacts/tool-annotation/pilot
python -m benchmarks.tool_annotation_effect review-export \
  --plan artifacts/tool-annotation/pilot --output artifacts/tool-annotation/review

python -m benchmarks.tool_annotation_effect power --margin .05 --task-difference-sd .2
python -m benchmarks.tool_annotation_effect power --margin .05 \
  --pilot-analysis artifacts/tool-annotation/pilot/analysis.json \
  --suite rag --model-id gpt-6-luna-openai --active purpose --control schema_only
```

`configs/rag-repeat-10.json` reproduces the single-task RAG setup with ten repetitions per arm (40 fresh trials). It uses the EC2 cached Podman image ID from live validation. Set both `OPENAI_API_KEY` and `OPENAI_BASE_URL=https://api.openai.com/v1` before execution. This repeated-run design still has only one task cluster; see [VALIDATION.md](VALIDATION.md) for the saved outcomes and limits.

`run --limit N` stops after N new assignments; unexecuted assignments remain missing in analysis. Completed failures are never retried by resume. After a crash, `--resume-partial` explicitly preserves the interrupted attempt and starts a fresh environment in the next attempt directory. The original attempt defines assigned-trial accounting; recovery attempts are exported and counted separately. A file lock prevents concurrent mutation. Status files use atomic replacement; immutable JSON artifacts use atomic hard-link publication. Modified manifests or implementation sources fail validation before execution.

`analyze --pricing PATH` accepts only an explicit versioned USD pricing file. The example contains placeholder rates that must be replaced; it is not a price estimate. Usage with unknown input/output/cache counts produces unknown cost where appropriate. Annotation token estimates never stand in for billed tokens. `analyze --tokenizer-file PATH` optionally uses the `tokenizers` library with a supplied local tokenizer JSON. No tokenizer is downloaded. The tokenizer hash and pricing version are recorded.

## Design and interpretation

A block fixes suite, task, provider/model settings, and repetition. Every arm starts in a fresh equivalent environment. An experiment seed deterministically shuffles arm order within blocks. A separate task-sampling seed freezes task IDs and strata before outcomes exist. Model executions are independent; replay is used only in software tests. Provider seeds are recorded without treating them or temperature zero as determinism guarantees.

All arms use identical model settings, native tool capabilities, step/timeout budgets, per-call completion-token limits, declared resource limits, context limit, native compaction policy, cache policy, and profiling configuration. Annotation tokens consume the same limits. Top-level trials execute sequentially. Tandem workers have independent environments and three scheduler slots to permit ordinary role interleaving. The primary design has no intentional contention. An annotation × contention design is not implemented and configuration rejects it.

The pilot uses five tasks per included suite, three repetitions, four arms, and one explicit model. Across all five suites it assigns 300 top-level trials and 420 nominal agent executions (tandem includes two workers plus its supervisor). Actual executions are counted independently; supervisors that fail to start workers do not inflate that count. This is feasibility screening, not evidence of equivalence. The expanded profile requires samples larger than five tasks and at least five repetitions. Supply larger frozen local fixture catalogs using `tasks_file` as well as published-suite catalogs; the five bundled fixtures alone are not an expanded sample. Keep each model/provider configuration separate.

Planned contrasts are schema-only minus baseline (schema exposure), purpose minus schema-only (requesting purpose), workstreams minus purpose (incremental grouping), and each active treatment versus baseline (total practical effect). Analysis labels comparisons exploratory, reports effect sizes and task-clustered intervals, and makes no equivalence claim. Predeclared practical margins are saved in example configurations. `power` supplies an approximate paired-task sample-size calculation from hypothetical SD or saved pilot task-average differences; the calculation is planning guidance, not a proof of sufficient power or equivalence.

The primary success summaries keep official/fixture verdicts, assigned counts, unknown counts, and success bounds separate. Confirmed-success differences treat unknown verdicts as unconfirmed for conservative assigned accounting, without replacing their exported unknown verdicts with zero. Complete-case sensitivity excludes infrastructure failures and missing verdicts. Repetitions are averaged within tasks; bootstrap draws resample whole tasks. A single task cannot produce a useful cluster interval. No heterogeneous suite scores are pooled into a headline score.

Trajectory comparisons include every baseline-baseline pair within task/model, every cross-arm pair, and within-treatment pairs across repetitions. Task clusters retain the dependency between comparisons. Treatment distance minus the same task's baseline-baseline distance quantifies excess over ordinary baseline variability; when baseline repeats are unavailable, excess is unknown. Metrics include longer-sequence-normalized tool LCS distance, normalized operation/target/argument edit distance, target overlap, action/validation-cycle differences, artifact hashes, and per-role trajectories. Purpose text and workstream IDs never enter these metrics. Shell commands remain compound strings; unknown tools stay unknown. Normalizer version and classification coverage are exposed. Unknown/missing trace coverage does not imply identical trajectories.

Overhead includes model calls/attempts, tool calls/errors, input/output/cache tokens, missing usage, annotation characters and optional tokenizer estimates, truncation, compaction, execution and setup/evaluation timing, and workstream reuse. Native timing starts at the treatment event after bootstrap; the top-level lifecycle/end-to-end measurement is saved separately. Tandem timing covers the interleaved workflow rather than summing concurrent durations. Profiler sessions are optional but fixed by configuration, and their artifacts are saved per attempt.

## Native integration

`agency/native_harness/annotations.py` augments schemas and extracts metadata. The ReAct loop applies it to both native built-ins and MCP discoveries. Metadata is captured before policy admission, including all calls in a model batch before the first starts, so denied/interrupted batches retain annotations. Annotation events, admission events, and result events share an event ID; admission/result carry the actual Agency call ID when a bridge exists. Model tool-call ID, exchange index, agent/run IDs, source/version/status, timestamps, batch size, and serial dispatch are recorded. No additional admission is issued by the MCP wrapper.

The native adapter transports settings through the daemon's credential-free configuration allow-list. External CLI harnesses reject active annotation configurations. They do not support augmentation of their private built-in schemas. Benchmark sandboxes mount only a temporary copy of Agency code; evaluator gold, host manifests and `.env` stay outside the mount. The native trace path is outside the task workspace and patches, and credentials are never intentionally included in artifacts. Transport secrets are redacted if echoed by an error or tool output. Per-agent logger/output directories retain Agency's ordinary exchange and profiler artifacts.

## Suite adapters and prerequisites

**SWE-bench:** optional `datasets` supplies Verified/Lite metadata during explicit preparation; `swebench` and its supported container runtime supply official evaluation. Preparation mirrors repositories on the host. Setup archives only the base commit into the agent environment; reference/test patches remain in host evaluator metadata. User-provided `setup_commands` and a compatible prebuilt `base_image` (or per-task `image`) must supply repository dependencies. The generic Python image is not a valid environment for every Verified repository. Patch extraction includes new files with `git add -N`, exports official `instance_id`, `model_name_or_path`, and `model_patch` JSONL, and invokes `swebench.harness.run_evaluation`. Every prediction evaluation gets a fresh UUID run ID; report import accepts per-instance and aggregate official formats. Missing official verdicts remain infrastructure failures.

**Terminal-Bench:** `terminal-bench@2.0` and **Harbor 0.23.0** are pinned. Preparation uses Harbor's dataset exporter and saves task-directory content hashes. Execution creates a one-task Harbor job with retries disabled and the custom `AgencyNativeAgent`. Harbor owns the task environment and verifier. Agency's real native loop runs externally and dispatches its built-in implementation through `BaseEnvironment.exec`; no nested Agency container or replacement agent is used. Only the stdlib native tool module is uploaded. Python 3 and bash must already be in the task image, plus dependencies needed by individual tools (for example webfetch). Task-provided Harbor MCP servers fail clearly; discovery through Agency's ordinary MCP client is supported by native Agency executions. Standalone Harbor calls have no Agency admission service, so their Agency call IDs are null rather than invented. Official verifier `reward` is imported, with exception/missing-verifier cases separated from task failure. Harbor's setup/agent/verifier timing and context usage remain in the saved official result.

**RAG:** five deterministic local tasks use retrieval/read custom `agtool` functions over a corpus with stale versions, multi-document owner/escalation links, region links, and distractors. Structured gold answers and supporting IDs remain on the host. Correctness and required evidence support are checked independently without a model judge.

**C → Rust:** five defined-domain C programs cover parsing, buffers, structs/state, error codes, and unsigned field extraction. The agent writes `solution.rs`. An evaluator host with `cc` and `rustc` compiles both programs and runs seeded bounded differential inputs. Compilation, behavioral equivalence over the declared test sample, and task success are separate outcomes. Compilation alone is insufficient; passing a finite differential sample is not a proof for all inputs. Execute untrusted evaluator binaries on a disposable benchmark host.

**Tandem:** current `Agent.run`, `agskill`, and user-defined `start_worker`, `collect_worker`, and `validate_ledger` tools coordinate independent ledger workers and a shared checked artifact. Both roles receive the primary treatment. `roles=supervisor` or `roles=worker` supports later role-specific studies. Worker native traces are saved per role and their artifacts are retained in collection tool results; the independently calculated final ledger determines task success.

Published-suite references inspected for this implementation:

- [Official SWE-bench evaluation](https://www.swebench.com/SWE-bench/guides/evaluation/)
- [Harbor custom-agent interface](https://docs.harborframework.com/core-concepts/agents/custom-agents)
- [Harbor task structure and verifier rewards](https://docs.harborframework.com/core-concepts/tasks/overview)
- [Pinned Harbor BaseAgent source](https://github.com/harbor-framework/harbor/blob/v0.23.0/src/harbor/agents/base.py)
- [Pinned Harbor BaseEnvironment source](https://github.com/harbor-framework/harbor/blob/v0.23.0/src/harbor/environments/base.py)

## Artifacts and verification

Each trial contains its assignment and numbered attempt directories. Attempts contain status, execution JSON, actual treatment manifest, normalized actions, per-role original model/tool traces, Agency logs and outputs, final artifacts/predictions, verifier responses, and failures. Root exports are `analysis.json`, `results.json`, `results.csv`, and `report.md`. Human review uses an arm/model/outcome-blinded raw-action packet first, a separate self-label packet second, and a private unblinding key. Human workstream assignments and utility/consistency ratings are blank for reviewers to fill; no annotator model launches automatically.

Offline tests use fake model clients, bridges, environments, evaluator processes, and saved verifier responses. They cover schema/prompt preservation, extraction without retries, interleaving/reuse, admission correlation, interrupted batches, daemon setting transport, schedules, atomic/idempotent artifacts, partial recovery, evaluator export/import, known statistical differences, missing usage/infrastructure, and blinded review.

```sh
python -m pytest tests/tool_annotation_effect tests/test_native_harness_tools.py \
  tests/test_native_harness_compaction.py tests/test_agconfig.py \
  tests/engine/test_harness_daemon_launcher.py -q
```

Official SWE-bench evaluation, Harbor jobs against published Terminal-Bench tasks, live migration compilation, and live tandem workflows require separate integration validation. Offline fake contracts do not establish those integrations work in every environment. The user-authorized four-arm EC2 RAG validation checked the actual native/daemon/MCP path only; it does not establish statistical treatment effects. See [VALIDATION.md](VALIDATION.md).
