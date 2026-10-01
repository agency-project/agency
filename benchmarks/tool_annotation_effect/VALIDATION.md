# Validation record — October 1, 2026

The implementation was developed on `eric/episode-traces`. The initial request restricted execution to offline tests; the user subsequently authorized SSH testing on EC2 and GPT Luna experiments using `.env`. The source was synced to the isolated `/home/eric/episode-traces` checkout before each EC2 run. The repository examples identify this model as `gpt-6-luna` with `reasoning_effort=none` for Chat Completions tool calls.

## Software tests

- **130 passed** locally and on EC2: new benchmark tests plus native tools, native compaction, configuration, and daemon-launcher tests.
- **38 passed, 1 skipped** locally: orchestrator lifecycle/edges, example imports/static contracts, and native profiler tests. The skipped test is the opt-in live tutorial suite.
- Ruff checks passed for changed runtime files, the benchmark package, and new tests. `git diff --check` passed.
- Official evaluators and Harbor environments were represented by fakes and saved verifier responses in these tests.

## Live native-path validation

The final validation (`validation-v4`) assigned **one local RAG task, one repetition, all four arms**, sequentially on EC2 using an already available Podman image pinned by its local image ID. No benchmark dataset was downloaded and no image was built or pulled. The task required current Atlas escalation contact and supporting document IDs. Its public contract explicitly declared the answer field names and types; the gold answer remained in the host evaluator. Models used the same limits, seed, context/compaction policy, resource limits, and cache policy across arms.

| Arm | Fixture success | Tool calls | Model calls | Input tokens | Output tokens | Valid annotations |
|---|---:|---:|---:|---:|---:|---:|
| baseline | true | 9 | 7 | 10,868 | 216 | 0 |
| schema_only | true | 10 | 6 | 10,241 | 246 | 0 |
| purpose | true | 9 | 7 | 12,506 | 378 | 9 |
| purpose_workstreams | true | 7 | 6 | 11,695 | 357 | 7 |

Every traced tool call had an actual Agency call ID. Schema-only calls omitted the optional metadata; purpose/workstream calls supplied valid metadata. Original model responses, executable actions, verifier verdicts, generated CSV/JSON/report files, and blinded review packets were saved. There is one task cluster and no baseline repeat, so this validation cannot estimate useful cluster intervals or excess trajectory distance over baseline variability. No equivalence or treatment-effect claim is made. No pricing file was used and no billed cost is claimed.

All development attempts remain under `artifacts/tool-annotation-ec2/` locally and `/home/eric/episode-traces-results/` on EC2:

- `validation-v1` revealed missing transport of trace/treatment settings through the daemon allow-list. Execution was interrupted; its assigned attempts were preserved.
- `validation-v2` verified the repaired transport but exposed an underspecified RAG answer-field contract. All assigned outcomes were retained; they are not evidence for the final fixture design.
- `validation-v3` recorded infrastructure failures before provider execution because a Docker image ID was supplied to the selected Podman runtime. These failures were preserved.
- `validation-v4` used the Podman image ID and the explicit fixture contract. All four assignments completed and passed fixture evaluation.

These are versioned software validation attempts, not a selectively rerun pilot. No completed trial was overwritten or hidden. The temporary EC2 credential file was removed after execution. `source-provenance.json` records the originating repository revision and exact final implementation hash; the isolated EC2 source checkout does not contain Git metadata.

## Unverified integration boundaries

SWE-bench official evaluation, Harbor 0.23.0 execution on published Terminal-Bench 2.0 tasks, live C→Rust compilation/differential behavior, and live tandem worker workflows were **not** executed. Harbor/SWE-bench are optional dependencies and absent from the EC2 test environment; `rustc` was also absent. Their fake contract tests establish export/import, dispatch, and failure-accounting behavior, not live compatibility or benchmark results. Task images and repository dependency setup must be prepared explicitly before a published-suite run.

## Ten repetitions per arm — October 1, 2026

The user requested n = 10 with the same setup. `configs/rag-repeat-10.json` changes only `repetitions` from 1 to 10: one frozen `atlas-escalation` task, all four arms, GPT-6 Luna, reasoning effort none, provider seed 42, identical budgets, Podman image, schemas, prompts, resource limits, and randomized block order. Source was synced before execution and its hash verified. The source differs from validation-v4 only by the previously documented formatting change in commit `daa9a88`; the frozen task is identical. Forty fresh model-backed assignments executed sequentially, with no recovery attempts or retries of completed trials.

| Arm | Success | Mean tool calls (range) | Mean input tokens | Mean output tokens | Mean total tokens | Mean agent seconds |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 10/10 | 8.9 (8–9) | 10,196.1 | 216.9 | 10,413.0 | 8.11 |
| schema_only | 10/10 | 8.4 (6–10) | 10,170.0 | 207.0 | 10,377.0 | 8.84 |
| purpose | 10/10 | 6.8 (5–9) | 10,034.1 | 281.1 | 10,315.2 | 7.10 |
| purpose_workstreams | 10/10 | 5.6 (5–6) | 9,572.4 | 274.9 | 9,847.3 | 6.62 |

All 40 final answer/evidence outputs had the same hash. All 297 tool actions had actual Agency call IDs. Purpose annotations were valid on 68/68 calls; purpose/workstream annotations were valid on 56/56 calls. Schema-only supplied no annotations on its 84 calls. One schema-only run requested nonexistent document ID `regions`, then recovered and passed. No input/output usage was missing, and no malformed annotation, compaction, or truncation was recorded. Agent timing excludes environment setup; mean setup times were 9.31–9.66 seconds across arms. No pricing file was supplied, so no billed dollar cost is estimated.

Purpose used 23.6% fewer tool calls and 0.9% fewer total tokens than baseline; purpose + workstreams used 37.1% fewer calls and 5.4% fewer tokens. These are observed mean differences on this task, not general performance claims. Mean normalized operation-edit distance was 0.180 across the 45 baseline-baseline pairs, 0.394 for baseline-purpose, and 0.470 for baseline-workstreams. Cross-arm distances thus exceeded observed baseline variability, while correctness was identical. Pairwise comparisons share runs and are not independent samples.

**Sample size:** 10 runs per arm, 40 runs total, one independent task cluster. Repetition estimates within-task variability but does not establish effects across tasks or providers. The task-clustered report correctly leaves confidence intervals unavailable with one cluster; no significance or equivalence claim is made. Passing 10/10 on this task does not establish a perfect underlying success rate.

Full artifacts, raw traces, CSV/JSON exports, descriptive means/sample SDs/ranges, provenance, and blinded review packets are in `artifacts/tool-annotation-ec2/rag-repeat-10-v2/` and `/home/eric/episode-traces-results/rag-repeat-10-v2/`. A failed initial launch is preserved separately in `rag-repeat-10/`: all 40 assignments failed before model calls because the remote environment lacked `OPENAI_BASE_URL`. The corrected launch explicitly set the same `https://api.openai.com/v1` endpoint used for validation-v4. The temporary remote credential file was removed after execution.
