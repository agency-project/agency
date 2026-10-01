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
