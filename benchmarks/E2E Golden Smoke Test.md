## OpenCode pre-LLM latency benchmark

### Methodology
- Ran on `goldentest@3.145.66.64` using the same Agency source snapshot, Docker image, and profiler settings throughout.
- Used the existing `test_golden_execution.py` E2E test with deterministic mock/replay responses; no real LLM APIs.
- Verified all harnesses, then ran one complete unmeasured warmup sweep.
- Ran 5 sequential measured rounds: 25 executions total. Rotated harness order so each occupied every position once.
- Measured `pre_llm_latency = first LLM attempt start − run_harness_attempt start`.
- Handled native’s two model-attempt spans separately from external harnesses’ initial attempt and follow-up.
- Validated successful execution, lifecycle completion, both model interactions, sandbox commit, and destroy. Preserved every trace; no measured samples were rerun.

### Results
All 25 measured executions passed.

| Harness | Mean pre-LLM latency |
|---|---:|
| Claude Code | 0.728 s |
| Codex | 0.697 s |
| Grok | 0.516 s |
| Native | 2.893 s |
| OpenCode | 6.747 s |

OpenCode’s five values: **7.550, 5.950, 5.702, 8.193, 6.342 seconds**.

OpenCode was consistently slower, although the delay was not consistently 7–8 seconds.

### Hypotheses—not confirmed causes
1. **Repeated cold startup:** Agency gives each OpenCode invocation a fresh `HOME` and deletes it afterward. Warmup therefore does not preserve OpenCode’s home-directory caches.
2. **Startup network or dependency work:** Model-catalog fetching or dependency resolution could contribute to the variable delay. Existing traces do not confirm these operations.
3. **Process-supervision overhead:** Agency’s ptrace supervision could amplify startup costs if OpenCode performs many intercepted operations.

Sandbox creation, daemon setup, and Docker commit occur outside the measured interval.

### Next diagnostic
Capture timestamped OpenCode startup logs and compare fresh versus reused cache directories using the same mock backend. Then isolate network waits and process-supervision overhead if needed.
