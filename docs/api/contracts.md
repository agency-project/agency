# Known API-contract issues

This is an evidence record, not an alternate intended API. The reference documents current behavior; Stage 2 changes inaccurate docstrings without changing execution.

| Priority | Demonstrated discrepancy | Evidence and consequence |
| --- | --- | --- |
| 1 | Pending agdata/agtask resolution loses the agerror/agcanceled subtype | [agdata._resolve](../../agency/agdata.py) copies only `_data`. The synthetic error example in [results](results.md) demonstrates it without infrastructure; [submission tests](../../tests/test_submission_orchestrator.py) establish that run returns bare agdata. Callers must inspect the reserved `error` field; subtype-based cancellation detection and AgError-on-output promises do not hold. Decide a future stable error/cancellation inspection API. |
| 1 | No automatic skill input validation | [agskill](../../agency/agskill.py), [engine](../../agency/engine/engine.py) and [native loop](../../agency/native_harness/react_loop.py) prepare inputs without calling validate_input. [Skill schema tests](../../tests/test_agskill.py) test explicit calls, not automatic submission enforcement. Invalid inputs can reach the model. Decide whether validation belongs at admission/materialization or stays caller-owned. |
| 2 | Initial orchestrator resource config is omitted | [GlobalAgentOrchestrator.__init__](../../agency/orchestrator/orchestrator.py) creates agResourcePool without agconfig; [pool construction](../../agency/orchestrator/agresources.py) makes defaults. Supplied idle resource limits do not initialize the global pool. Explicit change_config propagates them. Requires a runtime fix and constructor test separately. |
| 2 | Advertised timeouts are not overall execution deadlines | [agtool.__call__](../../agency/agtool.py) ignores timeout; host/sandbox MCP dispatch calls synchronously. [Orchestrator.flush](../../agency/orchestrator/orchestrator.py) ignores timeout_s. Result wait timeout applies to its immediate future and shutdown timeout does not bound every cleanup. Choose/remove/enforce timeout meanings individually; don't infer hard cancellation. |
| 3 | Process concurrency update does not resize executor | [orchestrator construction/change_config](../../agency/orchestrator/orchestrator.py) fixes ThreadPoolExecutor max_workers at construction. Increasing admission ceiling later can still queue in the original executor. Choose a documented immutable setting or implement safe resizing. |
| 3 | Configuration replacement has construction-time boundaries | [Agent.change_config](../../agency/agent.py) does not change chosen harness; output properties can change without updating the existing output mount; [sandbox change_config](../../agency/sandbox/agsandbox.py) does not rebuild image/mount selection; logger retains config/open connection. Decide which fields should be immutable versus explicitly recreated. |
| 3 | Recovery and schema typing are partial | [agschema](../../agency/agschema.py) validates selected shapes; preparation/recovery catch exceptions and retain values. agbinary may remain a path after failed recovery. The earlier agpath docstring referred to a removed plain-str field handler; current plain-str outputs are not dereferenced. Applications requiring strong types must explicitly check before/after execution. |

## Stage 3 findings requiring a code decision

| Priority | Current behavior | Evidence / decision |
| --- | --- | --- |
| 1 | False filesystem commit return is ignored by AgentEngine.execute | [Engine](../../agency/engine/engine.py) tests commit exceptions but does not check commit's bool. Decide whether false must fail before publishing session/output. Current docs promise ordering of invocation, not durable capture for every success. |
| 2 | Native and PTY inactivity can return ok=True with partial/empty output | [Native adapter](../../agency/harness/adapters/native.py), [PTY execution](../../agency/harness/adapters/pty/execution.py), and [timeout-as-completion tests](../../tests/harness/test_external_pty.py). Decide whether inactivity needs a distinct terminal status; raw outputs can otherwise appear successful. |
| 2 | Native pause is not excluded from its separate progress deadline | Native's 300-second polling deadline does not extend for pause; PTY does. Source-only finding: add/choose the supported long-pause semantics rather than promising an unbounded pause. |
| 3 | Failed-daemon attempt reader and callback drain have different limits | [Engine watchdog](../../agency/engine/engine.py) can abandon a blocked reader thread; [token retirement](../../agency/engine/host_servers/host_server_manager.py) waits for admitted callbacks. Decide acceptable leak/drain deadlines without violating attempt fencing. |

[Architecture](../architecture/index.md) explains the current package design and execution boundaries. These findings remain unresolved runtime choices.

## Evidence boundaries

Ordering, nested dependencies, timeout/async waiter protection, cancellation commit races, redirect request identity/fallback and stable fork are covered by the existing [agdata](../../tests/test_agdata.py), [submission](../../tests/test_submission_orchestrator.py), [transaction fence](../../tests/test_orchestrator_transaction_fence.py), [redirect](../../tests/test_redirect.py), [lifecycle](../../tests/test_orchestrator_lifecycle.py) and [workflow](../../tests/test_agsync.py) tests. These use local fixtures/mocked engines; they do not establish live external CLI/model behavior.

The input-validation omission, initial resource-config omission, executor resizing boundary and exact portable checkpoint restrictions were traced through source. No live model, Linux ptrace container, privileged ZFS/CRIU setup or Perfetto asset build was performed on the macOS ARM64 verification host. See the [refactor record](../archive/documentation-refactor.md#stage-2-verification) for executed checks and environment limitations.
