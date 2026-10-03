# Finished profiler views

The six primary screenshots use the explicitly synthetic three-agent demo.
Additional screenshots show recorded E1A.1 telemetry, the mobile layout, and
the trajectory's agent overlay.

## 1. Trajectory

![Trajectory with expandable evidence](../../assets/profiler/01-trajectory.jpg)

## 2. Resources & waits

![Action-aligned resource timeline](../../assets/profiler/02-resources.jpg)

## 3. Context & information

![Prompt composition and supervisor-worker context boundary](../../assets/profiler/03-context.jpg)

## 4. Multi-agent

![Agent relationships and concurrent work](../../assets/profiler/04-multi-agent.jpg)

## 5. Compare runs

![Aligned trajectories showing divergence and reconvergence](../../assets/profiler/05-compare.jpg)

## 6. Task progress

![Workflow obligations and supporting evidence](../../assets/profiler/06-task-progress.jpg)

## Recorded execution

![Real baseline resource telemetry](../../assets/profiler/07-recorded-resources.jpg)

## Mobile

![Responsive task progress layout](../../assets/profiler/08-mobile.jpg)

## Agent trajectory overlay

![Shared execution clock with three agent trajectories and linked action inspector](../../assets/profiler/09-trajectory-overlay.jpg)

## Real live trajectory

Actual local file/tool/subprocess work, with a running call and returned assertion.
The example invokes no model or autonomous agent.

![Real live trajectory with held evidence](../../assets/profiler/10-live-trajectory.jpg)

## Recorded trace replay

Real `psf__requests-1921` span data, replayed through the incremental reducer.
The final execution state was not recorded and is shown as unavailable.

![Recorded replay with retained evidence and unknown final state](../../assets/profiler/11-recorded-replay.jpg)

## Input and repeated failures

Explicitly synthetic edge-case fixture. A pattern worth inspecting and an
action request have different states, with inspectable supporting events.

![Synthetic attention scenario and source evidence](../../assets/profiler/12-replay-attention.jpg)

## Real local run after completion

![Read, failed checks, recorded edit, and a passing check](../../assets/profiler/13-live-review.jpg)

## Live trajectory on mobile

390px viewport, with the inspector below the activity feed and no horizontal overflow.

![Live trajectory at mobile width](../../assets/profiler/14-live-mobile.jpg)

## Real Codex and GPT Luna

SWE-bench Lite `psf__requests-1963`, executed through Agency's actual Codex
harness using `gpt-6-luna` and provider credentials. The official evaluator
reported the generated patch resolved the task.

![Real Codex execution while running](../../assets/profiler/16-real-codex-live.jpg)

The overlay inspector shows the actual focused test output: 6 passed.

![Real Codex overlay and returned test evidence](../../assets/profiler/17-real-codex-overlay.jpg)

An earlier attempt exposed a prompt-acknowledgment timeout in the Codex driver;
the failed run was retained and the driver fixed before the successful run.

![Retained real harness failure](../../assets/profiler/15-real-codex-harness-failure.jpg)

## Real two-agent SWE-bench workflow

Two GPT Luna / Codex agents worked on `psf__requests-1963`: one implemented the
fix, the other reproduced the bug with a regression test. Their combined patch
passed all 119 required official evaluator checks.

![Two real agents while running](../../assets/profiler/20-two-agent-live.jpg)

![Completed two-agent overlay with regression evidence and actual handoff](../../assets/profiler/21-two-agent-completed.jpg)

![Recorded regression-patch handoff in the real workflow](../../assets/profiler/22-two-agent-handoff.jpg)
