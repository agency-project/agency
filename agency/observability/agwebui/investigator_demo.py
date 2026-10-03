"""Deterministic, intentionally synthetic instrumentation for the research demo.

Kept separate from the saved-trace adapter. Every demo run is labeled synthetic
in the catalog, header, inspector and exported data.
"""

from __future__ import annotations

from .investigator import group_episodes

DEMO_ENTRIES = [
    {
        "id": "demo-baseline",
        "title": "Repair scheduler starvation",
        "subtitle": "Baseline · 3 agents · synthetic demo",
        "condition": "baseline",
        "task": "scheduler-starvation",
        "source": "synthetic",
        "model": "gpt-5.6-luna",
        "harness": "native",
        "duration": 142,
    },
    {
        "id": "demo-contention",
        "title": "Repair scheduler starvation",
        "subtitle": "CPU contention · synthetic demo",
        "condition": "cpu_contended",
        "task": "scheduler-starvation",
        "source": "synthetic",
        "model": "gpt-5.6-luna",
        "harness": "native",
        "duration": 218,
    },
]


def demo_run(run_id):
    contended = run_id == "demo-contention"
    entry = next(e for e in DEMO_ENTRIES if e["id"] == run_id)
    agents = [
        {
            "id": "supervisor",
            "label": "Supervisor",
            "role": "Coordinates and validates",
            "parent": None,
            "model": "gpt-5.6-luna",
            "harness": "native",
        },
        {
            "id": "scheduler",
            "label": "Scheduler worker",
            "role": "Investigates admission and fairness",
            "parent": "supervisor",
            "model": "gpt-5.6-luna",
            "harness": "native",
        },
        {
            "id": "tests",
            "label": "Validation worker",
            "role": "Builds regression coverage",
            "parent": "supervisor",
            "model": "gpt-5.6-luna",
            "harness": "native",
        },
    ]
    specifications = [
        (
            "a01",
            "supervisor",
            "search",
            0,
            3,
            "Map the execution path",
            'rg -n "dispatch|ready_queue" agency/orchestrator',
            "Located scheduler admission and dependency resolution in orchestrator.py.",
            "success",
        ),
        (
            "m01",
            "supervisor",
            "model",
            3,
            8,
            "Split implementation and regression work",
            "",
            "Delegate the queue investigation and regression coverage in parallel.",
            "success",
        ),
        (
            "a02",
            "scheduler",
            "read",
            12,
            5,
            "Inspect queue ordering",
            "sed -n '920,1040p' agency/orchestrator/orchestrator.py",
            "New ready requests are admitted before previously blocked requests.",
            "success",
        ),
        (
            "m02",
            "scheduler",
            "model",
            17,
            12,
            "Identify the starvation mechanism",
            "",
            "A released dependency re-enters behind newer requests; preserve submission order.",
            "success",
        ),
        (
            "a03",
            "tests",
            "search",
            12,
            4,
            "Locate scheduler tests",
            'rg -n "blocked|dispatch" tests/test_agorchestrator.py',
            "Found dependency admission fixtures; no fairness regression.",
            "success",
        ),
        (
            "m03",
            "tests",
            "model",
            16,
            9,
            "Design a deterministic regression",
            "",
            "Keep a worker occupied while a dependency becomes ready; assert admission order.",
            "success",
        ),
        (
            "a04",
            "tests",
            "edit",
            26,
            7,
            "Add a starvation regression",
            "apply_patch tests/test_agorchestrator.py",
            "Added test_dependency_ready_preserves_submission_order.",
            "success",
        ),
        (
            "a05",
            "scheduler",
            "edit",
            30,
            8,
            "Preserve stable queue priority",
            "apply_patch agency/orchestrator/orchestrator.py",
            "Ready queue now sorts by the original submission sequence.",
            "success",
        ),
        (
            "a06",
            "tests",
            "test",
            34,
            16,
            "Reproduce the original failure",
            "pytest tests/test_agorchestrator.py -k submission_order",
            "FAILED: request B started before dependency-ready request A.\n1 failed, 47 deselected",
            "failed",
        ),
        (
            "m04",
            "scheduler",
            "model",
            40,
            10,
            "Check cancellation and requeue invariants",
            "",
            "Retain the stable key across cancellation; notify on dependency release.",
            "success",
        ),
        (
            "a07",
            "scheduler",
            "read",
            51,
            6,
            "Inspect condition notifications",
            "sed -n '680,750p' agency/orchestrator/orchestrator.py",
            "Dependency completion wakes the scheduler; no polling is required.",
            "success",
        ),
        (
            "a08",
            "scheduler",
            "edit",
            58,
            6,
            "Keep requeue ordering consistent",
            "apply_patch agency/orchestrator/orchestrator.py",
            "Preserved priority across the ready → blocked → ready transition.",
            "success",
        ),
        (
            "m05",
            "tests",
            "model",
            65,
            7,
            "Validate the revised scheduler",
            "",
            "The worker patch is ready; rerun the regression and lifecycle cases.",
            "success",
        ),
        (
            "a09",
            "tests",
            "test",
            73,
            21,
            "Run fairness and lifecycle tests",
            "pytest tests/test_agorchestrator.py tests/test_orchestrator_lifecycle.py -q",
            "48 passed in 20.8s\nNo pending request leaks; stable admission order.",
            "success",
        ),
        (
            "m06",
            "supervisor",
            "model",
            98,
            11,
            "Review worker findings and artifacts",
            "",
            "Both workers agree on stable priority; review the diff before final validation.",
            "success",
        ),
        (
            "a10",
            "supervisor",
            "review",
            110,
            5,
            "Inspect the combined diff",
            "git diff --check && git diff --stat",
            "agency/orchestrator/orchestrator.py | 14 ++--\ntests/test_agorchestrator.py | 32 ++++\nDiff checks passed.",
            "success",
        ),
        (
            "a11",
            "supervisor",
            "test",
            116,
            22,
            "Validate integration",
            "pytest tests/test_orchestrator_lifecycle_edges.py -q",
            "17 passed in 21.5s\nIntegration validation passed.",
            "success",
        ),
    ]
    if contended:
        specifications.insert(
            12,
            (
                "extra",
                "scheduler",
                "read",
                65,
                8,
                "Recheck dispatch after delayed wakeup",
                "sed -n '920,1040p' agency/orchestrator/orchestrator.py",
                "Queue invariant still holds; dispatch is delayed by host CPU pressure.",
                "success",
            ),
        )
    actions = []
    previous = {}
    for (
        identifier,
        agent,
        kind,
        start,
        duration,
        intent,
        command,
        result,
        outcome,
    ) in specifications:
        if contended:
            start = start * 1.48 if start > 30 else start
            duration *= 1.65 if kind == "test" else 1.2
        context = None
        if kind == "model":
            base = [
                {
                    "id": f"system-{agent}",
                    "source": "System instructions",
                    "role": "system",
                    "chars": 3200,
                    "preview": "You are an Agency research agent. Inspect evidence, preserve scheduler semantics, and validate changes.",
                },
                {
                    "id": f"task-{agent}",
                    "source": "Original task"
                    if agent == "supervisor"
                    else "Supervisor instructions",
                    "role": "user",
                    "chars": 1900 if agent == "supervisor" else 780,
                    "preview": "Repair starvation when dependency-ready requests compete with newly submitted work."
                    if agent == "supervisor"
                    else "Investigate queue fairness. Preserve original submission order and return evidence. You have the task brief and scheduler files; the supervisor retains experiment constraints.",
                },
                {
                    "id": f"results-{identifier}",
                    "source": "Tool results",
                    "role": "tool",
                    "chars": 2600 + start * 46,
                    "preview": result,
                },
                {
                    "id": f"file-{agent}",
                    "source": "Files",
                    "role": "tool",
                    "chars": 6800,
                    "preview": "agency/orchestrator/orchestrator.py — admission, dependency wakeup, and stable queue ordering.",
                },
            ]
            if identifier == "m06":
                base.append(
                    {
                        "id": "summary",
                        "source": "Summary",
                        "role": "assistant",
                        "chars": 1400,
                        "preview": "Scheduler worker: preserved stable priority. Validation worker: regression fails before patch, 48 tests pass after patch.",
                    }
                )
            for block in base:
                block["change"] = "repeated" if block["id"] in previous.get(agent, {}) else "added"
            dropped = [
                {**b, "change": "dropped"}
                for key, b in previous.get(agent, {}).items()
                if key not in {b["id"] for b in base}
            ]
            previous[agent] = {b["id"]: b for b in base}
            context = {
                "blocks": base,
                "dropped": dropped,
                "chars": sum(b["chars"] for b in base),
                "source": "synthetic",
                "truncated": False,
            }
        actions.append(
            {
                "id": identifier,
                "agent": agent,
                "kind": kind,
                "name": "llm:attempt[0]"
                if kind == "model"
                else "Bash"
                if kind != "edit"
                else "apply_patch",
                "intent": intent,
                "start": start,
                "duration": duration,
                "outcome": outcome,
                "command": command,
                "result": result,
                "context": context,
                "files": ["agency/orchestrator/orchestrator.py"]
                if agent == "scheduler"
                else ["tests/test_agorchestrator.py"]
                if agent == "tests"
                else [],
                "metadata": {
                    "timing": "exact",
                    "provenance": "synthetic",
                    "model": "gpt-5.6-luna",
                    "cpu_ms": round(duration * (760 if kind == "test" else 35), 2),
                },
                "tokens": int(context["chars"] / 4) if context else None,
                "output_tokens": 340 if context else None,
                "source": "synthetic",
                "timing": "exact",
            }
        )
    actions.sort(key=lambda a: a["start"])
    episodes = group_episodes(actions)
    intervals = [
        {
            "id": a["id"],
            "agent": a["agent"],
            "start": a["start"],
            "duration": a["duration"],
            "kind": "model" if a["kind"] == "model" else "cpu" if a["kind"] == "test" else "tool",
            "label": a["intent"],
            "action": a["id"],
            "source": "synthetic",
        }
        for a in actions
    ]
    intervals.extend(
        [
            {
                "id": "wait-supervisor",
                "agent": "supervisor",
                "start": 12,
                "duration": 83 * (1.48 if contended else 1),
                "kind": "dependency",
                "label": "Waiting for scheduler and validation workers",
                "action": "a09",
                "source": "synthetic",
            },
            {
                "id": "queue-tests",
                "agent": "tests",
                "start": 51,
                "duration": 14 * (1.65 if contended else 1),
                "kind": "queue",
                "label": "Validation queued behind active engine capacity",
                "action": "a09",
                "source": "synthetic",
            },
        ]
    )
    edges = [
        {
            "from": "supervisor",
            "to": "scheduler",
            "kind": "delegation",
            "label": "Investigate queue fairness",
            "time": 11,
            "action": "a02",
            "source": "synthetic",
        },
        {
            "from": "supervisor",
            "to": "tests",
            "kind": "delegation",
            "label": "Create a regression",
            "time": 11,
            "action": "a03",
            "source": "synthetic",
        },
        {
            "from": "scheduler",
            "to": "tests",
            "kind": "handoff",
            "label": "Patch ready; rerun tests",
            "time": 65,
            "action": "m05",
            "source": "synthetic",
        },
        {
            "from": "tests",
            "to": "supervisor",
            "kind": "message",
            "label": "48 tests passed; integration remains",
            "time": 95,
            "action": "m06",
            "source": "synthetic",
        },
    ]
    obligations = [
        ("Locate scheduler admission", "completed", ["a01", "a02"]),
        ("Identify and reproduce starvation", "completed", ["m02", "a06"]),
        ("Preserve stable submission ordering", "completed", ["a05", "a08"]),
        ("Validate fairness and lifecycle", "completed", ["a09", "a11"]),
        ("Review combined artifacts", "completed", ["a10"]),
        ("Validate GPU contention behavior", "unvalidated", []),
    ]
    counters = {
        "workload_total cpu %": [
            [t, round(155 + 34 * math_sin(t) if contended else 22 + 65 * abs(math_sin(t)), 1)]
            for t in range(0, int(entry["duration"]), 2)
        ],
        "workload_total memory_mb": [
            [t, round(280 + t * 0.6 + 20 * math_sin(t), 1)]
            for t in range(0, int(entry["duration"]), 2)
        ],
    }
    return {
        **entry,
        "agents": agents,
        "actions": actions,
        "episodes": episodes,
        "intervals": intervals,
        "edges": edges,
        "counters": counters,
        "obligations": [
            {
                "id": f"obligation-{n}",
                "title": title,
                "status": status,
                "evidence": evidence,
                "inferred": False,
                "note": "Synthetic workflow obligation with linked evidence."
                if evidence
                else "No GPU experiment was run; requires independent validation.",
            }
            for n, (title, status, evidence) in enumerate(obligations)
        ],
        "summary": {},
        "coverage": {
            "events": 284,
            "spans": 76,
            "context": "Synthetic prompt composition and reported token usage.",
            "relationships": "Synthetic delegation, messages and dependency relationships.",
        },
        "handoff": {
            "from": "supervisor",
            "to": "scheduler",
            "transferred": [
                "Task brief: preserve submission order",
                "Scheduler file excerpt",
                "Expected output: patch + evidence",
            ],
            "retained": [
                "Full prior experiment transcript",
                "GPU validation requirements",
                "Other worker findings",
            ],
            "source": "synthetic",
        },
    }


def math_sin(t):
    import math

    return math.sin(t / 13)
