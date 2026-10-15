"""Explicitly synthetic edge cases; never supplement a recorded execution."""

ENTRY = {
    "id": "trajectory-scenarios",
    "title": "Trajectory edge cases",
    "subtitle": "Synthetic · input, repeated checks, incomplete call",
    "source": "synthetic",
    "model": "No model invoked",
    "harness": "Synthetic fixture",
    "duration": 90,
}

SCALE_ENTRY = {
    "id": "trajectory-scale",
    "title": "Trajectory scale check",
    "subtitle": "Synthetic · 5,000 calls across five actors",
    "source": "synthetic",
    "model": "No model invoked",
    "harness": "Synthetic fixture",
    "duration": 50,
}


def scale_events():
    rows = [
        {
            "id": "scale/start",
            "ts": 0,
            "type": "workload_started",
            "actor": "workflow",
            "payload": {"task": "Synthetic rendering stress check; no tools or agents executed."},
            "source": "synthetic_fixture",
        }
    ]
    for index in range(5000):
        for boundary in ("call", "result"):
            rows.append(
                {
                    "id": f"scale/{index}/{boundary}",
                    "ts": index / 100 + (0.005 if boundary == "result" else 0),
                    "type": f"tool_{boundary}",
                    "actor": f"fixture-{index % 5}",
                    "payload": {
                        "call_id": str(index),
                        "tool": "read_file",
                        "arguments": {"path": f"fixture-{index}.txt"},
                        **(
                            {"result": f"Synthetic file content {index}"}
                            if boundary == "result"
                            else {}
                        ),
                    },
                    "source": "synthetic_fixture",
                }
            )
    rows.append(
        {
            "id": "scale/end",
            "ts": 50,
            "type": "done",
            "actor": "workflow",
            "payload": {"status": "completed"},
            "source": "synthetic_fixture",
        }
    )
    return rows


def events():
    rows = []

    def add(time, kind, actor="supervisor", **payload):
        rows.append(
            {
                "id": f"synthetic/{len(rows)}",
                "ts": time,
                "type": kind,
                "actor": actor,
                "payload": payload,
                "source": "synthetic_fixture",
            }
        )

    add(0, "workload_started", task="Check a retry fix")
    add(
        1,
        "tool_call",
        call_id="read",
        tool="read_file",
        arguments={"path": "retry.py"},
        intent="Investigating retry behavior",
    )
    add(
        3,
        "tool_result",
        call_id="read",
        tool="read_file",
        result="Recorded synthetic file content: retry limit = 2",
    )
    add(
        4,
        "delegation",
        to="worker",
        task="Reproduce the target retry assertion; report the observed result",
    )
    for index in range(3):
        add(
            5 + index * 4,
            "tool_call",
            "worker",
            call_id=f"check{index}",
            tool="Bash",
            arguments={"command": "pytest test_retry.py"},
        )
        add(
            8 + index * 4,
            "tool_result",
            "worker",
            call_id=f"check{index}",
            tool="Bash",
            arguments={"command": "pytest test_retry.py"},
            result={"exit_code": 1, "stdout": "AssertionError: expected 3 retries"},
        )
    add(18, "input_required", message="Which retry limit is expected: 2 or 3?")
    add(30, "input_resolved", message="Use the documented limit of 3")
    add(
        31,
        "tool_call",
        "worker",
        call_id="edit",
        tool="apply_patch",
        arguments={"path": "retry.py"},
    )
    add(
        33,
        "tool_result",
        "worker",
        call_id="edit",
        tool="apply_patch",
        result="Patch applied",
        artifacts=[{"path": "retry.py", "change": "modified", "source": "synthetic_fixture"}],
    )
    add(
        35,
        "tool_call",
        "worker",
        call_id="long",
        tool="Bash",
        arguments={"command": "pytest test_retry.py"},
        intent="Checking the revised retry behavior",
    )
    add(
        75,
        "tool_result",
        "worker",
        call_id="long",
        tool="Bash",
        result={"exit_code": 0, "stdout": "1 passed"},
    )
    add(
        76,
        "handoff",
        "worker",
        to="supervisor",
        message="The target check passed; broader behavior is unverified",
    )
    add(
        80, "tool_call", call_id="missing", tool="Bash", arguments={"command": "pytest integration"}
    )
    add(90, "done", status="cancelled")
    return rows


MULTI_AGENT_ENTRIES = [
    {
        "id": f"trajectory-agents-{count}",
        "title": f"{count}-agent workspace example",
        "subtitle": "Synthetic · concurrent calls, recorded failures, "
        + ("explicit parent metadata" if count == 10 else "no parent metadata"),
        "source": "synthetic",
        "model": "Example model identity; no model invoked",
        "harness": "Synthetic fixture",
        "duration": 60,
        "agent_count": count,
    }
    for count in (10, 32)
]


def multi_agent_events(count, *, hierarchy=False):
    """Author deterministic recorded evidence only; never execute fixture commands."""
    if count not in (3, 10, 32):
        raise ValueError("Supported synthetic roster sizes are 3, 10, and 32.")
    rows = []

    def add(time, kind, actor="workflow", *, call_label=None, **payload):
        row = {
            "id": f"agents-{count}/{len(rows):04}",
            "ts": time,
            "type": kind,
            "actor": actor,
            "payload": payload,
            "source": "synthetic_fixture",
        }
        if call_label is not None:
            row["call_label"] = call_label
        rows.append(row)

    add(
        0,
        "workload_started",
        task=f"Synthetic {count}-agent workspace example; no agents, commands, or models executed.",
        model="Example model identity; no model invoked",
        harness="Synthetic fixture",
    )
    actors = ["coordinator", *(f"worker-{index:02}" for index in range(1, count))]
    for index, actor in enumerate(actors):
        metadata = {"harness": "Synthetic fixture"}
        if hierarchy and index:
            metadata["parent_agname"] = "coordinator" if index < 5 else "worker-01"
        add(0.1 + index * 0.001, "agent_registered", actor, **metadata)
        add(0.2 + index * 0.001, "agent_state", actor, state="running_harness")
    add(
        1,
        "agent_state",
        "coordinator",
        call_label="example-response",
        state="waiting_llm",
        model="Example model identity; no model invoked",
    )
    add(
        2,
        "model_result",
        "coordinator",
        call_id="example-response",
        model="Example model identity; no model invoked",
        result="Synthetic response requesting two independent tools. No model was invoked.",
        tool_call_ids=["example-tool-1", "example-tool-2"],
        tokens=32,
        output_tokens=14,
    )
    for index, actor in enumerate(actors):
        start = 3 + index * 0.05
        correlation = {"model_tool_call_id": "example-tool-1"} if index == 0 else {}
        add(
            start,
            "tool_call",
            actor,
            call_id="inspect",
            tool="read_file",
            arguments={"path": f"fixture/worker-{index:02}.txt"},
            intent="Inspect the recorded fixture input",
            **correlation,
        )
        correlation = {"model_tool_call_id": "example-tool-2"} if index == 0 else {}
        add(
            start + 0.25,
            "tool_call",
            actor,
            call_id="check",
            tool="Bash",
            arguments={"command": f"pytest fixture/test_worker_{index:02}.py"},
            intent="Inspect an authored check result; this command was not executed",
            **correlation,
        )
        failed = index in (2, count - 1)
        add(
            start + 2,
            "tool_result",
            actor,
            call_id="check",
            tool="Bash",
            result={
                "exit_code": 1 if failed else 0,
                "stdout": "AssertionError: synthetic expected value 3, observed 2"
                if failed
                else "Synthetic fixture: 1 passed; no command was executed.",
            },
        )
        output = f"Synthetic fixture input for worker {index:02}; no file was opened."
        if index == 1:
            output = "\n".join(
                f"Synthetic captured line {line:04}: bounded output viewer check."
                for line in range(2000)
            )
            output += "\n<script>This is untrusted recorded text, not executable HTML.</script>"
        add(start + 20, "tool_result", actor, call_id="inspect", tool="read_file", result=output)
        add(28 + index * 0.01, "agent_state", actor, state="completed")
    # A missing boundary stays incomplete. The terminal workload boundary must
    # not convert this authored gap into a successful call.
    add(
        30,
        "tool_call",
        actors[-1],
        call_id="unfinished",
        tool="Bash",
        arguments={"command": "inspect synthetic missing completion"},
        intent="Example of an unavailable result boundary",
    )
    add(60, "done", status="completed")
    return sorted(rows, key=lambda row: (row["ts"], row["id"]))
