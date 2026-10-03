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
