"""A stand-in worker for validating the runner at 6, 12 and 24 agents.

It speaks the same file protocol as ``worker`` (ready.json, start.json,
agent-finished.json, collected.json, task.json) and uses the real shared API
budget, but starts no container and calls no model. Behaviour comes from
``job["mock_worker"]`` (with optional per-slot overrides under ``by_slot``):

  duration            seconds of simulated agent work
  requests            synthetic model requests admitted through the shared budget
  request_tokens      size of each synthetic prompt (tokens, approximate)
  model_seconds       simulated model latency per request
  tool_seconds        simulated tool time after each request
  fail_setup          exit non-zero before becoming ready
  die                 "setup" | "running": exit abruptly without a final record
  ignore_deadline     keep running past the task deadline (tests the backstop)
  ignore_sigterm      also ignore SIGTERM (tests the kill path)
  write_outside       try to write into another agent's directory (must be caught)
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

from .api_budget import BudgetPolicy, DeadlineReached, SharedBudget
from .util import append_jsonl, atomic_json, clock_anchor, read_json, read_json_or


class Interrupted(Exception):
    pass


def synthetic_body(tokens: int, model: str = "mock") -> bytes:
    text = "alpha " * max(1, tokens)
    return json.dumps(
        dict(model=model, max_completion_tokens=64, messages=[dict(role="user", content=text)])
    ).encode()


def run(job: dict) -> int:
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    behaviour = dict(job.get("mock_worker") or {})
    behaviour.update((behaviour.pop("by_slot", None) or {}).get(str(job["slot"]), {}))
    if job.get("tiktoken_cache"):
        os.environ["TIKTOKEN_CACHE_DIR"] = job["tiktoken_cache"]
    state = dict(
        schema_version=1,
        run_id=job["run_id"],
        slot=job["slot"],
        agent_id=job["agent_id"],
        task_id=job["task"]["task_id"],
        task_digest=job["task"].get("digest"),
        status="setup",
        completed=False,
        censored=False,
        pid=os.getpid(),
        clock=clock_anchor(),
        setup_start_ns=time.monotonic_ns(),
        mock=True,
        host_affinity=sorted(os.sched_getaffinity(0)),
    )

    def save():
        atomic_json(directory / "task.json", state)

    save()

    def interrupted(signum, _frame):
        raise Interrupted(f"signal {signum}")

    if behaviour.get("ignore_sigterm"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if behaviour.get("fail_setup"):
        state.update(status="setup_failed", error="injected setup failure", finished=True)
        save()
        return 3
    if behaviour.get("die") == "setup":
        os._exit(9)
    time.sleep(behaviour.get("setup_seconds", 0))
    # The agent's private writable state.
    workspace = directory / "workspace"
    workspace.mkdir()
    (workspace / "owner.txt").write_text(job["agent_id"])
    atomic_json(
        directory / "isolation.json",
        dict(
            verified=True,
            problems=[],
            host_affinity=state["host_affinity"],
            writable_mounts=[dict(source=str(workspace), destination="/workspace", rw=True)],
        ),
    )
    state.update(status="ready", ready_ns=time.monotonic_ns(), isolation_verified=True)
    state["setup_seconds"] = (state["ready_ns"] - state["setup_start_ns"]) / 1e9
    save()
    atomic_json(
        directory / "ready.json",
        dict(
            ready_ns=state["ready_ns"],
            container_name=f"mock-{job['agent_id']}",
            host_cgroup=None,
            container_cgroup=None,
        ),
    )
    measured_start = deadline_ns = None
    try:
        gate = directory / "start.json"
        limit = time.monotonic() + job["timeouts"]["setup_seconds"]
        while not gate.exists():
            if time.monotonic() > limit:
                raise RuntimeError("Runner never released this agent")
            time.sleep(0.01)
        release = read_json(gate)
        if release.get("cancel"):
            state.update(status="not_started", censored=True, error="run stopped before release")
            return 0
        measured_start = time.monotonic_ns()
        deadline_ns = measured_start + int(job["timeouts"]["task_seconds"] * 1e9)
        if release.get("run_deadline_ns"):
            deadline_ns = min(deadline_ns, int(release["run_deadline_ns"]))
        state.update(status="running", start_ns=measured_start, deadline_ns=deadline_ns)
        save()
        if behaviour.get("die") == "running":
            os._exit(9)
        if behaviour.get("write_outside"):
            # A real container cannot do this; the mock records the attempt so
            # the suite can prove the cross-agent check notices shared writes.
            target = Path(behaviour["write_outside"])
            state["wrote_outside"] = str(target)
            state["isolation_verified"] = False
            atomic_json(
                directory / "isolation.json",
                dict(
                    verified=False,
                    problems=["shared writable mount"],
                    writable_mounts=[dict(source=str(target), destination="/shared", rw=True)],
                ),
            )
        budget = SharedBudget(
            job["api"]["directory"],
            BudgetPolicy.from_dict(job["api"]["policy"]),
            deadline=lambda: None if behaviour.get("ignore_deadline") else deadline_ns / 1e9,
        )
        body = synthetic_body(behaviour.get("request_tokens", 200))
        end_of_work = time.monotonic() + behaviour.get("duration", 0.2)
        for number in range(behaviour.get("requests", 1)):
            request_id = f"{job['agent_id']}-{number}"
            ready = time.monotonic_ns()
            admission = budget.admit(body, request_id=request_id, agent_id=job["agent_id"])
            admitted = time.monotonic_ns()
            time.sleep(behaviour.get("model_seconds", 0.01))
            first = time.monotonic_ns()
            prompt = admission.raw_prompt_tokens
            settlement = budget.reconcile(
                request_id, prompt_tokens=prompt, completion_tokens=5, agent_id=job["agent_id"]
            )
            done = time.monotonic_ns()
            append_jsonl(
                directory / "api-requests.jsonl",
                dict(
                    kind="request",
                    agent_id=job["agent_id"],
                    request_id=request_id,
                    outcome="completed",
                    ready_ns=ready,
                    admitted_ns=admitted,
                    first_chunk_ns=first,
                    completed_ns=done,
                    api_wait_seconds=(admitted - ready) / 1e9,
                    response_latency_seconds=(first - admitted) / 1e9,
                    time_to_first_chunk_seconds=(first - admitted) / 1e9,
                    stream_seconds=(done - first) / 1e9,
                    attempts=1,
                    retries=0,
                    http_429=0,
                    estimated_prompt_tokens=admission.estimated_prompt_tokens,
                    reserved_completion_tokens=admission.reserved_completion_tokens,
                    admission_charge=admission.charge,
                    reported_prompt_tokens=prompt,
                    reported_completion_tokens=5,
                    reported_cached_tokens=0,
                    reported_total_tokens=prompt + 5,
                    reconciliation=settlement,
                ),
            )
            time.sleep(behaviour.get("tool_seconds", 0))
        while time.monotonic() < end_of_work or behaviour.get("ignore_deadline"):
            if not behaviour.get("ignore_deadline") and time.monotonic_ns() >= deadline_ns:
                break
            time.sleep(0.01)
        end = time.monotonic_ns()
        if end >= deadline_ns:
            state.update(status="timeout", censored=True, end_ns=end)
        else:
            state.update(status="completed", completed=True, end_ns=end)
    except DeadlineReached:
        state.update(status="timeout", censored=True, end_ns=time.monotonic_ns())
    except Interrupted as exc:
        state.update(
            status="cancelled" if measured_start else "setup_interrupted",
            censored=measured_start is not None,
            error=str(exc),
            end_ns=time.monotonic_ns(),
        )
    except Exception as exc:
        state.update(
            status="agent_error" if measured_start else "setup_failed",
            error=f"{type(exc).__name__}: {exc}",
            end_ns=time.monotonic_ns(),
        )
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if measured_start and state.get("end_ns"):
            state["wall_seconds"] = (state["end_ns"] - measured_start) / 1e9
            save()
            atomic_json(
                directory / "agent-finished.json",
                dict(
                    end_ns=state["end_ns"],
                    status=state["status"],
                    container_name=f"mock-{job['agent_id']}",
                ),
            )
            limit = time.monotonic() + job["timeouts"]["collect_seconds"]
            while not (directory / "collected.json").exists() and time.monotonic() < limit:
                time.sleep(0.01)
        state["cleanup_end_ns"] = time.monotonic_ns()
        state["finished"] = True
        save()
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job = read_json_or(argv[0]) if argv else None
    if not isinstance(job, dict):
        print("Unreadable job file", file=sys.stderr)
        return 4
    return run(job)


if __name__ == "__main__":
    raise SystemExit(main())
