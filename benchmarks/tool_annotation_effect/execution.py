"""Current Agent/skill execution and local fixture verification.

This module is import-safe. All environments and models are created inside
execute(), called only by the explicit run command.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .common import ROOT, read_events
from .fixtures import CorpusTools, differential_inputs, rag_prompt
from .adapters import SweBenchAdapter, TerminalBenchAdapter

TRACE_PATH = "/var/run/agency_logs/experiment-events.jsonl"


def tandem_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "start_worker",
                "description": "Start an independent ledger worker.",
                "parameters": {
                    "type": "object",
                    "properties": {"role": {"type": "string", "enum": ["left", "right"]}},
                    "required": ["role"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "collect_worker",
                "description": "Collect a completed worker result; returns pending otherwise.",
                "parameters": {
                    "type": "object",
                    "properties": {"role": {"type": "string", "enum": ["left", "right"]}},
                    "required": ["role"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_ledger",
                "description": "Check a proposed shared ledger artifact.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "integer"},
                        "right": {"type": "integer"},
                        "total": {"type": "integer"},
                    },
                    "required": ["left", "right", "total"],
                },
            },
        },
    ]


def trace_metrics(events):
    exchanges = [event for event in events if event["kind"] == "model_exchange"]
    dispatches = [event for event in events if event["kind"] == "model_dispatch_complete"]
    # Dispatch completion includes compaction calls, while model_exchange contains
    # main turns only. Use dispatch telemetry where available; never double count.
    usages = [event.get("usage") for event in dispatches]
    if not dispatches:
        usages = [event["response"].get("usage") for event in exchanges]
    missing_input = sum(
        not isinstance(usage, dict) or usage.get("prompt_tokens") is None for usage in usages
    )
    missing_output = sum(
        not isinstance(usage, dict) or usage.get("completion_tokens") is None for usage in usages
    )
    annotations = [event["annotation"] for event in events if event["kind"] == "tool_annotation"]
    tokens = {
        "input_tokens": None
        if missing_input or not usages
        else sum(u["prompt_tokens"] for u in usages),
        "output_tokens": None
        if missing_output or not usages
        else sum(u["completion_tokens"] for u in usages),
    }
    workstreams = []
    for annotation in annotations:
        metadata = annotation.get("raw")
        if annotation["status"] == "valid" and isinstance(metadata, dict):
            workstreams.extend(metadata.get("workstream_ids", []))
    return {
        **tokens,
        "missing_input_usage": missing_input,
        "missing_output_usage": missing_output,
        "model_calls": len(dispatches) if dispatches else len(exchanges),
        "model_attempts": sum(event["kind"] == "model_attempt" for event in events),
        "tool_calls": len(annotations),
        "tool_errors": sum(
            event["kind"] == "tool_result" and event["category"] != "ok" for event in events
        ),
        "annotation_valid": sum(a["status"] == "valid" for a in annotations),
        "annotation_missing": sum(a["status"] == "missing" for a in annotations),
        "annotation_malformed": sum(a["status"].startswith("malformed") for a in annotations),
        "annotation_characters": sum(
            len(json.dumps(a["raw"], ensure_ascii=False))
            for a in annotations
            if a["raw"] is not None
        ),
        "workstream_memberships": len(workstreams),
        "workstream_unique": len(set(workstreams)),
        "workstream_reuses": len(workstreams) - len(set(workstreams)),
        "compactions": sum(event["kind"] == "compaction" for event in events),
        "truncations": sum(event.get("finish_reason") == "length" for event in dispatches),
        "cache_tokens": None
        if not usages
        or any(
            not isinstance(u, dict)
            or (u.get("prompt_tokens_details") or {}).get("cached_tokens") is None
            for u in usages
        )
        else sum(u["prompt_tokens_details"]["cached_tokens"] for u in usages),
    }


def agent_config(config, model, trial, directory, role):
    from agency.configs.agconfig import (
        agconfig,
        llmconfig,
        agentconfig,
        skillconfig,
        sandboxconfig,
        orchestratorconfig,
        resourcesconfig,
        dataloggerconfig,
    )

    settings = model.get("settings", {})
    return agconfig(
        llmconfig(
            provider=model["provider"],
            model=model["model"],
            base_url=os.environ[model["base_url_env"]],
            api_key=os.environ[model["api_key_env"]],
            context_limit=config["context_limit"],
            **settings,
        ),
        agentconfig(
            log_dir=str(directory / role / "logs"),
            output_dir=str(directory / role / "output"),
            harness="native",
            annotation_arm=role_arm(config, trial["arm"], role),
            native_trace_file=TRACE_PATH,
            experiment_run_id=trial["trial_id"],
            experiment_agent_id=role,
        ),
        skillconfig(react_max_steps=config["budgets"]["max_steps"]),
        sandboxconfig(
            backend=config.get("sandbox_backend", "auto"),
            base_image=config.get("base_image", "docker.io/library/python:3.12-slim"),
            gpu_passthrough=False,
            mounts={"_agency_package": (config["runtime_source"], "/opt/agency_pkg", "ro")}
            if config.get("runtime_source")
            else {},
        ),
        dataloggerconfig(db_path=str(directory / role / "logs" / "agent.sqlite")),
        orchestratorconfig(max_concurrent_engines=3 if trial["suite"] == "tandem" else 1),
        resourcesconfig(
            idle_cpus=config.get("resource_limits", {}).get("cpus", 8),
            idle_memory=config.get("resource_limits", {}).get("memory"),
        ),
    )


def role_arm(config, arm, role):
    roles = config.get("roles", "both")
    if roles == "supervisor" and role != "supervisor":
        return "baseline"
    if roles == "worker" and role == "supervisor":
        return "baseline"
    return arm


def host_tool(schema, handler):
    from agency import agdata, agtool

    function = schema["function"]
    return agtool(
        function["name"],
        function["description"],
        lambda arguments: agdata(**handler(arguments.to_dict())),
        params=function["parameters"],
    )


def verify_migration(task, directory, command_runner=None):
    """Compilation and behavior are reported separately; tests run on the host evaluator.

    Use a disposable evaluator host for untrusted artifacts in live benchmarks.
    """
    import subprocess
    import shutil

    run = command_runner or subprocess.run
    for executable in ("cc", "rustc"):
        if not shutil.which(executable):
            return {
                "success": None,
                "failure": "infrastructure",
                "error": f"Install {executable} on evaluator host",
            }
    directory = Path(directory)
    (directory / "reference.c").write_text(task["source"])
    c = run(
        [
            "cc",
            "-std=c11",
            "-O0",
            str(directory / "reference.c"),
            "-o",
            str(directory / "reference"),
        ],
        capture_output=True,
        text=True,
    )
    if c.returncode:
        return {
            "success": None,
            "failure": "infrastructure",
            "c_compilation": False,
            "error": c.stderr,
        }
    rust = run(
        [
            "rustc",
            "--edition=2021",
            str(directory / "solution.rs"),
            "-o",
            str(directory / "solution"),
        ],
        capture_output=True,
        text=True,
    )
    if rust.returncode:
        return {
            "success": False,
            "failure": "task",
            "rust_compilation": False,
            "behavioral": None,
            "error": rust.stderr,
        }
    mismatches = []
    cases = differential_inputs(task, task.get("test_seed", 0))
    for value in cases:
        try:
            c_result = run(
                [str(directory / "reference")],
                input=f"{value}\n",
                capture_output=True,
                text=True,
                timeout=2,
            )
            r_result = run(
                [str(directory / "solution")],
                input=f"{value}\n",
                capture_output=True,
                text=True,
                timeout=2,
            )
            matches = (
                c_result.returncode == r_result.returncode == 0
                and c_result.stdout == r_result.stdout
            )
        except subprocess.TimeoutExpired:
            matches = False
        if not matches:
            mismatches.append(value)
    return {
        "success": not mismatches,
        "failure": "task" if mismatches else None,
        "rust_compilation": True,
        "behavioral": not mismatches,
        "cases": len(cases),
        "mismatches": mismatches,
    }


def execute(task, trial, model, config, directory):
    if trial["suite"] == "terminalbench":
        return _execute(task, trial, model, config, directory)
    # Agency normally mounts the repository root. A benchmark must not expose
    # host manifests, evaluator gold or .env through that otherwise useful mount.
    with tempfile.TemporaryDirectory(prefix="agency-experiment-runtime-") as runtime:
        shutil.copytree(
            ROOT / "agency",
            Path(runtime) / "agency",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return _execute(task, trial, model, {**config, "runtime_source": runtime}, directory)


def _execute(task, trial, model, config, directory):
    from agency import Agent, agdata, agskill

    directory = Path(directory)
    if trial["suite"] == "terminalbench":
        started = time.monotonic()
        report = TerminalBenchAdapter().launch(task, trial, model, config, directory)
        event_files = sorted((directory / "harbor").rglob("events.jsonl"))
        events = [event for path in event_files for event in read_events(path)]
        native_files = sorted((directory / "harbor").rglob("native-result.json"))
        native_result = json.loads(native_files[0].read_text()) if len(native_files) == 1 else {}
        treatment = [event for event in events if event["kind"] == "treatment"]
        verified = bool(treatment) and all(event["arm"] == trial["arm"] for event in treatment)
        timings = {}
        for phase in ("environment_setup", "agent_setup", "agent_execution"):
            timing = report.get(phase) or {}
            if timing.get("started_at") and timing.get("finished_at"):
                timings[phase] = (
                    datetime.fromisoformat(timing["finished_at"])
                    - datetime.fromisoformat(timing["started_at"])
                ).total_seconds()
        return {
            "events": events,
            "metrics": trace_metrics(events),
            "agent_executions": 1,
            "harbor_report": report,
            "end_to_end_seconds": time.monotonic() - started,
            "agent_seconds": native_result.get("agent_seconds", timings.get("agent_execution")),
            "setup_seconds": sum(timings[phase] for phase in ("environment_setup", "agent_setup"))
            if all(phase in timings for phase in ("environment_setup", "agent_setup"))
            else None,
            "final_text": native_result.get("final_text", ""),
            "failure": (
                "budget"
                if "exceeded max_steps=" in native_result.get("error", "")
                or "dispatch deadline exhausted" in native_result.get("error", "")
                or (report.get("exception_info") or {}).get("exception_type") == "AgentTimeoutError"
                else None
            )
            if verified
            else "infrastructure",
        }
    agents = {}
    handles = {}
    started = time.monotonic()
    setup_started = started
    parent_role = "supervisor" if trial["suite"] == "tandem" else "agent"
    effective_config = {
        **config,
        "base_image": task.get(
            "image", config.get("base_image", "docker.io/library/python:3.12-slim")
        ),
    }
    cfg = agent_config(effective_config, model, trial, directory, parent_role)
    from .sandbox import EpisodeSandbox

    sandbox = EpisodeSandbox("annotation-" + trial["trial_id"], agconfig=cfg)
    parent = Agent(
        "annotation-" + trial["trial_id"], sandbox=sandbox, agconfig=cfg, harness="native"
    )
    agents[parent_role] = parent
    try:
        tools = []
        if trial["suite"] == "swebench":
            prompt = SweBenchAdapter().setup(task, sandbox)
        elif trial["suite"] == "rag":
            corpus = CorpusTools(task["corpus"])
            tools = [
                host_tool(schema, handler)
                for schema, handler in zip(corpus.schemas(), [corpus.retrieve, corpus.read])
            ]
            prompt = rag_prompt(task)
        elif trial["suite"] == "migration":
            sandbox.write_file("/workspace/reference.c", task["source"])
            prompt = (
                "Migrate reference.c to solution.rs. Read one integer from stdin; print the same integer result and newline. "
                f"Defined input domain: {task['domain']}. Use Rust edition 2021 and standard library only. Compilation alone is insufficient."
            )
        else:

            def start_worker(arguments):
                role = arguments["role"]
                if role not in ("left", "right"):
                    return {"error": "unknown role"}
                if role in handles:
                    return {"role": role, "started": True}
                worker_cfg = agent_config(config, model, trial, directory, role)
                worker = Agent(
                    "annotation-" + trial["trial_id"] + "-" + role,
                    sandbox=EpisodeSandbox(
                        "annotation-" + trial["trial_id"] + "-" + role,
                        agconfig=worker_cfg,
                    ),
                    agconfig=worker_cfg,
                    harness="native",
                )
                agents[role] = worker
                skill = agskill(
                    "ledger_worker",
                    'Sum the supplied values. Write JSON {"sum": integer} to /workspace/result.json.',
                    input_schema=agdata(values=list),
                    max_output_schema_retries=0,
                )
                handles[role] = worker.run(skill, agdata(values=task[role]))
                return {"role": role, "started": True}

            def collect_worker(arguments):
                role = arguments["role"]
                if role not in handles:
                    return {"error": "worker has not been started"}
                if handles[role].is_pending():
                    return {"role": role, "pending": True}
                handles[role].wait(timeout=1)
                try:
                    return {
                        "role": role,
                        "artifact": json.loads(
                            agents[role].sandbox.read_file("/workspace/result.json")
                        ),
                    }
                except (ValueError, OSError, RuntimeError) as error:
                    return {"error": str(error)}

            def validate_ledger(arguments):
                from .fixtures import evaluate_tandem

                return evaluate_tandem(task, json.dumps(arguments))

            tools = [
                host_tool(schema, handler)
                for schema, handler in zip(
                    tandem_schemas(), [start_worker, collect_worker, validate_ledger]
                )
            ]
            prompt = (
                "Start left and right workers, collect their independently computed sums, validate the shared ledger, "
                "then write /workspace/result.json containing left, right and total integer fields. "
                "The workers have their inputs; use their artifacts."
            )
    except BaseException:
        sandbox.stop()
        raise
    setup_seconds = time.monotonic() - setup_started
    execution_started = time.monotonic()
    execution_started_wall = time.time_ns()
    events = []
    handle = None
    try:
        skill = agskill(
            "annotation_task",
            config.get("system_prompt", "Complete the task using the available tools."),
            add_host_mcp_tools=tools,
            output_schema=agdata(answer=dict, evidence=list) if trial["suite"] == "rag" else None,
            max_output_schema_retries=0,
        )
        handle = parent.run(skill, agdata(task=prompt), max_steps=config["budgets"]["max_steps"])
        result = handle.wait(timeout=config["budgets"]["timeout_s"])
        failure = None
        result_data = result.to_dict()
        if "error" in result_data:
            failure = "budget" if "max_steps" in str(result_data["error"]) else "infrastructure"
        final_text = json.dumps(result_data)
        if trial["suite"] in ("migration", "tandem"):
            name = "solution.rs" if trial["suite"] == "migration" else "result.json"
            try:
                artifact = sandbox.read_file("/workspace/" + name)
                (directory / name).write_text(artifact)
                if trial["suite"] == "tandem":
                    final_text = artifact
            except (OSError, RuntimeError):
                failure = failure or "task"
        return_value = {
            "final_text": final_text,
            "failure": failure,
            "setup_seconds": setup_seconds,
            "agent_seconds": time.monotonic() - execution_started,
        }
    except TimeoutError:
        parent.cancel(handle)
        return_value = {
            "final_text": "",
            "failure": "budget",
            "setup_seconds": setup_seconds,
            "agent_seconds": time.monotonic() - execution_started,
        }
    finally:
        if handle is not None and handle.is_pending():
            parent.cancel(handle)
        for role, pending in handles.items():
            if pending.is_pending():
                agents[role].cancel(pending)
        for role, agent in agents.items():
            if agent.sandbox is not None:
                try:
                    host_trace = directory / role / "logs" / "experiment-events.jsonl"
                    raw = (
                        host_trace.read_bytes()
                        if host_trace.exists()
                        else agent.sandbox.read_file_bytes(TRACE_PATH)
                    )
                    trace_file = directory / role / "events.jsonl"
                    trace_file.parent.mkdir(parents=True, exist_ok=True)
                    trace_file.write_bytes(raw)
                    events.extend(read_events(trace_file))
                except (OSError, RuntimeError) as error:
                    events.append(
                        {"kind": "trace_unavailable", "agent_id": role, "error": str(error)}
                    )
                agent.sandbox.stop()
    # A wall-clock timeout still has a gradeable attempted patch. Collect it
    # after cancellation and teardown, just as for a normal or step-limited run.
    if trial["suite"] == "swebench":
        patch = SweBenchAdapter().extract_patch(sandbox)
        (directory / "prediction.patch").write_text(patch)
        SweBenchAdapter().export_prediction(
            task, patch, model["id"], directory / "prediction.jsonl"
        )
    events.sort(key=lambda event: event.get("timestamp_ns", 0))
    treatment_events = [event for event in events if event["kind"] == "treatment"]
    if not treatment_events or any(event["kind"] == "trace_unavailable" for event in events):
        return_value["failure"] = "infrastructure"
        return_value["instrumentation_error"] = (
            "Native trace missing; cannot verify assigned treatment"
        )
    else:
        for event in treatment_events:
            expected_arm = role_arm(config, trial["arm"], event["agent_id"])
            if event["arm"] != expected_arm:
                return_value["failure"] = "infrastructure"
                return_value["instrumentation_error"] = (
                    "Assigned treatment did not reach native harness"
                )
        first_native = min(event["timestamp_ns"] for event in treatment_events)
        last_native = max(event.get("timestamp_ns", first_native) for event in events)
        return_value["lifecycle_seconds"] = return_value["agent_seconds"]
        return_value["agent_seconds"] = (last_native - first_native) / 1e9
        return_value["setup_seconds"] += max(0, (first_native - execution_started_wall) / 1e9)
    return {
        **return_value,
        "events": events,
        "metrics": trace_metrics(events),
        "agent_executions": len(agents),
        "end_to_end_seconds": time.monotonic() - started,
    }
