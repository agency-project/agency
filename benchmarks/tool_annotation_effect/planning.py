"""Offline, task-blocked assignment and immutable experiment manifests."""

from __future__ import annotations

import importlib.metadata
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .common import ROOT, digest, immutable_json, native, read_json, reject_credentials, source_hash
from .fixtures import local_tasks, CorpusTools

SUITES = ("swebench", "terminalbench", "rag", "migration", "tandem")
HARBOR_VERSION = "0.23.0"
TERMINAL_RELEASE = "terminal-bench@2.0"


def validate(config):
    reject_credentials(config)
    harness = config.get("harness", "native")
    if harness not in ("native", "codex"):
        raise ValueError("Supported experiment harnesses are native and codex")
    if harness == "codex" and set(config.get("suites", {})) - {"swebench", "terminalbench"}:
        raise ValueError("Codex experiments support SWE-bench and Terminal Bench")
    profile = config.get("profile", "pilot")
    if profile not in ("pilot", "expanded", "custom"):
        raise ValueError("profile must be pilot, expanded or custom")
    repetitions = config.get("repetitions", 3)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be positive")
    if profile == "pilot" and repetitions != 3:
        raise ValueError("Pilot requires three repetitions")
    if profile == "expanded" and repetitions < 5:
        raise ValueError("Expanded requires at least five repetitions")
    models = config.get("models", [])
    if not models or (profile == "pilot" and len(models) != 1):
        raise ValueError(
            "Configure one explicit model for pilot; expanded models are kept separate"
        )
    names = [model.get("id") for model in models]
    if None in names or len(set(names)) != len(names):
        raise ValueError("Model IDs must be present and unique")
    for model in models:
        if not model.get("model") or model.get("provider") in (None, "mock", "replay"):
            raise ValueError("Independent model/provider must be explicit; replay is test-only")
        if not model.get("base_url_env") or not model.get("api_key_env"):
            raise ValueError("Configure base_url_env and api_key_env names")
    budgets = config.get("budgets", {})
    if budgets.get("max_steps", 0) < 1 or budgets.get("timeout_s", 0) < 1:
        raise ValueError("budgets.max_steps and timeout_s must be positive")
    if "context_limit" not in config:
        raise ValueError("Declare context_limit (null means unknown)")
    policy = "codex-default" if harness == "codex" else "native-v1"
    if config.get("compaction_policy", policy) != policy:
        raise ValueError(f"{harness} harness requires {policy} compaction")
    if config.get("contention", False):
        raise ValueError(
            "Primary design uses no intentional contention; factorial extension is future work"
        )
    if not config.get("cache_policy"):
        raise ValueError("Declare provider cache policy; cache usage may remain unknown")
    if config.get("roles", "both") not in ("both", "supervisor", "worker"):
        raise ValueError("roles must be both, supervisor or worker")
    for suite in config.get("suites", {}):
        if suite not in SUITES:
            raise ValueError(f"Unknown suite {suite}")
    if not config.get("suites"):
        raise ValueError("At least one suite is required")


def versions():
    result = {"python": sys.version.split()[0]}
    for name in ("agency", "harbor", "swebench", "mcp", "httpx"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def prerequisites(config):
    report = {
        "versions": versions(),
        "executables": {
            name: shutil.which(name)
            for name in ("docker", "podman", "cc", "rustc", "harbor", "ssh", "rsync")
        },
        "issues": [],
    }
    for suite, settings in config["suites"].items():
        if suite in ("swebench", "terminalbench") and not settings.get("tasks_file"):
            report["issues"].append(
                f"{suite}: prepare and supply a local tasks_file before planning"
            )
        if suite == "terminalbench" and report["versions"]["harbor"] != HARBOR_VERSION:
            report["issues"].append(
                f"Install optional harbor=={HARBOR_VERSION} before live execution"
            )
        if suite == "swebench" and report["versions"]["swebench"] is None:
            report["issues"].append("Install optional swebench for official evaluation")
    return report


def schemas_for(suite):
    schemas = list(native("tools").BUILTIN_TOOL_SCHEMAS.values())
    if suite == "rag":
        schemas += CorpusTools({}).schemas()
    if suite == "tandem":
        from .execution import tandem_schemas

        schemas += tandem_schemas()
    # Agent/skill's default host tools are serialized offline, without running agents.
    if suite != "terminalbench":
        from agency import agskill

        defaults = agskill("manifest", "").host_mcp_tools
        schemas += [tool.to_openai_tool() for tool in defaults]
    return schemas


def build_plan(config):
    validate(config)
    annotation = native("annotations")
    tasks = []
    sampling = random.Random(config.get("task_sampling_seed", 0))
    for suite, settings in config["suites"].items():
        if settings.get("tasks_file"):
            candidates = read_json(settings["tasks_file"])
        elif suite in ("rag", "migration", "tandem"):
            candidates = local_tasks(suite)
        else:
            if not settings.get("tasks_file"):
                raise ValueError(
                    f"{suite}: use prepare, then set tasks_file; planning never downloads data"
                )
            candidates = read_json(settings["tasks_file"])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{suite}: tasks_file must contain a nonempty JSON task array")
        by_id = {task["id"]: task for task in candidates}
        if len(by_id) != len(candidates):
            raise ValueError(f"{suite}: duplicate task IDs")
        selected_ids = settings.get("task_ids")
        if selected_ids is not None:
            candidates = [by_id[name] for name in selected_ids]
        count = settings.get(
            "sample_size", 5 if config.get("profile", "pilot") == "pilot" else len(candidates)
        )
        if config.get("profile", "pilot") == "pilot" and count != 5:
            raise ValueError("Pilot requires five tasks per included suite")
        if config.get("profile") == "expanded" and count <= 5:
            raise ValueError("Expanded samples must contain more than five tasks per suite")
        if count > len(candidates):
            raise ValueError(f"{suite}: requested {count} tasks, only {len(candidates)} supplied")
        chosen = sampling.sample(sorted(candidates, key=lambda task: task["id"]), count)
        for task in chosen:
            reject_credentials(task)
            tasks.append(
                {
                    "suite": suite,
                    "id": task["id"],
                    "stratum": task.get("stratum", "unknown"),
                    "hash": digest(task),
                    "data": task,
                }
            )
    schema_manifest = {}
    for suite in config["suites"]:
        # Codex discovers its own tools at runtime. Record those actual schemas
        # at the gateway rather than mislabeling native tool schemas as Codex's.
        base = [] if config.get("harness") == "codex" else schemas_for(suite)
        schema_manifest[suite] = {
            arm: annotation.augment_schemas(base, arm) for arm in annotation.ARMS
        }
    randomizer = random.Random(config.get("seed", 0))
    schedule = []
    for task in tasks:
        for model in config["models"]:
            for repetition in range(config.get("repetitions", 3)):
                block = {
                    "suite": task["suite"],
                    "task_id": task["id"],
                    "model_id": model["id"],
                    "repetition": repetition,
                }
                order = list(annotation.ARMS)
                randomizer.shuffle(order)
                for position, arm in enumerate(order):
                    trial = {
                        **block,
                        "arm": arm,
                        "position": position,
                        "block_id": digest(block)[:20],
                        "task_hash": task["hash"],
                    }
                    trial["trial_id"] = digest(trial)[:24]
                    schedule.append(trial)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return {
        "format_version": 1,
        "config": config,
        "tasks": tasks,
        "schedule": schedule,
        "instructions": {arm: annotation.instruction(arm) for arm in annotation.ARMS},
        "schemas": schema_manifest,
        "prerequisites": prerequisites(config),
        "repository_revision": revision.stdout.strip(),
        "implementation_hash": source_hash(),
        "repository_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
            ).stdout
        ),
        "run_counts": {
            "top_level_trials": len(schedule),
            "nominal_agent_executions": sum(3 if t["suite"] == "tandem" else 1 for t in schedule),
        },
        "annotation_version": annotation.VERSION,
        "terminal_release": TERMINAL_RELEASE,
        "harbor_version": HARBOR_VERSION,
    }


def save_plan(config, directory):
    manifest = build_plan(config)
    manifest["manifest_hash"] = digest(manifest)
    immutable_json(Path(directory) / "manifest.json", manifest)
    return manifest


def load_plan(directory):
    manifest = read_json(Path(directory) / "manifest.json")
    expected = manifest.pop("manifest_hash")
    if digest(manifest) != expected:
        raise ValueError("Manifest was modified after assignment")
    manifest["manifest_hash"] = expected
    return manifest
