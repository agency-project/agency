"""Frozen configuration, assignments, and artifact helpers (no model imports)."""

import hashlib
import json
import math
import os
import re
from pathlib import Path

from benchmarks.tool_annotation_effect.common import atomic_json, digest, reject_credentials

DATASET = "princeton-nlp/SWE-bench_Verified"
ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "condition_timeout_seconds": 1800,
    "termination_grace_seconds": 5,
    "context_limit": 196000,
    "max_steps": 50,
    "model": "gpt-6-luna",
    "model_settings": {"reasoning_effort": "none", "max_completion_tokens": 4096, "seed": 42},
    "api_key_env": "LITELLM_API_KEY",
    "base_url_env": "LITELLM_BASE_URL",
    "gateway_host": "host.docker.internal",
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config(path):
    raw = json.loads(Path(path).read_text())
    reject_credentials(raw)
    allowed = set(DEFAULTS) | {
        "condition",
        "manifest",
        "dataset_snapshot",
        "codex_binary",
        "codex_sha256",
        "codex_version",
    }
    if set(raw) - allowed:
        raise ValueError("Unknown configuration keys: " + ", ".join(sorted(set(raw) - allowed)))
    cfg = DEFAULTS | raw
    if cfg["condition"] not in ("DCB", "LSB"):
        raise ValueError("condition must be DCB or LSB")
    for key in (
        "condition_timeout_seconds",
        "termination_grace_seconds",
        "context_limit",
        "max_steps",
    ):
        if (
            isinstance(cfg[key], bool)
            or not isinstance(cfg[key], (float, int))
            or not math.isfinite(cfg[key])
            or cfg[key] <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("context_limit", "max_steps"):
        if not isinstance(cfg[key], int):
            raise ValueError(f"{key} must be an integer")
    if not re.fullmatch(r"[0-9a-f]{64}", cfg["codex_sha256"]):
        raise ValueError("Freeze the Linux Codex binary SHA256 first")
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", cfg["model"]):
        raise ValueError("Invalid model/provider identifier")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg["gateway_host"]):
        raise ValueError("gateway_host must be a hostname/IP, without credentials")
    for key in ("api_key_env", "base_url_env"):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", cfg[key]):
            raise ValueError("Credentials and endpoint must be environment variable references")
    for key in ("manifest", "codex_binary", "dataset_snapshot"):
        cfg[key] = str((Path(path).resolve().parent / cfg[key]).resolve())
    if set(cfg["model_settings"]) - {
        "reasoning_effort",
        "max_completion_tokens",
        "temperature",
        "seed",
    }:
        raise ValueError("Unsupported model_settings; do not place credentials here")
    return cfg


def validate_manifest(manifest):
    if manifest.get("dataset") != DATASET or not re.fullmatch(
        r"[0-9a-f]{40}", manifest.get("dataset_revision", "")
    ):
        raise ValueError("Use select with a pinned official Verified dataset snapshot")
    groups = manifest.get("agents", [])
    if len(groups) != 6 or any(len(group) != 5 for group in groups):
        raise ValueError("Exactly six fixed groups of five tasks are required")
    ids = []
    profile = set()
    for group in groups:
        for task in group:
            for key in (
                "instance_id",
                "repo",
                "version",
                "base_commit",
                "problem_statement",
                "image",
            ):
                if not isinstance(task.get(key), str) or not task[key]:
                    raise ValueError(f"Task is missing {key}")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", task["instance_id"]):
                raise ValueError("Unsafe instance ID")
            if not re.fullmatch(r"[0-9a-f]{40}", task["base_commit"]):
                raise ValueError("base_commit must be a full Git SHA")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", task["image"]):
                raise ValueError(
                    "Freeze prepared local Docker image IDs (sha256:...), not mutable tags"
                )
            if set(task) != {
                "instance_id",
                "repo",
                "version",
                "base_commit",
                "problem_statement",
                "image",
            }:
                raise ValueError(
                    "Manifest must contain only agent-visible fields; exclude gold patches"
                )
            ids.append(task["instance_id"])
            profile.add((task["repo"], task["version"]))
    if len(profile) != 1:
        raise ValueError("Use a homogeneous single repository/version workload")
    if manifest.get("replicated", False):
        if len(set(ids)) != 5 or any(group != groups[0] for group in groups):
            raise ValueError("Replicated mode requires the same five tasks in the same order")
    elif len(set(ids)) != 30:
        raise ValueError("Default workload requires 30 distinct instance IDs")
    return manifest


def assignments(manifest):
    return [
        {
            "attempt_id": f"a{agent}-t{index}",
            "agent": agent,
            "index": index,
            "instance_id": task["instance_id"],
            "status": "unattempted",
            "reason": None,
            "started": None,
            "finished": None,
            "duration_seconds": None,
            "censored": False,
            "success": None,
        }
        for agent, group in enumerate(manifest["agents"])
        for index, task in enumerate(group)
    ]


def safe_json(path, value):
    # Only this writer accepts errors/conversation-derived content.
    from benchmarks.tool_annotation_effect.common import native

    atomic_json(path, native("annotations").redact(value))


def comparison_identity(cfg, manifest, provenance):
    relevant = {
        k: v
        for k, v in cfg.items()
        if k not in ("condition", "manifest", "codex_binary", "dataset_snapshot")
    }
    return digest({"config": relevant, "manifest": manifest, "provenance": provenance})


def private_directory(path):
    os.umask(0o077)
    Path(path).mkdir(parents=True, exist_ok=False, mode=0o700)
