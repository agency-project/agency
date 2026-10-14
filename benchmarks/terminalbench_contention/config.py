"""Experiment configuration: one defaults tree, strict overrides, stable hash.

The configuration is a plain JSON-serializable dict so the exact values used
by a run are persisted verbatim. Overrides come from a JSON file and/or
``section.key=value`` pairs; unknown keys are rejected so a typo cannot
silently fall back to a default.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

from .api_budget import BudgetPolicy
from .util import REPOSITORY, sha256_json

SCHEMA_VERSION = 1

DEFAULTS: dict = {
    "schema_version": SCHEMA_VERSION,
    "benchmark": {
        "name": "terminal-bench",
        "dataset": "terminal-bench/terminal-bench@4.0.0",
        "version": "4.0.0",
        "harbor_version": "0.24.0",
        "tasks": 24,
    },
    # condition name -> maximum concurrently executing agents
    "conditions": {"LSB-6": 6, "LSB-12": 12, "LSB-24": 24},
    "repetitions": 3,
    "seed": 20261012,
    "cpu": {
        "experimental_physical_cores": 6,
        "reserved_physical_cores": 2,
        "smt_policy": "siblings",
        # Keep the rest of the user session (desktop, shells) on the reserved
        # CPUs for the duration of a run; restored afterwards.
        "confine_user_session": True,
        "scheduler": "linux-default",  # documented constant; nothing is changed
    },
    "memory": {
        # One ceiling for the whole experimental workload (all agents share
        # it), leaving the rest of the host for the OS and the runner.
        "workload_memory_max": "22G",
        # "max" lets the workload use the host's configured swap; "0" forbids it.
        "workload_swap_max": "max",
        # Per-container safety limit: the memory the task itself declares.
        "container_memory": "task",
        "container_swap": "unlimited",  # within the workload ceiling above
        "container_pids_limit": 4096,
        "workload_tasks_max": 32768,
        # Abort the run (preserving artifacts) if the host is about to fail.
        "host_min_available_mb": 1024,
        "host_min_available_seconds": 10,
        "min_free_disk_gb": 15,
    },
    "model": {
        "route": "litellm",
        "provider": "openai",  # Agency's OpenAI-compatible backend -> LiteLLM
        "model": "gpt-6-luna",
        "reasoning_effort": "none",
        "max_completion_tokens": 4096,
        "context_limit": 196000,
        "idle_timeout": 180,
        "stream_timeout": 180,
        "api_key_env": "LITELLM_API_KEY",
        "base_url_env": "LITELLM_BASE_URL",
        "env_file": ".env",
    },
    "agent": {
        "harness": "codex",
        "codex_version": "0.147.0",
        "codex_binary": ".venv/codex-0.147.0/package/vendor/x86_64-unknown-linux-musl/bin/codex",
        "harness_python": "cpython-3.12.15-linux-x86_64-gnu",
        "max_steps": None,
    },
    "timeouts": {
        # One wall-clock cap per task, measured from the moment the agent is
        # released. A task that reaches it is stopped, recorded as censored and
        # still graded on whatever it produced.
        "task_seconds": 1800,
        "setup_seconds": 600,
        "stop_grace_seconds": 60,
        "collect_seconds": 600,
        "grade_seconds": 3600,
        # Hard ceiling on the measured interval. null = derived:
        # ceil(tasks / agents) * (task_seconds + setup_seconds) + 600.
        "run_seconds": None,
    },
    "sandbox": {
        "engine": "podman",
        "network": "default",  # "default" (task policy: public) or "none"
        "label_prefix": "agency.tb",
    },
    "profiling": {
        "agprof_sample_hz": 2.0,
        "host_sample_seconds": 1.0,
        "ptrace_syscalls": True,
    },
    "grading": {"parallel": 2},
    "api": BudgetPolicy().to_dict(),
    "paths": {"artifact_root": "artifacts/terminalbench-contention"},
}

SMOKE_OVERRIDES: dict = {
    # Local mock endpoint, tiny caps. Never used for measured runs.
    "model": {"model": "mock-luna", "idle_timeout": 30, "stream_timeout": 30},
    "timeouts": {
        "task_seconds": 180,
        "setup_seconds": 180,
        "stop_grace_seconds": 20,
        "collect_seconds": 120,
        "grade_seconds": 600,
    },
    "cpu": {"confine_user_session": False},
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, path: str = "") -> dict:
    for key, value in override.items():
        where = f"{path}.{key}" if path else key
        if key not in base:
            raise ConfigError(f"Unknown configuration key {where!r}")
        if isinstance(base[key], dict) and key != "conditions":
            if not isinstance(value, dict):
                raise ConfigError(f"{where!r} must be an object")
            _merge(base[key], value, where)
        else:
            base[key] = value
    return base


def _coerce(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_assignments(pairs) -> dict:
    """['api.tokens_per_window=2000000'] -> {'api': {'tokens_per_window': 2000000}}."""
    tree: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"Override {pair!r} is not of the form section.key=value")
        dotted, value = pair.split("=", 1)
        node = tree
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value.strip())
    return tree


def validate(config: dict) -> dict:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError("Unsupported configuration schema_version")
    conditions = config["conditions"]
    if not isinstance(conditions, dict) or not conditions:
        raise ConfigError("conditions must map a name to a maximum agent count")
    for name, agents in conditions.items():
        if not isinstance(agents, int) or isinstance(agents, bool) or agents < 1:
            raise ConfigError(f"Condition {name!r} needs a positive integer agent count")
    if not isinstance(config["repetitions"], int) or config["repetitions"] < 1:
        raise ConfigError("repetitions must be a positive integer")
    if not isinstance(config["seed"], int):
        raise ConfigError("seed must be an integer")
    tasks = config["benchmark"]["tasks"]
    if not isinstance(tasks, int) or tasks < 1:
        raise ConfigError("benchmark.tasks must be a positive integer")
    cpu = config["cpu"]
    if cpu["smt_policy"] not in ("siblings", "primary"):
        raise ConfigError("cpu.smt_policy must be 'siblings' or 'primary'")
    if cpu["scheduler"] != "linux-default":
        raise ConfigError("This benchmark only runs the default Linux scheduler")
    for key in ("experimental_physical_cores", "reserved_physical_cores"):
        if not isinstance(cpu[key], int) or cpu[key] < 1:
            raise ConfigError(f"cpu.{key} must be a positive integer")
    for key, value in config["timeouts"].items():
        if value is None and key == "run_seconds":
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"timeouts.{key} must be a positive number")
    if config["sandbox"]["engine"] != "podman":
        raise ConfigError("Only the podman engine is supported on this host")
    if config["sandbox"]["network"] not in ("default", "none"):
        raise ConfigError("sandbox.network must be 'default' or 'none'")
    if config["memory"]["container_memory"] != "task" and not isinstance(
        config["memory"]["container_memory"], str
    ):
        raise ConfigError("memory.container_memory must be 'task' or a size such as '4g'")
    if config["agent"]["harness"] != "codex":
        raise ConfigError("This benchmark runs Agency's Codex harness")
    try:
        BudgetPolicy.from_dict(config["api"])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"api: {exc}") from exc
    if config["grading"]["parallel"] < 1:
        raise ConfigError("grading.parallel must be at least 1")
    return config


def load(config_file=None, assignments=None, *, smoke: bool = False) -> dict:
    config = copy.deepcopy(DEFAULTS)
    if smoke:
        _merge(config, copy.deepcopy(SMOKE_OVERRIDES))
    if config_file:
        try:
            override = json.loads(Path(config_file).read_text())
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Cannot read configuration {config_file}: {exc}") from exc
        if not isinstance(override, dict):
            raise ConfigError("Configuration file must contain a JSON object")
        _merge(config, override)
    _merge(config, parse_assignments(assignments))
    return validate(config)


def fingerprint(config: dict) -> str:
    return sha256_json(config)


def api_policy(config: dict) -> BudgetPolicy:
    return BudgetPolicy.from_dict(config["api"])


def artifact_root(config: dict) -> Path:
    root = Path(config["paths"]["artifact_root"])
    return root if root.is_absolute() else REPOSITORY / root


def condition_agents(config: dict, condition: str) -> int:
    try:
        return config["conditions"][condition]
    except KeyError:
        raise ConfigError(
            f"Unknown condition {condition!r}; choose from {sorted(config['conditions'])}"
        ) from None


def condition_for_agents(config: dict, agents: int) -> str:
    matches = [name for name, count in config["conditions"].items() if count == agents]
    if len(matches) != 1:
        raise ConfigError(
            f"No unique condition with {agents} agents; conditions are {config['conditions']}"
        )
    return matches[0]


def run_seconds(config: dict, agents: int) -> float:
    explicit = config["timeouts"]["run_seconds"]
    if explicit is not None:
        return float(explicit)
    tasks = config["benchmark"]["tasks"]
    waves = -(-tasks // agents)
    return waves * (config["timeouts"]["task_seconds"] + config["timeouts"]["setup_seconds"]) + 600


# ------------------------------------------------------------------ planning


def repetition_seed(config: dict, repetition: int) -> int:
    """Seed for one repetition's launch order. Independent of the condition, so
    every condition in a repetition launches tasks in the same order."""
    digest = hashlib.sha256(f"{config['seed']}:repetition:{repetition}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def launch_order(task_ids: list, seed: int) -> list:
    """Deterministic shuffle of the corpus (independent of input ordering)."""
    order = sorted(task_ids)
    random.Random(seed).shuffle(order)
    return order


def condition_sequence(config: dict, repetition: int) -> list:
    """Order in which conditions run within a repetition, rotated by repetition
    so no condition always runs first (or last) on a warm machine."""
    names = sorted(config["conditions"], key=lambda n: (config["conditions"][n], n))
    shift = (repetition - 1) % len(names)
    return names[shift:] + names[:shift]


def build_plan(config: dict, task_ids: list) -> list:
    """The full matrix: one entry per (repetition, condition), in run order."""
    if len(task_ids) != len(set(task_ids)):
        raise ConfigError("Task corpus contains duplicates")
    if len(task_ids) != config["benchmark"]["tasks"]:
        raise ConfigError(
            f"Corpus has {len(task_ids)} tasks; configuration expects "
            f"{config['benchmark']['tasks']}"
        )
    plan = []
    for repetition in range(1, config["repetitions"] + 1):
        seed = repetition_seed(config, repetition)
        order = launch_order(task_ids, seed)
        for condition in condition_sequence(config, repetition):
            agents = config["conditions"][condition]
            plan.append(
                dict(
                    index=len(plan) + 1,
                    run_label=f"{condition}-r{repetition}",
                    condition=condition,
                    agents=agents,
                    repetition=repetition,
                    order_seed=seed,
                    launch_order=order,
                    tasks=len(order),
                    task_timeout_seconds=config["timeouts"]["task_seconds"],
                    run_timeout_seconds=run_seconds(config, agents),
                )
            )
    return plan


def plan_entry(config: dict, task_ids: list, condition: str, repetition: int) -> dict:
    if not 1 <= repetition <= config["repetitions"]:
        raise ConfigError(f"Repetition must be between 1 and {config['repetitions']}")
    condition_agents(config, condition)
    for entry in build_plan(config, task_ids):
        if entry["condition"] == condition and entry["repetition"] == repetition:
            return entry
    raise ConfigError(f"No plan entry for {condition} repetition {repetition}")
