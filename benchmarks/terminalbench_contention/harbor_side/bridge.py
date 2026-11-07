"""Harbor-side operations, run with the pinned Harbor virtualenv's Python.

This file is executed as a script by the benchmark (never imported by it), so it
may import ``harbor`` but nothing from Agency. ``agency_podman_env`` beside it is
the only other Harbor-side module. It keeps Harbor responsible for
what Harbor defines:

  dataset   parse every task with Harbor's models and compute Harbor's content
            digest for each task directory
  oracle    run Harbor's OracleAgent in an existing container
  collect   gather the task's declared artifacts from an existing container
            with Harbor's ArtifactHandler, producing a regradable trial record
  regrade   run Harbor's regrade trial (separate verifier environment, Harbor's
            Verifier and reward parsing) against that record

The agent itself never runs here: Agency creates the container and drives
Codex. ``AgencyPodmanEnvironment`` in attach mode only lets Harbor's code exec/copy against
that already-running container.

Every command prints one JSON object on its last stdout line.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _harbor_version() -> str:
    from importlib.metadata import version

    return version("harbor")


def _tree_sha256(directory: Path) -> "str | None":
    if not directory.is_dir():
        return None
    outer = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        outer.update(path.relative_to(directory).as_posix().encode() + b"\0")
        outer.update(hashlib.sha256(path.read_bytes()).digest())
    return outer.hexdigest()


# ------------------------------------------------------------------ dataset


def command_dataset(args) -> dict:
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import (
        resolve_task_verifier_mode,
        resolve_verifier_environment_definition,
    )
    from harbor.publisher.packager import Packager

    tasks = []
    for task_dir in sorted(p for p in Path(args.dataset_dir).iterdir() if p.is_dir()):
        if not (task_dir / "task.toml").is_file():
            continue
        record = dict(directory=task_dir.name, path=str(task_dir.resolve()))
        try:
            task = Task(task_dir)
            config = task.config
            content_hash, files = Packager.compute_content_hash(task_dir)
            definition = resolve_verifier_environment_definition(config, task.paths)
            environment = config.environment
            compose = sorted(
                p.name for p in task.paths.environment_dir.glob("docker-compose*") if p.is_file()
            )
            record.update(
                valid=True,
                name=task.name,
                digest=f"sha256:{content_hash}",
                files=len(files),
                schema_version=config.schema_version,
                metadata=config.metadata,
                instruction_sha256=hashlib.sha256(
                    task.paths.instruction_path.read_bytes()
                ).hexdigest(),
                instruction_chars=len(task.instruction),
                # The text Harbor would hand an agent (canary markers stripped).
                rendered_instruction_sha256=hashlib.sha256(task.instruction.encode()).hexdigest(),
                environment_sha256=_tree_sha256(task.paths.environment_dir),
                tests_sha256=_tree_sha256(task.paths.tests_dir),
                environment=dict(
                    cpus=environment.cpus,
                    memory_mb=environment.memory_mb,
                    storage_mb=environment.storage_mb,
                    gpus=environment.gpus,
                    gpu_types=environment.gpu_types,
                    docker_image=environment.docker_image,
                    os=environment.os.value,
                    workdir=environment.workdir,
                    allow_internet=environment.allow_internet,
                    network_mode=environment.network_mode.value,
                    allowed_hosts=environment.allowed_hosts,
                    build_timeout_sec=environment.build_timeout_sec,
                    env_keys=sorted(environment.env),
                    mcp_servers=[s.name for s in environment.mcp_servers],
                    skills_dir=environment.skills_dir,
                    healthcheck=environment.healthcheck is not None,
                    compose_files=compose,
                    has_dockerfile=(task.paths.environment_dir / "Dockerfile").is_file(),
                ),
                agent=dict(
                    timeout_sec=config.agent.timeout_sec,
                    user=config.agent.user,
                    network_mode=(
                        config.agent.network_mode.value if config.agent.network_mode else None
                    ),
                ),
                verifier=dict(
                    timeout_sec=config.verifier.timeout_sec,
                    user=config.verifier.user,
                    environment_mode=resolve_task_verifier_mode(config).value,
                    env_keys=sorted(config.verifier.env),
                    collect_hooks=len(config.verifier.collect),
                    separate_definition=(
                        None
                        if definition is None
                        else dict(
                            directory=str(definition.directory),
                            bundled_tests=definition.bundled_tests,
                            docker_image=definition.config.docker_image,
                            cpus=definition.config.cpus,
                            memory_mb=definition.config.memory_mb,
                            gpus=definition.config.gpus,
                        )
                    ),
                    test_script=(task.paths.tests_dir / "test.sh").is_file(),
                ),
                artifacts=[
                    a if isinstance(a, str) else a.model_dump(mode="json") for a in config.artifacts
                ],
                steps=len(config.steps or []),
                has_solution=task.paths.solve_path.is_file(),
            )
            if args.instructions_dir:
                target = Path(args.instructions_dir) / f"{task_dir.name}.md"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(task.instruction)
                record["instruction_file"] = str(target.resolve())
        except Exception as exc:  # an unparsable task is reported, not hidden
            record.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        tasks.append(record)
    return dict(harbor_version=_harbor_version(), dataset_dir=str(args.dataset_dir), tasks=tasks)


# -------------------------------------------------------------- environment


def _make_attach_environment(task, container: str, trial_paths, engine: str, logger):
    """Harbor environment facade over a running container that Agency owns."""
    os.environ["AGENCY_TB_ATTACH_CONTAINER"] = container
    os.environ["AGENCY_TB_ENGINE"] = engine
    from agency_podman_env import AgencyPodmanEnvironment

    return AgencyPodmanEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=task.short_name,
        session_id=f"{trial_paths.trial_dir.name}__agent",
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
        logger=logger,
        enable_environment_dir_upload=False,
    )


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"agency-harbor-bridge.{path}")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ------------------------------------------------------------------- oracle


def command_oracle(args) -> dict:
    """Run Harbor's reference solution in an existing container (validation only)."""
    from harbor.agents.oracle import OracleAgent
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

    task = Task(Path(args.task_dir))
    trial_paths = TrialPaths(trial_dir=Path(args.trial_dir))
    trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
    trial_paths.agent_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(trial_paths.trial_dir / "bridge-oracle.log")

    async def run():
        environment = _make_attach_environment(
            task, args.container, trial_paths, args.engine, logger
        )
        await environment.start(force_build=False)
        env_paths = EnvironmentPaths.for_os(environment.os)
        await environment.ensure_dirs([env_paths.agent_dir, env_paths.artifacts_dir])
        agent = OracleAgent(
            logs_dir=trial_paths.agent_dir,
            task_dir=Path(args.task_dir),
            trial_paths=trial_paths,
            agent_timeout_sec=args.timeout,
            logger=logger,
        )
        await agent.run(task.instruction, environment, AgentContext())
        exit_code = trial_paths.agent_dir / "exit-code.txt"
        return int(exit_code.read_text()) if exit_code.exists() else 0

    return dict(ok=True, solve_exit_code=asyncio.run(run()))


# ------------------------------------------------------------------ collect


def command_collect(args) -> dict:
    from harbor.models.task.id import LocalTaskId
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import resolve_task_verifier_mode
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
    from harbor.models.trial.result import AgentInfo, ModelInfo, TrialResult
    from harbor.publisher.packager import Packager
    from harbor.trial.artifact_handler import ArtifactHandler

    task_dir = Path(args.task_dir).resolve()
    task = Task(task_dir)
    if task.config.steps:
        raise RuntimeError("Multi-step tasks are not supported by this benchmark")
    if task.config.verifier.collect:
        raise RuntimeError("Tasks with verifier collect hooks are excluded from this benchmark")
    trial_dir = Path(args.trial_dir).resolve()
    if trial_dir.exists() and any(trial_dir.iterdir()):
        raise FileExistsError(f"Trial record {trial_dir} already exists; refusing to overwrite")
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_paths.agent_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(trial_dir / "bridge-collect.log")
    started = datetime.now(timezone.utc)

    async def run():
        environment = _make_attach_environment(
            task, args.container, trial_paths, args.engine, logger
        )
        await environment.start(force_build=False)
        env_paths = EnvironmentPaths.for_os(environment.os)
        handler = ArtifactHandler(artifacts=list(task.config.artifacts), logger=logger)
        handler.begin_collection()
        collected = await handler.download_artifacts(
            environment,
            trial_paths.artifacts_dir,
            source_artifacts_dir=env_paths.artifacts_dir,
        )
        # For anything that was not collected, establish whether the agent
        # simply never produced it (absent) or collection itself went wrong.
        exists = {}
        for entry in collected.entries:
            if entry.status in ("failed", "skipped"):
                probe = await environment.exec(
                    f"test -e {shlex.quote(entry.source)}", user="root", timeout_sec=30
                )
                exists[entry.source] = probe.return_code == 0
        return collected, exists

    manifest, presence = asyncio.run(run())
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trial_name=trial_dir.name,
        trials_dir=trial_dir.parent,
        agent=AgentConfig(name="nop", model_name=args.model),
    )
    content_hash, _ = Packager.compute_content_hash(task_dir)
    result = TrialResult(
        task_name=task.name,
        trial_name=trial_dir.name,
        trial_uri=trial_dir.as_uri(),
        task_id=LocalTaskId(path=task_dir),
        task_checksum=content_hash,
        config=config,
        agent_info=AgentInfo(
            name="agency-codex",
            version=args.agent_version,
            model_info=ModelInfo(name=args.model, provider=args.provider),
        ),
        verifier_environment_mode=resolve_task_verifier_mode(task.config),
        started_at=started,
        finished_at=datetime.now(timezone.utc),
    )
    trial_paths.config_path.write_text(config.model_dump_json(indent=2))
    trial_paths.result_path.write_text(result.model_dump_json(indent=2))
    entries = manifest.to_json_data()
    missing = sorted(source for source, present in presence.items() if not present)
    broken = sorted(source for source, present in presence.items() if present)
    outcome = dict(
        # A declared output the agent never created is a task outcome, not a
        # collection failure; an existing path that could not be copied is.
        ok=not broken,
        trial_dir=str(trial_dir),
        task_name=task.name,
        task_digest=f"sha256:{content_hash}",
        artifacts=entries,
        missing_artifacts=missing,
        collection_failures=broken,
    )
    (trial_dir / "agency-collect.json").write_text(json.dumps(outcome, indent=2))
    return outcome


# ------------------------------------------------------------------ regrade


def command_regrade(args) -> dict:
    """Grade a collected record with Harbor's regrade trial.

    This is Harbor's own RegradeTrial (seed recorded artifacts into a separate
    verifier environment, run Harbor's Verifier, parse the reward) with one
    deliberate difference. Harbor's regrade refuses a record in which a declared
    artifact was not collected. For an agent that simply never produced an
    output file, a normal Harbor trial still runs the verifier (collection is
    best-effort) and the tests fail on the missing file. ``collect`` proves
    which declared artifacts were absent from the container; exactly those are
    tolerated here so such a task receives the verifier's verdict instead of an
    error. Every other coverage problem still fails as in Harbor.
    """
    from harbor.models.trial.config import (
        EnvironmentConfig,
        SourceTrialConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.trial.result import TrialResult
    from harbor.trial.regrade import RegradeTrial
    from harbor.trial.trial import Trial

    source = Path(args.source).resolve()
    trials_dir = Path(args.trials_dir).resolve()
    trials_dir.mkdir(parents=True, exist_ok=True)
    target = trials_dir / args.trial_name
    if target.exists():
        raise FileExistsError(f"Grading output {target} already exists; refusing to overwrite")
    os.environ.pop("AGENCY_TB_ATTACH_CONTAINER", None)
    os.environ["AGENCY_TB_ENGINE"] = args.engine
    if args.label:
        os.environ["AGENCY_TB_LABEL"] = args.label
    if args.cpuset:
        os.environ["AGENCY_TB_CPUSET"] = args.cpuset
    source_paths = TrialPaths(trial_dir=source)
    source_result = TrialResult.model_validate_json(source_paths.result_path.read_text())
    source_config = TrialConfig.model_validate_json(source_paths.config_path.read_text())
    collected = {}
    marker = source / "agency-collect.json"
    if marker.exists():
        collected = json.loads(marker.read_text())
    absent = set(collected.get("missing_artifacts") or [])
    config = TrialConfig(
        task=TaskConfig(path=Path(args.task_dir).resolve(), source=source_result.source),
        trial_name=args.trial_name,
        trials_dir=trials_dir,
        agent=source_config.agent,
        environment=EnvironmentConfig(import_path=args.environment),
        verifier=VerifierConfig(),
        artifacts=source_config.artifacts,
        source_trial=SourceTrialConfig(
            action="regrade", type="local", trial_id=source_result.id, path=source
        ),
    )
    tolerated: list = []

    class AbsentOutputTolerantRegrade(RegradeTrial):
        def _raise_artifact_coverage_error(self, problems):
            fatal = []
            for problem in problems:
                path = problem.split(":", 1)[0]
                if "collection failed in the source trial" in problem and path in absent:
                    tolerated.append(path)
                else:
                    fatal.append(problem)
            super()._raise_artifact_coverage_error(fatal)

    async def run():
        task, download = await Trial._load_task(config)
        trial = AbsentOutputTolerantRegrade(
            config, _task=task, _task_download_result=download, _source_trial_dir=source
        )
        return await asyncio.wait_for(trial.run(), timeout=args.timeout)

    verdict = dict(
        ok=False,
        harbor_version=_harbor_version(),
        grading="harbor RegradeTrial + Verifier in a separate verifier environment",
        environment=args.environment,
        trial_dir=str(target),
        timed_out=False,
        reward=None,
        rewards=None,
        exception=None,
        missing_declared_artifacts=sorted(absent),
    )
    try:
        result = asyncio.run(run())
    except asyncio.TimeoutError:
        verdict.update(
            timed_out=True, exception=dict(type="GradingTimeout", message=f"{args.timeout}s")
        )
        return verdict
    verdict["tolerated_absent_artifacts"] = sorted(set(tolerated))
    if result.exception_info is not None:
        verdict["exception"] = dict(
            type=result.exception_info.exception_type,
            message=result.exception_info.exception_message[:2000],
        )
    if result.verifier_result is not None and result.verifier_result.rewards is not None:
        rewards = result.verifier_result.rewards
        verdict.update(ok=True, rewards=rewards, reward=rewards.get("reward"))
    if result.verifier is not None:
        verdict["verifier_started_at"] = str(result.verifier.started_at)
        verdict["verifier_finished_at"] = str(result.verifier.finished_at)
    return verdict


def command_cleanup(args) -> dict:
    """Remove a container created by Harbor for one named grading trial only."""
    removed = []
    for needle in args.name:
        listing = subprocess.run(
            [args.engine, "ps", "-a", "--filter", f"name={needle}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        for name in listing.stdout.split():
            subprocess.run([args.engine, "rm", "--force", name], capture_output=True)
            removed.append(name)
    return dict(ok=True, removed=removed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset")
    dataset.add_argument("--dataset-dir", required=True)
    dataset.add_argument("--instructions-dir")

    for name in ("oracle", "collect"):
        sub = commands.add_parser(name)
        sub.add_argument("--task-dir", required=True)
        sub.add_argument("--container", required=True)
        sub.add_argument("--trial-dir", required=True)
        sub.add_argument("--engine", default="podman")
        if name == "oracle":
            sub.add_argument("--timeout", type=float, default=3600)
        else:
            sub.add_argument("--model", default="unknown")
            sub.add_argument("--provider", default="litellm")
            sub.add_argument("--agent-version", default="unknown")

    regrade = commands.add_parser("regrade")
    regrade.add_argument("--source", required=True)
    regrade.add_argument("--task-dir", required=True)
    regrade.add_argument("--trials-dir", required=True)
    regrade.add_argument("--trial-name", required=True)
    regrade.add_argument("--environment", default="agency_podman_env:AgencyPodmanEnvironment")
    regrade.add_argument("--engine", default="podman")
    regrade.add_argument("--label")
    regrade.add_argument("--cpuset")
    regrade.add_argument("--timeout", type=float, default=3600)

    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--engine", default="podman")
    cleanup.add_argument("--name", action="append", required=True)

    args = parser.parse_args(argv)
    if (
        args.command in ("oracle", "collect", "regrade")
        and shutil.which(getattr(args, "engine", "podman")) is None
    ):
        print(json.dumps(dict(ok=False, error="container engine not found")))
        return 3
    try:
        result = globals()[f"command_{args.command}"](args)
    except Exception as exc:
        print(
            json.dumps(
                dict(
                    ok=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
                )
            )
        )
        return 1
    print(json.dumps(result, default=str))
    return 0 if result.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
