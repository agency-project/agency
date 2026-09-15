"""Run the fixed-image Agency checkpoint-size experiment on the EC2 host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import shlex
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


os.environ["AGENCY_PROFILE"] = "0"

INSTANCE_ID = "psf__requests-1921"
REPOSITORY = "psf/requests"
REPOSITORY_COMMIT = "040ee8c95a4bd91dde598105d7b12bc83306488a"
DATASET_BASE_COMMIT = "3c88e520da24ae6f736929a750876e7654accc3d"
CREATION_IMAGE = "sha256:605aaf09a3a137a8c3a247d468046c5edbff3568eb99baf1415ce2d7a3df67a2"
# Docker 29's ID above is the OCI index. Its linux/amd64 manifest points to
# this config digest, which Podman uses as its image ID. These are the SAME
# image, not a rebuilt/tag-substituted benchmark base.
CREATION_PLATFORM_MANIFEST = (
    "sha256:baff84e829a563ab754843116a93bd1bec58c199deb9622009ca072064b52e18"
)
CREATION_CONFIG_IMAGE = "sha256:fdf3adde7ef9d0e04034282b01fce743954168fa73dffc8bcfbaeb43c7f9a21e"
OFFICIAL_IMAGE = (
    "swebench/sweb.eval.x86_64.psf_1776_requests-1921"
    "@sha256:2ee0a4c706f04d2926a1622ec51b53d4ce6010b80d3ee10d414ea2a276c26cc9"
)
AGENCY_GIT_HEAD = "ee7999fe6befea27e13f9db20b131e9c34bc2dc3"
SOURCE_MANIFEST_SHA256 = "d381de4aa0cc1bfdeba274d7de2821a326c84d441838c760fd70ef20b9cd0e2e"
PRIOR_PLAN_SHA256 = "14e8a33a4ebecbb9fc74dd67d0539d3d2de4de0141ae623e841afad30d9c46c8"

PAYLOAD_PATH = "/testbed/.agency-checkpoint-payload.bin"
AES_KEY = "7a3e12c6f55b904adcf8016b90e8f1a3c27d99e048bc61184d0a6b3fce752490"
AES_IV = "40670f28421b88c34719f80d15e6082b"
ORDER_SEED = 731_991
INITIAL_SIZES = [0, 64 * 1024, 128 * 1024**2, 1024**3]

SYSTEM_PROMPT = """You are executing one controlled filesystem operation, not solving a software issue. Submit the exact functions.exec JavaScript in payload_instruction without changing it. If functions.exec yields a cell ID, use functions.wait only to wait for that same program to finish. Do not inspect the repository or task. Do not run any other command. Do not modify any file except /testbed/.agency-checkpoint-payload.bin when the requested size is greater than zero. For zero bytes, do not create or modify that file and make no intentional filesystem modification. After the program succeeds, copy its complete output verbatim into summary and finish."""


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def output(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def executing_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def executing_source_digest():
    import agency

    root = Path(agency.__file__).parent
    accumulator = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        accumulator.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return accumulator.hexdigest()


def image_inspect(image: str, runtime: str = "docker") -> dict:
    return json.loads(output([runtime, "image", "inspect", image]))[0]


def host_docker_image_inspect(image: str) -> dict:
    return json.loads(
        output(["docker", "--host", "unix:///var/run/docker.sock", "image", "inspect", image])
    )[0]


def runtime_creation_image(runtime: str) -> str:
    return CREATION_CONFIG_IMAGE if runtime == "podman" else CREATION_IMAGE


def payload_command(size: int) -> str:
    if size == 0:
        return (
            "test ! -e /testbed/.agency-checkpoint-payload.bin && echo START_NS=0 && "
            "echo END_NS=0 && echo PAYLOAD_BYTES=0 && "
            "echo NONE /testbed/.agency-checkpoint-payload.bin"
        )
    return (
        "date +START_NS=%s%N; "
        f"head -c {size} /dev/zero | openssl enc -aes-256-ctr -K {AES_KEY} "
        f"-iv {AES_IV} -nosalt -out {PAYLOAD_PATH}; "
        "date +END_NS=%s%N; "
        f"stat -c PAYLOAD_BYTES=%s {PAYLOAD_PATH}; "
        f"sha256sum {PAYLOAD_PATH}"
    )


def payload_instruction(size: int) -> str:
    command = json.dumps(payload_command(size))
    program = (
        '// @exec: {"yield_time_ms": 120000, "max_output_tokens": 2000}\n'
        f"let current = await tools.exec_command({{cmd: {command}, yield_time_ms: 30000, "
        "max_output_tokens: 2000});\n"
        "let combined = current.output || '';\n"
        "while (current.session_id !== undefined) {\n"
        "  current = await tools.write_stdin({session_id: current.session_id, chars: '', "
        "yield_time_ms: 30000, max_output_tokens: 2000});\n"
        "  combined += current.output || '';\n"
        "}\n"
        "text(combined);"
    )
    return (
        f"Create exactly {size} bytes at {PAYLOAD_PATH} with the fixed deterministic "
        "AES-256-CTR stream. Submit this exact JavaScript to functions.exec. It starts "
        "one exact shell command and drains that command's session until exit:\n\n"
        f"{program}"
    )


def shuffled_conditions(sizes: list[int], replicates: tuple[int, ...], seed: int) -> list[dict]:
    conditions = [
        {"requested_bytes": size, "replicate": replicate}
        for replicate in replicates
        for size in sizes
    ]
    random.Random(seed).shuffle(conditions)
    for index, condition in enumerate(conditions, 1):
        condition["order"] = index
    return conditions


def image_history(image_id: str) -> list[dict]:
    image = image_id.removeprefix("sha256:")
    raw = output(
        [
            "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--unix-socket",
            "/var/run/docker.sock",
            f"http://localhost/v1.55/images/{image}/history",
        ]
    )
    return json.loads(raw)


def ec2_identity() -> dict:
    base = "http://169.254.169.254/latest/"
    try:
        token_request = urllib.request.Request(
            base + "api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_request, timeout=2).read().decode()
        request = urllib.request.Request(
            base + "dynamic/instance-identity/document",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return json.loads(urllib.request.urlopen(request, timeout=2).read())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def choose_sizes(free_bytes: int) -> tuple[list[int], dict | None]:
    requested = 1024**3
    # Leave enough room for the committed layer, temporary runtime data, and host safety margin.
    if free_bytes >= 10 * 1024**3 + 6 * requested:
        return list(INITIAL_SIZES), None
    for replacement in (512 * 1024**2, 256 * 1024**2):
        if free_bytes >= 10 * 1024**3 + 6 * replacement:
            sizes = [size for size in INITIAL_SIZES if size != requested] + [replacement]
            return sizes, {
                "requested_bytes": requested,
                "replacement_bytes": replacement,
                "reason": f"Only {free_bytes} free bytes before the matrix",
            }
    raise RuntimeError("Insufficient EC2 disk for a safe payload condition of at least 256 MiB")


def prepare(
    root: Path,
    prior_root: Path,
    *,
    checkpoint_backend: str = "image_commit",
    runtime: str = "docker",
    checkpoint_zfs_parent: str | None = None,
    checkpoint_fast_resume: bool = False,
) -> None:
    if root.exists():
        raise RuntimeError(f"Experiment directory already exists: {root}")
    root.mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "exact-prompts").mkdir()

    prior_plan = prior_root / "plan-codex.json"
    source_manifest = prior_root / "source-manifest.json"
    if digest(prior_plan) != PRIOR_PLAN_SHA256:
        raise RuntimeError("Prior measured-run plan drifted")
    if digest(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("Measured Agency source snapshot drifted")

    selected_image = (
        CREATION_CONFIG_IMAGE
        if checkpoint_backend == "cow_zfs"
        else runtime_creation_image(runtime)
    )
    creation = image_inspect(selected_image, runtime)
    if creation["Id"].removeprefix("sha256:") != selected_image.removeprefix("sha256:"):
        raise RuntimeError("Pinned creation image ID is not locally available")
    official = host_docker_image_inspect(OFFICIAL_IMAGE)
    if runtime == "podman":
        original = host_docker_image_inspect(CREATION_IMAGE)
        for field in ("RootFS", "Config", "Architecture", "Os"):
            if creation.get(field) != original.get(field):
                raise RuntimeError(f"Imported image {field} differs from the pinned Docker image")
    repo_probe = output(
        [
            runtime,
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            selected_image,
            "-lc",
            'cd /testbed && git rev-parse HEAD && test -z "$(git status --porcelain)"',
        ]
    ).splitlines()
    if repo_probe != [REPOSITORY_COMMIT]:
        raise RuntimeError(f"Creation-image repository mismatch: {repo_probe!r}")

    usage = shutil.disk_usage("/")
    sizes, substitution = choose_sizes(usage.free)
    conditions = shuffled_conditions(sizes, (1, 2, 3), ORDER_SEED)
    for size in sizes:
        (root / "exact-prompts" / f"payload-{size}.txt").write_text(
            payload_instruction(size) + "\n"
        )

    source = Path(__file__).resolve()
    destination = (root / "runner.py").resolve()
    if source != destination:
        shutil.copy2(source, destination)
    analysis_script = Path(__file__).with_name("analyze.py")
    if analysis_script.exists():
        shutil.copy2(analysis_script, root / "analyze.py")
    shutil.copy2(source_manifest, root / "measured-source-manifest.json")

    config = {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "temperature": 0,
        "max_completion_tokens": 4096,
        "context_limit": 64000,
        "max_steps": 80,
        "sample_hz": 5,
        "sample_gpu": False,
        "backend": runtime,
        "checkpoint_backend": checkpoint_backend,
        "checkpoint_fast_resume": checkpoint_fast_resume,
        "checkpoint_zfs_parent": checkpoint_zfs_parent,
        "sandbox_cpus": 4,
        "sandbox_memory": "8g",
        "harness": "codex",
        "reasoning_effort": "none",
        "checkpoint_diagnostics": True,
        "checkpoint_diagnostics_extended": True,
        "file_access": True,
        "max_concurrent_engines": 1,
    }
    environment = {
        "captured_wall_ns": time.time_ns(),
        "host": platform.uname()._asdict(),
        "ec2": ec2_identity(),
        "docker_version": json.loads(output(["docker", "version", "--format", "{{json .}}"])),
        "docker_info": json.loads(output(["docker", "info", "--format", "{{json .}}"])),
        "selected_runtime_version": output([runtime, "version", "--format", "{{json .}}"]),
        "selected_runtime_info": output([runtime, "info", "--format", "{{json .}}"]),
        "disk_usage": usage._asdict(),
        "creation_image_inspect": creation,
        "official_image_inspect": official,
        "running_processes": output(
            ["ps", "-eo", "user,pid,etimes,pcpu,pmem,args", "--sort=-pcpu"]
        ).splitlines()[:50],
    }
    if checkpoint_backend == "cow_zfs":
        environment["checkpoint_runtime"] = {
            "zfs_version": output(["zfs", "version"]),
        }
    save(root / "environment.json", environment)
    manifest = {
        "schema_version": 1,
        "experiment": "checkpoint-size-microbenchmark",
        "created_wall_ns": time.time_ns(),
        "control": {
            "instance_id": INSTANCE_ID,
            "repo": REPOSITORY,
            "repo_commit_in_creation_image": REPOSITORY_COMMIT,
            "dataset_base_commit": DATASET_BASE_COMMIT,
            "creation_image_id": CREATION_IMAGE,
            "runtime_creation_image_id": selected_image,
            "official_image_digest": OFFICIAL_IMAGE,
            "normal_issue_description_passed_to_agent": False,
            "base_image_rebuilt_or_mutated": False,
        },
        "agency_source": {
            "git_head": AGENCY_GIT_HEAD,
            "dirty": True,
            "description": "Original measured snapshot is identified by the retained manifest; executing_source fields identify the current implementation used for this cohort",
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "prior_measured_plan_sha256": PRIOR_PLAN_SHA256,
            "executing_source_git_head": executing_revision(),
            "executing_source_sha256": executing_source_digest(),
            "executing_runner_sha256": digest(Path(__file__)),
            "checkpoint_backend": checkpoint_backend,
        },
        "config": config,
        "payload": {
            "path": PAYLOAD_PATH,
            "generator": "OpenSSL AES-256-CTR over zeros; fixed key and IV; smaller outputs are prefixes",
            "aes_key_hex": AES_KEY,
            "aes_iv_hex": AES_IV,
            "system_prompt": SYSTEM_PROMPT,
        },
        "requested_sizes_bytes": INITIAL_SIZES,
        "actual_sizes_bytes": sizes,
        "one_gib_substitution": substitution,
        "order_seed": ORDER_SEED,
        "initial_conditions": conditions,
        "extra_conditions": [],
        "actual_execution_order": [],
        "validation_excluded_from_dataset": True,
    }
    save(root / "manifest.json", manifest)
    (root / "COMMAND.txt").write_text(
        f"PYTHONPATH={shlex.quote(str(source.parents[2]))} {shlex.quote(sys.executable)} "
        f"{root / 'runner.py'} cohort --root {root} --key-file {prior_root / '.openai-key'}\n"
    )


def refresh_protocol_after_failed_validation(root: Path) -> None:
    """Preserve a failed gate while refreshing prompts before any dataset run."""
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["actual_execution_order"]:
        raise RuntimeError("Protocol refresh is forbidden after a measured condition starts")
    failed = root / "validation"
    if failed.exists():
        index = 1
        destination = root / f"validation-failed-{index}"
        while destination.exists():
            index += 1
            destination = root / f"validation-failed-{index}"
        failed_run_path = failed / "run.json"
        if failed_run_path.exists():
            failed_run = json.loads(failed_run_path.read_text())
            reason = failed_run.get("exception_message", "Validation failed")
        else:
            reason = "Validation failed without run.json"
        failed.rename(destination)
        manifest.setdefault("validation_failures", []).append(
            {
                "path": destination.name,
                "reason": reason,
                "excluded_from_dataset": True,
            }
        )
    manifest["payload"]["system_prompt"] = SYSTEM_PROMPT
    manifest["protocol_refreshed_wall_ns"] = time.time_ns()
    for size in manifest["actual_sizes_bytes"]:
        (root / "exact-prompts" / f"payload-{size}.txt").write_text(
            payload_instruction(size) + "\n"
        )
    source = Path(__file__).resolve()
    destination = (root / "runner.py").resolve()
    if source != destination:
        shutil.copy2(source, destination)
    save(manifest_path, manifest)


def configure(root: Path, key_file: Path):
    import agency

    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    cfg = agency.agconfig()
    for name in (
        "provider",
        "model",
        "temperature",
        "max_completion_tokens",
        "context_limit",
        "reasoning_effort",
    ):
        setattr(cfg.llm, name, config[name])
    cfg.llm.api_key = key_file.read_text().strip()
    cfg.agent.harness = config["harness"]
    cfg.sandbox.backend = config["backend"]
    cfg.sandbox.checkpoint_backend = config.get("checkpoint_backend", "image_commit")
    cfg.sandbox.checkpoint_fast_resume = config.get("checkpoint_fast_resume", False)
    cfg.sandbox.checkpoint_zfs_parent = config.get("checkpoint_zfs_parent")
    cfg.sandbox.base_image = manifest["control"]["runtime_creation_image_id"]
    cfg.sandbox.checkpoint_diagnostics = True
    cfg.sandbox.checkpoint_diagnostics_extended = True
    cfg.ptrace.file_access = True
    cfg.orchestrator.max_concurrent_engines = 1
    cfg.resources.idle_cpus = config["sandbox_cpus"]
    cfg.resources.idle_memory = config["sandbox_memory"]
    return cfg


def validate_checkpoint_comparison(config: dict) -> None:
    """Never spend an LLM invocation on an invalid checkpoint comparison."""
    from agency.sandbox.checkpoint import CheckpointCapabilityError

    backend = config.get("checkpoint_backend", "image_commit")
    if backend not in {"image_commit", "cow_zfs"}:
        raise ValueError(f"Unknown checkpoint_backend {backend!r}")
    if backend == "cow_zfs" and not config.get("checkpoint_zfs_parent"):
        raise CheckpointCapabilityError("COW comparison requires checkpoint_zfs_parent")
    if config.get("checkpoint_fast_resume") and backend != "cow_zfs":
        raise CheckpointCapabilityError("CRIU fast resume requires cow_zfs")


def extract_payload_result(database: Path, summary: str = "") -> dict | None:
    start_pattern = re.compile(r"(?:^|\n)START_NS=(\d+)")
    end_pattern = re.compile(r"(?:^|\n)END_NS=(\d+)")
    bytes_pattern = re.compile(r"(?:^|\n)PAYLOAD_BYTES=(\d+)")
    digest_pattern = re.compile(
        r"(?:^|\n)([0-9a-f]{64}|NONE)\s+/testbed/\.agency-checkpoint-payload\.bin"
    )
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT type, payload FROM events WHERE type IN ('tool_result', 'llm_block') "
            "ORDER BY timestamp"
        ).fetchall()
    texts = [summary] if summary else []
    for event_type, payload in rows:
        decoded = json.loads(payload)
        if event_type == "tool_result":
            texts.append(decoded.get("result", ""))
        elif decoded.get("role") == "tool" and decoded.get("type") == "tool_result":
            texts.append(decoded.get("text", ""))
    matches = set()
    for text in texts:
        start = start_pattern.search(text)
        end = end_pattern.search(text)
        actual_bytes = bytes_pattern.search(text)
        sha256 = digest_pattern.search(text)
        if start and end and actual_bytes and sha256:
            matches.add(
                (
                    int(start.group(1)),
                    int(end.group(1)),
                    int(actual_bytes.group(1)),
                    sha256.group(1),
                )
            )
    if len(matches) != 1:
        return None
    start_ns, end_ns, actual_bytes, sha256 = next(iter(matches))
    return {
        "generation_seconds": (end_ns - start_ns) / 1e9,
        "generation_start_ns": start_ns,
        "generation_end_ns": end_ns,
        "actual_bytes_from_tool": actual_bytes,
        "sha256_from_tool": sha256,
    }


def verify_checkpoint_payload(image_id: str | None, requested_bytes: int, *, sandbox=None) -> dict:
    if requested_bytes == 0:
        command = f"test ! -e {PAYLOAD_PATH} && printf 'PAYLOAD_BYTES=0 PAYLOAD_SHA256=NONE\\n'"
    else:
        command = (
            f"actual=$(stat -c %s {PAYLOAD_PATH}); "
            f"digest=$(sha256sum {PAYLOAD_PATH} | cut -d' ' -f1); "
            f'test "$actual" -eq {requested_bytes}; '
            'printf \'PAYLOAD_BYTES=%s PAYLOAD_SHA256=%s\\n\' "$actual" "$digest"'
        )
    started = time.perf_counter_ns()
    if sandbox is None:
        raw = output(
            ["docker", "run", "--rm", "--entrypoint", "/bin/bash", image_id, "-lc", command]
        )
    elif image_id is None:
        raw, code = sandbox.exec(command)
        if code != 0:
            raise RuntimeError(f"Restored payload verification failed: {raw}")
        raw = raw.strip()
    else:
        backend = sandbox._backend
        raw = (
            backend._run(
                [
                    backend._runtime,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "/bin/bash",
                    image_id,
                    "-lc",
                    command,
                ],
                check=True,
            )
            .stdout.decode()
            .strip()
        )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    match = re.fullmatch(r"PAYLOAD_BYTES=(\d+) PAYLOAD_SHA256=([0-9a-f]+|NONE)", raw)
    if not match:
        raise RuntimeError(f"Unexpected checkpoint verification output: {raw!r}")
    return {
        "actual_bytes_from_checkpoint": int(match.group(1)),
        "sha256_from_checkpoint": match.group(2),
        "verification_seconds": elapsed,
    }


def run_one(
    root: Path, key_file: Path, requested_bytes: int, replicate: int, order: int, validation: bool
) -> dict:
    import agency

    manifest = json.loads((root / "manifest.json").read_text())
    validate_checkpoint_comparison(manifest["config"])
    if executing_source_digest() != manifest["agency_source"]["executing_source_sha256"]:
        raise RuntimeError("Agency source changed after cohort preparation")
    if digest(Path(__file__)) != manifest["agency_source"]["executing_runner_sha256"]:
        raise RuntimeError("Benchmark runner changed after cohort preparation")

    label = (
        "validation" if validation else f"order-{order:02d}-bytes-{requested_bytes}-r{replicate}"
    )
    directory = root / "validation" if validation else root / "runs" / label
    if directory.exists():
        raise RuntimeError(f"Refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    cfg = configure(root, key_file)
    cfg.agent.log_dir = str(directory / "logs")
    prompt = payload_instruction(requested_bytes)
    (directory / "prompt.txt").write_text(prompt + "\n")
    data = {
        "run_id": label,
        "requested_bytes": requested_bytes,
        "replicate": replicate,
        "order": order,
        "validation": validation,
        "status": "not_started",
        "start_wall_ns": time.time_ns(),
        "start_perf_ns": time.perf_counter_ns(),
        "creation_image_id": CREATION_IMAGE,
        "runtime_creation_image_id": cfg.sandbox.base_image,
        "instance_id": INSTANCE_ID,
        "repo": REPOSITORY,
        "repo_commit": REPOSITORY_COMMIT,
    }
    learner = None
    checkpoint_image = None
    cleanup_started = None
    try:
        with agency.agprof.session(
            directory / "profile", sample_hz=5, sample_gpu=False, auto_functions=False
        ):
            with agency.agprof.span("experiment:condition"):
                learner = agency.Agent(f"checkpoint_{requested_bytes}_{replicate}", agconfig=cfg)
                skill = agency.agskill(
                    name="checkpoint_size_microbenchmark",
                    prompt=SYSTEM_PROMPT,
                    input_schema=agency.agdata(payload_instruction=str),
                    output_schema=agency.agdata(summary=agency.agrawstring),
                )
                result = learner.run(
                    skill,
                    agency.agdata(payload_instruction=prompt),
                    max_steps=json.loads((root / "manifest.json").read_text())["config"][
                        "max_steps"
                    ],
                )
                result.wait()
                result_payload = result.to_dict()
                save(directory / "call-result.json", result_payload)
                if "error" in result_payload:
                    raise RuntimeError(f"Agent returned an error: {result_payload['error']}")

                backend = learner.sandbox._backend
                container_name = backend._name
                container = json.loads(
                    backend._run(
                        [backend._runtime, "container", "inspect", container_name], check=True
                    ).stdout
                )[0]
                if container["Image"].removeprefix(
                    "sha256:"
                ) != cfg.sandbox.base_image.removeprefix("sha256:"):
                    raise RuntimeError("Container creation image drifted")
                learner.data_logger.flush()
                history, inspect = [], {}
                if cfg.sandbox.checkpoint_backend == "cow_zfs":
                    checkpoint = backend._checkpointer.latest
                    if checkpoint is None or not backend._checkpointer.hibernated:
                        raise RuntimeError("Invocation did not produce a hibernated checkpoint")
                    report = {"checkpoint_id": checkpoint.reference, **checkpoint.stats}
                    report_path = directory / "cow-zfs-checkpoint.json"
                    save(report_path, report)
                    restore_started = time.perf_counter()
                    learner.sandbox.restore(checkpoint)
                    data["restore_seconds"] = time.perf_counter() - restore_started
                    # Restore annotates the same checkpoint stats with whether
                    # the optional CRIU cache was used and its restore stats.
                    # Persist the post-restore view for offline analysis.
                    report = {"checkpoint_id": checkpoint.reference, **checkpoint.stats}
                    save(report_path, report)
                    start_started = time.perf_counter()
                    learner.sandbox._backend._ensure_started()
                    data["container_start_seconds"] = time.perf_counter() - start_started
                    verification = verify_checkpoint_payload(
                        None, requested_bytes, sandbox=learner.sandbox
                    )
                    data["checkpoint_seconds"] = report["checkpoint_total_seconds"]
                else:
                    diagnostic_dir = directory / "logs" / "checkpoint-diagnostics"
                    reports = [
                        path
                        for path in diagnostic_dir.glob("*.json")
                        if not path.name.endswith((".reuse.json", ".state.json"))
                    ]
                    if len(reports) != 1:
                        raise RuntimeError(f"Expected one checkpoint report, got {len(reports)}")
                    report_path = reports[0]
                    report = json.loads(report_path.read_text())
                    if report.get("base_image_id", "").removeprefix(
                        "sha256:"
                    ) != cfg.sandbox.base_image.removeprefix("sha256:"):
                        raise RuntimeError("Checkpoint report base image drifted")
                    if not report.get("commit_success") or report.get("errors"):
                        raise RuntimeError(f"Checkpoint diagnostics failed: {report.get('errors')}")
                    checkpoint_image = report["image_id"]
                    inspect = json.loads(
                        backend._run(
                            [backend._runtime, "image", "inspect", checkpoint_image], check=True
                        ).stdout
                    )[0]
                    if backend._runtime == "docker":
                        history = image_history(checkpoint_image)
                    verification = verify_checkpoint_payload(
                        checkpoint_image, requested_bytes, sandbox=learner.sandbox
                    )
                    data["checkpoint_seconds"] = report["commit_seconds"]
                data["checkpoint_backend"] = cfg.sandbox.checkpoint_backend
                data["sandbox_runtime"] = cfg.sandbox.backend
                agent_databases = [
                    path
                    for path in (directory / "logs").glob("*_data.sqlite3")
                    if path.name != "global_data.sqlite3"
                ]
                if len(agent_databases) != 1:
                    raise RuntimeError(
                        f"Expected one agent event database, got {len(agent_databases)}"
                    )
                tool_result = extract_payload_result(
                    agent_databases[0], result_payload.get("summary", "")
                )
                if tool_result is None:
                    raise RuntimeError("Could not recover the unique payload-generation result")
                if tool_result["actual_bytes_from_tool"] != requested_bytes:
                    raise RuntimeError("Agent-reported payload size mismatch")
                if verification["actual_bytes_from_checkpoint"] != requested_bytes:
                    raise RuntimeError("Committed payload size mismatch")
                if tool_result["sha256_from_tool"] != verification["sha256_from_checkpoint"]:
                    raise RuntimeError("Committed payload digest mismatch")
                data.update(
                    status="completed",
                    container_name=container_name,
                    container_id=container["Id"],
                    container_creation_image_id=container["Image"],
                    checkpoint_report_path=str(report_path.relative_to(root)),
                    profiler_path=str((directory / "profile").relative_to(root)),
                    checkpoint_image_id=checkpoint_image,
                    checkpoint_image_size_bytes=inspect.get("Size"),
                    checkpoint_image_virtual_size_bytes=inspect.get("VirtualSize"),
                    checkpoint_rootfs_layers=inspect.get("RootFS", {}).get("Layers"),
                    docker_history_top_layer_bytes=history[0].get("Size") if history else None,
                    docker_history_top_layer_created_by=history[0].get("CreatedBy")
                    if history
                    else None,
                    payload=tool_result | verification,
                    report_checkpoint_id=report["checkpoint_id"],
                    report_commit_seconds=report.get("commit_seconds"),
                    report_collection_seconds=report.get("collection_seconds"),
                    report_image_size_delta_bytes=report.get(
                        "image_size_delta_from_creation_image_bytes"
                    ),
                )
                save(directory / "checkpoint-image-inspect.json", inspect)
                save(directory / "checkpoint-image-history.json", history)
    except Exception as exc:
        data["status"] = "failed"
        data["exception_type"] = type(exc).__name__
        data["exception_message"] = str(exc)
    finally:
        data["end_perf_ns"] = time.perf_counter_ns()
        data["end_wall_ns"] = time.time_ns()
        data["wall_seconds"] = (data["end_perf_ns"] - data["start_perf_ns"]) / 1e9
        try:
            records = agency.agprof.profile_records()
            save(directory / "records.json", records)
            if data.get("checkpoint_seconds") is not None:
                detach = sum(
                    row[3] / 1e9
                    for row in records
                    if len(row) > 3 and row[1] == "checkpoint.ptrace_detach"
                )
                data["boundary_ptrace_detach_seconds"] = detach
                data["checkpoint_including_handoff_seconds"] = data["checkpoint_seconds"] + detach
        except Exception as exc:
            data["records_error"] = f"{type(exc).__name__}: {exc}"
        cleanup_started = time.perf_counter_ns()
        try:
            if learner is not None and learner.sandbox is not None:
                storage = getattr(learner.sandbox._backend, "_checkpoint_storage", None)
                if storage is not None and storage.path is not None and storage.path.exists():
                    native = directory / "native-checkpoint"
                    native.mkdir(exist_ok=True)
                    for pattern in (
                        "graphroot/overlay-containers/*/userdata/*.log",
                        "graphroot/overlay-containers/*/userdata/checkpoint/*.log",
                    ):
                        for log in storage.path.glob(pattern):
                            (native / log.name).write_bytes(log.read_bytes())
                learner.sandbox.destroy()
            agency.get_orchestrator().shutdown()
        except Exception as exc:
            data["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            data["status"] = "failed"
        data["cleanup_seconds"] = (time.perf_counter_ns() - cleanup_started) / 1e9
        if checkpoint_image and cfg.sandbox.backend == "docker":
            data["checkpoint_image_present_after_normal_destroy"] = (
                subprocess.run(
                    ["docker", "image", "inspect", checkpoint_image],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
        save(directory / "run.json", data)
    return data


def invoke_one(root: Path, key_file: Path, condition: dict, validation: bool = False) -> dict:
    label = (
        "validation"
        if validation
        else f"order-{condition['order']:02d}-bytes-{condition['requested_bytes']}-r{condition['replicate']}"
    )
    unit = f"agency-checkpoint-size-{label}-{time.time_ns()}"
    command = [
        "systemd-run",
        *([] if os.geteuid() == 0 else ["--user"]),
        "--scope",
        "--quiet",
        "--unit",
        unit,
        sys.executable,
        str(root / "runner.py"),
        "one",
        "--root",
        str(root),
        "--key-file",
        str(key_file),
        "--requested-bytes",
        str(condition["requested_bytes"]),
        "--replicate",
        str(condition["replicate"]),
        "--order",
        str(condition["order"]),
    ]
    if validation:
        command.append("--validation")
    subprocess.run(command, check=True)
    directory = root / "validation" if validation else root / "runs" / label
    return json.loads((directory / "run.json").read_text())


def run_cohort(root: Path, key_file: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_checkpoint_comparison(manifest["config"])
    validation_condition = {"requested_bytes": 4096, "replicate": 0, "order": 0}
    validation = invoke_one(root, key_file, validation_condition, validation=True)
    if validation["status"] != "completed":
        raise RuntimeError("Validation checkpoint failed; no benchmark conditions started")
    manifest["validation"] = {
        "status": validation["status"],
        "requested_bytes": 4096,
        "run_path": "validation",
        "excluded_from_dataset": True,
    }
    save(manifest_path, manifest)

    for condition in manifest["initial_conditions"]:
        row = invoke_one(root, key_file, condition)
        manifest = json.loads(manifest_path.read_text())
        manifest["actual_execution_order"].append(
            {**condition, "run_id": row["run_id"], "status": row["status"]}
        )
        save(manifest_path, manifest)
        if row["status"] != "completed" or row.get("cleanup_error"):
            raise RuntimeError(f"Stopping after failed condition {row['run_id']}")

    by_size: dict[int, list[float]] = {}
    for path in (root / "runs").glob("*/run.json"):
        row = json.loads(path.read_text())
        by_size.setdefault(row["requested_bytes"], []).append(
            row.get("checkpoint_seconds", row.get("report_commit_seconds"))
        )
    triggered = []
    for size, durations in sorted(by_size.items()):
        if len(durations) != 3:
            raise RuntimeError(f"Expected three initial measurements for {size}")
        median = sorted(durations)[1]
        if (max(durations) - min(durations)) / median > 0.10:
            triggered.append(size)
    if triggered:
        extras = shuffled_conditions(triggered, (4, 5), ORDER_SEED + 1)
        offset = len(manifest["initial_conditions"])
        for condition in extras:
            condition["order"] += offset
        manifest["extra_conditions"] = extras
        save(manifest_path, manifest)
        for condition in extras:
            row = invoke_one(root, key_file, condition)
            manifest = json.loads(manifest_path.read_text())
            manifest["actual_execution_order"].append(
                {**condition, "run_id": row["run_id"], "status": row["status"]}
            )
            save(manifest_path, manifest)
            if row["status"] != "completed" or row.get("cleanup_error"):
                raise RuntimeError(f"Stopping after failed extra condition {row['run_id']}")
    manifest = json.loads(manifest_path.read_text())
    manifest["spread_triggered_extra_sizes_bytes"] = triggered
    manifest["completed_wall_ns"] = time.time_ns()
    manifest["status"] = "completed"
    save(manifest_path, manifest)
    (root / "REMOTE_COMPLETE").write_text(f"completed_wall_ns={time.time_ns()}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--prior-root", type=Path, required=True)
    prepare_parser.add_argument(
        "--checkpoint-backend", choices=("image_commit", "cow_zfs"), default="image_commit"
    )
    prepare_parser.add_argument("--runtime", choices=("docker", "podman"), default="docker")
    prepare_parser.add_argument("--checkpoint-zfs-parent")
    prepare_parser.add_argument("--checkpoint-fast-resume", action="store_true")
    cohort_parser = subparsers.add_parser("cohort")
    cohort_parser.add_argument("--root", type=Path, required=True)
    cohort_parser.add_argument("--key-file", type=Path, required=True)
    one_parser = subparsers.add_parser("one")
    one_parser.add_argument("--root", type=Path, required=True)
    one_parser.add_argument("--key-file", type=Path, required=True)
    one_parser.add_argument("--requested-bytes", type=int, required=True)
    one_parser.add_argument("--replicate", type=int, required=True)
    one_parser.add_argument("--order", type=int, required=True)
    one_parser.add_argument("--validation", action="store_true")
    refresh_parser = subparsers.add_parser("refresh-protocol")
    refresh_parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(
            args.root,
            args.prior_root,
            checkpoint_backend=args.checkpoint_backend,
            runtime=args.runtime,
            checkpoint_zfs_parent=args.checkpoint_zfs_parent,
            checkpoint_fast_resume=args.checkpoint_fast_resume,
        )
    elif args.mode == "cohort":
        run_cohort(args.root, args.key_file)
    elif args.mode == "refresh-protocol":
        refresh_protocol_after_failed_validation(args.root)
    else:
        result = run_one(
            args.root,
            args.key_file,
            args.requested_bytes,
            args.replicate,
            args.order,
            args.validation,
        )
        if result["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
