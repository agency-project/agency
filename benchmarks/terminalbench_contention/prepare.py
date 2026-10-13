"""Environment preparation, done once and outside every measured interval.

Preparation is identical for every condition: the same pinned tools, the same
dataset copy, the same digest-pinned images. It only ever adds to
``<artifact_root>/env``; it removes nothing and touches no other container
resource on the host.

Steps:
  tools        pinned Harbor virtualenv, portable harness Python, tokenizer file
  dataset      download terminal-bench/terminal-bench@4.0.0 through Harbor
  manifest     verify the frozen 24-task manifest against that download
  images       pull each task's digest-pinned environment and verifier image
  init check   start every environment image once (no agent, no model) and check
               it can host the Agency harness; check each verifier image
  oracle       optional: Harbor's reference solution -> collect -> regrade, to
               prove the grading path yields reward 1 per task
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

from . import corpus
from .util import PACKAGE, REPOSITORY, atomic_json, capture, checked, read_json_or, sha256_file

LOCKS = PACKAGE / "locks"
BRIDGE = PACKAGE / "harbor_side" / "bridge.py"
HARNESS_PYTHON_VERSION = "3.12.15"
HARNESS_PYTHON_DIR = "cpython-3.12.15-linux-x86_64-gnu"
HARBOR_PYTHON_VERSION = "3.12"
PREPARE_LABEL = "agency.tb.prepare"

INIT_PROBE = (
    "set -e; "
    '/opt/agency-python/bin/python3.12 -c "import fastapi, uvicorn, openai, httpx, mcp, '
    "pyseccomp, cloudpickle, pyte; print('harness-imports-ok')\"; "
    "{codex} --version; "
    "command -v bash >/dev/null && echo bash-ok; "
    # Agency's Codex hooks run `python3 <script>` with this PATH.
    "PATH=/usr/local/bin:/usr/bin:/bin python3 -c \"import json, sys; print('hook-python-ok')\"; "
    "id -u; pwd"
)


class PrepareError(RuntimeError):
    pass


def env_root(artifact_root: Path) -> Path:
    return Path(artifact_root) / "env"


def harbor_python(artifact_root: Path) -> Path:
    return env_root(artifact_root) / "harbor-venv" / "bin" / "python"


def harness_python_root(artifact_root: Path) -> Path:
    return env_root(artifact_root) / "harness-python" / HARNESS_PYTHON_DIR


def tiktoken_cache(artifact_root: Path) -> Path:
    return env_root(artifact_root) / "tiktoken-cache"


def bridge(artifact_root: Path, *arguments, timeout=600, env=None) -> dict:
    result = capture(
        [harbor_python(artifact_root), BRIDGE, *arguments],
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    lines = [line for line in result["stdout"].splitlines() if line.startswith("{")]
    if not lines:
        raise PrepareError(
            f"Harbor bridge produced no result: {result.get('error') or result['stderr'][-1500:]}"
        )
    return json.loads(lines[-1])


# ------------------------------------------------------------------- tools


def ensure_tools(artifact_root: Path, log=print) -> dict:
    root = env_root(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.which("uv") is None:
        raise PrepareError("uv is required to build the pinned environments")
    venv_python = harbor_python(artifact_root)
    if not venv_python.exists():
        log("creating pinned Harbor virtualenv")
        checked(
            ["uv", "venv", "--quiet", "--python", HARBOR_PYTHON_VERSION, root / "harbor-venv"],
            timeout=600,
        )
    checked(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            venv_python,
            "--require-hashes",
            "-r",
            LOCKS / "harbor-requirements.txt",
        ],
        timeout=1800,
    )
    harness = harness_python_root(artifact_root) / "bin" / "python3.12"
    if not harness.exists():
        log("installing portable harness Python")
        checked(
            [
                "uv",
                "python",
                "install",
                "--quiet",
                HARNESS_PYTHON_VERSION,
                "--install-dir",
                root / "harness-python",
            ],
            timeout=1800,
        )
    checked(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            harness,
            "--break-system-packages",
            "--require-hashes",
            "-r",
            LOCKS / "harness-requirements.txt",
        ],
        timeout=1800,
    )
    runner_python = REPOSITORY / ".venv" / "bin" / "python"
    checked(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            runner_python,
            "--require-hashes",
            "-r",
            LOCKS / "runner-requirements.txt",
        ],
        timeout=600,
    )
    cache = tiktoken_cache(artifact_root)
    cache.mkdir(exist_ok=True)
    # Loads (and on first use downloads and hash-checks) the encoding file.
    encoding = checked(
        [
            runner_python,
            "-c",
            "import tiktoken; e = tiktoken.get_encoding('o200k_base'); print(e.n_vocab)",
        ],
        timeout=300,
        env={**os.environ, "TIKTOKEN_CACHE_DIR": str(cache)},
    )
    return dict(
        harbor_python=str(venv_python),
        harbor_version=checked(
            [venv_python, "-c", "from importlib.metadata import version; print(version('harbor'))"]
        ),
        harness_python=str(harness_python_root(artifact_root)),
        harness_python_version=checked([harness, "--version"]),
        tiktoken_cache=str(cache),
        tokenizer=dict(
            encoding="o200k_base",
            vocabulary=int(encoding),
            files={p.name: sha256_file(p) for p in sorted(cache.iterdir()) if p.is_file()},
        ),
        locks={p.name: sha256_file(p) for p in sorted(LOCKS.glob("*.txt"))},
    )


# ----------------------------------------------------------------- dataset


def ensure_dataset(artifact_root: Path, dataset_ref: str, log=print) -> dict:
    root = env_root(artifact_root)
    dataset_dir = root / "dataset" / dataset_ref.split("/", 1)[0]
    if not dataset_dir.is_dir() or not any(dataset_dir.iterdir()):
        log(f"downloading {dataset_ref} through Harbor")
        (root / "dataset").mkdir(parents=True, exist_ok=True)
        harbor = harbor_python(artifact_root).parent / "harbor"
        checked([harbor, "download", dataset_ref, "-o", root / "dataset"], timeout=1800)
    parsed = bridge(
        artifact_root,
        "dataset",
        "--dataset-dir",
        dataset_dir,
        "--instructions-dir",
        root / "instructions",
    )
    parsed["dataset_ref"] = dataset_ref
    atomic_json(root / "dataset.json", parsed)
    return parsed


# ------------------------------------------------------------------ images


def inspect_image(reference: str) -> "dict | None":
    result = capture(["podman", "image", "inspect", reference], timeout=120)
    if result["returncode"] != 0:
        return None
    data = json.loads(result["stdout"])[0]
    return dict(
        id=data["Id"],
        digest=data.get("Digest"),
        repo_digests=data.get("RepoDigests"),
        size_bytes=data.get("Size"),
        architecture=data.get("Architecture"),
        os=data.get("Os"),
        user=(data.get("Config") or {}).get("User") or "",
        working_dir=(data.get("Config") or {}).get("WorkingDir") or "/",
    )


def image_has_harness_python(reference: str) -> "bool | None":
    """Whether the image offers python3 on the PATH Agency's Codex hooks run with.

    None when the image is not present locally (not inspected)."""
    if inspect_image(reference) is None:
        return None
    result = capture(
        [
            "podman",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            f"--label={PREPARE_LABEL}=probe",
            "--entrypoint",
            "sh",
            reference,
            "-c",
            f"PATH={corpus.HARNESS_PATH} python3 -c 'import json, sys'",
        ],
        timeout=300,
    )
    return result["returncode"] == 0


def ensure_image(reference: str, *, pull: bool, log=print) -> dict:
    info = inspect_image(reference)
    if info is None:
        if not pull:
            raise PrepareError(f"Image {reference} is not present (run prepare without --no-pull)")
        log(f"pulling {reference}")
        checked(["podman", "pull", "--quiet", reference], timeout=3600)
        info = inspect_image(reference)
        if info is None:
            raise PrepareError(f"Image {reference} is still missing after pull")
    wanted = reference.split("@", 1)[1]
    digests = [d.split("@", 1)[1] for d in info.get("repo_digests") or []]
    if wanted != info.get("digest") and wanted not in digests:
        raise PrepareError(f"Image {reference} resolved to a different digest {info.get('digest')}")
    return info


def check_environment_init(task: dict, artifact_root: Path, codex_binary: Path) -> dict:
    """Start the task environment once, with the harness runtime mounted, no agent."""
    name = f"agency-tb-init-{uuid.uuid4().hex[:10]}"
    harness = harness_python_root(artifact_root)
    codex_root = codex_binary.parents[3]
    started = time.monotonic()
    run = capture(
        [
            "podman",
            "run",
            "-d",
            "--pull=never",
            "--init",
            "--name",
            name,
            f"--label={PREPARE_LABEL}=init",
            "--network=none",
            f"--memory={task['environment']['memory_mb']}m",
            "-v",
            f"{harness}:/opt/agency-python:ro",
            "-v",
            f"{codex_root}:{codex_root}:ro",
            task["environment"]["image"],
            "tail",
            "-f",
            "/dev/null",
        ],
        timeout=300,
    )
    report = dict(started=run["returncode"] == 0, start_seconds=time.monotonic() - started)
    try:
        if run["returncode"] != 0:
            report["error"] = (run.get("error") or run["stderr"])[-800:]
            return report
        probe = capture(
            ["podman", "exec", name, "sh", "-c", INIT_PROBE.format(codex=codex_binary)],
            timeout=180,
        )
        lines = probe["stdout"].split("\n")
        report.update(
            probe_returncode=probe["returncode"],
            harness_imports="harness-imports-ok" in probe["stdout"],
            codex_version=next((line for line in lines if "codex" in line.lower()), None),
            bash="bash-ok" in probe["stdout"],
            hook_python="hook-python-ok" in probe["stdout"],
            uid=next((line for line in lines if line.strip().isdigit()), None),
            working_directory=lines[-2] if len(lines) >= 2 else None,
            stderr=probe["stderr"][-800:],
        )
        report["ok"] = bool(
            probe["returncode"] == 0
            and report["harness_imports"]
            and report["bash"]
            and report["hook_python"]
            and report["uid"] == "0"
            and report["codex_version"]
        )
    finally:
        capture(["podman", "rm", "--force", "--time", "0", name], timeout=120)
    return report


def check_verifier(task: dict) -> dict:
    """The verifier image must exist and carry Harbor's test entry point."""
    result = capture(
        [
            "podman",
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            f"--label={PREPARE_LABEL}=verifier",
            "--entrypoint",
            "sh",
            task["verifier"]["image"],
            "-c",
            "test -f /tests/test.sh && command -v bash",
        ],
        timeout=300,
    )
    return dict(
        ok=result["returncode"] == 0,
        returncode=result["returncode"],
        stderr=(result.get("error") or result["stderr"])[-500:],
    )


# ------------------------------------------------------------------ oracle


def oracle_roundtrip(
    task: dict, task_dir: Path, artifact_root: Path, out: Path, *, solve: bool, timeout: float
) -> dict:
    """Harbor oracle (or nothing) -> Harbor collect -> Harbor regrade."""
    kind = "oracle" if solve else "noop"
    name = f"agency-tb-{kind}-{uuid.uuid4().hex[:10]}"
    record = dict(task_id=task["task_id"], kind=kind)
    started = time.monotonic()
    run = capture(
        [
            "podman",
            "run",
            "-d",
            "--pull=never",
            "--init",
            "--name",
            name,
            f"--label={PREPARE_LABEL}={kind}",
            f"--memory={task['environment']['memory_mb']}m",
            task["environment"]["image"],
            "tail",
            "-f",
            "/dev/null",
        ],
        timeout=300,
    )
    if run["returncode"] != 0:
        return dict(record, ok=False, error=run["stderr"][-800:])
    try:
        capture(
            [
                "podman",
                "exec",
                name,
                "sh",
                "-c",
                "mkdir -p /logs/agent /logs/artifacts /logs/verifier",
            ],
            timeout=60,
        )
        if solve:
            record["oracle"] = bridge(
                artifact_root,
                "oracle",
                "--task-dir",
                task_dir,
                "--container",
                name,
                "--trial-dir",
                out / f"{kind}-agent",
                "--timeout",
                str(timeout),
                timeout=timeout + 120,
            )
        record["collect"] = bridge(
            artifact_root,
            "collect",
            "--task-dir",
            task_dir,
            "--container",
            name,
            "--trial-dir",
            out / f"{kind}-source",
            "--model",
            "oracle",
            timeout=1200,
        )
    finally:
        capture(["podman", "rm", "--force", "--time", "0", name], timeout=120)
    if not record["collect"].get("ok"):
        return dict(record, ok=False, error="artifact collection failed")
    record["verdict"] = bridge(
        artifact_root,
        "regrade",
        "--source",
        out / f"{kind}-source",
        "--task-dir",
        task_dir,
        "--trials-dir",
        out / "grading",
        "--trial-name",
        kind,
        "--timeout",
        str(timeout),
        "--label",
        f"{PREPARE_LABEL}=grade",
        timeout=timeout + 300,
    )
    record["reward"] = record["verdict"].get("reward")
    record["ok"] = bool(record["verdict"].get("ok"))
    record["seconds"] = time.monotonic() - started
    return record


# --------------------------------------------------------------------- main


def prepare(config: dict, artifact_root: Path, *, pull=True, init_check=True, log=print) -> dict:
    tools = ensure_tools(artifact_root, log)
    dataset = ensure_dataset(artifact_root, config["benchmark"]["dataset"], log)
    manifest = corpus.load_manifest()
    problems = corpus.verify_against_dataset(manifest, dataset)
    if problems:
        raise PrepareError(
            "Frozen task manifest does not match the dataset: " + "; ".join(problems)
        )
    by_name = {task.get("name"): task for task in dataset["tasks"]}
    codex = REPOSITORY / config["agent"]["codex_binary"]
    if not codex.is_file():
        raise PrepareError(f"Pinned Codex binary is missing: {codex}")
    tasks, failures = {}, []
    for task in manifest["tasks"]:
        parsed = by_name[task["task_id"]]
        environment = ensure_image(task["environment"]["image"], pull=pull, log=log)
        verifier = ensure_image(task["verifier"]["image"], pull=pull, log=log)
        record = dict(
            task_dir=parsed["path"],
            instruction_file=parsed["instruction_file"],
            instruction_sha256=parsed["rendered_instruction_sha256"],
            environment_image_id=environment["id"],
            verifier_image_id=verifier["id"],
            environment_image=environment,
            verifier_image=verifier,
        )
        if environment["user"] not in ("", "root", "0", "0:0", "root:root"):
            failures.append(f"{task['task_id']}: image user {environment['user']!r} is not root")
        if environment["architecture"] != "amd64":
            failures.append(f"{task['task_id']}: image architecture {environment['architecture']}")
        if init_check:
            log(f"init check {task['task_id']}")
            record["init_check"] = check_environment_init(task, artifact_root, codex)
            record["verifier_check"] = check_verifier(task)
            if not record["init_check"].get("ok"):
                failures.append(f"{task['task_id']}: environment cannot host the harness")
            if not record["verifier_check"].get("ok"):
                failures.append(f"{task['task_id']}: verifier image has no /tests/test.sh")
        tasks[task["task_id"]] = record
    prepared = dict(
        schema_version=1,
        prepared_at_ns=time.time_ns(),
        dataset=config["benchmark"]["dataset"],
        dataset_dir=dataset["dataset_dir"],
        corpus_sha256=manifest["corpus_sha256"],
        codex_binary=str(codex),
        codex_sha256=sha256_file(codex),
        podman=checked(["podman", "--version"]),
        init_checked=init_check,
        tasks=tasks,
        failures=failures,
        **tools,
    )
    atomic_json(env_root(artifact_root) / "prepared.json", prepared)
    if failures:
        raise PrepareError("Preparation found problems: " + "; ".join(failures))
    return prepared


def load_prepared(artifact_root: Path) -> dict:
    prepared = read_json_or(env_root(artifact_root) / "prepared.json")
    if not isinstance(prepared, dict) or not prepared.get("tasks"):
        raise PrepareError("The environment is not prepared; run the `prepare` command first")
    return prepared


def verify_prepared(prepared: dict, manifest: dict) -> list:
    """Cheap re-check before a run: nothing prepared has gone missing or changed."""
    problems = []
    if prepared.get("corpus_sha256") != manifest["corpus_sha256"]:
        problems.append("prepared environment was built for a different task manifest")
    for key in ("harbor_python", "harness_python", "tiktoken_cache", "codex_binary"):
        if not prepared.get(key) or not Path(prepared[key]).exists():
            problems.append(f"prepared {key} is missing: {prepared.get(key)}")
    if prepared.get("failures"):
        problems.append("preparation recorded failures")
    for task in manifest["tasks"]:
        record = prepared["tasks"].get(task["task_id"])
        if record is None:
            problems.append(f"{task['task_id']}: not prepared")
            continue
        for kind in ("environment", "verifier"):
            info = inspect_image(task[kind]["image"])
            if info is None:
                problems.append(f"{task['task_id']}: {kind} image is no longer present")
            elif info["id"] != record[f"{kind}_image_id"]:
                problems.append(f"{task['task_id']}: {kind} image ID changed since preparation")
        if not Path(record["instruction_file"]).is_file():
            problems.append(f"{task['task_id']}: instruction file is missing")
        elif sha256_file(record["instruction_file"]) != record["instruction_sha256"]:
            problems.append(f"{task['task_id']}: instruction text changed since preparation")
        if not Path(record["task_dir"]).is_dir():
            problems.append(f"{task['task_id']}: task directory is missing")
    return problems
