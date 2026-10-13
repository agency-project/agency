"""The frozen 24-task corpus and how it was chosen.

Selection has two stages, both recorded in the generated manifest:

1. Eligibility rules, applied mechanically to every task in the published
   dataset using fields parsed by Harbor (never by guessing).
2. A fixed choice of 24 eligible tasks that spreads over the benchmark's
   categories, languages and execution styles. Tasks were not chosen for being
   CPU-heavy; the ``workload`` label is an expectation from reading the task
   definition, not a measurement.

``manifests/tasks.json`` is the frozen result. ``load_manifest`` verifies it
against the local dataset copy before any run.
"""

from __future__ import annotations

from pathlib import Path

from .util import PACKAGE, read_json, sha256_json

MANIFEST_DIR = PACKAGE / "manifests"
TASKS_MANIFEST = MANIFEST_DIR / "tasks.json"
BENCHMARK_MANIFEST = MANIFEST_DIR / "benchmark.json"

MAX_TASK_MEMORY_MB = 8192
MAX_TASK_CPUS = 6

# name -> (workload expectation, primary language/toolchain, rationale)
SELECTED = {
    "session-window-debug": (
        "mixed",
        "python",
        "Debug a stream-processing library against its design document; read/edit/test loop.",
    ),
    "wal-recovery-ordering": (
        "mixed",
        "python",
        "Repair write-ahead-log recovery with structural and performance gates.",
    ),
    "mvcc-lsm-compaction": (
        "computation-heavy",
        "c++",
        "Diagnose an MVCC visibility bug in a C++ storage model; compile and stress-test cycles.",
    ),
    "bun-sourcemap-leak": (
        "mixed",
        "typescript/bun",
        "Fix a Bun/TypeScript release pipeline; JavaScript toolchain builds.",
    ),
    "data-anonymization": (
        "mixed",
        "python",
        "Build a policy-driven CSV anonymisation CLI across related tables.",
    ),
    "ontology-kg-querying": (
        "mixed",
        "python/sparql",
        "Build an RDF/OWL pipeline and SPARQL queries; data wrangling with rdflib.",
    ),
    "photonic-waveguide-routing": (
        "computation-heavy",
        "python",
        "Constrained geometric routing; search and optimisation runs.",
    ),
    "vf2-speedup-networkx": (
        "computation-heavy",
        "python/rust/c++",
        "Write a faster graph-isomorphism library; native builds and benchmarks.",
    ),
    "ks-solver-cpp": (
        "computation-heavy",
        "c++",
        "Numerical PDE solver in C++; compile and long numeric runs.",
    ),
    "coq-block-bound": (
        "mixed",
        "coq",
        "Complete a Coq proof; proof search with repeated coqc checks.",
    ),
    "foodstuff-beta-activity": (
        "inference-heavy",
        "python",
        "Radiochemistry calculation from provided data; mostly reasoning.",
    ),
    "hof-topology-interpenetration": (
        "computation-heavy",
        "python",
        "Derive network topology and interpenetration of crystal frameworks from CIF files.",
    ),
    "sound-change-cascade": (
        "mixed",
        "python",
        "Infer ordered sound-change rules; hypothesis search with scripted checks.",
    ),
    "batched-eval-parity": (
        "mixed",
        "python",
        "Repair a batched ML evaluation harness to match single-example semantics.",
    ),
    "embedding-drift-monitor": (
        "mixed",
        "python",
        "Fix statistical drift detectors (KS, PSI, MMD) and an alert debouncer.",
    ),
    "pretrain-shard-corruption": (
        "computation-heavy",
        "python/pytorch",
        "Find and fix corrupted pretraining shards; CPU training runs.",
    ),
    "mp-checkpoint-consolidation": (
        "computation-heavy",
        "python/pytorch",
        "Consolidate parallel-training checkpoint shards; tensor processing.",
    ),
    "cargo-flight-dispatch": (
        "inference-heavy",
        "python",
        "Fix a flight-dispatch planner from operational rules; domain reasoning.",
    ),
    "fin-saccr-rwa": (
        "inference-heavy",
        "python",
        "Counterparty credit-risk capital calculation with spreadsheet output.",
    ),
    "production-planning": (
        "mixed",
        "python/sql",
        "Produce a production plan through a database gateway tool.",
    ),
    "interleaved-vigenere": (
        "mixed",
        "python",
        "Build a classical cipher cracker; cryptanalysis with scripted search.",
    ),
    "html-js-filter": (
        "mixed",
        "python",
        "Write an HTML sanitiser that survives a browser-based XSS suite.",
    ),
    "retro-console-soc": (
        "computation-heavy",
        "verilog",
        "Implement an 8-bit SoC in Verilog; simulation and synthesis tools.",
    ),
    "music-harmony": (
        "inference-heavy",
        "python/musicxml",
        "Harmonise a chorale from a PDF score; mostly reasoning.",
    ),
}

WORKLOADS = ("inference-heavy", "computation-heavy", "mixed")


HARNESS_PATH = "/usr/local/bin:/usr/bin:/bin"


def exclusion_reasons(
    task: dict, image_users: "dict | None" = None, image_python: "dict | None" = None
) -> list:
    """Why a dataset task cannot run in this experiment (empty = eligible)."""
    reasons = []
    if not task.get("valid"):
        return [f"Harbor could not parse the task: {task.get('error')}"]
    environment, verifier = task["environment"], task["verifier"]
    if (environment["gpus"] or 0) > 0:
        reasons.append("requires a GPU")
    if environment["compose_files"]:
        reasons.append("multi-container environment (docker-compose sidecar services)")
    if verifier["collect_hooks"]:
        reasons.append("verifier collect hooks must run inside the agent environment")
    if environment["mcp_servers"]:
        reasons.append("requires task-provided MCP servers")
    if environment["skills_dir"]:
        reasons.append("requires task-provided agent skills")
    if environment["healthcheck"]:
        reasons.append("requires an environment healthcheck/startup service")
    if (environment["memory_mb"] or 0) > MAX_TASK_MEMORY_MB:
        reasons.append(
            f"declares {environment['memory_mb']} MB of memory (limit {MAX_TASK_MEMORY_MB} MB "
            "for a 30 GiB host shared by up to 24 agents)"
        )
    if (environment["cpus"] or 0) > MAX_TASK_CPUS:
        reasons.append(f"declares {environment['cpus']} CPUs, more than the six-core pool")
    if environment["os"] != "linux":
        reasons.append("non-Linux task")
    if task["steps"]:
        reasons.append("multi-step task")
    if verifier["environment_mode"] != "separate":
        reasons.append("verifier is not separate-mode (cannot be graded from artifacts)")
    if not verifier["test_script"]:
        reasons.append("no tests/test.sh verifier script")
    for label, image in (
        ("environment", environment["docker_image"]),
        ("verifier", (verifier["separate_definition"] or {}).get("docker_image")),
    ):
        if not image or "@sha256:" not in image:
            reasons.append(f"no digest-pinned prebuilt {label} image")
    if task["agent"]["user"] not in (None, "root", 0, "0"):
        reasons.append(f"agent must run as non-root user {task['agent']['user']!r}")
    if environment["network_mode"] != "public":
        reasons.append(f"non-default network policy {environment['network_mode']!r}")
    if verifier["env_keys"]:
        reasons.append("verifier needs environment variables (possible external service)")
    user = (image_users or {}).get(task["directory"])
    if user not in (None, "", "root", "0", "0:0", "root:root"):
        reasons.append(f"environment image runs as non-root user {user!r}")
    # Agency installs its Codex hooks as `python3 <script>` and runs the CLI with
    # the fixed PATH above; without a python3 there the harness never sees the
    # prompt acknowledged. None means the image was not inspected.
    if (image_python or {}).get(task["directory"]) is False:
        reasons.append(
            f"environment image has no python3 on {HARNESS_PATH}, which Agency's Codex "
            "harness hooks require"
        )
    return reasons


def pinned_reference(image: str) -> str:
    """'repo:tag@sha256:x' -> 'docker.io/repo@sha256:x' (the digest is the pin)."""
    repository, digest = image.split("@", 1)
    repository = repository.split(":", 1)[0] if "/" in repository else repository
    if repository.count("/") < 2 and not repository.startswith("docker.io/"):
        repository = "docker.io/" + repository
    return f"{repository}@{digest}"


def build_manifest(
    dataset: dict, *, dataset_ref: str, image_users=None, image_python=None, images=None
) -> dict:
    """Produce the task manifest from Harbor's parse of the downloaded dataset."""
    by_directory = {task["directory"]: task for task in dataset["tasks"]}
    missing = sorted(set(SELECTED) - set(by_directory))
    if missing:
        raise RuntimeError(f"Selected tasks are absent from the dataset: {missing}")
    tasks, excluded, eligible_unselected = [], [], []
    for directory, task in sorted(by_directory.items()):
        reasons = exclusion_reasons(task, image_users, image_python)
        if directory in SELECTED:
            if reasons:
                raise RuntimeError(f"Selected task {directory} is not eligible: {reasons}")
            workload, language, rationale = SELECTED[directory]
            environment, verifier = task["environment"], task["verifier"]
            environment_image = pinned_reference(environment["docker_image"])
            verifier_image = pinned_reference(verifier["separate_definition"]["docker_image"])
            metadata = task.get("metadata") or {}
            record = dict(
                task_id=task["name"],
                directory=directory,
                version=dataset_ref.split("@", 1)[-1],
                digest=task["digest"],
                instruction_sha256=task["instruction_sha256"],
                environment_sha256=task["environment_sha256"],
                tests_sha256=task["tests_sha256"],
                category=metadata.get("category"),
                subcategory=metadata.get("subcategory"),
                tags=metadata.get("tags") or [],
                language=language,
                environment=dict(
                    image=environment_image,
                    image_declared=environment["docker_image"],
                    cpus=environment["cpus"],
                    memory_mb=environment["memory_mb"],
                    storage_mb=environment["storage_mb"],
                    gpus=environment["gpus"] or 0,
                    network_mode=environment["network_mode"],
                    privileged=False,
                    external_services=[],
                ),
                verifier=dict(
                    image=verifier_image,
                    image_declared=verifier["separate_definition"]["docker_image"],
                    mode=verifier["environment_mode"],
                    timeout_sec=verifier["timeout_sec"],
                    cpus=verifier["separate_definition"]["cpus"],
                    memory_mb=verifier["separate_definition"]["memory_mb"],
                    bundled_tests=verifier["separate_definition"]["bundled_tests"],
                ),
                agent=dict(official_timeout_sec=task["agent"]["timeout_sec"], user="root"),
                artifacts=task["artifacts"],
                expected_resources=dict(
                    workload=workload,
                    declared_cpus=environment["cpus"],
                    declared_memory_mb=environment["memory_mb"],
                    declared_storage_mb=environment["storage_mb"],
                    expert_time_estimate_hours=metadata.get("expert_time_estimate_hours"),
                    source="task.toml [environment]/[metadata]; workload label from reading "
                    "the task definition, not measured",
                ),
                selection_rationale=rationale,
            )
            for key, value in (images or {}).get(directory, {}).items():
                record[key] = value
            tasks.append(record)
        elif reasons:
            excluded.append(dict(task_id=task.get("name", directory), reasons=reasons))
        else:
            eligible_unselected.append(task["name"])
    counts = {w: sum(t["expected_resources"]["workload"] == w for t in tasks) for w in WORKLOADS}
    categories: dict = {}
    for task in tasks:
        categories[task["category"]] = categories.get(task["category"], 0) + 1
    manifest = dict(
        schema_version=1,
        dataset=dataset_ref,
        harbor_version=dataset["harbor_version"],
        task_count=len(tasks),
        tasks=tasks,
        composition=dict(by_workload=counts, by_category=categories),
        selection=dict(
            eligibility_rules=[
                "no GPU",
                "single container (no docker-compose sidecars)",
                "no verifier collect hooks, MCP servers, agent skills or healthcheck services",
                f"declared memory <= {MAX_TASK_MEMORY_MB} MB and CPUs <= {MAX_TASK_CPUS}",
                "Linux, single step, separate-mode verifier with tests/test.sh",
                "digest-pinned prebuilt environment and verifier images",
                "agent and image default user is root",
                f"environment image provides python3 on {HARNESS_PATH} (Agency Codex hooks)",
                "public (default) network policy; verifier needs no credentials",
            ],
            method=(
                "24 of the eligible tasks, fixed by hand to cover every benchmark category "
                "and a mix of languages and of inference-heavy, computation-heavy and mixed "
                "work. No task was chosen for CPU intensity alone."
            ),
            eligible_not_selected=sorted(eligible_unselected),
            eligible_not_selected_note=(
                "Image-level rules (root user, python3 on the harness PATH) were only "
                "checked for images that had been pulled; unpulled tasks in this list "
                "passed every task.toml rule."
            ),
            excluded=excluded,
        ),
    )
    manifest["corpus_sha256"] = sha256_json(
        [
            (t["task_id"], t["digest"], t["environment"]["image"], t["verifier"]["image"])
            for t in tasks
        ]
    )
    return manifest


class ManifestError(RuntimeError):
    pass


def load_manifest(path: Path = TASKS_MANIFEST) -> dict:
    try:
        manifest = read_json(path)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"Cannot read task manifest {path}: {exc}") from exc
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ManifestError("Task manifest contains no tasks")
    identifiers = [t["task_id"] for t in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ManifestError("Task manifest contains duplicate task IDs")
    expected = sha256_json(
        [
            (t["task_id"], t["digest"], t["environment"]["image"], t["verifier"]["image"])
            for t in tasks
        ]
    )
    if manifest.get("corpus_sha256") != expected:
        raise ManifestError("Task manifest corpus hash does not match its contents")
    for task in tasks:
        for kind in ("environment", "verifier"):
            if "@sha256:" not in task[kind]["image"]:
                raise ManifestError(f"{task['task_id']}: {kind} image is not digest-pinned")
    return manifest


def task_ids(manifest: dict) -> list:
    return [task["task_id"] for task in manifest["tasks"]]


def task_by_id(manifest: dict) -> dict:
    return {task["task_id"]: task for task in manifest["tasks"]}


def verify_against_dataset(manifest: dict, dataset: dict) -> list:
    """Compare the frozen manifest with Harbor's parse of the local dataset."""
    problems = []
    local = {task.get("name"): task for task in dataset["tasks"] if task.get("valid")}
    for task in manifest["tasks"]:
        found = local.get(task["task_id"])
        if found is None:
            problems.append(f"{task['task_id']}: missing from the local dataset copy")
            continue
        for key in ("digest", "instruction_sha256", "environment_sha256", "tests_sha256"):
            if found[key] != task[key]:
                problems.append(f"{task['task_id']}: {key} differs from the frozen manifest")
        if exclusion_reasons(found):
            problems.append(f"{task['task_id']}: no longer eligible: {exclusion_reasons(found)}")
    return problems
