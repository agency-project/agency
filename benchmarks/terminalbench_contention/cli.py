"""Command line for the Terminal-Bench 4.0 contention benchmark.

    scripts/agency-bench <command> [options]
    .venv/bin/python -m benchmarks.terminalbench_contention <command> [options]

Commands
  prepare    build pinned tools, download the dataset, pull images, check tasks
  validate   check that this host and environment can run the experiment
  plan       print/write the complete nine-run plan; launches nothing (dry run)
  run        execute ONE condition and repetition (the only command, besides
             `matrix --execute`, that makes model requests)
  matrix     show the nine commands; run them all only with --execute
  grade      grade a finished run's collected artifacts
  report     summarise finished runs
  smoke      end-to-end check against a local mock model (no paid requests)
  cleanup    remove this benchmark's leftover containers and restore placement
  calibrate-tokens   compare the token estimator with recorded usage, offline

Exit codes
  0  success
  2  usage or configuration error
  3  precondition failed; nothing was measured
  4  run aborted or its evidence is unusable
  5  run completed with infrastructure or grading failures (see summary.json)
  130 interrupted by a signal
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import config as config_module
from . import corpus, prepare, runner, topology
from .api_budget import BudgetPolicy
from .placement import restore_stale, systemd_available
from .tokens import tokenizer_fingerprint
from .util import (
    PACKAGE,
    REPOSITORY,
    atomic_json,
    capture,
    git_state,
    load_env_file,
    read_json_or,
    sha256_file,
    sha256_tree,
)

TERMINAL_BENCH_TAG = dict(
    repository="https://github.com/harbor-framework/terminal-bench",
    tag="v4.0.0",
    commit="452bf305c6daa62fc59061d22133a7cbc7c1572e",
    note="The registry release adds digest-pinned prebuilt image references to each "
    "task.toml, so registry task digests differ from this tag's dataset.toml.",
)


def say(*parts) -> None:
    print(*parts, flush=True)


def load_config(args, *, smoke=False) -> dict:
    return config_module.load(
        getattr(args, "config", None), getattr(args, "set", None), smoke=smoke
    )


def add_config_options(parser) -> None:
    parser.add_argument("--config", help="JSON file overriding default configuration keys")
    parser.add_argument(
        "--set",
        action="append",
        metavar="SECTION.KEY=VALUE",
        help="override one configuration value (repeatable)",
    )


# ------------------------------------------------------------------ checks


def environment_report(config: dict) -> dict:
    """Host facts for manifests/environment.json. Read-only."""

    def text(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    podman = capture(["podman", "info", "--format", "json"], timeout=60)
    info = {}
    if podman["returncode"] == 0:
        data = json.loads(podman["stdout"])
        host = data.get("host", {})
        info = dict(
            version=data.get("version", {}).get("Version"),
            cgroup_version=host.get("cgroupVersion"),
            cgroup_manager=host.get("cgroupManager"),
            rootless=host.get("security", {}).get("rootless"),
            oci_runtime=host.get("ociRuntime", {}).get("name"),
            graph_driver=data.get("store", {}).get("graphDriverName"),
            cgroup_controllers=host.get("cgroupControllers"),
        )
    uid = os.getuid()
    delegation = capture(
        ["systemctl", "show", f"user@{uid}.service", "-p", "Delegate", "-p", "DelegateControllers"]
    )
    try:
        layout = topology.detect_layout(config["cpu"]).to_dict()
        layout_error = None
    except (topology.InsufficientCpuError, OSError, ValueError) as exc:
        layout, layout_error = None, str(exc)
    return dict(
        generated_at_ns=time.time_ns(),
        kernel=text("/proc/sys/kernel/osrelease"),
        cpu_model=next(
            (
                line.split(":", 1)[1].strip()
                for line in (text("/proc/cpuinfo") or "").splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        cpu_layout=layout,
        cpu_layout_error=layout_error,
        smt_active=text("/sys/devices/system/cpu/smt/active"),
        memory_total_kb=next(
            (
                int(line.split()[1])
                for line in (text("/proc/meminfo") or "").splitlines()
                if line.startswith("MemTotal:")
            ),
            None,
        ),
        swap_total_kb=next(
            (
                int(line.split()[1])
                for line in (text("/proc/meminfo") or "").splitlines()
                if line.startswith("SwapTotal:")
            ),
            None,
        ),
        psi={r: text(f"/proc/pressure/{r}") is not None for r in ("cpu", "memory", "io")},
        cgroup_controllers=text("/sys/fs/cgroup/cgroup.controllers"),
        user_delegation=delegation["stdout"].strip().splitlines(),
        podman=info,
        systemd_user_manager=systemd_available(),
        python=sys.version,
        git=git_state(),
        disk_free_gb=shutil.disk_usage(REPOSITORY).free / 1024**3,
    )


def run_checks(config: dict, *, require_credentials: bool, deep: bool = True) -> list:
    """Every check returns (name, ok, detail). Nothing here starts an agent."""
    checks = []

    def add(name, ok, detail="", optional=False):
        # An optional check documents a degraded measurement; it never blocks a run.
        checks.append(dict(name=name, ok=bool(ok), detail=str(detail), optional=optional))

    root = config_module.artifact_root(config)
    # Configuration and plan.
    try:
        manifest = corpus.load_manifest()
        add(
            "task manifest loads and its corpus hash matches",
            True,
            f"{manifest['task_count']} tasks, {manifest['corpus_sha256'][:12]}",
        )
    except corpus.ManifestError as exc:
        add("task manifest loads and its corpus hash matches", False, exc)
        return checks
    identifiers = corpus.task_ids(manifest)
    try:
        plan = config_module.build_plan(config, identifiers)
        add("experiment plan builds", True, f"{len(plan)} runs")
    except config_module.ConfigError as exc:
        add("experiment plan builds", False, exc)
    add(
        "corpus has 24 distinct tasks",
        len(identifiers) == len(set(identifiers)) == config["benchmark"]["tasks"],
        len(identifiers),
    )
    # CPU.
    try:
        layout = topology.detect_layout(config["cpu"])
        add(
            "CPU topology provides reserved cores and the experimental pool",
            True,
            f"pool {layout.pool_arg} ({len(layout.experimental_cores)} cores), "
            f"reserved {layout.reserved_arg}",
        )
        add(
            "experimental pool does not share a physical core with reserved CPUs",
            not (
                {c for g in layout.reserved_cores for c in g}
                & {c for g in layout.experimental_cores for c in g}
            ),
        )
        add("single NUMA node", len(layout.numa_nodes) == 1, layout.numa_nodes)
        add("pool CPUs report one maximum frequency", not layout.heterogeneous_max_frequency)
    except (topology.InsufficientCpuError, OSError, ValueError) as exc:
        layout = None
        add("CPU topology provides reserved cores and the experimental pool", False, exc)
    # Host facilities.
    add("user systemd manager available", systemd_available())
    uid = os.getuid()
    delegated = capture(
        ["systemctl", "show", f"user@{uid}.service", "-p", "DelegateControllers", "--value"]
    )["stdout"].split()
    for controller in ("cpu", "cpuset", "memory", "pids"):
        add(
            f"cgroup controller '{controller}' delegated to the user manager",
            controller in delegated,
            " ".join(delegated),
        )
    add(
        "cgroup 'io' controller delegated (per-agent io.stat)",
        "io" in delegated,
        "not delegated: per-agent I/O falls back to /proc/<pid>/io sums"
        if "io" not in delegated
        else "",
        optional=True,
    )
    for resource in ("cpu", "memory", "io"):
        add(
            f"pressure stall information for {resource}",
            Path(f"/proc/pressure/{resource}").exists(),
        )
    add("/proc/schedstat readable (runqueue delay)", Path("/proc/schedstat").exists())
    podman = capture(
        [
            "podman",
            "info",
            "--format",
            "{{.Host.CgroupsVersion}} {{.Host.CgroupManager}} {{.Host.Security.Rootless}}",
        ],
        timeout=60,
    )
    add(
        "podman with cgroup v2 and the systemd cgroup manager",
        podman["returncode"] == 0 and podman["stdout"].split()[:2] == ["v2", "systemd"],
        podman["stdout"].strip() or podman.get("error") or podman["stderr"][-200:],
    )
    add(
        "no leftover benchmark containers",
        not runner.stale_containers(config["sandbox"]["label_prefix"]),
        runner.stale_containers(config["sandbox"]["label_prefix"]),
    )
    lease = read_json_or(root / "placement-lease.json") or {}
    add("no unrestored session CPU placement from an earlier run", not lease.get("active"))
    free_gb = shutil.disk_usage(REPOSITORY).free / 1024**3
    add(
        "free disk space above the configured minimum",
        free_gb >= config["memory"]["min_free_disk_gb"],
        f"{free_gb:.1f} GiB free",
    )
    # Prepared environment.
    try:
        prepared = prepare.load_prepared(root)
        add("environment prepared", True, f"{len(prepared['tasks'])} tasks")
    except prepare.PrepareError as exc:
        prepared = None
        add("environment prepared", False, exc)
    if prepared is not None:
        problems = prepare.verify_prepared(prepared, manifest) if deep else []
        add(
            "prepared images, instructions and tools are intact",
            not problems,
            "; ".join(problems[:5]),
        )
        add(
            "every task passed its environment init check",
            prepared.get("init_checked")
            and all((t.get("init_check") or {}).get("ok") for t in prepared["tasks"].values()),
        )
        add(
            "every task's verifier image is available with /tests/test.sh",
            prepared.get("init_checked")
            and all((t.get("verifier_check") or {}).get("ok") for t in prepared["tasks"].values()),
        )
        os.environ["TIKTOKEN_CACHE_DIR"] = prepared["tiktoken_cache"]
        fingerprint = tokenizer_fingerprint(config["api"]["estimator"]["encoding"])
        add(
            "tokenizer loads from the local cache",
            fingerprint["available"],
            fingerprint.get("error") or f"vocabulary {fingerprint.get('vocabulary_size')}",
        )
        harbor = capture(
            [
                prepared["harbor_python"],
                "-c",
                "from importlib.metadata import version; print(version('harbor'))",
            ]
        )
        add(
            "pinned Harbor version installed",
            harbor["stdout"].strip() == config["benchmark"]["harbor_version"],
            harbor["stdout"].strip(),
        )
    codex = REPOSITORY / config["agent"]["codex_binary"]
    version = capture([codex, "--version"]) if codex.is_file() else dict(stdout="", returncode=1)
    add(
        "pinned Codex binary present",
        config["agent"]["codex_version"] in version["stdout"],
        version["stdout"].strip(),
    )
    # Agency integration (imports only).
    try:
        from agency.harness.adapters.codex import CodexAdapter  # noqa: F401
        from agency.llm.openai import _OpenAICompatibleBackend
        from agency.observability.profiler import agprof

        add("Agency Codex adapter, OpenAI-compatible backend and profiler import", True)
        add(
            "Agency backend exposes the hooks the API instrumentation needs",
            all(
                hasattr(_OpenAICompatibleBackend, name)
                for name in ("make_client", "_call_backend_stream", "_call_backend")
            ),
        )
        add(
            "Agency profiler session API available",
            all(hasattr(agprof, name) for name in ("start", "stop", "summary_metrics", "span")),
        )
    except Exception as exc:
        add("Agency Codex adapter, OpenAI-compatible backend and profiler import", False, exc)
    # API configuration.
    try:
        policy = BudgetPolicy.from_dict(config["api"])
        add(
            "API admission policy valid",
            True,
            f"{policy.tokens_per_window} tokens / {policy.window_seconds:g} s, "
            f"reconciliation={policy.reconciliation}",
        )
    except Exception as exc:
        add("API admission policy valid", False, exc)
    environment = {**load_env_file(REPOSITORY / config["model"]["env_file"]), **os.environ}
    have = all(environment.get(config["model"][key]) for key in ("api_key_env", "base_url_env"))
    add(
        "model gateway credentials configured"
        + ("" if require_credentials else " (informational)"),
        have or not require_credentials,
        "present"
        if have
        else f"{config['model']['api_key_env']} / {config['model']['base_url_env']} not set",
    )
    return checks


# ---------------------------------------------------------------- commands


def command_prepare(args) -> int:
    config = load_config(args)
    root = config_module.artifact_root(config)
    try:
        prepared = prepare.prepare(
            config, root, pull=not args.no_pull, init_check=not args.no_init_check, log=say
        )
    except (prepare.PrepareError, RuntimeError, corpus.ManifestError) as exc:
        say(f"prepare failed: {exc}")
        return runner.EXIT_PRECONDITION
    say(f"prepared {len(prepared['tasks'])} tasks under {prepare.env_root(root)}")
    return runner.EXIT_OK


def command_freeze_manifest(args) -> int:
    """Maintainer step: regenerate the committed manifests from the dataset."""
    config = load_config(args)
    root = config_module.artifact_root(config)
    tools = prepare.ensure_tools(root, say)
    dataset = prepare.ensure_dataset(root, config["benchmark"]["dataset"], say)
    users, python = {}, {}
    for task in dataset["tasks"]:
        if not task.get("valid") or not task["environment"].get("docker_image"):
            continue
        reference = corpus.pinned_reference(task["environment"]["docker_image"])
        if task["directory"] in corpus.SELECTED:
            prepare.ensure_image(reference, pull=True, log=say)
        info = prepare.inspect_image(reference)
        if info is not None:  # image-level rules are checked wherever the image is local
            users[task["directory"]] = info["user"]
            python[task["directory"]] = prepare.image_has_harness_python(reference)
    manifest = corpus.build_manifest(
        dataset,
        dataset_ref=config["benchmark"]["dataset"],
        image_users=users,
        image_python=python,
    )
    atomic_json(corpus.TASKS_MANIFEST, manifest)
    codex = REPOSITORY / config["agent"]["codex_binary"]
    benchmark = dict(
        schema_version=1,
        benchmark="Terminal-Bench",
        dataset=config["benchmark"]["dataset"],
        dataset_registry="Harbor registry (harbor download)",
        terminal_bench_source=TERMINAL_BENCH_TAG,
        harbor=dict(
            version=tools["harbor_version"],
            lock="locks/harbor-requirements.txt",
            lock_sha256=tools["locks"]["harbor-requirements.txt"],
        ),
        harness_runtime=dict(
            python=tools["harness_python_version"],
            lock="locks/harness-requirements.txt",
            lock_sha256=tools["locks"]["harness-requirements.txt"],
        ),
        runner_dependencies=dict(
            lock="locks/runner-requirements.txt",
            lock_sha256=tools["locks"]["runner-requirements.txt"],
        ),
        tokenizer=tools["tokenizer"],
        codex=dict(
            version=config["agent"]["codex_version"],
            binary=config["agent"]["codex_binary"],
            sha256=sha256_file(codex) if codex.is_file() else None,
        ),
        model={k: v for k, v in config["model"].items() if not k.endswith("_env")},
        conditions=config["conditions"],
        repetitions=config["repetitions"],
        seed=config["seed"],
        corpus_sha256=manifest["corpus_sha256"],
        task_count=manifest["task_count"],
        api_policy=config["api"],
        cpu=config["cpu"],
        timeouts=config["timeouts"],
    )
    atomic_json(corpus.BENCHMARK_MANIFEST, benchmark)
    say(
        f"wrote {corpus.TASKS_MANIFEST} ({manifest['task_count']} tasks) and "
        f"{corpus.BENCHMARK_MANIFEST}"
    )
    return runner.EXIT_OK


def write_manifests(config: dict, root: Path) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for source in (corpus.TASKS_MANIFEST, corpus.BENCHMARK_MANIFEST):
        if source.exists():
            shutil.copy2(source, manifests / source.name)
    atomic_json(manifests / "environment.json", environment_report(config))


def command_validate(args) -> int:
    config = load_config(args)
    root = config_module.artifact_root(config)
    checks = run_checks(config, require_credentials=args.require_credentials)
    write_manifests(config, root)
    report = dict(
        generated_at_ns=time.time_ns(),
        ok=all(c["ok"] or c["optional"] for c in checks),
        checks=checks,
    )
    atomic_json(root / "validation" / "preflight.json", report)
    for check in checks:
        mark = "ok  " if check["ok"] else ("warn" if check["optional"] else "FAIL")
        say(f"[{mark}] {check['name']}" + (f" - {check['detail']}" if check["detail"] else ""))
    say(
        "validation "
        + ("passed" if report["ok"] else "FAILED")
        + f"; report: {root / 'validation' / 'preflight.json'}"
    )
    return runner.EXIT_OK if report["ok"] else runner.EXIT_PRECONDITION


def run_command_line(entry: dict) -> str:
    return f"scripts/agency-bench run --agents {entry['agents']} --rep {entry['repetition']}"


def command_plan(args) -> int:
    """Dry run: the complete plan, with no task or model request launched."""
    config = load_config(args)
    try:
        manifest = corpus.load_manifest()
        plan = config_module.build_plan(config, corpus.task_ids(manifest))
        layout = topology.detect_layout(config["cpu"]).to_dict()
    except (config_module.ConfigError, corpus.ManifestError, topology.InsufficientCpuError) as exc:
        say(f"cannot build the plan: {exc}")
        return runner.EXIT_PRECONDITION
    root = config_module.artifact_root(config)
    document = dict(
        dry_run=True,
        generated_at_ns=time.time_ns(),
        config_fingerprint=config_module.fingerprint(config),
        dataset=config["benchmark"]["dataset"],
        corpus_sha256=manifest["corpus_sha256"],
        tasks=corpus.task_ids(manifest),
        conditions=config["conditions"],
        repetitions=config["repetitions"],
        seed=config["seed"],
        cpu=dict(
            pool=layout["pool_arg"],
            reserved=layout["reserved_arg"],
            smt_policy=layout["smt_policy"],
        ),
        api_policy=config["api"],
        timeouts=config["timeouts"],
        runs=[
            dict(
                entry,
                command=run_command_line(entry),
                artifact_directory=str(root / "runs" / entry["run_label"]),
            )
            for entry in plan
        ],
        matched_orderings="Within a repetition every condition uses the same launch order "
        "(same seed); orders differ between repetitions.",
    )
    if args.out or args.write:
        target = Path(args.out) if args.out else root / "manifests" / "plan.json"
        atomic_json(target, document)
        say(f"plan written to {target}")
    if args.json:
        say(json.dumps(document, indent=2))
    else:
        say(
            f"{len(plan)} runs over {len(document['tasks'])} tasks; pool CPUs "
            f"{layout['pool_arg']}; budget {config['api']['tokens_per_window']} tokens/"
            f"{config['api']['window_seconds']:g}s"
        )
        for entry in plan:
            say(
                f"  {entry['index']}. {entry['run_label']:10} agents={entry['agents']:<2} "
                f"seed={entry['order_seed']}  first: "
                f"{', '.join(t.split('/')[-1] for t in entry['launch_order'][:3])}, ...  "
                f"-> {run_command_line(entry)}"
            )
    return runner.EXIT_OK


def start_run(
    config,
    entry,
    *,
    mock=None,
    grade=True,
    attempt_parent=None,
    manifest=None,
    prepared=None,
    use_systemd=True,
) -> int:
    root = config_module.artifact_root(config)
    manifest = manifest or corpus.load_manifest()
    prepared = prepared or prepare.load_prepared(root)
    layout = topology.detect_layout(config["cpu"])
    with runner.RunLock(root / "run.lock"):
        leftovers = runner.stale_containers(config["sandbox"]["label_prefix"])
        if leftovers:
            raise runner.PreconditionError(
                f"Leftover benchmark containers exist ({leftovers[:3]}); run `cleanup` first"
            )
        run = runner.build_runner(
            config,
            manifest,
            entry,
            prepared,
            layout,
            mock=mock,
            attempt_parent=attempt_parent,
            use_systemd=use_systemd,
        )
        say(
            f"run {run.run_id}: {entry['agents']} concurrent agents over "
            f"{len(entry['launch_order'])} tasks; artifacts in {run.run_dir}"
        )
        code = run.execute(grade=grade)
    summary = read_json_or(run.run_dir / "summary.json") or {}
    say(
        f"run {run.run_id} finished: status={summary.get('status')} exit={code} "
        f"makespan={summary.get('makespan_seconds')} counts={summary.get('counts')}"
    )
    return code


def command_run(args) -> int:
    config = load_config(args)
    if (args.agents is None) == (args.condition is None):
        say("give exactly one of --agents or --condition")
        return runner.EXIT_USAGE
    try:
        condition = args.condition or config_module.condition_for_agents(config, args.agents)
        manifest = corpus.load_manifest()
        entry = config_module.plan_entry(config, corpus.task_ids(manifest), condition, args.rep)
    except (config_module.ConfigError, corpus.ManifestError) as exc:
        say(f"invalid run request: {exc}")
        return runner.EXIT_USAGE
    checks = run_checks(config, require_credentials=True)
    blocking = [c for c in checks if not c["ok"] and not c["optional"]]
    if blocking:
        for check in blocking:
            say(f"[FAIL] {check['name']} - {check['detail']}")
        say("preconditions not met; nothing was started")
        return runner.EXIT_PRECONDITION
    write_manifests(config, config_module.artifact_root(config))
    try:
        return start_run(config, entry, grade=not args.no_grade, manifest=manifest)
    except (runner.PreconditionError, prepare.PrepareError) as exc:
        say(f"cannot start: {exc}")
        return runner.EXIT_PRECONDITION


def command_matrix(args) -> int:
    config = load_config(args)
    manifest = corpus.load_manifest()
    plan = config_module.build_plan(config, corpus.task_ids(manifest))
    if not args.execute:
        say("The full matrix was NOT started. Commands, in plan order:")
        for entry in plan:
            say(f"  {run_command_line(entry)}")
        say("Re-run with --execute to run all of them in sequence.")
        return runner.EXIT_OK
    checks = run_checks(config, require_credentials=True)
    blocking = [c for c in checks if not c["ok"] and not c["optional"]]
    if blocking:
        for check in blocking:
            say(f"[FAIL] {check['name']} - {check['detail']}")
        return runner.EXIT_PRECONDITION
    write_manifests(config, config_module.artifact_root(config))
    results = []
    for entry in plan:
        try:
            code = start_run(config, entry, grade=not args.no_grade, manifest=manifest)
        except (runner.PreconditionError, prepare.PrepareError) as exc:
            say(f"cannot start {entry['run_label']}: {exc}")
            code = runner.EXIT_PRECONDITION
        results.append(dict(run_label=entry["run_label"], exit_code=code))
        atomic_json(
            config_module.artifact_root(config) / "reports" / "matrix-progress.json", results
        )
        # No retries and no rescheduling: stop at the first run that did not
        # produce usable evidence and leave the decision to the operator.
        if code not in (runner.EXIT_OK, runner.EXIT_DEGRADED) or (
            code == runner.EXIT_DEGRADED and args.stop_on_degraded
        ):
            say(f"stopping the matrix after {entry['run_label']} (exit {code})")
            return code
    return max((r["exit_code"] for r in results), default=runner.EXIT_OK)


def resolve_run_dir(config, value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        path = config_module.artifact_root(config) / "runs" / value
    if (path / "summary.json").exists() or (path / "run.json").exists():
        return path
    attempts = sorted(path.glob("attempt-[0-9][0-9]"))
    if not attempts:
        raise FileNotFoundError(f"No run found at {value}")
    return attempts[-1]


def command_grade(args) -> int:
    from . import analysis

    config = load_config(args)
    try:
        run_dir = resolve_run_dir(config, args.run)
    except FileNotFoundError as exc:
        say(exc)
        return runner.EXIT_USAGE
    record = read_json_or(run_dir / "run.json")
    if not record:
        say(f"{run_dir} has no run.json; the run did not finish")
        return runner.EXIT_INVALID
    prepared = prepare.load_prepared(config_module.artifact_root(config))
    manifest = corpus.load_manifest()
    by_id = corpus.task_by_id(manifest)
    grader = runner.HarborGrader(
        prepared["harbor_python"],
        config["timeouts"]["grade_seconds"],
        f"{config['sandbox']['label_prefix']}.grade={record['run_label']}",
    )
    verdicts = read_json_or(run_dir / "grading" / "verdicts.json") or {}
    results = run_dir / "grading" / "results"
    for task in record["tasks"]:
        name = task["directory"]
        if not task["collection_ok"] or (verdicts.get(name) or {}).get("ok"):
            continue
        if (results / name).exists():  # keep the earlier attempt's evidence
            (results / name).rename(results / f"{name}.superseded-{time.time_ns()}")
        merged = {**by_id[task["task_id"]], **prepared["tasks"][task["task_id"]]}
        say(f"grading {name}")
        verdicts[name] = grader(name, merged, run_dir / "grading" / "source" / name, run_dir)
        atomic_json(run_dir / "grading" / "verdicts.json", verdicts)
    summary = analysis.finalize(run_dir, record, verdicts, config)
    previous = read_json_or(run_dir / "summary.json") or {}
    summary["exit_code"] = previous.get("exit_code")
    atomic_json(run_dir / "summary.json", summary)
    say(
        f"graded {summary['counts']['graded']} of {summary['counts']['collected']} collected "
        f"tasks; passed {summary['counts']['passed']}"
    )
    return runner.EXIT_DEGRADED if summary["counts"]["grading_failures"] else runner.EXIT_OK


def command_report(args) -> int:
    config = load_config(args)
    root = config_module.artifact_root(config)
    rows = []
    for summary_path in sorted((root / "runs").glob("*/attempt-*/summary.json")):
        summary = read_json_or(summary_path) or {}
        cross = (summary.get("cross_resource") or {}).get("inference_tool_overlap") or {}
        resources = summary.get("resources") or {}
        api = summary.get("api") or {}
        rows.append(
            dict(
                run=str(summary_path.parent.relative_to(root)),
                run_label=summary.get("run_label"),
                condition=summary.get("condition"),
                repetition=summary.get("repetition"),
                agents=summary.get("agents"),
                status=summary.get("status"),
                exit_code=summary.get("exit_code"),
                makespan_seconds=summary.get("makespan_seconds"),
                counts=summary.get("counts"),
                mean_reward_corpus=(summary.get("rewards") or {}).get("mean_reward_corpus"),
                pool_cpu_utilization=(resources.get("pool_cpu_utilization") or {}).get("mean"),
                cpu_stall_seconds=(resources.get("stall_seconds") or {}).get("workload_cpu_some"),
                api_wait_seconds=(api.get("admission_wait_seconds") or {}).get("total"),
                api_budget_utilization=(resources.get("api_budget_utilization") or {}).get("mean"),
                inference_tool_overlap_seconds=cross.get("seconds_both_present"),
            )
        )
    document = dict(
        generated_at_ns=time.time_ns(),
        runs=rows,
        note="One row per attempt. Attempts with status other than 'completed' "
        "are listed, not hidden; exclude them explicitly in analysis.",
    )
    atomic_json(root / "reports" / "runs.json", document)
    if args.json:
        say(json.dumps(document, indent=2))
    else:
        for row in rows:
            say(
                f"{row['run']:28} status={row['status']:<10} agents={row['agents']} "
                f"makespan={row['makespan_seconds']} completed="
                f"{(row['counts'] or {}).get('completed')} passed="
                f"{(row['counts'] or {}).get('passed')}"
            )
        say(f"{len(rows)} run attempt(s); details in {root / 'reports' / 'runs.json'}")
    return runner.EXIT_OK


def command_cleanup(args) -> int:
    """Remove only this benchmark's labelled containers; restore session placement."""
    config = load_config(args)
    root = config_module.artifact_root(config)
    prefix = config["sandbox"]["label_prefix"]
    removed = []
    for label in (f"{prefix}.run", f"{prefix}.grade", prepare.PREPARE_LABEL):
        listing = capture(
            ["podman", "ps", "-a", "--filter", f"label={label}", "--format", "{{.Names}}"],
            timeout=60,
        )
        for name in listing["stdout"].split():
            capture(["podman", "rm", "--force", "--time", "0", name], timeout=120)
            removed.append(name)
    restored = restore_stale(root / "placement-lease.json")
    units = capture(
        ["systemctl", "--user", "list-units", "--all", "--plain", "--no-legend", "agencytb-*"]
    )["stdout"].split("\n")
    stopped = []
    for line in units:
        if line.split():
            unit = line.split()[0]
            capture(["systemctl", "--user", "stop", unit], timeout=60)
            stopped.append(unit)
    say(
        f"removed containers: {removed or 'none'}; stopped units: {stopped or 'none'}; "
        f"placement restored: {restored if restored is not None else 'nothing to restore'}"
    )
    return runner.EXIT_OK


def command_smoke(args) -> int:
    """End-to-end run against a local mock model; no paid request is made."""
    from . import mock_model

    overrides = [
        f"benchmark.tasks={args.tasks}",
        f"conditions={json.dumps({f'SMOKE-{args.agents}': args.agents})}",
        "repetitions=1",
        f"timeouts.task_seconds={args.task_seconds}",
    ]
    config = config_module.load(args.config, [*overrides, *(args.set or [])], smoke=True)
    root = config_module.artifact_root(config)
    manifest = corpus.load_manifest()
    prepared = prepare.load_prepared(root)
    wanted = args.task or []
    chosen = (
        [t for t in manifest["tasks"] if t["directory"] in wanted]
        if wanted
        else sorted(
            manifest["tasks"],
            key=lambda t: prepared["tasks"][t["task_id"]]["environment_image"]["size_bytes"],
        )[: args.tasks]
    )
    if len(chosen) != args.tasks:
        say(f"need {args.tasks} tasks, found {len(chosen)}")
        return runner.EXIT_USAGE
    subset = dict(manifest, tasks=chosen, task_count=len(chosen))
    entry = config_module.plan_entry(
        config, [t["task_id"] for t in chosen], f"SMOKE-{args.agents}", 1
    )
    parent = root / "validation" / "smoke-tests" / (args.name or entry["run_label"])
    mock_arguments = [
        "--command",
        args.command,
        "--model",
        config["model"]["model"],
        "--log",
        str(parent / "mock-model.jsonl"),
    ]
    for flag in ("fail_429", "fail_500", "hang_first", "latency", "stream_seconds"):
        value = getattr(args, flag)
        if value:
            mock_arguments += [f"--{flag.replace('_', '-')}", str(value)]
    parent.mkdir(parents=True, exist_ok=True)
    server, base_url, _state = mock_model.start(mock_arguments)
    say(f"mock model endpoint at {base_url}")
    try:
        return start_run(
            config,
            entry,
            mock=dict(base_url=base_url),
            grade=not args.no_grade,
            attempt_parent=parent,
            manifest=subset,
            prepared=prepared,
            use_systemd=not args.no_systemd,
        )
    except (runner.PreconditionError, prepare.PrepareError) as exc:
        say(f"cannot start: {exc}")
        return runner.EXIT_PRECONDITION
    finally:
        server.shutdown()


def command_oracle(args) -> int:
    """Validate grading per task with Harbor's reference solutions (no model)."""
    config = load_config(args)
    root = config_module.artifact_root(config)
    manifest = corpus.load_manifest()
    prepared = prepare.load_prepared(root)
    out_root = root / "validation" / "oracle"
    results = read_json_or(out_root / "results.json") or {}
    selected = [t for t in manifest["tasks"] if not args.task or t["directory"] in args.task]
    failures = 0
    for task in selected:
        directory = task["directory"]
        if directory in results and results[directory].get("complete") and not args.force:
            continue
        out = out_root / directory / f"attempt-{time.time_ns()}"
        task_dir = Path(prepared["tasks"][task["task_id"]]["task_dir"])
        record = dict(task_id=task["task_id"])
        try:
            for kind, solve in (("noop", False), ("oracle", True)):
                if kind == "oracle" and args.noop_only:
                    continue
                say(f"{directory}: {kind}")
                record[kind] = prepare.oracle_roundtrip(
                    task, task_dir, root, out, solve=solve, timeout=args.timeout
                )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        noop, oracle = record.get("noop") or {}, record.get("oracle") or {}
        record["grading_path_ok"] = bool(noop.get("ok"))
        record["noop_reward"] = noop.get("reward")
        record["oracle_reward"] = oracle.get("reward")
        record["oracle_passes"] = oracle.get("reward") == 1.0 if oracle else None
        record["complete"] = bool(noop.get("ok")) and (args.noop_only or bool(oracle.get("ok")))
        results[directory] = record
        atomic_json(out_root / "results.json", results)
        if not record["complete"]:
            failures += 1
        say(
            f"{directory}: noop reward={record['noop_reward']} oracle reward="
            f"{record['oracle_reward']}"
        )
    return runner.EXIT_DEGRADED if failures else runner.EXIT_OK


def command_calibrate(args) -> int:
    from . import calibrate

    config = load_config(args)
    os.environ.setdefault(
        "TIKTOKEN_CACHE_DIR", str(prepare.tiktoken_cache(config_module.artifact_root(config)))
    )
    arguments = []
    for source in args.source:
        arguments += ["--source", source]
    if args.out:
        arguments += ["--out", args.out]
    if args.limit:
        arguments += ["--limit", str(args.limit)]
    return calibrate.main(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agency-bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("prepare", help="build tools, download dataset, pull images")
    add_config_options(sub)
    sub.add_argument("--no-pull", action="store_true", help="fail instead of pulling images")
    sub.add_argument("--no-init-check", action="store_true")
    sub.set_defaults(handler=command_prepare)

    sub = commands.add_parser("freeze-manifest", help="(maintainer) regenerate manifests")
    add_config_options(sub)
    sub.set_defaults(handler=command_freeze_manifest)

    sub = commands.add_parser("validate", help="check host, environment and integration")
    add_config_options(sub)
    sub.add_argument(
        "--require-credentials",
        action="store_true",
        help="fail if the model gateway credentials are not configured",
    )
    sub.set_defaults(handler=command_validate)

    sub = commands.add_parser("plan", help="dry run: print the full experiment plan")
    add_config_options(sub)
    sub.add_argument("--json", action="store_true")
    sub.add_argument("--write", action="store_true", help="write manifests/plan.json")
    sub.add_argument("--out")
    sub.set_defaults(handler=command_plan)

    sub = commands.add_parser("run", help="execute one condition and repetition")
    add_config_options(sub)
    sub.add_argument("--agents", type=int, help="maximum concurrent agents (6, 12 or 24)")
    sub.add_argument("--condition", help="condition name, e.g. LSB-12")
    sub.add_argument("--rep", type=int, required=True, help="repetition number, starting at 1")
    sub.add_argument("--no-grade", action="store_true", help="collect only; grade later")
    sub.set_defaults(handler=command_run)

    sub = commands.add_parser("matrix", help="list the nine runs; --execute runs them all")
    add_config_options(sub)
    sub.add_argument("--execute", action="store_true")
    sub.add_argument("--no-grade", action="store_true")
    sub.add_argument("--stop-on-degraded", action="store_true")
    sub.set_defaults(handler=command_matrix)

    sub = commands.add_parser("grade", help="grade a finished run")
    add_config_options(sub)
    sub.add_argument("--run", required=True, help="run label (e.g. LSB-6-r1) or attempt path")
    sub.set_defaults(handler=command_grade)

    sub = commands.add_parser("report", help="summarise finished runs")
    add_config_options(sub)
    sub.add_argument("--json", action="store_true")
    sub.set_defaults(handler=command_report)

    sub = commands.add_parser("cleanup", help="remove leftover benchmark containers")
    add_config_options(sub)
    sub.set_defaults(handler=command_cleanup)

    sub = commands.add_parser("smoke", help="end-to-end test with a local mock model")
    add_config_options(sub)
    sub.add_argument("--agents", type=int, default=1)
    sub.add_argument("--tasks", type=int, default=1)
    sub.add_argument("--task", action="append", help="task directory name (repeatable)")
    sub.add_argument("--name")
    sub.add_argument("--task-seconds", type=float, default=180)
    sub.add_argument(
        "--command",
        default=(
            "echo agency-smoke > /logs/artifacts/agency-smoke.txt && nproc && "
            "grep Cpus_allowed_list /proc/self/status"
        ),
    )
    sub.add_argument("--no-grade", action="store_true")
    sub.add_argument("--no-systemd", action="store_true")
    sub.add_argument("--fail-429", type=int, default=0)
    sub.add_argument("--fail-500", type=int, default=0)
    sub.add_argument("--hang-first", type=int, default=0)
    sub.add_argument("--latency", type=float, default=0)
    sub.add_argument("--stream-seconds", type=float, default=0)
    sub.set_defaults(handler=command_smoke)

    sub = commands.add_parser("oracle", help="validate grading with reference solutions")
    add_config_options(sub)
    sub.add_argument("--task", action="append")
    sub.add_argument("--timeout", type=float, default=1800)
    sub.add_argument("--noop-only", action="store_true")
    sub.add_argument("--force", action="store_true")
    sub.set_defaults(handler=command_oracle)

    sub = commands.add_parser("calibrate-tokens", help="offline estimator calibration")
    add_config_options(sub)
    sub.add_argument("--source", action="append", required=True)
    sub.add_argument("--out")
    sub.add_argument("--limit", type=int)
    sub.set_defaults(handler=command_calibrate)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except config_module.ConfigError as exc:
        say(f"configuration error: {exc}")
        return runner.EXIT_USAGE
    except KeyboardInterrupt:
        return runner.EXIT_INTERRUPTED


__all__ = ["main", "PACKAGE", "sha256_tree"]

if __name__ == "__main__":
    raise SystemExit(main())
