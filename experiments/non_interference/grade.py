"""Official SWE-bench evaluator, one independent bounded container per recorded patch."""

from pathlib import Path
import asyncio
import json
import os
import sys
import traceback
from settings import validate_host

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def grade(directory):
    configuration = json.loads((ROOT / "configuration.json").read_text())
    actual_host, _ = validate_host(ROOT)
    assert actual_host == configuration["execution_host"], "Prepared host identity changed"
    sys.path.insert(0, configuration["grader"])
    import docker
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    result = json.loads((directory / "summary.json").read_text())
    patch = directory / "prediction.patch"
    report = directory / "official-report.json"
    if report.exists():
        return
    if not patch.exists():
        write(
            report,
            {
                "resolved": None,
                "grading_error": "No patch could be recovered",
                "agent_status": result["status"],
            },
        )
        return
    if not patch.read_text().strip():
        write(report, {"resolved": False, "empty_patch": True, "agent_status": result["status"]})
        return
    task = next(
        t
        for t in json.loads((ROOT / "private/tasks.json").read_text())
        if t["instance_id"] == result["instance_id"]
    )
    client = docker.from_env()
    from docker.models.containers import ContainerCollection

    original = ContainerCollection.create

    def bounded(collection, *args, **kwargs):
        kwargs.update(nano_cpus=2000000000, mem_limit="4g", memswap_limit="4g")
        kwargs["image"] = configuration["images"][task["instance_id"]]["image_id"]
        container = original(collection, *args, **kwargs)
        container.reload()
        limits = {
            k: container.attrs["HostConfig"].get(k) for k in ("NanoCpus", "Memory", "MemorySwap")
        }
        assert (
            limits["NanoCpus"] == 2000000000
            and limits["Memory"] == limits["MemorySwap"] == 4294967296
        )
        write(directory / "grader-resources.json", limits)
        return container

    ContainerCollection.create = bounded
    os.chdir(ROOT / "private")
    spec = make_test_spec(task, namespace="swebench")
    prediction = {
        "instance_id": result["instance_id"],
        "model_name_or_path": "haiku-non-interference-" + result["run_id"],
        "model_patch": patch.read_text(),
    }
    try:
        evaluated = run_instance(
            spec,
            prediction,
            rm_image=False,
            force_rebuild=False,
            client=client,
            run_id="ni-" + directory.parent.name + "-a" + directory.name,
            timeout=600,
        )
        if evaluated is None:
            raise RuntimeError("Official evaluator returned no report")
        write(report, evaluated[1][task["instance_id"]])
    except Exception as error:
        write(
            report,
            {"resolved": None, "grading_error": str(error), "traceback": traceback.format_exc()},
        )


async def batch(diagnostic_prefix=None):
    directory = ROOT / ("diagnostics" if diagnostic_prefix else "attempts")
    semaphore = asyncio.Semaphore(4)

    async def job(path):
        async with semaphore:
            if (path / "official-report.json").exists():
                return
            with (path / "grading.log").open("ab") as output:
                process = await asyncio.create_subprocess_exec(
                    PYTHON, str(ROOT / "grade.py"), str(path), stdout=output, stderr=output
                )
                await process.wait()
            print(path.relative_to(ROOT), flush=True)

    pattern = (diagnostic_prefix + "*" if diagnostic_prefix else "*") + "/*/summary.json"
    await asyncio.gather(*(job(path.parent) for path in sorted(directory.glob(pattern))))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "batch" and not sys.argv[1].startswith("diagnostics"):
        grade(Path(sys.argv[1]).resolve())
    else:
        asyncio.run(
            batch(
                sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("diagnostics") else None
            )
        )
