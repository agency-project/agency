"""Validate the approved tasks against the immutable Verified dataset before freezing."""

from pathlib import Path
import hashlib
import json
import os

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
TASK_IDS = [
    "matplotlib__matplotlib-25960",
    "pydata__xarray-3305",
    "django__django-15563",
    "sphinx-doc__sphinx-7748",
    "sympy__sympy-13615",
]
APPROVAL = Path(os.environ.get("NI_APPROVED_TASK_LIST", "approved-task-list.txt"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    destination = ROOT / "task_manifest.json"
    if destination.exists():
        raise SystemExit("Manifest already exists; refusing to modify frozen tasks")
    source = ROOT / "evidence/swe-bench-verified.parquet"
    metadata = json.loads((ROOT / "evidence/dataset-metadata.json").read_text())
    assert metadata["sha"] == REVISION
    assert (
        sha(source.read_bytes())
        == "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
    )
    rows = pq.read_table(source).to_pylist()
    index = {row["instance_id"]: row for row in rows}
    assert len(rows) == len(index) == 500
    assert all(task in index for task in TASK_IDS), (
        "Task membership failed: stop without replacements"
    )
    approval = APPROVAL.read_bytes()
    assert all(f"`{task}`" in approval.decode() for task in TASK_IDS)
    tasks = []
    for task_id in TASK_IDS:
        row = index[task_id]
        tasks.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "instance_id",
                        "repo",
                        "base_commit",
                        "version",
                        "environment_setup_commit",
                    )
                },
                "dataset_row_sha256": sha(canonical(row)),
                "problem_statement_sha256": sha(row["problem_statement"].encode()),
                "test_patch_sha256": sha(row["test_patch"].encode()),
                "reference_patch_sha256": sha(row["patch"].encode()),
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "dataset_revision": REVISION,
        "dataset_config": "default",
        "dataset_split": "test",
        "dataset_population": len(rows),
        "source_path": "data/test-00000-of-00001.parquet",
        "source_sha256": sha(source.read_bytes()),
        "approved_task_list_sha256": sha(approval),
        "canonicalization": "UTF-8 JSON; sorted object keys; compact separators; ensure_ascii=false; no trailing newline",
        "selection": "User-approved fixed list in specified order; no sampling or replacements",
        "tasks": tasks,
    }
    with destination.open("xb") as output:
        output.write(canonical(manifest))
    print(
        json.dumps({"tasks": TASK_IDS, "manifest_sha256": sha(destination.read_bytes())}, indent=2)
    )


if __name__ == "__main__":
    main()
