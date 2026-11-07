"""Explicit offline preparation commands; never called by run or preflight."""

import json
import random
import re
from pathlib import Path

from .config import DATASET, atomic_json, digest, private_directory, validate_manifest
from .linux import require_linux


def fetch(directory, revision):
    require_linux()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin the official Hugging Face dataset repository commit SHA")
    from datasets import load_dataset

    directory = Path(directory)
    private_directory(directory)
    rows = load_dataset(DATASET, revision=revision, split="test")
    rows.save_to_disk(str(directory / "dataset"))
    atomic_json(directory / "rows.json", list(rows))
    atomic_json(
        directory / "source.json",
        {"dataset": DATASET, "revision": revision, "content_hash": digest(list(rows))},
    )


def select(directory, images_file, repository, version, seed, replicated, output):
    from datasets import load_from_disk

    directory = Path(directory)
    source = json.loads((directory / "source.json").read_text())
    rows = list(load_from_disk(str(directory / "dataset")))
    if source["dataset"] != DATASET or digest(rows) != source["content_hash"]:
        raise ValueError("Dataset snapshot provenance/hash mismatch")
    candidates = sorted(
        [row for row in rows if row["repo"] == repository and row["version"] == version],
        key=lambda row: row["instance_id"],
    )
    count = 5 if replicated else 30
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} tasks match this repository/version; need {count}"
        )
    chosen = random.Random(seed).sample(candidates, count)
    if images_file is None:
        if Path(output).exists():
            raise ValueError("Refusing to overwrite an image map")
        atomic_json(
            output, {row["instance_id"]: "REPLACE_WITH_PREPARED_IMAGE_ID" for row in chosen}
        )
        return {
            "image_map_template": str(output),
            "instance_ids": [row["instance_id"] for row in chosen],
        }
    images = json.loads(Path(images_file).read_text())
    tasks = []
    for row in chosen:
        if row["instance_id"] not in images:
            raise ValueError(f"Prepare and freeze an image for {row['instance_id']}")
        tasks.append(
            {
                key: row[key]
                for key in ("instance_id", "repo", "version", "base_commit", "problem_statement")
            }
            | {"image": images[row["instance_id"]]}
        )
    groups = (
        [tasks[:] for _ in range(6)]
        if replicated
        else [tasks[n * 5 : (n + 1) * 5] for n in range(6)]
    )
    manifest = {
        "dataset": DATASET,
        "dataset_revision": source["revision"],
        "dataset_hash": source["content_hash"],
        "selection_seed": seed,
        "profile": {"repo": repository, "version": version},
        "replicated": replicated,
        "agents": groups,
    }
    validate_manifest(manifest)
    if Path(output).exists():
        raise ValueError("Refusing to overwrite an assignment manifest")
    atomic_json(output, manifest)
    return {"manifest": str(output), "hash": digest(manifest), "assigned": 30}


def verify_snapshot(cfg, manifest):
    rows = json.loads(Path(cfg["dataset_snapshot"]).read_text())
    if digest(rows) != manifest["dataset_hash"]:
        raise ValueError("Official evaluator snapshot hash differs from the manifest")
    by_id = {row["instance_id"]: row for row in rows}
    for group in manifest["agents"]:
        for task in group:
            source = by_id.get(task["instance_id"])
            if source is None or any(
                source.get(key) != task[key]
                for key in ("repo", "version", "base_commit", "problem_statement")
            ):
                raise ValueError("Assignment does not match the frozen official Verified row")
