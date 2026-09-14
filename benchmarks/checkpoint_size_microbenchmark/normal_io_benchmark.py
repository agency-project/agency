"""Small matched-runtime normal I/O comparison for ext4-overlayfs and ZFS."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
import uuid
from pathlib import Path


WORKLOADS = {
    "small_files": (
        "from pathlib import Path\n"
        "p=Path('/tmp/agency-io'); p.mkdir(exist_ok=True)\n"
        "for i in range(5000): (p/f'f{i}').write_bytes((str(i)*64).encode()[:256])\n"
        "assert sum(x.stat().st_size for x in p.iterdir()) > 0\n"
        "for x in p.iterdir(): x.unlink()\n"
    ),
    "metadata": (
        "from pathlib import Path\n"
        "p=Path('/tmp/agency-io'); p.mkdir(exist_ok=True)\n"
        "for i in range(10000): (p/f'm{i}').touch()\n"
        "for i in range(10000):\n"
        " q=p/f'm{i}'; q.chmod(0o640); q.stat(); q.rename(p/f'n{i}')\n"
        "for x in p.iterdir(): x.unlink()\n"
    ),
    "sequential_write": (
        "import hashlib, pathlib\n"
        "p=pathlib.Path('/tmp/agency-io/stream')\n"
        "block=hashlib.sha256(b'agency-cow-io').digest()*32768\n"
        "with p.open('wb', buffering=0) as f:\n"
        " for _ in range(256): f.write(block)\n"
        " f.flush()\n"
        " import os; os.fsync(f.fileno())\n"
        "assert p.stat().st_size == 268435456\n"
        "p.unlink()\n"
    ),
}


def run(command, *, env=None):
    return subprocess.run(command, check=True, capture_output=True, text=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--ext4-host", default="unix:///var/run/docker.sock")
    parser.add_argument("--zfs-host", default="unix:///run/agency-docker-zfs.sock")
    args = parser.parse_args()

    backends = {"docker_ext4": args.ext4_host, "docker_zfs": args.zfs_host}
    containers = {}
    rows = []
    try:
        for backend, host in backends.items():
            name = f"agency-io-{backend}-{uuid.uuid4().hex[:8]}"
            run(
                [
                    "docker",
                    "--host",
                    host,
                    "run",
                    "-d",
                    "--network",
                    "none",
                    "--name",
                    name,
                    args.image,
                    "tail",
                    "-f",
                    "/dev/null",
                ]
            )
            containers[backend] = (host, name)
        conditions = [
            (backend, workload, repetition)
            for repetition in range(1, args.repetitions + 1)
            for workload in WORKLOADS
            for backend in backends
        ]
        random.Random(912731).shuffle(conditions)
        for order, (backend, workload, repetition) in enumerate(conditions, 1):
            host, name = containers[backend]
            started = time.perf_counter()
            run(["docker", "--host", host, "exec", name, "python", "-c", WORKLOADS[workload]])
            rows.append(
                {
                    "backend": backend,
                    "workload": workload,
                    "repetition": repetition,
                    "order": order,
                    "seconds": time.perf_counter() - started,
                }
            )
    finally:
        for host, name in containers.values():
            subprocess.run(["docker", "--host", host, "rm", "-f", name], capture_output=True)

    summary = {}
    for backend in backends:
        summary[backend] = {}
        for workload in WORKLOADS:
            values = [
                row["seconds"]
                for row in rows
                if row["backend"] == backend and row["workload"] == workload
            ]
            summary[backend][workload] = {
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
                "samples": len(values),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
