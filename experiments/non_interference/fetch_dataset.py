"""Fetch only the frozen dataset revision; never silently replace a bad local copy."""

from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent


def verify(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Dataset SHA-256 mismatch; refusing task substitution")


def main():
    manifest = json.loads((ROOT / "task_manifest.json").read_text())
    target = ROOT / "evidence/swe-bench-verified.parquet"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        url = f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/{manifest['dataset_revision']}/{manifest['source_path']}"
        request = urllib.request.Request(url, headers={"User-Agent": "agency-non-interference/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        temporary = target.with_suffix(".download")
        temporary.write_bytes(payload)
        verify(temporary, manifest["source_sha256"])
        temporary.replace(target)
    verify(target, manifest["source_sha256"])
    metadata = {
        "dataset": manifest["dataset"],
        "sha": manifest["dataset_revision"],
        "source_sha256": manifest["source_sha256"],
    }
    (target.parent / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Frozen dataset verified; generated task contents remain outside Git")


if __name__ == "__main__":
    main()
