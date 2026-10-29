"""Export reproducible study evidence after scanning for existing account credentials."""

from pathlib import Path
import hashlib
import json
import tarfile
import time
import os
from settings import credential, file_credentials

ROOT = Path(__file__).resolve().parent


def main():
    credentials = [credential(ROOT).encode()]
    for source in (os.environ, file_credentials(ROOT)):
        for name in ("LITE_LLM_KEY", "LITELLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            if len(source.get(name, "")) >= 8:
                credentials.append(source[name].encode())
    folders = {
        "evidence",
        "attempts",
        "diagnostics",
        "private",
        "job_logs",
        "capture",
        "driver",
        "launcher",
        "support",
        "figures",
    }
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] not in folders and len(relative.parts) != 1:
            continue
        if len(relative.parts) == 1 and path.suffix == ".log":
            continue  # The running pipeline appends to its own log during export/validation.
        if any(
            part in ("state", "__pycache__") or part.startswith("._") for part in relative.parts
        ):
            continue
        if (
            ".sqlite3" in path.name
            or path.name.startswith(".env")
            or path.name
            in (
                "host.local.json",
                "recorder.json",
                "results-evidence.tar.gz",
                "export_manifest.json",
                "run_integrity.json",
                "analysis_runtime.json",
                "archive_checksum.json",
            )
        ):
            continue
        # Old deployment archives contain no account credentials and preserve diagnostic code versions.
        data = path.read_bytes()
        if any(secret in data for secret in credentials):
            raise RuntimeError("Account credential found in export candidate: " + str(relative))
        files.append((path, relative, hashlib.sha256(data).hexdigest(), len(data)))
    manifest = {
        "exported_unix_s": time.time(),
        "account_credential_scan": "passed for all exported files",
        "excluded": [
            "account credentials outside study",
            "ephemeral per-run recorder authentication files",
            "SQLite databases",
            "state/checkpoints",
            "binary/runtime installs",
            "__pycache__",
        ],
        "files": {
            str(relative): {"sha256": digest, "bytes": size} for _, relative, digest, size in files
        },
    }
    output = ROOT / "export_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(ROOT / "results-evidence.tar.gz", "w:gz") as archive:
        for path, relative, _, _ in files:
            archive.add(path, arcname=str(relative), recursive=False)
        archive.add(output, arcname="export_manifest.json")
    print(
        json.dumps(
            {
                "files": len(files),
                "archive_bytes": (ROOT / "results-evidence.tar.gz").stat().st_size,
                "credential_scan": "passed",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
