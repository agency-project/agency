"""Benchmark-only native Stop barrier; never sends verifier results to Claude."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    payload = json.load(sys.stdin)
    root = Path("/study-control")
    config = json.loads((root / "config.json").read_text())
    marker = root / "native-stop.json"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            dict(
                monotonic_s=time.monotonic(),
                unix_s=time.time(),
                event=payload.get("hook_event_name"),
                session_id=payload.get("session_id"),
            )
        )
    )
    temporary.replace(marker)
    deadline = time.monotonic() + config["gate_timeout_s"]
    while not (root / "release").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Benchmark verifier barrier was not released")
        time.sleep(0.05)
    # The original native hook receives the identical payload and no reward.
    result = subprocess.run(shlex.split(sys.argv[1]), input=json.dumps(payload), text=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
