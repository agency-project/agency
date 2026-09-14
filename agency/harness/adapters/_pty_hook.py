"""Standalone Codex/Grok lifecycle observer; runs inside the sandbox."""

import json
import os
import time
import uuid
from pathlib import Path


def main():
    import sys

    payload = json.load(sys.stdin)
    # Never stamp an event with the host's *current* turn. A delayed stop must
    # retain the CLI's original turn identity across interrupt-and-submit.
    root = Path(os.environ["AGENCY_PTY_STATE"]) / "events"
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex}"
    temporary = root / (name + ".tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(root / (name + ".json"))


if __name__ == "__main__":
    main()
