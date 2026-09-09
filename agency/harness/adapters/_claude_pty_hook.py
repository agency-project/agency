"""Standalone Claude lifecycle hook, copied into the attempt's config directory."""

import json
import os
import time
import uuid
from pathlib import Path
import sys


def main():
    root = Path(os.environ["AGENCY_CLAUDE_STATE"])
    payload = json.load(sys.stdin)
    state = json.loads((root / "agency-turn.json").read_text())
    event = {"turn_id": state["turn_id"], "payload": payload}
    destination = root / "events" / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(event))
    temporary.replace(destination)


if __name__ == "__main__":
    main()
