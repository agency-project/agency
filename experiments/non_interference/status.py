"""Read-only compact study status; never reads credentials."""

from collections import Counter
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parent


def main():
    counts = {}
    for folder in ("diagnostics", "attempts"):
        results = [json.loads(p.read_text()) for p in (ROOT / folder).glob("*/*/summary.json")]
        latest = {}
        for result in results:
            if (
                result["run_id"] not in latest
                or result["attempt"] > latest[result["run_id"]]["attempt"]
            ):
                latest[result["run_id"]] = result
        counts[folder] = {
            "attempts": len(results),
            "unique": len(latest),
            "statuses": dict(Counter(r["status"] for r in latest.values())),
        }
    counts["graded"] = len(list((ROOT / "attempts").glob("*/*/official-report.json")))
    counts["gate_passed"] = (ROOT / "diagnostic_gate.json").exists()
    counts["batch_complete"] = (ROOT / "batch_complete.json").exists()
    counts["utc_unix_s"] = time.time()
    print(json.dumps(counts))
    for name in ("diagnostics-v4.log", "pipeline-v4.log"):
        p = ROOT / name
        if p.exists():
            print(name + ": " + p.read_text()[-1600:])


if __name__ == "__main__":
    main()
