"""Cost-free live telemetry smoke: real local tools, no agent/model invocation.

Start the web server separately with the same --run-dir, then run this file.
Normal Agent/harness executions use these same HostInteractionServer events.
The local test task is deliberately tiny; the tool I/O, results and durations
are observed, not synthetic. For a full agent use examples/10_harnesses_and_webui.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from agency.agpolicy import agpolicy
from agency.configs.agconfig import agconfig
from agency.engine.host_servers.host_interaction_server import HostInteractionServer
from agency.observability.agdatalogger import agDataLogger


def logger(path, name):
    config = agconfig()
    config.data_logger.db_path = str(path)
    result = agDataLogger(config, default_name=name, default_object="trajectory-local-tools")
    result.start()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/tmp/agency-trajectory-live/logs"))
    parser.add_argument("--delay", type=float, default=3)
    args = parser.parse_args()
    if any(args.run_dir.glob("*.sqlite3")):
        parser.error(
            "Use a fresh --run-dir. Existing telemetry is preserved; choose a new directory."
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    workspace = args.run_dir / "local-tools-workspace"
    workspace.mkdir(exist_ok=True)
    target = workspace / "retry.py"
    target.write_text("RETRIES = 2\n")
    (workspace / "test_retry.py").write_text(
        "import time, unittest\nfrom retry import RETRIES\n"
        "class RetryTest(unittest.TestCase):\n"
        "    def test_retry_limit(self):\n"
        f"        time.sleep({max(0, args.delay)!r})\n"
        "        self.assertEqual(RETRIES, 3, 'expected 3 retries')\n"
    )
    global_log = logger(args.run_dir / "global_data.sqlite3", "workflow")
    agent_log = logger(args.run_dir / "local-tools.sqlite3", "local-tools")
    server = HostInteractionServer(
        SimpleNamespace(policy=agpolicy()),
        agent_log,
        "local-tools",
        profile_attributes={"request_id": "local-tools-smoke"},
    )
    global_log.record_event(
        "agent_registered",
        {"db_path": str(Path(agent_log.db_path).resolve())},
        name="local-tools",
        update_latest_snapshot=True,
        flush=True,
    )
    global_log.record_event(
        "workload_started",
        {
            "task": "Local tool smoke: verify a retry constant. No model or autonomous agent is invoked."
        },
        flush=True,
    )

    def run_tool(tool, arguments, operation):
        admission = server.admit_tool_call(tool, arguments)
        result = operation()
        server.complete_tool_call(admission["call_id"], result)
        return result

    def check():
        result = subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "test_retry"],
            cwd=workspace,
            capture_output=True,
            text=True,
        )
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    status = "completed"
    try:
        run_tool("read_file", {"path": str(target)}, lambda: {"content": target.read_text()})
        for _ in range(3):
            run_tool("Bash", {"command": f"{sys.executable} -B -m unittest test_retry"}, check)

        def edit():
            target.write_text("RETRIES = 3\n")
            return {
                "result": "Wrote the retry constant",
                "artifacts": [
                    {
                        "path": str(target),
                        "change": "modified",
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "source": "observed_local_file",
                    }
                ],
            }

        run_tool("edit", {"path": str(target)}, edit)
        run_tool("Bash", {"command": f"{sys.executable} -B -m unittest test_retry"}, check)
        print(json.dumps({"status": status, "run_dir": str(args.run_dir)}))
    except Exception:
        status = "failed"
        raise
    except BaseException:
        status = "cancelled"
        raise
    finally:
        global_log.record_event("done", {"status": status}, update_latest_snapshot=True, flush=True)
        agent_log.stop()
        global_log.stop()


if __name__ == "__main__":
    main()
