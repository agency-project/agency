"""Lesson 7: use a sandbox directly, without a model-backed request."""

from __future__ import annotations

import time

from agency import agSandbox, get_container_runtime

from _common import run_example, tutorial_config


def main() -> None:
    cfg, run_dir = tutorial_config("07_sandbox_api", require_llm=False)
    sandbox = agSandbox("sandbox-api", agconfig=cfg)
    forked = None
    checkpoint_tag = "agency/tutorial-sandbox-api"
    try:
        print(f"container runtime: {get_container_runtime()}")
        sandbox.write_file("/workspace/note.txt", "before checkpoint")
        sandbox.write_file_bytes("/workspace/blob.bin", b"\x00\x01agency\xff")
        assert sandbox.read_file("/workspace/note.txt") == "before checkpoint"
        assert sandbox.read_file_bytes("/workspace/blob.bin") == b"\x00\x01agency\xff"
        print("file round-trip: text='before checkpoint', binary=9 bytes")

        output, return_code = sandbox.exec("wc -c /workspace/blob.bin")
        assert return_code == 0
        print(f"exec output: {output.strip()}")

        sandbox.exec_detached("sleep 1; printf detached > /workspace/detached.txt")
        for _ in range(20):
            time.sleep(0.25)
            _, ready = sandbox.exec("test -f /workspace/detached.txt")
            if ready == 0:
                break
        assert sandbox.read_file("/workspace/detached.txt") == "detached"
        print("detached process: wrote /workspace/detached.txt")

        sandbox.update_limits(cpus=1.0, memory="1g")
        print("resource limits: cpus=1.0, memory=1g")
        assert sandbox.commit(checkpoint_tag)
        sandbox.write_file("/workspace/note.txt", "after checkpoint")
        sandbox.restore(checkpoint_tag)
        assert sandbox.read_file("/workspace/note.txt") == "before checkpoint"
        print("checkpoint restore: reverted 'after checkpoint' to 'before checkpoint'")

        forked = sandbox.fork("sandbox-api-fork")
        assert forked.read_file("/workspace/note.txt") == "before checkpoint"
        forked.write_file("/workspace/fork-only.txt", "independent")
        assert forked.read_file("/workspace/fork-only.txt") == "independent"
        _, original_has_fork_file = sandbox.exec("test -f /workspace/fork-only.txt")
        assert original_has_fork_file != 0
        print("sandbox fork: inherited checkpoint and kept later writes independent")

        copied = sandbox.get_config_copy()
        copied.sandbox.hibernation_diagnostics = True
        sandbox.change_config(copied)
        assert sandbox.get_config_copy().sandbox.hibernation_diagnostics is True
        print("sandbox config update: hibernation_diagnostics=True")
        sandbox.stop()
        print(f"sandbox stopped cleanly: kind={sandbox.image_kind}")
        print(f"artifacts: {run_dir}")
    finally:
        if forked is not None:
            forked.destroy()
        sandbox.destroy()


if __name__ == "__main__":
    run_example(main)
