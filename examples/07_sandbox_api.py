"""Lesson 7: write files and run a command without calling a model."""

from agency import agSandbox, get_container_runtime

from _common import run_example, tutorial_config


def main() -> None:
    cfg, run_dir = tutorial_config("07_sandbox_api", require_llm=False)
    sandbox = agSandbox("sandbox-api", agconfig=cfg)
    try:
        print(f"container runtime: {get_container_runtime()}")
        sandbox.write_file("/workspace/note.txt", "Hello from Agency!")
        print(f"text file: {sandbox.read_file('/workspace/note.txt')}")

        sandbox.write_file_bytes("/workspace/blob.bin", b"agency")
        print(f"binary file: {sandbox.read_file_bytes('/workspace/blob.bin')!r}")
        output, return_code = sandbox.exec("wc -c /workspace/blob.bin")
        print(f"command output: {output.strip()}; exit code: {return_code}")
        if return_code:
            raise RuntimeError("Sandbox command failed")

        sandbox.update_limits(cpus=1.0, memory="1g")
        print("resource limits: cpus=1.0, memory=1g")
        print(f"artifacts: {run_dir}")
    finally:
        sandbox.destroy()


if __name__ == "__main__":
    run_example(main)
