"""Opt-in destructive tests confined to a freshly created private ZFS dataset.

Run as root on Linux with AGENCY_COW_CRIU_TEST=1 and AGENCY_TEST_ZFS_PARENT
pointing to an existing test dataset. Select Podman (the default) or Docker
with AGENCY_TEST_RUNTIME. The configured base image must already be loaded in
that rootful runtime. Enabled tests fail, rather than skip, on missing
capabilities. PTY tests use Agency's actual manager/controller/supervisor.
"""

import json
import os
import shlex
import time

import pytest

from agency.configs.agconfig import agconfig
from agency.sandbox.agsandbox import agSandbox
from agency.utils.agutil import AGENCY_PACKAGE_CONTAINER_MOUNT


pytestmark = pytest.mark.skipif(
    os.environ.get("AGENCY_COW_CRIU_TEST") != "1",
    reason="requires opt-in rootful Linux Docker/Podman + ZFS + CRIU host",
)

RUNTIME = os.environ.get("AGENCY_TEST_RUNTIME", "podman")


@pytest.fixture
def sandbox(request, tmp_path):
    cfg = agconfig()
    cfg.sandbox.backend = RUNTIME
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_fast_resume = True
    cfg.sandbox.checkpoint_zfs_parent = os.environ["AGENCY_TEST_ZFS_PARENT"]
    cfg.sandbox.base_image = os.environ.get("AGENCY_TEST_IMAGE", "agency-sandbox:latest")
    if RUNTIME == "docker":
        # Moby's experimental restore cannot currently recreate its managed
        # network namespace. Host networking keeps this opt-in test focused on
        # process, PTY, and filesystem continuity.
        cfg.sandbox.flags.append("--network=host")
    cfg.agent.harness = "native"
    cfg.resources.idle_cpus = 1
    sb = agSandbox("cow-criu-integration", agconfig=cfg)
    try:
        sb.exec("true")
        yield sb
    finally:
        storage = sb._backend._checkpoint_storage
        if storage.path is not None and storage.path.exists():
            from pathlib import Path

            destination = (
                Path(os.environ.get("AGENCY_TEST_ARTIFACT_DIR", str(tmp_path))) / request.node.name
            )
            destination.mkdir(parents=True, exist_ok=True)
            if sb._backend._container_running():
                for source in ("/workspace/pty-log", "/var/log/agency-daemon.log"):
                    output, code = sb._backend._container_exec_started("cat " + shlex.quote(source))
                    if code == 0:
                        (destination / Path(source).name).write_text(output)
            for extra in ("restore-mounts.json", "restore-runtime-mounts.py"):
                file = storage.fast_root / extra
                if file.exists():
                    (destination / extra).write_bytes(file.read_bytes())
            for pattern in (
                "graphroot/overlay-containers/*/userdata/*.log",
                "graphroot/overlay-containers/*/userdata/checkpoint/*.log",
            ):
                for log in storage.path.glob(pattern):
                    (destination / log.name).write_bytes(log.read_bytes())
            if sb._backend._checkpointer.latest is not None:
                (destination / "checkpoint-stats.json").write_text(
                    json.dumps(sb._backend._checkpointer.latest.stats, indent=2)
                )
        sb.destroy()


def execute(sandbox, command):
    out, code = sandbox.exec(command)
    assert code == 0, out
    return out.strip()


MEMORY_SERVER = """
import json, os, socket, uuid
token = uuid.uuid4().hex
counter = 0
os.environ['CHECKPOINT_MEMORY_TOKEN'] = token
os.chdir('/etc')
server = socket.socket(socket.AF_UNIX)
server.bind('/workspace/memory.sock')
server.listen()
while True:
    connection, _ = server.accept()
    counter += 1
    with open('/workspace/combined-state', 'w') as f:
        f.write(str(counter))
    connection.sendall(json.dumps({'token': token, 'counter': counter, 'pid': os.getpid(),
        'cwd': os.getcwd(), 'env': os.environ['CHECKPOINT_MEMORY_TOKEN']}).encode())
    connection.close()
"""


def memory_state(sandbox):
    command = """import socket
s=socket.socket(socket.AF_UNIX)
s.connect('/workspace/memory.sock')
print(s.recv(4096).decode())
s.close()
"""
    return json.loads(execute(sandbox, "python3 -c " + shlex.quote(command)))


def launch_memory_server(sandbox):
    sandbox.exec_detached("python3 -c " + shlex.quote(MEMORY_SERVER))
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if execute(sandbox, "test -S /workspace/memory.sock && echo ready || true") == "ready":
            return
        time.sleep(0.05)
    pytest.fail("memory server failed to start")


def test_full_rootfs_files_survive_hibernation_and_restore(sandbox):
    for path in ("/workspace/unique-state", "/etc/agency-state", "/usr/local/lib/agency-state"):
        sandbox.write_file(path, "checkpoint-unique-content")
    execute(sandbox, "printf '\n127.0.0.9 agency-checkpoint-test\n' >> /etc/hosts")
    handle = sandbox.checkpoint()
    assert not sandbox._backend._container_running()
    sandbox.stop()
    sandbox.restore(handle)
    for path in ("/workspace/unique-state", "/etc/agency-state", "/usr/local/lib/agency-state"):
        assert sandbox.read_file(path) == "checkpoint-unique-content"
    assert "agency-checkpoint-test" in sandbox.read_file("/etc/hosts")


def test_process_memory_and_files_continue_across_three_cycles(sandbox):
    launch_memory_server(sandbox)
    prior = memory_state(sandbox)
    assert prior["cwd"] == "/etc"
    assert prior["env"] == prior["token"]
    for _ in range(3):
        checkpoint = sandbox.checkpoint()
        assert not sandbox._backend._container_running()
        sandbox.restore(checkpoint)
        assert sandbox.read_file("/workspace/combined-state") == str(prior["counter"])
        current = memory_state(sandbox)
        assert current == {**prior, "counter": prior["counter"] + 1}
        prior = current


def test_discard_rolls_back_both_memory_and_rootfs(sandbox):
    launch_memory_server(sandbox)
    saved = memory_state(sandbox)
    handle = sandbox.checkpoint()
    sandbox.restore(handle)
    assert memory_state(sandbox)["counter"] == saved["counter"] + 1
    sandbox.write_file("/workspace/uncommitted", "discard me")
    sandbox.rm_container()
    # The ordinary exec/read API must trigger CRIU restore, never a cold start.
    assert execute(sandbox, "test ! -e /workspace/uncommitted && echo clean") == "clean"
    assert memory_state(sandbox) == {**saved, "counter": saved["counter"] + 1}


def test_actual_harness_daemon_and_control_socket_survive(sandbox):
    from agency.engine.harness_daemon_launcher import ensure_harness_daemon
    from agency.utils.agutil import new_uds_path

    host_path = new_uds_path("host")
    handle = ensure_harness_daemon(sandbox, host_path, "native", "native")
    with handle.client() as client:
        assert client.is_ready()
        before = client.daemon_identity()
        client.pause_harness()
    checkpoint = sandbox.checkpoint()
    sandbox.restore(checkpoint)
    restored = ensure_harness_daemon(sandbox, host_path, "native", "native")
    assert restored == handle
    with restored.client() as client:
        assert client.is_ready()
        after = client.daemon_identity()
        assert before[0] == after[0]  # Same container PID, not a replacement daemon.
        client.resume_harness()


def test_criu_restore_failure_falls_back_to_zfs_and_fresh_harness(sandbox, monkeypatch):
    from agency.engine.harness_daemon_launcher import ensure_harness_daemon
    from agency.utils.agutil import new_uds_path

    host_path = new_uds_path("fallback-host")
    original = ensure_harness_daemon(sandbox, host_path, "native", "native")
    with original.client() as client:
        before = client.daemon_identity()
    sandbox.write_file("/workspace/durable-fallback-state", "saved-on-zfs")
    checkpoint = sandbox.checkpoint()
    assert checkpoint.stats["fast_resume_available"] is True

    real_run = sandbox._backend._run

    def fail_process_restore(args, **kwargs):
        if args[:3] == ["podman", "container", "restore"] or args[:2] == [
            "docker",
            "start",
        ]:
            raise RuntimeError("forced CRIU restore failure")
        return real_run(args, **kwargs)

    monkeypatch.setattr(sandbox._backend, "_run", fail_process_restore)
    sandbox.restore(checkpoint)
    monkeypatch.setattr(sandbox._backend, "_run", real_run)

    assert checkpoint.stats["fast_resume_used"] is False
    assert checkpoint.stats["fast_resume_restore_error"] == "RuntimeError"
    assert sandbox.read_file("/workspace/durable-fallback-state") == "saved-on-zfs"
    replacement = ensure_harness_daemon(sandbox, host_path, "native", "native")
    with replacement.client() as client:
        after = client.daemon_identity()
        assert client.is_ready()
    assert before is not None and after is not None and after[0] != before[0]


def test_actual_agency_pty_accepts_commands_after_three_restores(sandbox, tmp_path):
    from pathlib import Path
    from agency.engine.clients import HarnessInteractionClient
    from agency.harness.protocol import HarnessAttemptRequest, PromptPayload
    from agency.utils.agutil import new_uds_path
    from agency.engine.harness_daemon_launcher import DaemonHandle, _container_socket_path

    from agency.utils.agutil import ensure_python_packages_in_container

    ensure_python_packages_in_container(
        sandbox,
        ["fastapi", "uvicorn", "openai", "httpx", "mcp", "pyseccomp", "cloudpickle", "pyte"],
        timeout_s=180,
    )
    host_socket = new_uds_path("pty-test-host")
    control_socket = new_uds_path("pty-test-control")
    sandbox.write_file(
        "/workspace/pty-daemon.py",
        Path(__file__).with_name("fixtures").joinpath("checkpoint_pty_daemon.py").read_text(),
    )
    sandbox.exec_detached(
        f"PYTHONPATH={AGENCY_PACKAGE_CONTAINER_MOUNT} python3 /workspace/pty-daemon.py "
        + shlex.quote(_container_socket_path(Path(control_socket)))
        + " "
        + shlex.quote(_container_socket_path(Path(host_socket)))
        + " >/workspace/pty-log 2>&1"
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with HarnessInteractionClient(control_socket, timeout_s=1) as client:
                if client.is_ready():
                    break
        except Exception:
            time.sleep(0.05)
    else:
        pytest.fail(execute(sandbox, "cat /workspace/pty-log"))
    sandbox._backend._agency_harness_daemon_handles = {
        "pty-test": DaemonHandle(
            host_uds_path=str(host_socket),
            sandbox_uds_path=str(control_socket),
            container_host_uds_path=_container_socket_path(Path(host_socket)),
            container_sandbox_uds_path=_container_socket_path(Path(control_socket)),
            engine_name="pty-test",
        )
    }

    def command(text, number):
        with HarnessInteractionClient(control_socket, timeout_s=30) as client:
            result = client.run_harness_attempt(
                HarnessAttemptRequest(
                    prompt=PromptPayload("", text),
                    harness="native",
                    request_id=f"request-{number}",
                    attempt_token=f"token-{number}",
                )
            )
        assert result.ok, result.error_message
        return result.final_text.split("|"), result.session_id

    prior, session = command(
        "exec 9<>/dev/tty; printf agency-pty >/proc/$$/comm; SESSION_RAM=$RANDOM$RANDOM; COUNT=0; RAM_TTY=$(readlink /proc/$$/fd/0); cd /etc; printf 0 >/workspace/pty-count",
        0,
    )
    assert prior[3] == "/etc" and prior[4].startswith("/dev/pts/")
    states = [prior]
    for number in range(1, 4):
        # The successful controller return leaves the same CLI alive. The
        # checkpoint lifecycle detaches its profiler immediately before CRIU.
        checkpoint = sandbox.checkpoint()
        assert checkpoint.stats["sessions"][0]["root_pid"] == int(prior[0])
        sandbox.restore(checkpoint)
        assert sandbox.read_file("/workspace/pty-count") == str(number - 1)
        current, current_session = command(
            "[[ $(readlink /proc/$$/fd/9) == /dev/tty ]] && COUNT=$((COUNT+1)); printf %s $COUNT >/workspace/pty-count",
            number,
        )
        assert current_session == session
        assert current == [prior[0], prior[1], str(number), prior[3], prior[4]]
        prior = current
        states.append(current)
    destination = (
        Path(os.environ.get("AGENCY_TEST_ARTIFACT_DIR", str(tmp_path))) / "pty-continuity.json"
    )
    destination.write_text(
        json.dumps(
            {
                "session_id": session,
                "columns": ["pid", "ram_token", "counter", "cwd", "tty"],
                "states": states,
            },
            indent=2,
        )
    )


@pytest.mark.skipif(
    os.environ.get("AGENCY_COW_CRIU_LLM_TEST") != "1",
    reason="requires opt-in model calls and the pinned SWE-bench image",
)
def test_codex_agent_run_continues_same_cli_after_restore(tmp_path):
    """Actual Codex adapter + host LLM/MCP + engine run/commit/restore, no test dialect."""
    from pathlib import Path
    import agency
    from benchmarks.checkpoint_size_microbenchmark.runner import (
        CREATION_CONFIG_IMAGE,
        SYSTEM_PROMPT,
        payload_instruction,
    )

    cfg = agconfig()
    cfg.sandbox.backend = RUNTIME
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_fast_resume = True
    cfg.sandbox.checkpoint_zfs_parent = os.environ["AGENCY_TEST_ZFS_PARENT"]
    cfg.sandbox.base_image = CREATION_CONFIG_IMAGE
    if RUNTIME == "docker":
        cfg.sandbox.flags.append("--network=host")
    cfg.agent.harness = "codex"
    cfg.agent.log_dir = str(tmp_path / "logs")
    cfg.llm.api_key = Path(os.environ["AGENCY_TEST_API_KEY_FILE"]).read_text().strip()
    cfg.llm.provider = "openai"
    cfg.llm.model = "gpt-5.6-luna"
    cfg.llm.temperature = 0
    cfg.llm.reasoning_effort = "none"
    cfg.ptrace.file_access = True
    cfg.llm.max_completion_tokens = 4096
    cfg.llm.context_limit = 64000
    cfg.resources.idle_cpus = 4
    cfg.resources.idle_memory = "8g"
    agent = agency.Agent("real-codex-continuity", agconfig=cfg)
    skill = agency.agskill(
        name="checkpoint_size_microbenchmark",
        prompt=SYSTEM_PROMPT,
        input_schema=agency.agdata(payload_instruction=str),
        output_schema=agency.agdata(summary=agency.agrawstring),
    )
    try:
        with agency.agprof.session(
            tmp_path / "profile", sample_hz=5, sample_gpu=False, auto_functions=False
        ):
            identities = []
            for size in (0, 4096, 8192):
                result = agent.run(
                    skill,
                    agency.agdata(payload_instruction=payload_instruction(size)),
                    max_steps=80,
                )
                result.wait()
                assert "error" not in result.to_dict(), result.to_dict()
                checkpoint = agent.sandbox._backend._checkpointer.latest
                assert agent.sandbox._backend._checkpointer.hibernated
                session = checkpoint.stats["sessions"][0]
                assert session["root_pid"]
                identities.append((session["daemon_pid"], session["root_pid"]))
                (tmp_path / "codex-continuity.json").write_text(
                    json.dumps(
                        {
                            "identities": identities,
                            "last_checkpoint": checkpoint.stats,
                        },
                        indent=2,
                    )
                )
            assert identities[0] == identities[1] == identities[2]
            assert (
                execute(agent.sandbox, "stat -c %s /testbed/.agency-checkpoint-payload.bin")
                == "8192"
            )
    finally:
        try:
            storage = agent.sandbox._backend._checkpoint_storage
            if storage.path is not None:
                for pattern in (
                    "graphroot/overlay-containers/*/userdata/*.log",
                    "graphroot/overlay-containers/*/userdata/checkpoint/*.log",
                ):
                    for log in storage.path.glob(pattern):
                        (tmp_path / log.name).write_bytes(log.read_bytes())
        finally:
            agent.sandbox.destroy()
            agency.get_orchestrator().shutdown()


@pytest.mark.skipif(RUNTIME != "podman", reason="tests Podman's shared immutable image seed")
def test_sandboxes_share_one_image_snapshot_but_isolate_writes(sandbox):
    from agency.sandbox.checkpoint import _command

    other = agSandbox("cow-criu-second-clone", agconfig=sandbox.agconfig)
    try:
        execute(other, "true")
        first_store = sandbox._backend._checkpoint_storage
        second_store = other._backend._checkpoint_storage
        assert first_store.dataset != second_store.dataset
        origins = [
            _command(["zfs", "get", "-H", "-o", "value", "origin", store.dataset]).decode().strip()
            for store in (first_store, second_store)
        ]
        assert origins[0] == origins[1] and "@seed-v1" in origins[0]
        sandbox.write_file("/workspace/clone-only", "first sandbox")
        assert execute(other, "test ! -e /workspace/clone-only && echo isolated") == "isolated"
    finally:
        other.destroy()


@pytest.mark.skipif(RUNTIME != "podman", reason="tests Podman's private dataset teardown")
def test_destroy_retries_real_busy_dataset_reader(sandbox, monkeypatch):
    """A host directory reader must not turn normal sandbox cleanup into a leak."""
    from agency.sandbox import checkpoint as cp

    storage = sandbox._backend._checkpoint_storage
    descriptor = os.open(storage.path, os.O_RDONLY | os.O_DIRECTORY)
    original = cp._command
    observed_busy = []

    def command(args, **kwargs):
        nonlocal descriptor
        try:
            return original(args, **kwargs)
        except cp.CheckpointCapabilityError as exc:
            if args == ["zfs", "destroy", "-r", storage.dataset] and "dataset is busy" in str(exc):
                observed_busy.append(str(exc))
                if descriptor is not None:
                    os.close(descriptor)
                    descriptor = None
            raise

    monkeypatch.setattr(cp, "_command", command)
    try:
        sandbox.destroy()
        assert observed_busy
        assert not storage.created
    finally:
        if descriptor is not None:
            os.close(descriptor)
