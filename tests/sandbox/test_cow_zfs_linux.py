"""Native Linux correctness tests for filesystem-only ZFS hibernation."""

import os
import platform
import secrets

import pytest

from agency.configs.agconfig import agconfig
from agency.sandbox.agsandbox import agSandbox


def _runtime_cases():
    raw = os.environ.get("AGENCY_TEST_COW_RUNTIMES", "podman")
    return [value.strip() for value in raw.split(",") if value.strip()]


@pytest.fixture(params=_runtime_cases())
def sandbox(request):
    parent = os.environ.get("AGENCY_TEST_ZFS_PARENT")
    if platform.system() != "Linux" or os.geteuid() != 0 or not parent:
        pytest.skip("requires root Linux and AGENCY_TEST_ZFS_PARENT")
    cfg = agconfig()
    cfg.sandbox.backend = request.param
    cfg.sandbox.checkpoint_backend = "cow_zfs"
    cfg.sandbox.checkpoint_zfs_parent = parent
    sandbox = agSandbox(f"cow-zfs-{request.param}", agconfig=cfg)
    try:
        yield sandbox
    finally:
        sandbox.destroy()


def _run(sandbox, command):
    output, returncode = sandbox.exec(command, timeout=180)
    assert returncode == 0, output
    return output


def test_full_mutable_rootfs_survives_and_process_state_does_not(sandbox):
    token = secrets.token_hex(32)
    _run(
        sandbox,
        "set -eu; "
        "mkdir -p /workspace/state /root/.config/agency /opt/agency-venv; "
        f"printf %s {token} >/workspace/state/value; "
        "chmod 0640 /workspace/state/value; "
        "ln -s value /workspace/state/link; "
        "ln /workspace/state/value /workspace/state/hard; "
        "printf persisted >/etc/agency-cow-test; "
        "printf configured >/root/.config/agency/value; "
        "printf '\nexport AGENCY_PERSISTED_TEST=foo\n' >>/root/.bashrc; "
        "python -m venv /opt/agency-venv; "
        "/opt/agency-venv/bin/pip install --no-cache-dir six; "
        "apt-get update -qq; apt-get install -y -qq --no-install-recommends ed; "
        "rm -f /etc/issue; "
        "export AGENCY_EPHEMERAL_TEST=foo; cd /tmp; "
        "sleep 600 & printf %s $! >/workspace/state/old-background-pid",
    )
    handle = sandbox.checkpoint()
    assert handle.backend == "cow_zfs"
    assert sandbox._backend._container_running() is False

    sandbox.restore(handle)
    output = _run(
        sandbox,
        "set -eu; "
        'test "$(cat /workspace/state/value)" = ' + token + "; "
        'test "$(cat /workspace/state/link)" = ' + token + "; "
        'test "$(stat -c %h /workspace/state/value)" = 2; '
        'test "$(stat -c %a /workspace/state/value)" = 640; '
        'test "$(cat /etc/agency-cow-test)" = persisted; '
        'test "$(cat /root/.config/agency/value)" = configured; '
        "test -x /opt/agency-venv/bin/python; "
        "/opt/agency-venv/bin/python -c 'import six'; "
        "command -v ed >/dev/null; "
        "bash -ic 'test \"$AGENCY_PERSISTED_TEST\" = foo'; "
        'test -z "${AGENCY_EPHEMERAL_TEST:-}"; '
        'test "$PWD" = /workspace; '
        "test ! -e /etc/issue; "
        "! grep -l '^Name:[[:space:]]*sleep$' /proc/[0-9]*/status >/dev/null 2>&1; "
        "printf restored",
    )
    assert output.strip().endswith("restored")


def test_three_cycles_accumulate_disk_state_with_new_container_processes(sandbox):
    init_pids = []
    for generation in range(3):
        _run(sandbox, f"printf {generation} >>/workspace/generations")
        init_pids.append(
            sandbox._backend._run(
                [
                    sandbox._backend._runtime,
                    "inspect",
                    "--format",
                    "{{.State.Pid}}",
                    sandbox._backend._name,
                ],
                check=True,
            ).stdout.strip()
        )
        handle = sandbox.checkpoint()
        sandbox.restore(handle)
    assert _run(sandbox, "cat /workspace/generations").strip() == "012"
    assert len(set(init_pids)) == 3
