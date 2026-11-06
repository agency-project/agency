"""Exec paths that end at a mapping's last byte must not kill the tracer."""

import os
import sys
import textwrap
from types import SimpleNamespace

import pytest

import agency.harness.ptrace._ctypes_defs as ctypes_defs
from agency.configs.agconfig import agconfig
from agency.harness.ptrace.supervisor import agProxyPtrace, ptrace_available

pytestmark = pytest.mark.skipif(not ptrace_available(), reason="requires Linux x86_64 ptrace")

_CHILD = textwrap.dedent(
    """
    import ctypes, mmap, sys
    libc = ctypes.CDLL(None, use_errno=True)
    page = mmap.PAGESIZE
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    base = libc.mmap(None, 2 * page, 3, 0x22, -1, 0)
    libc.munmap(ctypes.c_void_p(base + page), page)
    path = b"/bin/true\\x00" if sys.argv[1] == "aligned" else b"/bin/true\\x00\\x00\\x00"
    addr = base + page - len(path)
    ctypes.memmove(addr, path, len(path))
    argv = (ctypes.c_char_p * 2)(b"/bin/true", None)
    libc.execve(ctypes.c_void_p(addr), argv, None)
    sys.exit(1)
    """
)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("force_peekdata", [False, True])
@pytest.mark.parametrize("layout", ["aligned", "unaligned"])
def test_exec_path_at_mapping_end(tmp_path, monkeypatch, force_peekdata, layout):
    if force_peekdata:

        def deny(*_args):
            raise PermissionError(1, "process_vm_readv failed")

        monkeypatch.setattr(ctypes_defs, "_read_bytes_vm_readv", deny)
    child = tmp_path / "child.py"
    child.write_text(_CHILD)
    paths = []

    def check(_agent, event):
        if event.syscall == "execve":
            paths.append(event.path)
        return True

    handle = agProxyPtrace(agconfig(), allow_initial_exec=True).launch(
        [sys.executable, str(child), layout],
        dict(os.environ),
        policy=SimpleNamespace(check=check),
        ag=None,
    )
    _, _, returncode = handle.wait(timeout=30)
    handle.close()
    assert returncode == 0
    assert "/bin/true" in paths
