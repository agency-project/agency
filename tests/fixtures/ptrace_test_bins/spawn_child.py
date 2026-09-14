"""Test target for agproxy_ptrace's multi-exec tracking: forks a child that
execs /bin/true, waits for it, then execs /bin/echo itself. Exercises
PTRACE_O_TRACEFORK auto-attach of the grandchild plus argv resolution across
more than one execve in the same traced tree.

Not a pytest file -- invoked as a subprocess target by
tests/harness/test_agproxy_ptrace.py, e.g. `[sys.executable, THIS_FILE]`.
"""

import os

if __name__ == "__main__":
    pid = os.fork()
    if pid == 0:
        os.execv("/bin/true", ["/bin/true"])
        os._exit(1)
    os.waitpid(pid, 0)
    os.execv("/bin/echo", ["/bin/echo", "child-ran"])
    os._exit(1)
