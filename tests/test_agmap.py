"""Tests for agmap — concurrent mapping of non-agent functions.

Container tests that create real sandboxes are gated on a reachable container
runtime (docker/podman) and skipped otherwise.
"""

from __future__ import annotations

import subprocess
import time

import pytest

from agency.agdata import agdata
from agency.agmap import agmap, agtask
from agency.agsync import agsync


def _runtime_available() -> bool:
    try:
        from agency.agsandbox import get_container_runtime

        rt = get_container_runtime()
        return subprocess.run([rt, "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


container = pytest.mark.skipif(
    not _runtime_available(), reason="container runtime (docker/podman) not reachable"
)


# ---------------------------------------------------------------------------
# Pairing modes
# ---------------------------------------------------------------------------


def test_one_fn_many_items():
    r = agmap(lambda x: {"sq": x * x}, [1, 2, 3])
    assert isinstance(r, list)
    assert [x.result for x in r] == [{"sq": 1}, {"sq": 4}, {"sq": 9}]


def test_many_fns_one_item():
    r = agmap([lambda x: x + 1, lambda x: x * 10], 7)
    assert [x.result for x in r] == [8, 70]


def test_zip_equal_lengths():
    r = agmap([lambda x: x + 1, lambda x: x * 10], [5, 6])
    assert [x.result for x in r] == [6, 60]


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        agmap([lambda x: x, lambda x: x], [1, 2, 3])


# ---------------------------------------------------------------------------
# Return shape mirrors input; value wrapping
# ---------------------------------------------------------------------------


def test_scalar_in_scalar_out():
    one = agmap(lambda x: x + 100, 5)
    assert not isinstance(one, list)
    assert one.result == 105


def test_list_in_list_out_even_single():
    r = agmap(lambda x: x, [42])
    assert isinstance(r, list) and len(r) == 1 and r[0].result == 42


def test_agdata_return_passed_through():
    r = agmap(lambda x: agdata(a=x, b="two"), 9)
    assert r.a == 9 and r.b == "two"
    assert "result" not in r.to_dict()  # not re-wrapped


def test_forwards_the_item():
    r = agmap(lambda pair: {"sum": pair[0] + pair[1]}, [(1, 2), (3, 4)])
    assert [x.result["sum"] for x in r] == [3, 7]


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


def test_error_isolated_per_item():
    r = agmap([lambda x: 1 / x, lambda x: x], [0, 9])
    assert "error" in r[0].to_dict()
    assert "division by zero" in r[0].to_dict()["error"]
    assert r[1].result == 9  # sibling unaffected


# ---------------------------------------------------------------------------
# Concurrency: sync blocks-but-overlaps; async returns immediately
# ---------------------------------------------------------------------------


def test_sync_runs_concurrently():
    def slow(x):
        time.sleep(0.3)
        return {"x": x}

    t0 = time.perf_counter()
    r = agmap(slow, [1, 2, 3, 4])
    elapsed = time.perf_counter() - t0
    assert [x.result["x"] for x in r] == [1, 2, 3, 4]
    assert elapsed < 0.3 * 4, "tasks must overlap, not run serially"


def test_results_are_agtasks():
    r = agmap(lambda x: {"v": x}, [1, 2])
    assert all(isinstance(p, agtask) for p in r)
    assert all(isinstance(p, agdata) for p in r)  # agtask is-an agdata


def test_async_returns_immediately_and_agsync_joins_targets():
    done = []

    def slow(x):
        time.sleep(0.3)
        done.append(x)
        return {"x": x}

    pend = agmap(slow, [1, 2, 3], is_asynchronous=True)
    assert done == [], "async agmap must not block"
    agsync(pend)  # explicit barrier, like agsync(team)
    assert sorted(done) == [1, 2, 3]
    assert [p.result["x"] for p in pend] == [1, 2, 3]


def test_agsync_accepts_single_agtask():
    one = agmap(lambda x: {"v": x}, 7, is_asynchronous=True)
    agsync(one)
    assert one.result == {"v": 7}


def test_agsync_rejects_plain_agdata():
    with pytest.raises(TypeError, match="agsync"):
        agsync(agdata(x=1))  # strictness preserved


def test_async_joinable_via_wait_all():
    pend = agmap(lambda x: {"v": x}, [1, 2], is_asynchronous=True)
    agdata.wait_all(pend)
    assert [p.result["v"] for p in pend] == [1, 2]


def test_bare_agsync_inside_worker_does_not_deadlock():
    """Regression: with the old global registry, a mapped fn calling agsync()
    self-joined on its own future and hung forever (and poisoned every later
    agsync call). With local futures, bare agsync() never sees agmap tasks."""

    def worker(x):
        agsync()  # must NOT wait on this task itself
        return {"x": x}

    pend = agmap(worker, [1], is_asynchronous=True)
    agdata.wait_all(pend)  # completes instead of hanging
    assert pend[0].result == {"x": 1}
    agsync()  # later bare agsync not poisoned


# ---------------------------------------------------------------------------
# Real sandboxes — one private fork per item, in parallel
# ---------------------------------------------------------------------------


@container
def _seed_base():
    from agency.agsandbox import agSandbox

    base = agSandbox("agmap-base")
    base.write_file("/workspace/repo/data.txt", "hello-from-checkpoint\n")
    # commit() checkpoints for forks; stop() hibernates (old stop(commit=True)).
    base.commit()
    base.stop()
    return base


@container
def test_agmap_sandbox_fork_per_item():
    base = _seed_base()
    try:

        def check(i):
            sb = base.fork(f"agmap-val-{i}")
            try:
                out, rc = sb.exec("cat /workspace/repo/data.txt")
                return agdata(i=i, ok=("hello-from-checkpoint" in out), rc=rc)
            finally:
                sb.destroy()

        r = agmap(check, [0, 1, 2])
        assert [x.i for x in r] == [0, 1, 2]  # order preserved
        assert all(x.ok for x in r)  # each fork has the repo
        assert all(x.rc == 0 for x in r)
    finally:
        base.destroy()


@container
def test_agmap_isolates_writes_between_forks():
    base = _seed_base()
    try:

        def mutate(tag):
            sb = base.fork(f"agmap-iso-{tag}")
            try:
                sb.exec(f"echo {tag} >> /workspace/repo/data.txt")
                out, _ = sb.exec("cat /workspace/repo/data.txt")
                return agdata(tag=tag, lines=len(out.strip().splitlines()))
            finally:
                sb.destroy()

        r = agmap(mutate, ["X", "Y"])
        # each fork started from the same 1-line checkpoint and appended one line;
        # a shared container would show 2 appended lines.
        assert all(x.lines == 2 for x in r)
    finally:
        base.destroy()
