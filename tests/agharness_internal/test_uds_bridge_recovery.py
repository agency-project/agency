"""Every host-side UDS bridge must survive losing its socket file.

This is a cross-service contract, so it lives in its own module rather than
being copied into each service's tests.

The failure it pins down: an external cleanup deleted a *live* terminus socket
(a Unix socket's mtime never updates, so age-based reapers treat every
long-lived one as stale). `ensure_uds_started()` kept returning its cached
path, nothing ever rebuilt the listener, and because every request across the
bridge begins with a token-validation POST to that socket, all traffic failed
with a bare `httpx.ConnectError: ENOENT` -- unauthorized and authorized
requests alike, indistinguishably, as HTTP 500s from inside the container.

Rebinding the SAME path is the load-bearing half: a container is told its
bridge's socket name once, at launch, and neither docker nor podman can change
a running container's mounts, so a rebuild under a fresh random name would be
invisible to every container already pointed at the old one.
"""

from __future__ import annotations

import os

import pytest

from agency.agharness_internal.agharness_messenger import agHarnessMessenger
from agency.agharness_internal.agllm_terminus import agLLMTerminus
from agency.agharness_internal.agmcp_server import agMCPServer
from agency.agharness_internal.agprof_ingest import agProfilerIngest
from agency.agharness_internal.agproxy_llm import agProxyLLM

_SERVICES = [
    pytest.param(agLLMTerminus, id="agllm_terminus"),
    pytest.param(agHarnessMessenger, id="agharness_messenger"),
    pytest.param(agMCPServer, id="agmcp_server"),
    pytest.param(agProxyLLM, id="agproxy_llm"),
    pytest.param(agProfilerIngest, id="agprof_ingest"),
]


@pytest.mark.parametrize("service_cls", _SERVICES)
def test_reaped_socket_is_rebuilt_at_the_same_path(service_cls):
    """Delete the socket underneath a running server -- exactly what the
    external cleaner did -- and the next ensure_uds_started() must restore a
    working listener at the identical path."""
    service = service_cls()
    try:
        path = service.ensure_uds_started()
        assert os.path.exists(path)

        os.remove(path)
        assert not os.path.exists(path)

        recovered = service.ensure_uds_started()

        assert recovered == path, (
            "rebuilt under a new name -- launched containers point at the old one"
        )
        assert os.path.exists(recovered), "returned a path with no socket behind it"
    finally:
        service.stop_uds()


@pytest.mark.parametrize("service_cls", _SERVICES)
def test_repeated_calls_do_not_relaunch_a_healthy_listener(service_cls):
    """The guard must not cost idempotency: a healthy bridge is returned as
    is, without tearing down and rebinding a socket that works."""
    service = service_cls()
    try:
        path = service.ensure_uds_started()
        inode = os.stat(path).st_ino

        assert service.ensure_uds_started() == path
        assert os.stat(path).st_ino == inode, "relaunched a healthy listener"
    finally:
        service.stop_uds()


@pytest.mark.parametrize("service_cls", _SERVICES)
def test_socket_lives_in_this_runs_gateway_dir(service_cls):
    """Run-scoping is what bounds cleanup and stops one run's container (the
    directory is bind-mounted rw into every container) from reaching another
    run's sockets."""
    from agency.agutil import agency_run_id, agharness_llm_gateway_dir

    service = service_cls()
    try:
        path = service.ensure_uds_started()

        assert os.path.dirname(path) == str(agharness_llm_gateway_dir())
        assert os.path.basename(os.path.dirname(path)) == agency_run_id()
    finally:
        service.stop_uds()
