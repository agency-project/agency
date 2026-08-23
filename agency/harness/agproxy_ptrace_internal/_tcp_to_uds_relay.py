"""Standalone in-container TCP-to-Unix-domain-socket relay.

Self-contained (stdlib only, no `agency` package imports) for the same
reason `_in_container_entrypoint.py` is: written into the sandbox
container's filesystem and run there via `docker/podman exec`, where
nothing beyond a `python3` interpreter can be assumed present.

Why this exists: a harness launched inside the container (see
`_in_container_entrypoint.py`) needs to reach `agproxy_llm`'s gateway,
which runs on the host. A plain network hop across the container boundary
turned out not to be reliable in every environment -- this host's rootless
Docker setup refused connections via both the bridge gateway IP and
`host.docker.internal`. `agsandbox` already bind-mounts a fixed host
directory into every container unconditionally (see
`agutil.agharness_llm_gateway_dir`), and a Unix domain socket placed in
that directory crosses the container boundary as a filesystem object
instead of a network connection -- unaffected by whatever networking mode
the container runtime happens to use. This relay is the last piece: the
harness itself only ever speaks plain TCP/HTTP (`ANTHROPIC_BASE_URL` is a
URL, not a socket path), so something inside the container has to listen
on a container-local TCP port and forward each connection to the
bind-mounted socket -- both ends of that forwarding live in the same
network namespace as the harness, so this hop is never subject to the
container-to-host networking question at all.

Usage: `python3 _tcp_to_uds_relay.py <uds_path> <tcp_port>` -- binds
127.0.0.1:<tcp_port> and forwards every connection to <uds_path>. Runs
until killed; intended to be launched as a background process before the
harness itself starts, and killed once the harness invocation finishes
(the harness backend that launches it owns that lifecycle, not this
script).
"""

from __future__ import annotations

import socket
import sys
import threading


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _handle_connection(tcp_conn: socket.socket, uds_path: str) -> None:
    try:
        uds_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        uds_conn.connect(uds_path)
    except OSError as exc:
        sys.stderr.write(f"_tcp_to_uds_relay: failed to connect to {uds_path!r}: {exc!r}\n")
        tcp_conn.close()
        return

    t1 = threading.Thread(target=_pump, args=(tcp_conn, uds_conn), daemon=True)
    t2 = threading.Thread(target=_pump, args=(uds_conn, tcp_conn), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    tcp_conn.close()
    uds_conn.close()


def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: _tcp_to_uds_relay.py <uds_path> <tcp_port>\n")
        sys.exit(2)
    uds_path, tcp_port = sys.argv[1], int(sys.argv[2])

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", tcp_port))
    listener.listen(64)
    # Signal readiness on stdout -- the launcher polls for this line rather
    # than assuming a fixed startup delay, same rationale as
    # agProxyLLM.start()'s server.started poll loop.
    print("READY", flush=True)

    while True:
        try:
            conn, _addr = listener.accept()
        except OSError:
            break
        threading.Thread(target=_handle_connection, args=(conn, uds_path), daemon=True).start()


if __name__ == "__main__":
    main()
