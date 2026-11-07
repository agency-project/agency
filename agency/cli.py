"""Host setup is opt-in; importing Agency never provisions the machine."""

import argparse
import os
import subprocess
import sys

from agency.host.profile import DEFAULT_PROFILE, read_profile


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agency")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup-host", help="Provision an isolated rootful Ubuntu ZFS host")
    setup.add_argument("--runtime", choices=["podman", "docker"], default="podman")
    setup.add_argument("--fast-resume", action="store_true", help="Require CRIU in the smoke test")
    setup.add_argument("--pool-size-gib", type=int, choices=range(8, 17), default=16)
    setup.add_argument(
        "--dry-run", action="store_true", help="Print the plan without changing the host"
    )
    setup.add_argument(
        "--skip-install", action="store_true", help="Require preinstalled system tools"
    )
    run = commands.add_parser("run", help="Run Python with the validated rootful host profile")
    run.add_argument(
        "python_args", nargs=argparse.REMAINDER, help="Python script or -m module and args"
    )
    for name, label in (("claude", "Claude Code"), ("codex", "Codex")):
        commands.add_parser(
            name,
            add_help=False,
            help=f"Run {label} in your terminal with Agency observation (see: agency {name} --help)",
        )
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] in (["claude"], ["codex"]):
        # Their own options (and the CLI's, after --) are parsed by the
        # session. They run as the invoking user with their own CLI login;
        # never root.
        if argv[0] == "claude":
            from agency.sessions.claude_code import main as interactive_session
        else:
            from agency.sessions.codex import main as interactive_session

        return interactive_session(argv[1:])
    args = parser.parse_args(argv)
    try:
        if args.command == "setup-host":
            from agency.host.setup import setup_host

            setup_host(args)
            return 0
        if os.geteuid() != 0:
            raise ValueError(
                "ZFS currently requires root: sudo .venv/bin/agency run your_script.py"
            )
        python_args = args.python_args
        if python_args[:1] == ["--"]:
            python_args = python_args[1:]
        if not python_args:
            raise ValueError("Provide a Python script, or -- -m module")
        profile = read_profile(DEFAULT_PROFILE)
        env = os.environ.copy()
        # Do not allow ambient remote-runtime settings to redirect the local profile.
        for key in (
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
            "CONTAINER_HOST",
            "CONTAINER_CONNECTION",
        ):
            env.pop(key, None)
        env["AGENCY_HOST_CONFIG"] = str(DEFAULT_PROFILE)
        if profile["docker_host"]:
            env["DOCKER_HOST"] = profile["docker_host"]
        return subprocess.call([sys.executable, *python_args], env=env)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"agency: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
