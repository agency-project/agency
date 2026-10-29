"""Machine-specific paths and credentials; experiment settings stay in configuration.json."""

from pathlib import Path
import json
import os
import platform
import urllib.request

DEFAULT_ROUTER = "https://router.js-park.info"


def host_settings(root, required=False):
    path = Path(os.environ.get("NI_HOST_CONFIG", str(root / "host.local.json")))
    if not path.exists():
        if required:
            raise RuntimeError(
                "Copy host.example.json to host.local.json and configure the target host"
            )
        return {}
    value = json.loads(path.read_text())
    if any(key in value for key in ("api_key", "LITE_LLM_KEY", "LITELLM_API_KEY")):
        raise ValueError(
            "Keep actual keys in the environment or an external private credentials file"
        )
    return value


def file_credentials(root):
    filename = host_settings(root).get("credentials_file")
    if not filename or not Path(filename).expanduser().is_file():
        return {}
    values = {}
    for line in Path(filename).expanduser().read_text().splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and not name.startswith("#"):
            values[name.strip()] = value.strip().strip("\"'")
    return values


def credential(root):
    for name in ("LITE_LLM_KEY", "LITELLM_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    source = file_credentials(root)
    for name in ("LITE_LLM_KEY", "LITELLM_API_KEY"):
        if source.get(name):
            return source[name]
    raise RuntimeError(
        "LiteLLM credential unavailable; set LITE_LLM_KEY or configure credentials_file"
    )


def validate_host(root):
    """Fail closed on laptops or the wrong remote host; never fall back from EC2 to Linux."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Benchmark execution requires the configured Linux x86_64 remote host")
    host = host_settings(root, required=True)
    kind = host.get("host_kind", "ec2")
    if kind == "linux":
        expected = host.get("expected_hostname")
        if not expected or platform.node() != expected:
            raise RuntimeError("Configured remote hostname does not match this machine")
        return {"kind": kind, "hostname": platform.node()}, {}
    if kind != "ec2" or not host.get("expected_instance_id"):
        raise RuntimeError(
            "Configure host_kind and the expected EC2 instance ID or remote hostname"
        )
    request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        token = response.read().decode()
    request = urllib.request.Request(
        "http://169.254.169.254/latest/dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        identity = json.load(response)
    if identity["instanceId"] != host["expected_instance_id"]:
        raise RuntimeError("Configured EC2 instance ID does not match this machine")
    return {"kind": kind, "instance_id": identity["instanceId"]}, identity
