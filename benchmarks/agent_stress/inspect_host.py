"""Read-only Linux inventory. Never starts a container or benchmark workload."""

import datetime
import json
from pathlib import Path
import subprocess
import urllib.request


def capture(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=25)
        return {
            "command": argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "error": str(exc)}


def inventory():
    commands = {
        "hostname": ["hostname"],
        "kernel": ["uname", "-a"],
        "os": ["cat", "/etc/os-release"],
        "cpu": ["lscpu", "-J"],
        "topology": ["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"],
        "memory": ["free", "-b"],
        "uptime": ["uptime"],
        "process_cpu": [
            "ps",
            "-eo",
            "user,pid,ppid,psr,pcpu,pmem,rss,etime,stat,comm",
            "--sort=-pcpu",
        ],
        "process_tree": [
            "ps",
            "-e",
            "--forest",
            "-o",
            "user,pid,ppid,pcpu,pmem,rss,etime,stat,comm",
        ],
        "gpu": ["nvidia-smi"],
        "gpu_details": [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,driver_version,utilization.gpu,utilization.memory,power.draw",
            "--format=csv",
        ],
        "gpu_processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
        "docker_version": ["docker", "version"],
        "docker_info": ["docker", "info", "--format", "json"],
        "containers": ["docker", "ps", "-a", "--no-trunc", "--format", "json"],
        "container_stats": ["docker", "stats", "--no-stream", "--format", "json"],
        "images": ["docker", "image", "ls", "--digests", "--format", "json"],
        "podman": ["podman", "version"],
        "python": ["python3", "--version"],
        "disk_space": ["df", "-hT"],
        "disks": ["lsblk", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL"],
        "io": ["iostat", "-xz", "1", "3"],
        "load_samples": ["vmstat", "1", "3"],
        "interfaces": ["ip", "-brief", "address"],
        "network_stats": ["ip", "-s", "link"],
        "logged_in": ["who"],
        "timers": ["systemctl", "list-timers", "--all", "--no-pager"],
        "clock": ["timedatectl", "status"],
        "numa": ["numactl", "--hardware"],
    }
    data = {
        "collected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Read-only inventory; no benchmark or container launched",
        "commands": {name: capture(cmd) for name, cmd in commands.items()},
    }
    data["sysfs"] = {}
    for pattern in [
        "cpu[0-9]*/topology/thread_siblings_list",
        "cpu[0-9]*/cpufreq/scaling_governor",
        "cpu[0-9]*/cpufreq/scaling_cur_freq",
    ]:
        for path in Path("/sys/devices/system/cpu").glob(pattern):
            data["sysfs"][str(path)] = path.read_text().strip()
    for name in [
        "/proc/meminfo",
        "/proc/loadavg",
        "/proc/pressure/cpu",
        "/proc/pressure/memory",
        "/proc/pressure/io",
        "/proc/self/status",
        "/sys/fs/cgroup/cgroup.controllers",
    ]:
        path = Path(name)
        if path.exists():
            data["sysfs"][name] = path.read_text()
    # Retrieve only the identity document, never credentials or arbitrary metadata.
    try:
        base = "http://169.254.169.254/latest/"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            base + "api/token", method="PUT", headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}
        )
        with opener.open(request, timeout=3) as response:
            token = response.read().decode()
        request = urllib.request.Request(
            base + "dynamic/instance-identity/document", headers={"X-aws-ec2-metadata-token": token}
        )
        with opener.open(request, timeout=3) as response:
            identity = json.load(response)
        data["ec2"] = {
            key: identity.get(key)
            for key in [
                "instanceId",
                "instanceType",
                "architecture",
                "region",
                "availabilityZone",
                "imageId",
            ]
        }
    except Exception as exc:
        data["ec2"] = {"error": str(exc)}
    return data


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
