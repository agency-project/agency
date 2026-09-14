"""Metadata-only, best-effort access evidence for Linux traced process trees."""

import os
import time

ACCESS_SYSCALLS = ("open", "openat", "read", "pread64", "readv")
READ_SYSCALLS = ("read", "pread64", "readv")


def fd_metadata(pid, fd):
    try:
        link = f"/proc/{pid}/fd/{fd}"
        path = os.readlink(link)
        if not path.startswith("/"):
            return None
        info = os.stat(link)
        return {
            "path": path,
            "device": info.st_dev,
            "inode": info.st_ino,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
            "size_bytes": info.st_size,
        }
    except OSError:
        return None


def access_result(stop, result):
    metadata = getattr(stop, "file_access", None)
    if stop.syscall in ("open", "openat") and result >= 0:
        metadata = fd_metadata(stop.pid, result)
    if metadata is None or result < 0:
        return None
    return {
        **metadata,
        "syscall": stop.syscall,
        "pid": stop.pid,
        "started_wall_ns": round(stop.timestamp * 1e9),
        "ended_wall_ns": time.time_ns(),
        "returned_bytes": result if stop.syscall in READ_SYSCALLS else 0,
        "successful_open": stop.syscall in ("open", "openat"),
        "coverage": "observed ptrace tree only; mmap/io_uring/other read variants unobserved",
        "version_match_basis": "inode/device/size/mtime/ctime metadata, not content hash",
    }
