"""Deadline reconciliation and the three primary metrics; pure functions."""

import csv
import io


def reconcile(rows, events, batch_start, cutoff, reason):
    """Only events before the cutoff count. Late cleanup cannot complete an attempt."""
    result = [dict(row) for row in rows]
    by_id = {row["attempt_id"]: row for row in result}
    for event in events:
        row = by_id[event["attempt_id"]]
        when = event["time"]
        if when < batch_start or when >= cutoff:
            continue
        if event["kind"] == "started" and row["started"] is None:
            row.update(started=when, status="running")
        elif event["kind"] == "finished" and row["started"] is not None:
            row.update(finished=when, status="completed", returncode=event["returncode"])
    for row in result:
        if row["started"] is None:
            row.update(status="unattempted", reason=reason, success=False)
        elif row["finished"] is None:
            row.update(
                status="timed_out" if reason == "deadline" else "interrupted",
                reason=reason,
                finished=cutoff,
                censored=True,
                success=False,
            )
        if row["started"] is not None:
            row["duration_seconds"] = max(0, row["finished"] - row["started"])
    return result


def summarize(rows, elapsed, timeout):
    if len(rows) != 30 or len({r["attempt_id"] for r in rows}) != 30:
        raise ValueError("A condition must retain all 30 assigned attempts")
    elapsed = min(timeout, max(0, elapsed))
    durations = [r["duration_seconds"] for r in rows if r["duration_seconds"] is not None]
    completed = sum(r["status"] == "completed" for r in rows)
    successful = sum(r["status"] == "completed" and r["success"] is True for r in rows)
    pending = sum(r["status"] == "completed" and r["success"] is None for r in rows)
    return {
        "assigned": 30,
        "completed_attempts": completed,
        "successful_attempts": successful,
        "grading_pending": pending,
        "pass_rate": None if pending else successful / 30,
        "average_task_wall_seconds": sum(durations) / len(durations) if durations else None,
        "throughput_attempts_per_hour": completed * 3600 / elapsed if elapsed else 0,
        "successful_tasks_per_hour": None
        if pending
        else (successful * 3600 / elapsed if elapsed else 0),
        "batch_elapsed_seconds": elapsed,
        "observed_tasks": len(durations),
        "censored_tasks": sum(r["censored"] for r in rows),
        "unattempted_tasks": sum(r["status"] == "unattempted" for r in rows),
        "timed_out_tasks": sum(r["reason"] == "deadline" for r in rows),
    }


def csv_text(rows):
    stream = io.StringIO()
    fields = sorted(set().union(*(row.keys() for row in rows)))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()
