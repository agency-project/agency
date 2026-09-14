import sqlite3

from benchmarks.checkpoint_size_microbenchmark.analyze import linear_fit, logical_dirty
from benchmarks.checkpoint_size_microbenchmark.runner import extract_payload_result


def test_logical_dirty_counts_only_regular_files_with_known_sizes():
    report = {
        "changes": [
            {"kind": "file", "size_bytes": 10},
            {"kind": "file", "size_bytes": 5},
            {"kind": "non_regular", "size_bytes": None},
        ]
    }

    assert logical_dirty(report) == (2, 15)


def test_linear_fit_recovers_intercept_and_slope():
    rows = [
        {"requested_bytes": 0, "runtime_commit_seconds": 6.0},
        {"requested_bytes": 10, "runtime_commit_seconds": 8.0},
        {"requested_bytes": 20, "runtime_commit_seconds": 10.0},
    ]

    fit = linear_fit(rows, "requested_bytes")

    assert fit["intercept_seconds"] == 6.0
    assert fit["slope_seconds_per_byte"] == 0.2
    assert fit["r_squared"] == 1.0


def test_payload_result_uses_summary_before_async_event_delivery(tmp_path):
    database = tmp_path / "events.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE events (type TEXT, payload TEXT, timestamp REAL)")
    summary = (
        "START_NS=100\nEND_NS=250\nPAYLOAD_BYTES=4096\n"
        "4ecea2e55ae4e836b84afd024a0d90cc16ed116d440b4aa9aaabd53790384835  "
        "/testbed/.agency-checkpoint-payload.bin"
    )

    result = extract_payload_result(database, summary)

    assert result["generation_seconds"] == 0.00000015
    assert result["actual_bytes_from_tool"] == 4096
