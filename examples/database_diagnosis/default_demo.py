"""Default SQLite demo database for the simplified diagnosis example."""
from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .sqlite_access import database_session


# ---------------------------------------------------------------------------
# Demo database identity
# ---------------------------------------------------------------------------

APPLICATION_ID = 0x44424447
ORDER_INSERT_SQL = """
INSERT INTO orders
    (order_id, customer_id, status, total_cents, channel, created_at)
VALUES (?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def _insert_order_batch(
    conn: sqlite3.Connection,
    batch: list[tuple[int, int, str, int, str, str]],
) -> None:
    conn.executemany(ORDER_INSERT_SQL, batch)
    batch.clear()


# ---------------------------------------------------------------------------
# Default database construction
# ---------------------------------------------------------------------------

def create_default_database(
    path: str | Path,
    *,
    rows: int = 120_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a deterministic SQLite database with one omitted useful index."""
    if rows < 100:
        raise ValueError("--rows must be at least 100 when creating the default database.")

    db_path = Path(path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(seed)
    customer_count = max(50, min(10_000, rows // 18))
    hot_customer_count = max(3, min(12, customer_count // 10))
    start = datetime(2025, 1, 1, 8, 0, 0)
    statuses = ["created", "paid", "packed", "shipped", "returned", "cancelled"]
    regions = ["north", "south", "east", "west", "central"]
    channels = ["web", "mobile", "marketplace", "sales"]

    with database_session(db_path, write_enabled=True) as conn:
        conn.executescript(
            f"""
            PRAGMA application_id = {APPLICATION_ID};
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            PRAGMA temp_store = MEMORY;

            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                region TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                total_cents INTEGER NOT NULL,
                channel TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );

            CREATE INDEX idx_orders_status_created_at
                ON orders (status, created_at DESC);
            """
        )

        customers = []
        for customer_id in range(1, customer_count + 1):
            created_at = start - timedelta(days=rng.randint(10, 900))
            region = regions[(customer_id + rng.randint(0, 8)) % len(regions)]
            customers.append(
                (
                    customer_id,
                    f"customer{customer_id:05d}@example.test",
                    region,
                    created_at.isoformat(),
                )
            )
        conn.executemany(
            """
            INSERT INTO customers (customer_id, email, region, created_at)
            VALUES (?, ?, ?, ?)
            """,
            customers,
        )

        batch = []
        for order_id in range(1, rows + 1):
            if order_id <= rows // 3:
                customer_id = 1 + ((order_id - 1) % hot_customer_count)
            else:
                customer_id = rng.randint(1, customer_count)
            status = rng.choices(statuses, weights=[8, 18, 20, 38, 6, 10])[0]
            created_at = start + timedelta(minutes=order_id)
            total_cents = rng.randint(1_500, 125_000)
            channel = rng.choice(channels)
            batch.append(
                (order_id, customer_id, status, total_cents, channel, created_at.isoformat())
            )
            if len(batch) >= 5_000:
                _insert_order_batch(conn, batch)
        if batch:
            _insert_order_batch(conn, batch)
        conn.execute("ANALYZE")

    return {
        "db_path": str(db_path),
        "rows": rows,
        "customers": customer_count,
        "seed": seed,
        "intentionally_omitted_index": (
            "An index on orders(customer_id, created_at DESC) is intentionally omitted "
            "so the analyzer can discover it."
        ),
    }
