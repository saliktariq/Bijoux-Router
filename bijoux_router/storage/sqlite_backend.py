"""SQLite implementation of the Bijoux storage backend."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from bijoux_router.exceptions.errors import StorageError
from bijoux_router.storage.base import StorageBackend
from bijoux_router.utils.logging import get_logger

logger = get_logger("storage.sqlite")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS usage_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_name TEXT NOT NULL,
    window_key TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    request_id TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_provider_window
    ON usage_records(provider_name, window_key);

CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    provider_name TEXT NOT NULL,
    window_key TEXT NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reservations_provider_window
    ON reservations(provider_name, window_key);

CREATE TABLE IF NOT EXISTS cooldowns (
    provider_name TEXT PRIMARY KEY,
    until_timestamp REAL NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS failure_counters (
    provider_name TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kv_store (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteStorage(StorageBackend):
    """Thread-safe SQLite storage backend.

    Notes on concurrency:
    - Uses WAL mode for better read concurrency.
    - A threading lock serializes writes within the same process.
    - SQLite has a single-writer limitation; for multi-process writes,
      consider PostgreSQL instead.
    """

    def __init__(self, db_path: str = "bijoux_state.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("Storage not initialized. Call initialize() first.")
        return self._conn

    def initialize(self) -> None:
        try:
            # Ensure parent directory exists
            db_dir = Path(self._db_path).parent
            if db_dir != Path("."):
                db_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
            logger.info("SQLite storage initialized at %s", self._db_path)
        except Exception as exc:
            raise StorageError(f"Failed to initialize SQLite storage: {exc}") from exc

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Usage records ------------------------------------------------------

    def record_usage(
        self,
        provider_name: str,
        window_key: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        request_id: str,
        model: str,
        timestamp: float,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO usage_records (provider_name, window_key, prompt_tokens, completion_tokens, total_tokens, request_id, model, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (provider_name, window_key, prompt_tokens, completion_tokens, total_tokens, request_id, model, timestamp),
            )
            conn.commit()

    def get_window_usage(self, provider_name: str, window_key: str) -> dict[str, int]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), "
            "COALESCE(SUM(total_tokens), 0), COUNT(*) "
            "FROM usage_records WHERE provider_name = ? AND window_key = ?",
            (provider_name, window_key),
        ).fetchone()
        return {
            "prompt_tokens": row[0],
            "completion_tokens": row[1],
            "total_tokens": row[2],
            "request_count": row[3],
        }

    def delete_window_usage(self, provider_name: str, window_key: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM usage_records WHERE provider_name = ? AND window_key = ?",
                (provider_name, window_key),
            )
            conn.commit()

    # -- Reservations -------------------------------------------------------

    def create_reservation(
        self, reservation_id: str, provider_name: str, window_key: str, estimated_tokens: int, timestamp: float,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO reservations (reservation_id, provider_name, window_key, estimated_tokens, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (reservation_id, provider_name, window_key, estimated_tokens, timestamp),
            )
            conn.commit()

    def release_reservation(self, reservation_id: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM reservations WHERE reservation_id = ?", (reservation_id,))
            conn.commit()

    def get_active_reservations(self, provider_name: str, window_key: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COALESCE(SUM(estimated_tokens), 0) FROM reservations WHERE provider_name = ? AND window_key = ?",
            (provider_name, window_key),
        ).fetchone()
        return row[0]

    # -- Cooldown state -----------------------------------------------------

    def set_cooldown(self, provider_name: str, until_timestamp: float, reason: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO cooldowns (provider_name, until_timestamp, reason) VALUES (?, ?, ?)",
                (provider_name, until_timestamp, reason),
            )
            conn.commit()

    def get_cooldown(self, provider_name: str) -> float | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT until_timestamp FROM cooldowns WHERE provider_name = ?",
            (provider_name,),
        ).fetchone()
        if row is None:
            return None
        if row[0] <= time.time():
            # Cooldown expired — clean up
            self.clear_cooldown(provider_name)
            return None
        return row[0]

    def clear_cooldown(self, provider_name: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM cooldowns WHERE provider_name = ?", (provider_name,))
            conn.commit()

    # -- Failure counters ---------------------------------------------------

    def increment_failure(self, provider_name: str) -> int:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO failure_counters (provider_name, count) VALUES (?, 1) "
                "ON CONFLICT(provider_name) DO UPDATE SET count = count + 1",
                (provider_name,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT count FROM failure_counters WHERE provider_name = ?", (provider_name,),
            ).fetchone()
            return row[0] if row else 1

    def reset_failures(self, provider_name: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM failure_counters WHERE provider_name = ?", (provider_name,))
            conn.commit()

    def get_failure_count(self, provider_name: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT count FROM failure_counters WHERE provider_name = ?", (provider_name,),
        ).fetchone()
        return row[0] if row else 0

    # -- Fairness cursor ----------------------------------------------------

    def get_last_used_provider(self) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = 'last_used_provider'",
        ).fetchone()
        return row[0] if row else None

    def set_last_used_provider(self, provider_name: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('last_used_provider', ?)",
                (provider_name,),
            )
            conn.commit()

    # -- Bulk / admin -------------------------------------------------------

    def reset_provider_usage(self, provider_name: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM usage_records WHERE provider_name = ?", (provider_name,))
            conn.execute("DELETE FROM reservations WHERE provider_name = ?", (provider_name,))
            conn.execute("DELETE FROM cooldowns WHERE provider_name = ?", (provider_name,))
            conn.execute("DELETE FROM failure_counters WHERE provider_name = ?", (provider_name,))
            conn.commit()
            logger.info("Reset all state for provider '%s'", provider_name)

    def get_all_provider_states(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        # Get all known provider names from usage + cooldowns + failures
        rows = conn.execute(
            "SELECT DISTINCT provider_name FROM usage_records "
            "UNION SELECT provider_name FROM cooldowns "
            "UNION SELECT provider_name FROM failure_counters"
        ).fetchall()
        states: list[dict[str, Any]] = []
        for (name,) in rows:
            cooldown = self.get_cooldown(name)
            failures = self.get_failure_count(name)
            states.append({
                "provider_name": name,
                "cooldown_until": cooldown,
                "failure_count": failures,
            })
        return states
