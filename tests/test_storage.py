"""Tests for the SQLite storage backend."""

from __future__ import annotations

import time

from bijoux_router.storage.sqlite_backend import SQLiteStorage


class TestSQLiteStorage:
    def test_record_and_get_usage(self, storage: SQLiteStorage) -> None:
        storage.record_usage("p1", "w1", 10, 20, 30, "req-1", "model-a", time.time())
        storage.record_usage("p1", "w1", 5, 10, 15, "req-2", "model-a", time.time())
        usage = storage.get_window_usage("p1", "w1")
        assert usage["prompt_tokens"] == 15
        assert usage["completion_tokens"] == 30
        assert usage["total_tokens"] == 45
        assert usage["request_count"] == 2

    def test_empty_window_usage(self, storage: SQLiteStorage) -> None:
        usage = storage.get_window_usage("nobody", "nowhere")
        assert usage["total_tokens"] == 0
        assert usage["request_count"] == 0

    def test_reservations(self, storage: SQLiteStorage) -> None:
        storage.create_reservation("r1", "p1", "w1", 500, time.time())
        storage.create_reservation("r2", "p1", "w1", 300, time.time())
        assert storage.get_active_reservations("p1", "w1") == 800
        storage.release_reservation("r1")
        assert storage.get_active_reservations("p1", "w1") == 300
        storage.release_reservation("r2")
        assert storage.get_active_reservations("p1", "w1") == 0

    def test_cooldown(self, storage: SQLiteStorage) -> None:
        future = time.time() + 3600
        storage.set_cooldown("p1", future, "test")
        assert storage.get_cooldown("p1") is not None
        assert storage.get_cooldown("p1") == future
        storage.clear_cooldown("p1")
        assert storage.get_cooldown("p1") is None

    def test_expired_cooldown_auto_clears(self, storage: SQLiteStorage) -> None:
        past = time.time() - 10
        storage.set_cooldown("p1", past, "expired")
        assert storage.get_cooldown("p1") is None

    def test_failure_counter(self, storage: SQLiteStorage) -> None:
        assert storage.get_failure_count("p1") == 0
        assert storage.increment_failure("p1") == 1
        assert storage.increment_failure("p1") == 2
        assert storage.get_failure_count("p1") == 2
        storage.reset_failures("p1")
        assert storage.get_failure_count("p1") == 0

    def test_last_used_provider(self, storage: SQLiteStorage) -> None:
        assert storage.get_last_used_provider() is None
        storage.set_last_used_provider("p1")
        assert storage.get_last_used_provider() == "p1"
        storage.set_last_used_provider("p2")
        assert storage.get_last_used_provider() == "p2"

    def test_reset_provider_usage(self, storage: SQLiteStorage) -> None:
        storage.record_usage("p1", "w1", 10, 20, 30, "r1", "m", time.time())
        storage.create_reservation("res1", "p1", "w1", 100, time.time())
        storage.set_cooldown("p1", time.time() + 3600, "test")
        storage.increment_failure("p1")
        storage.reset_provider_usage("p1")
        assert storage.get_window_usage("p1", "w1")["total_tokens"] == 0
        assert storage.get_active_reservations("p1", "w1") == 0
        assert storage.get_cooldown("p1") is None
        assert storage.get_failure_count("p1") == 0

    def test_delete_window_usage(self, storage: SQLiteStorage) -> None:
        storage.record_usage("p1", "w1", 10, 20, 30, "r1", "m", time.time())
        storage.record_usage("p1", "w2", 5, 5, 10, "r2", "m", time.time())
        storage.delete_window_usage("p1", "w1")
        assert storage.get_window_usage("p1", "w1")["total_tokens"] == 0
        assert storage.get_window_usage("p1", "w2")["total_tokens"] == 10

    def test_persistence_across_reopen(self, tmp_db: str) -> None:
        """Verify data persists when storage is closed and reopened."""
        s1 = SQLiteStorage(tmp_db)
        s1.initialize()
        s1.record_usage("p1", "w1", 10, 20, 30, "r1", "m", time.time())
        s1.set_last_used_provider("p1")
        s1.close()

        s2 = SQLiteStorage(tmp_db)
        s2.initialize()
        usage = s2.get_window_usage("p1", "w1")
        assert usage["total_tokens"] == 30
        assert s2.get_last_used_provider() == "p1"
        s2.close()
