"""Tests for concurrent storage access scenarios."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from bijoux_router.storage.sqlite_backend import SQLiteStorage


class TestConcurrentStorage:
    def test_concurrent_usage_recording(self, tmp_db: str) -> None:
        """Multiple threads recording usage concurrently should not lose data."""
        storage = SQLiteStorage(tmp_db)
        storage.initialize()
        n_threads = 10
        n_records = 50

        def worker(thread_id: int) -> None:
            for i in range(n_records):
                storage.record_usage(
                    "p1", "w1",
                    prompt_tokens=1, completion_tokens=1, total_tokens=2,
                    request_id=f"t{thread_id}-r{i}", model="m",
                    timestamp=time.time(),
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        usage = storage.get_window_usage("p1", "w1")
        expected = n_threads * n_records * 2  # 2 total_tokens per record
        assert usage["total_tokens"] == expected
        assert usage["request_count"] == n_threads * n_records
        storage.close()

    def test_concurrent_reservation_create_release(self, tmp_db: str) -> None:
        """Concurrent reservation creates and releases should be consistent."""
        storage = SQLiteStorage(tmp_db)
        storage.initialize()
        n_threads = 10

        def worker(thread_id: int) -> None:
            r_id = f"res-{thread_id}"
            storage.create_reservation(r_id, "p1", "w1", 100, time.time())
            time.sleep(0.01)
            storage.release_reservation(r_id)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reservations should be released
        assert storage.get_active_reservations("p1", "w1") == 0
        storage.close()

    def test_concurrent_failure_increment(self, tmp_db: str) -> None:
        storage = SQLiteStorage(tmp_db)
        storage.initialize()
        n_threads = 20

        def worker() -> None:
            storage.increment_failure("p1")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert storage.get_failure_count("p1") == n_threads
        storage.close()
