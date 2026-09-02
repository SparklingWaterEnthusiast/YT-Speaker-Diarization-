"""Unit tests for v0.2.1 multi-queue support (db.py queues, ordering,
priority scheduling, migration)."""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from ytscribe.db import DEFAULT_QUEUE_ID, JobStore


class QueueTestBase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db_path = Path(self.td.name) / "jobs.sqlite3"
        self.store = JobStore(self.db_path)

    def tearDown(self):
        self.store.close()
        self.td.cleanup()


class TestQueues(QueueTestBase):
    def test_default_queue_exists(self):
        queues = self.store.queues()
        self.assertEqual(queues[0]["id"], DEFAULT_QUEUE_ID)
        self.assertEqual(queues[0]["name"], "Main")
        self.assertEqual(queues[0]["folder"], "")

    def test_create_rename_delete(self):
        qid = self.store.create_queue("Priority", "Priority")
        self.assertEqual(len(self.store.queues()), 2)
        self.store.rename_queue(qid, "Rush", "Rush")
        self.assertEqual(self.store.get_queue(qid)["name"], "Rush")
        self.store.add("v1", "u", queue_id=qid)
        self.store.delete_queue(qid)
        self.assertIsNone(self.store.get_queue(qid))
        # jobs cascade with their queue
        self.assertIsNone(self.store.get("v1"))

    def test_same_video_in_two_queues(self):
        q2 = self.store.create_queue("Priority", "Priority")
        self.assertTrue(self.store.add("v1", "u", queue_id=DEFAULT_QUEUE_ID))
        self.assertTrue(self.store.add("v1", "u", queue_id=q2))
        self.assertFalse(self.store.add("v1", "u", queue_id=q2))  # dup in queue
        self.assertEqual(sorted(self.store.queues_containing("v1")),
                         sorted([DEFAULT_QUEUE_ID, q2]))

    def test_per_queue_status_updates(self):
        q2 = self.store.create_queue("P", "P")
        self.store.add("v1", "u", queue_id=DEFAULT_QUEUE_ID)
        self.store.add("v1", "u", queue_id=q2)
        self.store.mark_done("v1", 10.0, queue_id=q2)
        self.assertEqual(self.store.get("v1", q2)["status"], "done")
        self.assertEqual(self.store.get("v1", DEFAULT_QUEUE_ID)["status"], "queued")

    def test_tab_order_persisted(self):
        q2 = self.store.create_queue("B", "B")
        q3 = self.store.create_queue("C", "C")
        self.store.set_tab_order([q3, DEFAULT_QUEUE_ID, q2])
        self.assertEqual([q["id"] for q in self.store.queues()],
                         [q3, DEFAULT_QUEUE_ID, q2])


class TestPriorityScheduling(QueueTestBase):
    def test_most_recent_queue_wins_then_falls_back(self):
        for v in ("bulk1", "bulk2"):
            self.store.add(v, "u", queue_id=DEFAULT_QUEUE_ID)
        prio = self.store.create_queue("Priority", "Priority")
        self.store.add("urgent", "u", queue_id=prio)

        self.store.set_queue_active(DEFAULT_QUEUE_ID, True)
        time.sleep(0.01)  # activation timestamps must differ
        self.store.set_queue_active(prio, True)

        nxt = self.store.next_queued_jobs(2)
        self.assertEqual(nxt[0]["video_id"], "urgent")   # newest queue first
        self.assertEqual(nxt[1]["video_id"], "bulk1")    # then bulk resumes

        self.store.mark_done("urgent", 1.0, queue_id=prio)
        self.assertEqual(self.store.next_queued_jobs(1)[0]["video_id"], "bulk1")

    def test_inactive_queues_are_skipped(self):
        self.store.add("v1", "u")
        self.assertEqual(self.store.next_queued_jobs(), [])
        self.store.set_queue_active(DEFAULT_QUEUE_ID, True)
        self.assertEqual(self.store.next_queued_jobs()[0]["video_id"], "v1")
        self.store.set_queue_active(DEFAULT_QUEUE_ID, False)
        self.assertEqual(self.store.next_queued_jobs(), [])

    def test_jobs_follow_manual_order(self):
        for v in ("a", "b", "c", "d"):
            self.store.add(v, "u")
        self.store.set_queue_active(DEFAULT_QUEUE_ID, True)
        ids = {j["video_id"]: j["id"] for j in self.store.all_jobs()}
        self.store.move_jobs([ids["d"]], "top")
        self.assertEqual(self.store.next_queued_jobs(1)[0]["video_id"], "d")
        self.store.move_jobs([ids["d"]], "down")
        self.assertEqual([j["video_id"] for j in self.store.all_jobs()],
                         ["a", "d", "b", "c"])


class TestReorderAndBulkOps(QueueTestBase):
    def _ids(self):
        return {j["video_id"]: j["id"] for j in self.store.all_jobs()}

    def test_move_multiple_preserves_relative_order(self):
        for v in ("a", "b", "c", "d", "e"):
            self.store.add(v, "u")
        ids = self._ids()
        self.store.move_jobs([ids["b"], ids["d"]], "bottom")
        self.assertEqual([j["video_id"] for j in self.store.all_jobs()],
                         ["a", "c", "e", "b", "d"])
        self.store.move_jobs([ids["b"], ids["d"]], "up")
        self.assertEqual([j["video_id"] for j in self.store.all_jobs()],
                         ["a", "c", "b", "d", "e"])

    def test_remove_skips_running(self):
        self.store.add("v1", "u")
        self.store.add("v2", "u")
        self.store.set_stage("v1", "transcribe")  # running
        ids = self._ids()
        removed = self.store.remove_jobs([ids["v1"], ids["v2"]])
        self.assertEqual(removed, 1)
        self.assertIsNotNone(self.store.get("v1"))

    def test_requeue_selected_and_cancelled(self):
        self.store.add("f", "u")
        self.store.add("c", "u")
        self.store.add("q", "u")
        self.store.mark_failed("f", "boom")
        self.store.mark_cancelled("c")
        ids = self._ids()
        n = self.store.requeue_jobs([ids["f"], ids["c"], ids["q"]])
        self.assertEqual(n, 2)  # queued job untouched
        self.assertEqual(self.store.get("c")["status"], "queued")

    def test_requeue_failed_includes_cancelled(self):
        self.store.add("f", "u")
        self.store.add("c", "u")
        self.store.mark_failed("f", "x")
        self.store.mark_cancelled("c")
        self.assertEqual(self.store.requeue_failed(), 2)


class TestMigrationFromV1(unittest.TestCase):
    def test_v1_database_is_upgraded_in_place(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "jobs.sqlite3"
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT UNIQUE NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT DEFAULT '', channel TEXT DEFAULT '',
                    upload_date TEXT DEFAULT '', duration REAL DEFAULT 0,
                    status TEXT DEFAULT 'queued', stage TEXT DEFAULT 'acquire',
                    error TEXT DEFAULT '', retries INTEGER DEFAULT 0,
                    batch TEXT DEFAULT '', wall_seconds REAL DEFAULT 0,
                    added_at REAL, updated_at REAL);
                INSERT INTO jobs (video_id, url, title, status, stage)
                    VALUES ('old1', 'u1', 'T1', 'done', 'complete'),
                           ('old2', 'u2', 'T2', 'queued', 'acquire');
            """)
            conn.commit()
            conn.close()

            store = JobStore(db_path)
            try:
                jobs = store.all_jobs()
                self.assertEqual(len(jobs), 2)
                self.assertTrue(all(j["queue_id"] == DEFAULT_QUEUE_ID
                                    for j in jobs))
                self.assertEqual(store.get("old1")["status"], "done")
                self.assertEqual([j["video_id"] for j in jobs],
                                 ["old1", "old2"])  # order preserved
                # same video can now join a second queue
                q2 = store.create_queue("New", "New")
                self.assertTrue(store.add("old1", "u1", queue_id=q2))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
