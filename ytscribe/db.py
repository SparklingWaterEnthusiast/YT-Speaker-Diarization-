"""SQLite-backed job store: the processing queue and per-video state.

One row per video. Stage progression:
    pending -> acquire -> transcribe -> diarize -> merge -> export -> complete
status: queued | running | done | failed | cancelled

Thread-safety: a single connection guarded by a lock; every write commits
immediately so a crash never loses more than the in-flight stage (whose
artifacts are cached on disk anyway).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

STAGES = ("acquire", "transcribe", "diarize", "merge", "export")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT UNIQUE NOT NULL,
    url TEXT NOT NULL,
    title TEXT DEFAULT '',
    channel TEXT DEFAULT '',
    upload_date TEXT DEFAULT '',
    duration REAL DEFAULT 0,
    status TEXT DEFAULT 'queued',
    stage TEXT DEFAULT 'acquire',
    error TEXT DEFAULT '',
    retries INTEGER DEFAULT 0,
    batch TEXT DEFAULT '',
    wall_seconds REAL DEFAULT 0,
    added_at REAL,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- queue management -------------------------------------------------

    def add(self, video_id: str, url: str, title: str = "", duration: float = 0.0,
            channel: str = "", upload_date: str = "", batch: str = "") -> bool:
        """Insert a job; returns False if the video is already known."""
        now = time.time()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO jobs (video_id,url,title,duration,channel,upload_date,"
                    "batch,added_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (video_id, url, title, duration, channel, upload_date, batch, now, now))
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def pending(self) -> list[dict]:
        return self._rows("SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY id")

    def failed(self) -> list[dict]:
        return self._rows("SELECT * FROM jobs WHERE status='failed' ORDER BY id")

    def all_jobs(self) -> list[dict]:
        return self._rows("SELECT * FROM jobs ORDER BY id")

    def batch_jobs(self, batch: str) -> list[dict]:
        return self._rows("SELECT * FROM jobs WHERE batch=? ORDER BY id", (batch,))

    def get(self, video_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM jobs WHERE video_id=?", (video_id,))
        return rows[0] if rows else None

    # -- state transitions -------------------------------------------------

    def set_stage(self, video_id: str, stage: str, status: str = "running") -> None:
        self._update(video_id, stage=stage, status=status)

    def set_meta(self, video_id: str, title: str, duration: float,
                 channel: str, upload_date: str) -> None:
        self._update(video_id, title=title, duration=duration,
                     channel=channel, upload_date=upload_date)

    def mark_done(self, video_id: str, wall_seconds: float) -> None:
        self._update(video_id, status="done", stage="complete",
                     error="", wall_seconds=wall_seconds)

    def mark_failed(self, video_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='failed', error=?, retries=retries+1, "
                "updated_at=? WHERE video_id=?", (error[:2000], time.time(), video_id))
            self._conn.commit()

    def mark_cancelled(self, video_id: str) -> None:
        self._update(video_id, status="cancelled")

    def requeue_failed(self) -> int:
        """Move failed jobs back to the queue (retry queue -> main queue)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='queued', error='', updated_at=? "
                "WHERE status='failed'", (time.time(),))
            self._conn.commit()
            return cur.rowcount

    def recover_interrupted(self) -> int:
        """On startup, demote jobs left 'running' by a crash back to 'queued'."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='queued', updated_at=? WHERE status='running'",
                (time.time(),))
            self._conn.commit()
            return cur.rowcount

    def remove(self, video_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE video_id=?", (video_id,))
            self._conn.commit()

    def clear_finished(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE status IN ('done','cancelled')")
            self._conn.commit()
            return cur.rowcount

    # -- internals ----------------------------------------------------------

    def _update(self, video_id: str, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {cols} WHERE video_id=?",
                               (*fields.values(), video_id))
            self._conn.commit()

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
