"""SQLite-backed job store: named queues (UI tabs) and per-video job state.

Since v0.2.1 jobs belong to queues. Each queue is a UI tab with its own
output subfolder; the same video may appear in several queues (a priority
tab often duplicates one video of a bulk channel queue — the second run is
nearly free because pipeline stages are cached by video_id).

Job stage progression:
    pending -> acquire -> transcribe -> diarize -> merge -> export -> complete
status: queued | running | done | failed | cancelled

Queue scheduling: queues with a non-NULL activated_at are "running"; the
runner always takes the next job from the MOST RECENTLY activated queue
(a priority stack — starting a small queue mid-run preempts the bulk queue
at the next video boundary, and the bulk queue resumes when it drains).

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

DEFAULT_QUEUE_ID = 1
SCHEMA_VERSION = 2

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS queues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    folder TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    activated_at REAL,
    tab_order INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    queue_id INTEGER NOT NULL DEFAULT {DEFAULT_QUEUE_ID}
        REFERENCES queues(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,
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
    updated_at REAL,
    UNIQUE(video_id, queue_id)
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(queue_id, position);
"""


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Create/upgrade the schema. v1 (single flat jobs table) -> v2
        (queues + per-queue positions + UNIQUE(video_id, queue_id))."""
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        has_jobs = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone() is not None
        if version < SCHEMA_VERSION and has_jobs:
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(jobs)")}
            if "queue_id" not in cols:  # genuine v1 database — rebuild jobs
                self._conn.executescript(f"""
                    CREATE TABLE IF NOT EXISTS queues (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        folder TEXT NOT NULL DEFAULT '',
                        created_at REAL NOT NULL,
                        activated_at REAL,
                        tab_order INTEGER NOT NULL DEFAULT 0
                    );
                    INSERT OR IGNORE INTO queues (id, name, folder, created_at, tab_order)
                        VALUES ({DEFAULT_QUEUE_ID}, 'Main', '', {time.time()}, 0);
                    ALTER TABLE jobs RENAME TO jobs_v1;
                """)
                self._conn.executescript(_SCHEMA)
                self._conn.execute(f"""
                    INSERT INTO jobs (id, video_id, queue_id, position, url, title,
                        channel, upload_date, duration, status, stage, error,
                        retries, batch, wall_seconds, added_at, updated_at)
                    SELECT id, video_id, {DEFAULT_QUEUE_ID}, id, url, title,
                        channel, upload_date, duration, status, stage, error,
                        retries, batch, wall_seconds, added_at, updated_at
                    FROM jobs_v1
                """)
                self._conn.execute("DROP TABLE jobs_v1")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO queues (id, name, folder, created_at, tab_order) "
            "VALUES (?, 'Main', '', ?, 0)", (DEFAULT_QUEUE_ID, time.time()))
        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # -- queues (UI tabs) ---------------------------------------------------

    def create_queue(self, name: str, folder: str) -> int:
        now = time.time()
        with self._lock:
            order = self._conn.execute(
                "SELECT COALESCE(MAX(tab_order), -1) + 1 FROM queues").fetchone()[0]
            qid = self._conn.execute(
                "INSERT INTO queues (name, folder, created_at, tab_order) "
                "VALUES (?,?,?,?)", (name, folder, now, order)).lastrowid
            self._conn.commit()
            return qid

    def queues(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT * FROM queues ORDER BY tab_order, id")]

    def get_queue(self, queue_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM queues WHERE id=?",
                                     (queue_id,)).fetchone()
            return dict(row) if row else None

    def rename_queue(self, queue_id: int, name: str, folder: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE queues SET name=?, folder=? WHERE id=?",
                               (name, folder, queue_id))
            self._conn.commit()

    def delete_queue(self, queue_id: int) -> None:
        """Remove a queue and its jobs (cache artifacts stay on disk)."""
        with self._lock:
            self._conn.execute("DELETE FROM queues WHERE id=?", (queue_id,))
            self._conn.commit()

    def set_queue_active(self, queue_id: int, active: bool) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE queues SET activated_at=? WHERE id=?",
                (time.time() if active else None, queue_id))
            self._conn.commit()

    def set_tab_order(self, ordered_ids: list[int]) -> None:
        with self._lock:
            for i, qid in enumerate(ordered_ids):
                self._conn.execute("UPDATE queues SET tab_order=? WHERE id=?",
                                   (i, qid))
            self._conn.commit()

    # -- queue management -------------------------------------------------

    def add(self, video_id: str, url: str, title: str = "", duration: float = 0.0,
            channel: str = "", upload_date: str = "", batch: str = "",
            queue_id: int = DEFAULT_QUEUE_ID) -> bool:
        """Insert a job; returns False if the video is already in that queue."""
        now = time.time()
        with self._lock:
            try:
                pos = self._conn.execute(
                    "SELECT COALESCE(MAX(position), 0) + 1 FROM jobs "
                    "WHERE queue_id=?", (queue_id,)).fetchone()[0]
                self._conn.execute(
                    "INSERT INTO jobs (video_id,queue_id,position,url,title,duration,"
                    "channel,upload_date,batch,added_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (video_id, queue_id, pos, url, title, duration, channel,
                     upload_date, batch, now, now))
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def next_queued_jobs(self, limit: int = 2) -> list[dict]:
        """Next job(s) to process: active queues form a priority stack
        (most recently activated first), jobs in queue order within each."""
        return self._rows(
            "SELECT j.* FROM jobs j JOIN queues q ON q.id = j.queue_id "
            "WHERE j.status='queued' AND q.activated_at IS NOT NULL "
            "ORDER BY q.activated_at DESC, j.position, j.id LIMIT ?", (limit,))

    def pending(self, queue_id: int | None = None) -> list[dict]:
        return self._filtered("status IN ('queued','running')", queue_id)

    def failed(self, queue_id: int | None = None) -> list[dict]:
        return self._filtered("status='failed'", queue_id)

    def all_jobs(self, queue_id: int | None = None) -> list[dict]:
        return self._filtered("1=1", queue_id)

    def _filtered(self, cond: str, queue_id: int | None) -> list[dict]:
        if queue_id is None:
            return self._rows(f"SELECT * FROM jobs WHERE {cond} "
                              "ORDER BY queue_id, position, id")
        return self._rows(f"SELECT * FROM jobs WHERE {cond} AND queue_id=? "
                          "ORDER BY position, id", (queue_id,))

    def get(self, video_id: str, queue_id: int | None = None) -> dict | None:
        """A job for this video (in a specific queue, or any queue)."""
        if queue_id is None:
            rows = self._rows("SELECT * FROM jobs WHERE video_id=? LIMIT 1",
                              (video_id,))
        else:
            rows = self._rows("SELECT * FROM jobs WHERE video_id=? AND queue_id=?",
                              (video_id, queue_id))
        return rows[0] if rows else None

    def queues_containing(self, video_id: str) -> list[int]:
        return [r["queue_id"] for r in self._rows(
            "SELECT queue_id FROM jobs WHERE video_id=?", (video_id,))]

    # -- ordering & bulk operations (context menu) ---------------------------

    def move_jobs(self, job_ids: list[int], direction: str) -> None:
        """Reorder within a queue: 'up' | 'down' | 'top' | 'bottom'."""
        if not job_ids:
            return
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, queue_id FROM jobs WHERE id IN (%s)"
                % ",".join("?" * len(job_ids)), job_ids).fetchall()
            by_queue: dict[int, set[int]] = {}
            for r in rows:
                by_queue.setdefault(r["queue_id"], set()).add(r["id"])
            for qid, selected in by_queue.items():
                ordered = [r["id"] for r in self._conn.execute(
                    "SELECT id FROM jobs WHERE queue_id=? ORDER BY position, id",
                    (qid,))]
                ordered = _reorder(ordered, selected, direction)
                for pos, jid in enumerate(ordered, 1):
                    self._conn.execute("UPDATE jobs SET position=? WHERE id=?",
                                       (pos, jid))
            self._conn.commit()

    def remove_jobs(self, job_ids: list[int]) -> int:
        if not job_ids:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE id IN (%s) AND status != 'running'"
                % ",".join("?" * len(job_ids)), job_ids)
            self._conn.commit()
            return cur.rowcount

    def requeue_jobs(self, job_ids: list[int]) -> int:
        """Requeue selected failed/cancelled/done jobs."""
        if not job_ids:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='queued', error='', updated_at=? "
                "WHERE id IN (%s) AND status IN ('failed','cancelled','done')"
                % ",".join("?" * len(job_ids)), (time.time(), *job_ids))
            self._conn.commit()
            return cur.rowcount

    # -- state transitions -------------------------------------------------

    def set_stage(self, video_id: str, stage: str, status: str = "running",
                  queue_id: int | None = None) -> None:
        self._update(video_id, stage=stage, status=status, queue_id=queue_id)

    def set_meta(self, video_id: str, title: str, duration: float,
                 channel: str, upload_date: str) -> None:
        self._update(video_id, title=title, duration=duration,
                     channel=channel, upload_date=upload_date)

    def mark_done(self, video_id: str, wall_seconds: float,
                  queue_id: int | None = None) -> None:
        self._update(video_id, status="done", stage="complete",
                     error="", wall_seconds=wall_seconds, queue_id=queue_id)

    def mark_failed(self, video_id: str, error: str,
                    queue_id: int | None = None) -> None:
        cond, params = _vid_cond(video_id, queue_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET status='failed', error=?, retries=retries+1, "
                f"updated_at=? WHERE {cond}",
                (error[:2000], time.time(), *params))
            self._conn.commit()

    def mark_cancelled(self, video_id: str, queue_id: int | None = None) -> None:
        self._update(video_id, status="cancelled", queue_id=queue_id)

    def requeue_failed(self, include_cancelled: bool = True) -> int:
        """Move failed (and cancelled) jobs back to the queue."""
        statuses = "('failed','cancelled')" if include_cancelled else "('failed')"
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE jobs SET status='queued', error='', updated_at=? "
                f"WHERE status IN {statuses}", (time.time(),))
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

    def clear_finished(self, queue_id: int | None = None) -> int:
        with self._lock:
            if queue_id is None:
                cur = self._conn.execute(
                    "DELETE FROM jobs WHERE status IN ('done','cancelled')")
            else:
                cur = self._conn.execute(
                    "DELETE FROM jobs WHERE status IN ('done','cancelled') "
                    "AND queue_id=?", (queue_id,))
            self._conn.commit()
            return cur.rowcount

    # -- internals ----------------------------------------------------------

    def _update(self, video_id: str, queue_id: int | None = None, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        cond, params = _vid_cond(video_id, queue_id)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {cols} WHERE {cond}",
                               (*fields.values(), *params))
            self._conn.commit()

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _vid_cond(video_id: str, queue_id: int | None) -> tuple[str, tuple]:
    if queue_id is None:
        return "video_id=?", (video_id,)
    return "video_id=? AND queue_id=?", (video_id, queue_id)


def _reorder(ordered: list[int], selected: set[int], direction: str) -> list[int]:
    """Move `selected` ids within `ordered` one step or to an end,
    preserving their relative order."""
    sel = [i for i in ordered if i in selected]
    rest = [i for i in ordered if i not in selected]
    if direction == "top":
        return sel + rest
    if direction == "bottom":
        return rest + sel
    result = list(ordered)
    indices = [result.index(i) for i in sel]
    if direction == "up":
        for idx in indices:
            if idx > 0 and result[idx - 1] not in selected:
                result[idx - 1], result[idx] = result[idx], result[idx - 1]
    elif direction == "down":
        for idx in reversed(indices):
            if idx < len(result) - 1 and result[idx + 1] not in selected:
                result[idx + 1], result[idx] = result[idx], result[idx + 1]
    return result
