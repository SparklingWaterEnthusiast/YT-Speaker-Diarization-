"""QThread wrappers around the pure-Python pipeline.

All pipeline callbacks are re-emitted as Qt signals, which are thread-safe
and delivered to the UI thread via queued connections.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from .. import pipeline
from ..config import Config
from ..db import JobStore


class EnqueueWorker(QThread):
    """Resolves a URL (video/playlist/channel) and adds jobs — can take a
    while for large channels, so it never runs on the UI thread."""

    log = Signal(str)
    done = Signal(str, int)      # batch name, number added
    failed = Signal(str)

    def __init__(self, url: str, cfg: Config, store: JobStore,
                 queue_id: int, parent=None):
        super().__init__(parent)
        self.url, self.cfg, self.store = url, cfg, store
        self.queue_id = queue_id

    def run(self):
        try:
            batch, added = pipeline.enqueue(self.url, self.cfg, self.store,
                                            self.log.emit, self.queue_id)
            self.done.emit(batch, added)
        except Exception as exc:
            self.failed.emit(str(exc))


class RunWorker(QThread):
    """Owns the QueueRunner (and therefore the GPU) for a queue run."""

    log = Signal(str)
    queue_progress = Signal(int, int, float, float)   # done, total, eta_s, rtf
    job_start = Signal(str, str)                      # video_id, title
    stage = Signal(str, str)                          # video_id, stage
    item_progress = Signal(str, str, float, str)      # video_id, stage, fraction, label
    job_done = Signal(str, str, str)                  # video_id, status, error
    queue_done = Signal(dict)

    def __init__(self, cfg: Config, store: JobStore, parent=None):
        super().__init__(parent)
        cb = pipeline.Callbacks(
            on_log=self.log.emit,
            on_queue_progress=self.queue_progress.emit,
            on_job_start=self.job_start.emit,
            on_stage=self.stage.emit,
            on_item_progress=self.item_progress.emit,
            on_job_done=self.job_done.emit,
            on_queue_done=self.queue_done.emit,
        )
        self.runner = pipeline.QueueRunner(cfg, store, cb)

    def run(self):
        try:
            self.runner.run()
        except Exception as exc:  # never let a crash take down the app silently
            self.log.emit(f"FATAL queue error: {type(exc).__name__}: {exc}")
            self.queue_done.emit({"completed": 0, "failed": 1,
                                  "wall_seconds": 0, "combined": [], "avg_rtf": 0})
