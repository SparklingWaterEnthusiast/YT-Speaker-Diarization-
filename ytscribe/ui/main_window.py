"""Main application window: queue, progress, resource monitors, log, controls."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QFont
from PySide6.QtWidgets import (QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget,
                               QWidgetAction)

from .. import resources, voices
from ..config import APP_DIR, load_config, save_config
from ..db import JobStore
from ..export import fmt_ts
from .settings_dialog import SettingsDialog
from .workers import EnqueueWorker, RunWorker

STATUS_COLORS = {"done": "#2e7d32", "failed": "#c62828", "running": "#1565c0",
                 "queued": "#616161", "cancelled": "#8d6e63"}
MAX_LOG_BLOCKS = 2000


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YTScribe — YouTube Speaker-Diarized Transcripts")
        self.resize(1180, 780)

        self.cfg = load_config()
        save_config(self.cfg)  # materialize defaults on first launch
        self.store = JobStore(APP_DIR / "jobs.sqlite3")
        recovered = self.store.recover_interrupted()

        self.run_worker: RunWorker | None = None
        self.enqueue_worker: EnqueueWorker | None = None
        self.run_started_at: float | None = None
        self.voice_db = voices.VoiceDB(APP_DIR / "voices.sqlite3")
        self._row_jobs: list[dict] = []

        self._build_ui()
        self.refresh_queue()
        if recovered:
            self.log_line(f"Recovered {recovered} interrupted job(s); press Start to resume.")

        self.res_timer = QTimer(self)
        self.res_timer.timeout.connect(self._update_resources)
        self.res_timer.start(1000)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_elapsed)
        self.clock_timer.start(1000)

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        # URL input row
        row = QHBoxLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(
            "Paste a YouTube video, playlist, or channel URL…")
        self.url_edit.returnPressed.connect(self.add_url)
        self.add_btn = QPushButton("Add to Queue")
        self.add_btn.clicked.connect(self.add_url)
        row.addWidget(self.url_edit, 1)
        row.addWidget(self.add_btn)
        root.addLayout(row)

        split = QSplitter(Qt.Orientation.Vertical)

        # queue table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Title", "Duration", "Status", "Stage", "Progress", "Error"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellClicked.connect(self._row_clicked)
        self.table.setToolTip("Click a completed video to review or rename "
                              "its speakers")
        split.addWidget(self.table)

        # bottom: current item + log
        bottom = QWidget()
        bl = QHBoxLayout(bottom)

        cur = QGroupBox("Current")
        grid = QGridLayout(cur)
        self.cur_title = QLabel("—")
        self.cur_title.setWordWrap(True)
        font = QFont(); font.setBold(True)
        self.cur_title.setFont(font)
        self.cur_stage = QLabel("idle")
        self.item_bar = QProgressBar(); self.item_bar.setRange(0, 1000)
        self.item_label = QLabel("")
        self.total_bar = QProgressBar(); self.total_bar.setRange(0, 1000)
        self.elapsed_label = QLabel("Elapsed: —")
        self.eta_label = QLabel("Remaining: —")
        self.speed_label = QLabel("Speed: —")
        grid.addWidget(self.cur_title, 0, 0, 1, 2)
        grid.addWidget(QLabel("Stage:"), 1, 0); grid.addWidget(self.cur_stage, 1, 1)
        grid.addWidget(self.item_bar, 2, 0, 1, 2)
        grid.addWidget(self.item_label, 3, 0, 1, 2)
        grid.addWidget(QLabel("Queue:"), 4, 0); grid.addWidget(self.total_bar, 4, 1)
        grid.addWidget(self.elapsed_label, 5, 0)
        grid.addWidget(self.eta_label, 5, 1)
        grid.addWidget(self.speed_label, 6, 0, 1, 2)
        grid.setRowStretch(7, 1)
        bl.addWidget(cur, 1)

        logbox = QGroupBox("Log")
        lv = QVBoxLayout(logbox)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_BLOCKS)
        lv.addWidget(self.log)
        bl.addWidget(logbox, 2)

        split.addWidget(bottom)
        split.setSizes([420, 300])
        root.addWidget(split, 1)

        # controls
        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_queue)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.cancel_btn = QPushButton("Cancel Current")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_current)
        self.retry_btn = QPushButton("Retry Failed")
        self.retry_btn.clicked.connect(self.retry_failed)
        self.clear_btn = QPushButton("Clear Finished")
        self.clear_btn.clicked.connect(self.clear_finished)
        self.open_btn = QPushButton("Open Output Folder")
        self.open_btn.clicked.connect(self.open_output)
        self.settings_btn = QPushButton("Settings…")
        self.settings_btn.clicked.connect(self.open_settings)
        for b in (self.start_btn, self.pause_btn, self.cancel_btn, self.retry_btn,
                  self.clear_btn, self.open_btn, self.settings_btn):
            controls.addWidget(b)
        controls.addStretch(1)
        root.addLayout(controls)

        # resource strip
        res = QHBoxLayout()
        self.res_labels = {}
        for key, text in (("gpu", "GPU: —"), ("vram", "VRAM: —"), ("cpu", "CPU: —"),
                          ("ram", "RAM: —"), ("disk", "Disk free: —")):
            lbl = QLabel(text)
            self.res_labels[key] = lbl
            res.addWidget(lbl)
            res.addSpacing(18)
        res.addStretch(1)
        root.addLayout(res)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------- actions --

    def add_url(self):
        url = self.url_edit.text().strip()
        if not url:
            return
        if self.enqueue_worker and self.enqueue_worker.isRunning():
            QMessageBox.information(self, "Busy", "Still resolving the previous URL.")
            return
        self.add_btn.setEnabled(False)
        self.statusBar().showMessage("Resolving URL…")
        self.enqueue_worker = EnqueueWorker(url, self.cfg, self.store, self)
        self.enqueue_worker.log.connect(self.log_line)
        self.enqueue_worker.done.connect(self._enqueue_done)
        self.enqueue_worker.failed.connect(self._enqueue_failed)
        self.enqueue_worker.start()

    def _enqueue_done(self, batch: str, added: int):
        self.add_btn.setEnabled(True)
        self.url_edit.clear()
        self.statusBar().showMessage(f"Added {added} video(s).", 5000)
        self.refresh_queue()

    def _enqueue_failed(self, error: str):
        self.add_btn.setEnabled(True)
        self.statusBar().showMessage("Could not add URL.", 5000)
        self.log_line(f"ERROR: {error}")
        QMessageBox.warning(self, "Could not add URL", error)

    def start_queue(self):
        if self.run_worker and self.run_worker.isRunning():
            return
        if not [j for j in self.store.pending() if j["status"] == "queued"]:
            QMessageBox.information(self, "Queue empty",
                                    "Add a video, playlist, or channel URL first.")
            return
        self.cfg = load_config()  # pick up any settings changes
        self.run_worker = RunWorker(self.cfg, self.store, self)
        w = self.run_worker
        w.log.connect(self.log_line)
        w.job_start.connect(self._job_start)
        w.stage.connect(self._stage)
        w.item_progress.connect(self._item_progress)
        w.job_done.connect(self._job_done)
        w.queue_progress.connect(self._queue_progress)
        w.queue_done.connect(self._queue_done)
        w.start()
        self.run_started_at = time.time()
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.statusBar().showMessage("Processing…")

    def toggle_pause(self):
        if not self.run_worker:
            return
        runner = self.run_worker.runner
        if runner.pause_event.is_set():
            runner.resume()
            self.pause_btn.setText("Pause")
            self.statusBar().showMessage("Processing…")
        else:
            runner.pause()
            self.pause_btn.setText("Resume")
            self.statusBar().showMessage("Paused")

    def cancel_current(self):
        if self.run_worker:
            self.run_worker.runner.cancel_current_job()

    def retry_failed(self):
        n = self.store.requeue_failed()
        self.log_line(f"Requeued {n} failed job(s).")
        self.refresh_queue()

    def clear_finished(self):
        n = self.store.clear_finished()
        self.log_line(f"Removed {n} finished job(s) from the list.")
        self.refresh_queue()

    def open_output(self):
        path = self.cfg.output_path
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))  # noqa: S606 — desktop app opening its own folder

    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self.cfg = load_config()
            self.log_line("Settings saved.")
            if self.run_worker and self.run_worker.isRunning():
                self.log_line("Note: model/device changes apply to the next queue run.")

    # ------------------------------------------------------- speaker editor --

    def _row_clicked(self, row: int, _col: int):
        if row >= len(self._row_jobs):
            return
        job = self._row_jobs[row]
        if job["status"] != "done":
            return
        self._show_speaker_menu(job)

    def _show_speaker_menu(self, job: dict):
        vdir = Path(self.cfg.cache_dir) / job["video_id"]
        merged_file = vdir / "merged.json"
        if not merged_file.exists():
            self.statusBar().showMessage(
                "No speaker data cached for this video.", 4000)
            return
        merged = json.loads(merged_file.read_text(encoding="utf-8"))
        labels = merged.get("speakers", [])
        if not labels:
            self.statusBar().showMessage("No speakers detected in this video.", 4000)
            return
        names = voices.display_names(self.cfg, vdir, labels)
        mapping = voices.load_speaker_map(vdir)

        menu = QMenu(self)
        header = menu.addAction(f"Speakers — {(job['title'] or job['video_id'])[:48]}")
        header.setEnabled(False)
        menu.addSeparator()
        for label in labels:
            menu.addAction(self._speaker_row_action(menu, vdir, label,
                                                    names[label], mapping))
        menu.exec(QCursor.pos())

    def _speaker_row_action(self, menu: QMenu, vdir: Path, label: str,
                            display: str, mapping: dict) -> QWidgetAction:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(10, 2, 10, 2)

        entry = mapping.get(label) or {}
        text = display if display == label else f"{display}  ({label})"
        if entry.get("source") == "auto":
            text += f"  · auto {entry.get('score', 0):.2f}"
        name_lbl = QLabel(text)
        name_lbl.setMinimumWidth(220)
        lay.addWidget(name_lbl)

        play = QPushButton("▶")
        play.setFixedWidth(30)
        play.setToolTip("Play a short voice sample")
        sample = voices.sample_path(vdir, label)
        play.setEnabled(sample is not None)
        play.clicked.connect(lambda _=False, p=sample: self._play_sample(p))
        lay.addWidget(play)

        edit = QLineEdit()
        edit.setPlaceholderText("rename… (Enter)")
        edit.setMinimumWidth(160)
        edit.returnPressed.connect(
            lambda l=label, e=edit, m=menu, v=vdir: self._apply_rename(
                v, l, e.text(), m))
        lay.addWidget(edit)

        action = QWidgetAction(menu)
        action.setDefaultWidget(row)
        return action

    def _play_sample(self, path: Path | None):
        if path is None:
            return
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(str(path),
                               winsound.SND_FILENAME | winsound.SND_ASYNC)

    def _apply_rename(self, vdir: Path, label: str, new_name: str, menu: QMenu):
        menu.close()
        new_name = new_name.strip()
        if not new_name:
            return
        try:
            voices.apply_rename(self.cfg, vdir, label, new_name,
                                self.voice_db, self.log_line)
        except Exception as exc:
            self.log_line(f"ERROR renaming {label}: {exc}")
            QMessageBox.warning(self, "Rename failed", str(exc))
            return
        self.statusBar().showMessage(
            f"{label} is now '{new_name}' — transcripts updated, voice "
            "profile saved.", 6000)

    # ---------------------------------------------------------- run events --

    def _job_start(self, video_id: str, title: str):
        self.cur_title.setText(title)
        self.cur_stage.setText("starting")
        self.item_bar.setValue(0)
        self.refresh_queue()

    def _stage(self, video_id: str, stage: str):
        self.cur_stage.setText(stage)
        self.item_bar.setValue(0)
        self.item_label.setText("")
        self.refresh_queue()

    def _item_progress(self, video_id: str, stage: str, fraction: float, label: str):
        self.item_bar.setValue(int(fraction * 1000))
        self.item_label.setText(f"{stage}: {label}")

    def _job_done(self, video_id: str, status: str, error: str):
        self.refresh_queue()

    def _queue_progress(self, done: int, total: int, eta_s: float, rtf: float):
        if total:
            self.total_bar.setValue(int(done / total * 1000))
        self.total_bar.setFormat(f"{done}/{total} videos")
        self.eta_label.setText(f"Remaining: ~{fmt_ts(eta_s)}" if eta_s else "Remaining: —")
        self.speed_label.setText(
            f"Speed: {rtf:.2f}× realtime" if rtf else "Speed: —")

    def _queue_done(self, summary: dict):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.cancel_btn.setEnabled(False)
        self.run_started_at = None
        self.cur_stage.setText("idle")
        self.statusBar().showMessage(
            f"Queue finished: {summary.get('completed', 0)} completed, "
            f"{summary.get('failed', 0)} failed.")
        self.refresh_queue()
        if summary.get("failed"):
            self.log_line(f"WARNING: {summary['failed']} video(s) failed — "
                          "use 'Retry Failed' to requeue them.")

    # ------------------------------------------------------------- display --

    def refresh_queue(self):
        jobs = self.store.all_jobs()
        self._row_jobs = jobs
        self.table.setRowCount(len(jobs))
        for r, j in enumerate(jobs):
            title = j["title"] or j["url"]
            dur = fmt_ts(j["duration"]) if j["duration"] else "—"
            cells = [title, dur, j["status"], j["stage"],
                     "✔" if j["status"] == "done" else "",
                     j["error"] or ""]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if c == 2:
                    color = STATUS_COLORS.get(j["status"])
                    if color:
                        item.setForeground(QColor(color))
                self.table.setItem(r, c, item)
        done = sum(1 for j in jobs if j["status"] == "done")
        if jobs:
            self.total_bar.setValue(int(done / len(jobs) * 1000))
            self.total_bar.setFormat(f"{done}/{len(jobs)} videos")

    def log_line(self, msg: str):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _update_elapsed(self):
        if self.run_started_at:
            self.elapsed_label.setText(
                f"Elapsed: {fmt_ts(time.time() - self.run_started_at)}")

    def _update_resources(self):
        snap = resources.sample(self.cfg.cache_dir)
        r = self.res_labels
        r["gpu"].setText(f"GPU: {snap.gpu_percent:.0f}%" if snap.gpu_available
                         else "GPU: n/a")
        r["vram"].setText(
            f"VRAM: {snap.vram_used_gb:.1f}/{snap.vram_total_gb:.0f} GB"
            if snap.gpu_available else "VRAM: n/a")
        r["cpu"].setText(f"CPU: {snap.cpu_percent:.0f}%")
        r["ram"].setText(f"RAM: {snap.ram_used_gb:.1f}/{snap.ram_total_gb:.0f} GB")
        r["disk"].setText(f"Disk free: {snap.disk_free_gb:.0f} GB")

    # ------------------------------------------------------------- closing --

    def closeEvent(self, event):
        if self.run_worker and self.run_worker.isRunning():
            answer = QMessageBox.question(
                self, "Quit", "Processing is running. Stop after the current "
                "stage and quit?\nCompleted work is saved and will resume next time.")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.run_worker.runner.stop()
            self.run_worker.runner.cancel_current_job()
            self.run_worker.wait(15000)
        event.accept()
