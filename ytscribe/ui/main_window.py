"""Main application window: queue, progress, resource monitors, log, controls."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSplitter, QTabBar, QTableWidget,
                               QTableWidgetItem, QToolButton, QVBoxLayout,
                               QWidget, QWidgetAction)

from .. import export, resources, voices
from ..config import APP_DIR, load_config, save_config
from ..db import JobStore
from ..export import fmt_ts
from .settings_dialog import SettingsDialog
from .workers import EnqueueWorker, RunWorker

STATUS_COLORS = {"done": "#2e7d32", "failed": "#c62828", "running": "#1565c0",
                 "queued": "#616161", "cancelled": "#8d6e63"}
MAX_LOG_BLOCKS = 2000


def _safe_folder(name: str) -> str:
    """Queue name -> filesystem-safe output subfolder name."""
    folder = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    return re.sub(r"\s+", " ", folder).strip()[:80] or "queue"


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
        from .. import media
        for warning in media.environment_report():
            self.log_line(f"WARNING: {warning}")

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

        # queue tabs (File-Explorer style: one queue per tab)
        tab_row = QHBoxLayout()
        tab_row.setSpacing(0)
        self.tabs = QTabBar()
        self.tabs.setMovable(True)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setDocumentMode(True)
        self.tabs.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tabs.currentChanged.connect(lambda _i: self.refresh_queue())
        self.tabs.tabBarDoubleClicked.connect(self._rename_queue_tab)
        self.tabs.customContextMenuRequested.connect(self._tab_context_menu)
        self.tabs.tabMoved.connect(self._tabs_reordered)
        add_tab = QToolButton()
        add_tab.setText("+")
        add_tab.setToolTip("New queue tab")
        add_tab.clicked.connect(self._new_queue_tab)
        tab_row.addWidget(self.tabs, 1)
        tab_row.addWidget(add_tab)
        root.addLayout(tab_row)
        self._tab_queue_ids: list[int] = []
        self._reload_tabs()

        split = QSplitter(Qt.Orientation.Vertical)

        # queue table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Title", "Duration", "Status", "Stage", "Progress", "Error"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_context_menu)
        self.table.cellDoubleClicked.connect(self._row_double_clicked)
        self.table.setToolTip("Double-click a completed video to edit its "
                              "speakers; right-click for more options")
        QShortcut(QKeySequence.StandardKey.Delete, self.table,
                  activated=self._delete_selected)
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
        self.enqueue_worker = EnqueueWorker(url, self.cfg, self.store,
                                            self.current_queue_id(), self)
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
        qid = self.current_queue_id()
        queue = self.store.get_queue(qid) or {"name": "?"}
        if not [j for j in self.store.pending(qid) if j["status"] == "queued"]:
            QMessageBox.information(self, "Queue empty",
                                    "This tab has no queued videos — add a "
                                    "video, playlist, or channel URL first.")
            return
        self.store.set_queue_active(qid, True)
        if self.run_worker and self.run_worker.isRunning():
            # priority stack: the runner switches to this queue after the
            # current video finishes, and returns to the old one when done
            self.log_line(f"Queue '{queue['name']}' started — it takes "
                          "priority after the current video finishes.")
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
        n = self.store.requeue_failed(include_cancelled=True)
        self.log_line(f"Requeued {n} failed/cancelled job(s).")
        self.refresh_queue()

    def clear_finished(self):
        n = self.store.clear_finished(self.current_queue_id())
        self.log_line(f"Removed {n} finished job(s) from this tab.")
        self.refresh_queue()

    def open_output(self):
        self._open_folder(self._queue_out_dir(self.current_queue_id()))

    def open_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self.cfg = load_config()
            self.log_line("Settings saved.")
            if self.run_worker and self.run_worker.isRunning():
                self.log_line("Note: model/device changes apply to the next queue run.")

    # ------------------------------------------------------------ queue tabs --

    def current_queue_id(self) -> int:
        idx = self.tabs.currentIndex()
        if 0 <= idx < len(self._tab_queue_ids):
            return self._tab_queue_ids[idx]
        return 1

    def _reload_tabs(self):
        queues = self.store.queues()
        self.tabs.blockSignals(True)
        while self.tabs.count():
            self.tabs.removeTab(0)
        self._tab_queue_ids = []
        for q in queues:
            idx = self.tabs.addTab(q["name"])
            self.tabs.setTabData(idx, q["id"])
            self._tab_queue_ids.append(q["id"])
        self.tabs.blockSignals(False)

    def _new_queue_tab(self):
        name = time.strftime("%Y-%m-%d %H.%M")
        qid = self.store.create_queue(name, _safe_folder(name))
        self._reload_tabs()
        self.tabs.setCurrentIndex(self._tab_queue_ids.index(qid))
        self.refresh_queue()
        self.log_line(f"New queue '{name}' — double-click its tab to rename; "
                      "its transcripts go to a subfolder of the same name.")

    def _rename_queue_tab(self, index: int):
        if not (0 <= index < len(self._tab_queue_ids)):
            return
        qid = self._tab_queue_ids[index]
        queue = self.store.get_queue(qid)
        if not queue:
            return
        name, ok = QInputDialog.getText(self, "Rename queue", "Queue name:",
                                        text=queue["name"])
        name = name.strip()
        if not ok or not name or name == queue["name"]:
            return
        new_folder = _safe_folder(name)
        old_folder = queue["folder"]
        # keep transcripts together: rename the output subfolder with the tab
        if old_folder:
            old_path = self.cfg.output_path / old_folder
            new_path = self.cfg.output_path / new_folder
            if old_path.is_dir() and old_path != new_path:
                try:
                    old_path.rename(new_path)
                except OSError as exc:
                    QMessageBox.warning(
                        self, "Folder not renamed",
                        f"Queue renamed, but its output folder could not be "
                        f"moved ({exc}). New transcripts will use the new "
                        f"folder; existing files stay in '{old_folder}'.")
        elif qid == 1:
            # the original Main queue writes to the output root; renaming it
            # moves only future exports into a subfolder
            self.log_line("Note: existing transcripts of the Main queue stay "
                          "in the output root; new ones go to the subfolder "
                          f"'{new_folder}'.")
        self.store.rename_queue(qid, name, new_folder)
        self._reload_tabs()
        self.tabs.setCurrentIndex(index)

    def _tabs_reordered(self, *_):
        # QTabBar has already moved the tab; persist the new visual order
        new_ids = [self.tabs.tabData(i) for i in range(self.tabs.count())]
        self._tab_queue_ids = new_ids
        self.store.set_tab_order(new_ids)

    def _tab_context_menu(self, pos):
        index = self.tabs.tabAt(pos)
        if index < 0:
            return
        qid = self._tab_queue_ids[index]
        queue = self.store.get_queue(qid)
        if not queue:
            return
        menu = QMenu(self)
        menu.addAction("Rename…", lambda: self._rename_queue_tab(index))
        menu.addAction("Open output folder",
                       lambda: self._open_folder(self._queue_out_dir(qid)))
        if queue["activated_at"]:
            menu.addAction("Stop this queue",
                           lambda: self._stop_queue(qid, queue["name"]))
        menu.addSeparator()
        act_del = menu.addAction("Delete queue…",
                                 lambda: self._delete_queue_tab(index))
        act_del.setEnabled(len(self._tab_queue_ids) > 1)
        menu.exec(self.tabs.mapToGlobal(pos))

    def _stop_queue(self, qid: int, name: str):
        self.store.set_queue_active(qid, False)
        self.log_line(f"Queue '{name}' stopped — remaining videos stay "
                      "queued; press Start on its tab to continue.")

    def _delete_queue_tab(self, index: int):
        qid = self._tab_queue_ids[index]
        queue = self.store.get_queue(qid)
        jobs = self.store.all_jobs(qid)
        if any(j["status"] == "running" for j in jobs):
            QMessageBox.information(self, "Queue busy",
                                    "This queue is processing — cancel the "
                                    "current item first.")
            return
        if jobs:
            answer = QMessageBox.question(
                self, "Delete queue",
                f"Delete queue '{queue['name']}' and its {len(jobs)} "
                "list entries?\nTranscripts and cached results stay on disk.")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.store.delete_queue(qid)
        self._reload_tabs()
        self.tabs.setCurrentIndex(max(0, index - 1))
        self.refresh_queue()
        self.log_line(f"Queue '{queue['name']}' deleted.")

    def _queue_out_dir(self, qid: int) -> Path:
        queue = self.store.get_queue(qid) or {}
        folder = queue.get("folder") or ""
        return self.cfg.output_path / folder if folder else self.cfg.output_path

    def _open_folder(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))  # noqa: S606

    # ----------------------------------------------- table interactions --

    def _selected_jobs(self) -> list[dict]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self._row_jobs[r] for r in rows if r < len(self._row_jobs)]

    def _row_double_clicked(self, row: int, _col: int):
        if row >= len(self._row_jobs):
            return
        job = self._row_jobs[row]
        if job["status"] == "done":
            self._show_speaker_menu(job)

    def _table_context_menu(self, pos):
        jobs = self._selected_jobs()
        if not jobs:
            return
        menu = QMenu(self)
        single = jobs[0] if len(jobs) == 1 else None

        if single and single["status"] == "done":
            menu.addAction("Edit speakers…",
                           lambda: self._show_speaker_menu(single))
            menu.addAction("Open transcript",
                           lambda: self._open_transcript(single))
        menu.addAction("Open output folder",
                       lambda: self._open_folder(
                           self._queue_out_dir(self.current_queue_id())))
        menu.addSeparator()

        retryable = [j for j in jobs if j["status"] in ("failed", "cancelled")]
        redoable = [j for j in jobs if j["status"] == "done"]
        if retryable:
            menu.addAction(
                f"Retry {len(retryable)} failed/cancelled",
                lambda: self._requeue_jobs([j["id"] for j in retryable]))
        if redoable:
            menu.addAction(
                f"Reprocess {len(redoable)} completed",
                lambda: self._requeue_jobs([j["id"] for j in redoable]))
        menu.addSeparator()

        ids = [j["id"] for j in jobs]
        move = menu.addMenu("Move")
        move.addAction("Up", lambda: self._move_selected(ids, "up"))
        move.addAction("Down", lambda: self._move_selected(ids, "down"))
        move.addAction("To top", lambda: self._move_selected(ids, "top"))
        move.addAction("To bottom", lambda: self._move_selected(ids, "bottom"))
        menu.addSeparator()
        menu.addAction(f"Remove {len(jobs)} from queue\tDel",
                       self._delete_selected)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _requeue_jobs(self, ids: list[int]):
        n = self.store.requeue_jobs(ids)
        self.log_line(f"Requeued {n} video(s).")
        self.refresh_queue()

    def _move_selected(self, ids: list[int], direction: str):
        self.store.move_jobs(ids, direction)
        selected = set(ids)
        self.refresh_queue()
        # keep the moved rows selected so repeated moves feel natural
        self.table.clearSelection()
        for r, job in enumerate(self._row_jobs):
            if job["id"] in selected:
                for c in range(self.table.columnCount()):
                    item = self.table.item(r, c)
                    if item:
                        item.setSelected(True)

    def _delete_selected(self):
        jobs = self._selected_jobs()
        if not jobs:
            return
        running = [j for j in jobs if j["status"] == "running"]
        removable = [j["id"] for j in jobs if j["status"] != "running"]
        if running:
            self.log_line("The currently processing video was skipped — "
                          "use Cancel Current first.")
        if removable:
            n = self.store.remove_jobs(removable)
            self.log_line(f"Removed {n} video(s) from the queue "
                          "(transcripts and cache stay on disk).")
        self.refresh_queue()

    def _open_transcript(self, job: dict):
        vdir = Path(self.cfg.cache_dir) / job["video_id"]
        meta_file = vdir / "meta.json"
        if not meta_file.exists():
            self.statusBar().showMessage("No transcript found.", 4000)
            return
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        out_dir = self._queue_out_dir(job["queue_id"])
        base = out_dir / export.safe_filename(meta)
        md = base.parent / f"{base.name}.md"
        if md.exists():
            os.startfile(str(md))  # noqa: S606
        else:
            self.statusBar().showMessage(
                "Transcript file not found in this queue's folder.", 4000)

    # ------------------------------------------------------- speaker editor --

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
        # rewrite the transcripts in every queue folder this video was
        # exported to (a video can sit in several tabs since v0.2.1)
        out_dirs = [self._queue_out_dir(qid)
                    for qid in self.store.queues_containing(vdir.name)] or None
        try:
            voices.apply_rename(self.cfg, vdir, label, new_name,
                                self.voice_db, self.log_line,
                                out_dirs=out_dirs)
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
        jobs = self.store.all_jobs(self.current_queue_id())
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
