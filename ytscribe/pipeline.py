"""Queue orchestration: runs each job through the five pipeline stages.

Pure Python (no Qt imports) so it is testable headless and reusable from the
CLI; the UI drives it from a worker thread through the Callbacks bag and the
pause/cancel/stop events.

Concurrency model (sized for 8 GB VRAM):
  - exactly one video occupies the GPU at a time;
  - while the GPU works, a prefetch thread downloads the next item's audio;
  - randomized sleep between downloads keeps YouTube request rates human-ish.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import diarize, export, media, merge, transcribe, voices
from .config import APP_DIR, Config
from .db import JobStore


def _noop(*args, **kwargs):
    pass


@dataclass
class Callbacks:
    on_log: Callable = _noop                # (msg)
    on_queue_progress: Callable = _noop     # (done, total, eta_s, avg_rtf)
    on_job_start: Callable = _noop          # (video_id, title)
    on_stage: Callable = _noop              # (video_id, stage)
    on_item_progress: Callable = _noop      # (video_id, stage, fraction, label)
    on_job_done: Callable = _noop           # (video_id, status, error)
    on_queue_done: Callable = _noop         # (summary: dict)
    on_jobs_changed: Callable = _noop       # () — queue contents changed


def enqueue(url: str, cfg: Config, store: JobStore,
            log: Callable[[str], None] = print) -> tuple[str, int]:
    """Expand a URL and add its videos to the queue.

    Returns (batch_name, number_of_new_jobs).
    """
    kind = media.classify_url(url)
    if kind == "unknown":
        raise media.DownloadError(f"Not a recognizable YouTube URL: {url}")
    log(f"Resolving {kind}: {url}")
    entries = media.enumerate_videos(url, cfg, log)
    if not entries:
        raise media.DownloadError("No processable videos found at that URL.")
    batch = entries[0].get("channel") or "queue"
    batch = f"{batch} {time.strftime('%Y-%m-%d %H%M')}"
    added = 0
    for e in entries:
        if store.add(e["video_id"], e["url"], e["title"], e["duration"],
                     e.get("channel", ""), batch=batch):
            added += 1
    log(f"Queued {added} new video(s) ({len(entries) - added} already known).")
    return batch, added


class QueueRunner:
    def __init__(self, cfg: Config, store: JobStore, callbacks: Callbacks | None = None):
        self.cfg = cfg
        self.store = store
        self.cb = callbacks or Callbacks()
        self.pause_event = threading.Event()     # set -> paused
        self.cancel_current = threading.Event()  # set -> abandon current job
        self.stop_event = threading.Event()      # set -> finish current, stop queue
        self._transcriber: transcribe.Transcriber | None = None
        self._diarizer: diarize.Diarizer | None = None
        self._voice_db: voices.VoiceDB | None = None
        self._rtf_ema: float | None = None       # audio-seconds per wall-second
        self._prefetch: dict = {}

    # -- public control ------------------------------------------------------

    def pause(self):
        self.pause_event.set()
        self.cb.on_log("Paused. Current stage will finish, then the queue waits.")

    def resume(self):
        self.pause_event.clear()
        self.cb.on_log("Resumed.")

    def cancel_current_job(self):
        self.cancel_current.set()
        self.cb.on_log("Cancelling current item at the next checkpoint...")

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()

    # -- main loop -------------------------------------------------------------

    def run(self) -> dict:
        recovered = self.store.recover_interrupted()
        if recovered:
            self.cb.on_log(f"Recovered {recovered} interrupted job(s) from a previous run.")

        session_done: list[str] = []
        failed = 0
        queue_started = time.time()
        first_download_done = False

        while not self.stop_event.is_set():
            self._wait_if_paused()
            pending = [j for j in self.store.pending() if j["status"] == "queued"]
            if not pending:
                break
            job = pending[0]
            nxt = pending[1] if len(pending) > 1 else None
            self.cancel_current.clear()

            # polite spacing between YouTube downloads
            if first_download_done and not self._prefetched(job["video_id"]):
                self._sleep_between_downloads()

            started = time.time()
            self.cb.on_job_start(job["video_id"], job["title"] or job["url"])
            try:
                self._process(job, nxt)
            except media.CancelledError:
                self.store.mark_cancelled(job["video_id"])
                self.cb.on_job_done(job["video_id"], "cancelled", "")
                self.cb.on_log(f"Cancelled: {job['title'] or job['video_id']}")
                continue
            except Exception as exc:  # any stage failure -> retry queue, keep going
                error = f"{type(exc).__name__}: {exc}"
                self.store.mark_failed(job["video_id"], error)
                failed += 1
                self.cb.on_job_done(job["video_id"], "failed", error)
                self.cb.on_log(f"FAILED {job['video_id']}: {error}")
                continue
            finally:
                first_download_done = True

            wall = time.time() - started
            self.store.mark_done(job["video_id"], wall)
            session_done.append(job["video_id"])
            self._update_rtf(job["duration"], wall)
            self.cb.on_job_done(job["video_id"], "done", "")
            self._emit_queue_progress()

        summary = self._finish_queue(session_done, failed, time.time() - queue_started)
        self.cb.on_queue_done(summary)
        return summary

    def run_single(self, video_id: str) -> None:
        """Process one known video through the pipeline regardless of its
        queue status (missing stages run, cached stages are reused).
        Used by --seed; does not touch other queued jobs."""
        job = self.store.get(video_id)
        if job is None:
            raise ValueError(f"video {video_id} is not in the job store")
        started = time.time()
        self.cb.on_job_start(video_id, job["title"] or job["url"])
        self._process(job, None)
        self.store.mark_done(video_id, time.time() - started)
        self.cb.on_job_done(video_id, "done", "")

    # -- per-job processing ---------------------------------------------------

    def _process(self, job: dict, next_job: dict | None) -> None:
        vid = job["video_id"]
        vdir = self.cfg.cache_path / vid
        ffmpeg = self.cfg.resolve_ffmpeg()

        # housekeeping: half-written artifacts from a crash mid-stage
        if vdir.is_dir():
            for leftover in vdir.glob("*.tmp*"):
                leftover.unlink(missing_ok=True)

        # Stage 1: acquire ---------------------------------------------------
        self._enter_stage(vid, "acquire")
        audio = self._take_prefetched(vid)
        if audio is None:
            audio = media.download_audio(
                job["url"], vdir, self.cfg,
                progress_cb=lambda f, s: self.cb.on_item_progress(vid, "acquire", f, s),
                cancelled=self.cancel_current.is_set)
        wav = media.to_wav(audio, ffmpeg)
        meta = media.load_meta(vdir)
        duration = float(meta.get("duration") or 0) or media.wav_duration(wav)
        self.store.set_meta(vid, meta.get("title") or job["title"], duration,
                            meta.get("channel") or "", meta.get("upload_date") or "")
        self._checkpoint()

        # start prefetching the next item's audio while the GPU works
        if next_job:
            self._start_prefetch(next_job)

        # Stage 2: transcribe --------------------------------------------------
        self._enter_stage(vid, "transcribe")
        if self._transcriber is None:
            self._transcriber = transcribe.Transcriber(self.cfg, self.cb.on_log)
        asr = transcribe.run_stage(
            wav, vdir / "transcript.json", self._transcriber, duration,
            progress_cb=lambda f, s: self.cb.on_item_progress(vid, "transcribe", f, s),
            cancelled=self.cancel_current.is_set)
        self._checkpoint()

        # Stage 3: diarize ----------------------------------------------------
        self._enter_stage(vid, "diarize")
        self._invalidate_stale_diarization(vdir)
        if self._diarizer is None:
            self._diarizer = diarize.Diarizer(self.cfg, self.cb.on_log)
        dia = diarize.run_stage(
            wav, vdir / "diarization.json", self._diarizer,
            progress_cb=lambda f, s: self.cb.on_item_progress(vid, "diarize", f, s))

        # speaker identification: playback samples must be cut and voices
        # matched now, while audio.wav still exists (cache policy may delete it)
        voices.extract_samples(wav, dia, vdir / "samples")
        if self.cfg.recognition_enabled:
            if self._voice_db is None:
                self._voice_db = voices.VoiceDB(APP_DIR / "voices.sqlite3")
            voices.auto_match(vdir, dia, self._voice_db,
                              self.cfg.recognition_threshold, self.cb.on_log)
        self._checkpoint()

        # Stage 4: merge -----------------------------------------------------
        self._enter_stage(vid, "merge")
        merged = merge.run_stage(asr, dia, vdir / "merged.json", self.cfg)

        # Stage 5: export ------------------------------------------------------
        self._enter_stage(vid, "export")
        names = voices.display_names(self.cfg, vdir, merged.get("speakers", []))
        files = export.export_video(merged, meta, self.cfg, self.cfg.output_path,
                                    names=names)
        self.cb.on_log(f"Exported {len(files)} file(s) for '{meta.get('title', vid)}'")

        if self.cfg.cache_policy == "delete_after_video":
            self._delete_audio(vdir)

    def _invalidate_stale_diarization(self, vdir: Path) -> None:
        """Pre-v0.2 diarization artifacts lack embeddings; recomputing them may
        relabel speakers, so the merged artifact must be rebuilt with them."""
        diar_file = vdir / "diarization.json"
        if not diar_file.exists():
            return
        try:
            stale = "embeddings" not in json.loads(
                diar_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stale = True
        if stale:
            self.cb.on_log("Recomputing diarization (older artifact without "
                           "voice embeddings).")
            diar_file.unlink(missing_ok=True)
            (vdir / "merged.json").unlink(missing_ok=True)

    def _enter_stage(self, vid: str, stage: str) -> None:
        self._wait_if_paused()
        if self.cancel_current.is_set():
            raise media.CancelledError("cancelled")
        self.store.set_stage(vid, stage)
        self.cb.on_stage(vid, stage)

    def _checkpoint(self) -> None:
        self._wait_if_paused()
        if self.cancel_current.is_set():
            raise media.CancelledError("cancelled")

    def _wait_if_paused(self) -> None:
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.2)

    # -- prefetch --------------------------------------------------------------

    def _start_prefetch(self, job: dict) -> None:
        vid = job["video_id"]
        if vid in self._prefetch:
            return
        slot = {"path": None, "error": None, "thread": None}
        self._prefetch = {vid: slot}  # keep at most one prefetch alive

        def work():
            try:
                self._sleep_between_downloads(quiet=True)
                slot["path"] = media.download_audio(
                    job["url"], self.cfg.cache_path / vid, self.cfg,
                    cancelled=self.stop_event.is_set)
            except Exception as exc:
                slot["error"] = exc

        t = threading.Thread(target=work, daemon=True, name=f"prefetch-{vid}")
        slot["thread"] = t
        t.start()

    def _prefetched(self, vid: str) -> bool:
        return vid in self._prefetch

    def _take_prefetched(self, vid: str) -> Path | None:
        slot = self._prefetch.pop(vid, None)
        if not slot:
            return None
        slot["thread"].join()
        if slot["error"]:
            self.cb.on_log(f"Prefetch failed ({slot['error']}); retrying inline.")
            return None
        return slot["path"]

    # -- helpers -----------------------------------------------------------------

    def _sleep_between_downloads(self, quiet: bool = False) -> None:
        lo, hi = (self.cfg.sleep_between_downloads_min,
                  self.cfg.sleep_between_downloads_max)
        delay = random.uniform(lo, max(hi, lo))
        if delay <= 0:
            return
        if not quiet:
            self.cb.on_log(f"Waiting {delay:.0f}s before next download "
                           "(YouTube rate courtesy)...")
        deadline = time.time() + delay
        while time.time() < deadline and not self.stop_event.is_set():
            time.sleep(0.2)

    def _delete_audio(self, vdir: Path) -> None:
        for f in vdir.glob("audio.*"):
            f.unlink(missing_ok=True)

    def _update_rtf(self, audio_seconds: float, wall_seconds: float) -> None:
        if audio_seconds <= 0 or wall_seconds <= 0:
            return
        rtf = audio_seconds / wall_seconds
        self._rtf_ema = rtf if self._rtf_ema is None else 0.7 * self._rtf_ema + 0.3 * rtf

    def _emit_queue_progress(self) -> None:
        jobs = self.store.all_jobs()
        done = sum(1 for j in jobs if j["status"] == "done")
        active = [j for j in jobs if j["status"] in ("queued", "running")]
        remaining_audio = sum(j["duration"] or 1800 for j in active)  # assume 30 min if unknown
        eta = remaining_audio / self._rtf_ema if self._rtf_ema else 0
        self.cb.on_queue_progress(done, len(jobs), eta, self._rtf_ema or 0)

    def _finish_queue(self, session_done: list[str], failed: int,
                      wall: float) -> dict:
        # combined transcript over everything completed this session, queue order
        combined_files: list[str] = []
        if self.cfg.combined_transcript and len(session_done) > 1:
            items = []
            for vid in session_done:
                vdir = self.cfg.cache_path / vid
                mfile, meta_file = vdir / "merged.json", vdir / "meta.json"
                if mfile.exists() and meta_file.exists():
                    merged = json.loads(mfile.read_text(encoding="utf-8"))
                    names = voices.display_names(self.cfg, vdir,
                                                 merged.get("speakers", []))
                    items.append((merged,
                                  json.loads(meta_file.read_text(encoding="utf-8")),
                                  names))
            if items:
                job = self.store.get(session_done[0])
                batch = (job.get("batch") or "").rsplit(" ", 2)[0] if job else ""
                name = batch or f"queue {time.strftime('%Y-%m-%d')}"
                combined_files = [str(p) for p in export.export_combined(
                    items, self.cfg, self.cfg.output_path, name)]
                self.cb.on_log(f"Combined transcript written ({len(items)} videos).")

        if self.cfg.cache_policy == "delete_after_queue":
            # every finished job, not just this session's: a queue completed
            # across several sessions must still clean up earlier audio
            for j in self.store.all_jobs():
                if j["status"] == "done":
                    self._delete_audio(self.cfg.cache_path / j["video_id"])

        return {"completed": len(session_done), "failed": failed,
                "wall_seconds": wall, "combined": combined_files,
                "avg_rtf": self._rtf_ema or 0}

    def unload_models(self) -> None:
        if self._transcriber:
            self._transcriber.unload()
        if self._diarizer:
            self._diarizer.unload()
