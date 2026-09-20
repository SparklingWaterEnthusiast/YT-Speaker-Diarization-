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
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import diarize, export, media, merge, transcribe, voices
from .config import APP_DIR, Config
from .db import DEFAULT_QUEUE_ID, JobStore
from .telemetry import span, event, Recorder, current_recorder, use_recorder
from .worker_lock import processing_lease


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
            log: Callable[[str], None] = print,
            queue_id: int = DEFAULT_QUEUE_ID) -> tuple[str, int]:
    """Expand a URL and add its videos to one queue (UI tab).

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
                     e.get("channel", ""), batch=batch, queue_id=queue_id):
            added += 1
    log(f"Queued {added} new video(s) ({len(entries) - added} already in "
        "this queue).")
    return batch, added


class QueueRunner:
    def __init__(self, cfg: Config, store: JobStore, callbacks: Callbacks | None = None):
        self.cfg = copy.deepcopy(cfg)
        problems = self.cfg.validate()
        if problems:
            raise ValueError('Invalid processing configuration: ' + '; '.join(problems))
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
        with processing_lease(self.cfg.cache_path, self.store.db_path.parent):
            with self._recording():
                try:
                    summary = self._run()
                finally:
                    self.close()
                self.cb.on_queue_done(summary)
                return summary

    def _recording(self):
        if current_recorder() is not None:
            return use_recorder(current_recorder())
        path = None
        if self.cfg.profiling_enabled:
            import os
            path = self.cfg.cache_path / 'diagnostics' / f'{time.time_ns()}-{os.getpid()}.jsonl'
            self.cb.on_log(f'Performance trace: {path}')
        return Recorder(path, disk_path=self.cfg.cache_dir,
                        interval=self.cfg.profiling_interval)

    def _run(self) -> dict:
        recovered = self.store.recover_interrupted()
        if recovered:
            self.cb.on_log(f"Recovered {recovered} interrupted job(s) from a previous run.")

        completed = 0
        failed = 0
        queue_started = time.perf_counter()
        first_download_done = False

        while not self.stop_event.is_set():
            self._wait_if_paused()
            # active queues form a priority stack: most recently started first
            with span("queue.transition"):
                upcoming = self.store.next_queued_jobs(2)
            if not upcoming:
                break
            job = upcoming[0]
            nxt = upcoming[1] if len(upcoming) > 1 else None
            qid = job["queue_id"]
            self.cancel_current.clear()

            # polite spacing between YouTube downloads
            if (first_download_done and self._needs_audio(job["video_id"])
                    and not media._find_audio(self.cfg.cache_path / job["video_id"])
                    and not self._prefetched(job["video_id"])):
                self._sleep_between_downloads()

            started = time.perf_counter()
            self.cb.on_job_start(job["video_id"], job["title"] or job["url"])
            try:
                with span("item.total", video_id=job['video_id'], queue_id=qid,
                          audio_duration=job['duration']):
                    self._process(job, nxt)
            except (media.CancelledError, InterruptedError):
                if self.stop_event.is_set():
                    current = self.store.get(job['video_id'], queue_id=qid)
                    self.store.set_stage(job['video_id'], current['stage'],
                                         status='queued', queue_id=qid)
                    self.cb.on_job_done(job['video_id'], 'queued', '')
                    self.cb.on_log('Interrupted stage queued for resume after restart.')
                    break
                self.store.mark_cancelled(job["video_id"], queue_id=qid)
                self.cb.on_job_done(job["video_id"], "cancelled", "")
                self.cb.on_log(f"Cancelled: {job['title'] or job['video_id']}")
                self._maybe_finalize_queue(qid)
                continue
            except Exception as exc:  # any stage failure -> retry queue, keep going
                error = f"{type(exc).__name__}: {exc}"
                self.store.mark_failed(job["video_id"], error, queue_id=qid)
                failed += 1
                self.cb.on_job_done(job["video_id"], "failed", error)
                self.cb.on_log(f"FAILED {job['video_id']}: {error}")
                self._maybe_finalize_queue(qid)
                continue
            finally:
                first_download_done = True

            wall = time.perf_counter() - started
            self.store.mark_done(job["video_id"], wall, queue_id=qid)
            completed += 1
            # Metadata from flat channel enumeration can be missing; use the
            # resolved duration. Cache-only re-exports must not inflate ETA.
            if self._did_inference:
                resolved = self.store.get(job['video_id'], queue_id=qid)
                self._update_rtf(resolved['duration'], wall)
            self.cb.on_job_done(job["video_id"], "done", "")
            self._emit_queue_progress()
            self._maybe_finalize_queue(qid)

        summary = self._finish_run(completed, failed, time.perf_counter() - queue_started)
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
        with processing_lease(self.cfg.cache_path, self.store.db_path.parent):
            with self._recording():
                try:
                    self._process(job, None)
                finally:
                    self.close()
        self.store.mark_done(video_id, time.time() - started,
                             queue_id=job["queue_id"])
        self.cb.on_job_done(video_id, "done", "")

    # -- per-job processing ---------------------------------------------------

    def _process(self, job: dict, next_job: dict | None) -> None:
        vid = job["video_id"]
        qid = job.get("queue_id", DEFAULT_QUEUE_ID)
        vdir = self.cfg.cache_path / vid
        self._did_inference = False
        event("item.startup", video_id=vid, queue_id=qid)

        # housekeeping: half-written artifacts from a crash mid-stage
        if vdir.is_dir():
            for name in ('transcript.tmp', 'diarization.tmp', 'merged.tmp',
                         'audio.tmp.wav'):
                (vdir / name).unlink(missing_ok=True)

        self._invalidate_stale_diarization(vdir)
        asr = self._read_artifact(vdir / 'transcript.json', 'segments')
        dia = self._read_artifact(vdir / 'diarization.json', 'embeddings')
        if asr is None:
            (vdir / 'transcript.json').unlink(missing_ok=True)
        if dia is None:
            (vdir / 'diarization.json').unlink(missing_ok=True)
        if asr is None or dia is None:
            (vdir / 'merged.json').unlink(missing_ok=True)
        need_audio = asr is None or dia is None or not (vdir / 'meta.json').exists()
        # Stage 1: acquire ---------------------------------------------------
        self._enter_stage(vid, "acquire", qid)
        wav = vdir / 'audio.wav'
        with span('acquire', cache_hit=not need_audio):
            if need_audio:
                audio = self._take_prefetched(vid)
                if not (wav.exists() and wav.stat().st_size > 44 and (vdir / 'meta.json').exists()):
                    if audio is None:
                        audio = media.download_audio(
                            job["url"], vdir, self.cfg,
                            progress_cb=lambda f, s: self.cb.on_item_progress(vid, "acquire", f, s),
                            cancelled=self.cancel_current.is_set)
                    wav = media.to_wav(audio, self.cfg.resolve_ffmpeg())
            meta = media.load_meta(vdir)
            duration = float(meta.get("duration") or job.get('duration') or 0)
            if not duration and wav.exists():
                duration = media.wav_duration(wav)
        self.store.set_meta(vid, meta.get("title") or job["title"], duration,
                            meta.get("channel") or "", meta.get("upload_date") or "")
        self._checkpoint()

        # start prefetching the next item's audio while the GPU works
        if next_job:
            self._start_prefetch(next_job)

        # Stage 2: transcribe --------------------------------------------------
        self._enter_stage(vid, "transcribe", qid)
        with span('transcribe', cache_hit=asr is not None, audio_duration=duration):
            if asr is None:
                self._did_inference = True
                if self.cfg.model_residency == 'stage' and self._diarizer:
                    self._diarizer.unload()
                if self._transcriber is None:
                    self._transcriber = transcribe.Transcriber(self.cfg, self.cb.on_log)
                asr = transcribe.run_stage(
                    wav, vdir / "transcript.json", self._transcriber, duration,
                    progress_cb=lambda f, s: self.cb.on_item_progress(vid, "transcribe", f, s),
                    cancelled=self.cancel_current.is_set)
        self._checkpoint()

        # Stage 3: diarize ----------------------------------------------------
        self._enter_stage(vid, "diarize", qid)
        with span('diarize', cache_hit=dia is not None, audio_duration=duration):
            if dia is None:
                self._did_inference = True
                if self.cfg.model_residency == 'stage' and self._transcriber:
                    self._transcriber.unload()
                if self._diarizer is None:
                    self._diarizer = diarize.Diarizer(self.cfg, self.cb.on_log)
                dia = diarize.run_stage(
                    wav, vdir / "diarization.json", self._diarizer,
                    progress_cb=lambda f, s: self.cb.on_item_progress(vid, "diarize", f, s))

        # speaker identification: playback samples must be cut and voices
        # matched now, while audio.wav still exists (cache policy may delete it)
        with span('speaker_matching'):
            if wav.exists():
                voices.extract_samples(wav, dia, vdir / "samples")
            if self.cfg.recognition_enabled:
                if self._voice_db is None:
                    self._voice_db = voices.VoiceDB(APP_DIR / "voices.sqlite3")
                voices.auto_match(vdir, dia, self._voice_db,
                                  self.cfg.recognition_threshold, self.cb.on_log)
        self._checkpoint()

        # Stage 4: merge -----------------------------------------------------
        self._enter_stage(vid, "merge", qid)
        with span('merge'):
            merged = merge.run_stage(asr, dia, vdir / "merged.json", self.cfg)

        # Stage 5: export ------------------------------------------------------
        self._enter_stage(vid, "export", qid)
        names = voices.display_names(self.cfg, vdir, merged.get("speakers", []))
        with span('export'):
            files = export.export_video(merged, meta, self.cfg,
                                        self.queue_out_dir(qid), names=names)
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

    def queue_out_dir(self, queue_id: int) -> Path:
        """Output folder for a queue: output_dir/<queue folder> (root when
        the folder is empty — the default 'Main' queue)."""
        queue = self.store.get_queue(queue_id)
        folder = (queue or {}).get("folder") or ""
        return self.cfg.output_path / folder if folder else self.cfg.output_path

    def _enter_stage(self, vid: str, stage: str, queue_id: int | None = None) -> None:
        self._wait_if_paused()
        if self.cancel_current.is_set():
            raise media.CancelledError("cancelled")
        self.store.set_stage(vid, stage, queue_id=queue_id)
        self.cb.on_stage(vid, stage)

    def _checkpoint(self) -> None:
        self._wait_if_paused()
        if self.cancel_current.is_set():
            raise media.CancelledError("cancelled")

    def _wait_if_paused(self) -> None:
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.2)

    # -- prefetch --------------------------------------------------------------

    @staticmethod
    def _read_artifact(path, key):
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            return result if isinstance(result, dict) and key in result else None
        except (OSError, ValueError):
            return None

    def _needs_audio(self, vid):
        vdir = self.cfg.cache_path / vid
        return (not (vdir / 'meta.json').exists()
                or self._read_artifact(vdir / 'transcript.json', 'segments') is None
                or self._read_artifact(vdir / 'diarization.json', 'embeddings') is None)

    def _start_prefetch(self, job: dict) -> None:
        vid = job["video_id"]
        if not self._needs_audio(vid) or self.stop_event.is_set():
            return
        vdir = self.cfg.cache_path / vid
        wav = vdir / 'audio.wav'
        if wav.exists() and wav.stat().st_size > 44 and (vdir / 'meta.json').exists():
            return
        if vid in self._prefetch:
            return
        # A priority change can choose a different successor. Never discard a
        # live thread's ownership: at most one prefetch exists, including stop.
        if any(slot['thread'].is_alive() for slot in self._prefetch.values()):
            return
        slot = {"path": None, "error": None, "thread": None}
        self._prefetch = {vid: slot}  # keep at most one prefetch alive
        recorder = current_recorder()

        def work():
            try:
                with use_recorder(recorder):
                    if not media._find_audio(self.cfg.cache_path / vid):
                        self._sleep_between_downloads(quiet=True)
                    if not self.stop_event.is_set():
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
        if self.cfg.cache_policy == 'keep_forever':
            return
        # Only application-owned names, not arbitrary files beside the cache.
        allowed = {'audio.' + ext for ext in ('m4a','webm','opus','mp4','mp3','ogg','aac','wav')}
        with span('cleanup'):
            for f in vdir.glob('audio.*'):
                if f.name in allowed or f.name.endswith(('.part', '.ytdl', '.tmp.wav')):
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

    def _maybe_finalize_queue(self, queue_id: int) -> None:
        """When a queue runs out of queued jobs: write its combined
        transcript, apply the cache policy, and deactivate it (the next
        queue on the priority stack takes over automatically)."""
        remaining = [j for j in self.store.pending(queue_id)
                     if j["status"] == "queued"]
        if remaining:
            return
        queue = self.store.get_queue(queue_id) or {}
        done = [j for j in self.store.all_jobs(queue_id) if j["status"] == "done"]

        if self.cfg.combined_transcript and len(done) > 1:
            items = []
            for j in done:
                vdir = self.cfg.cache_path / j["video_id"]
                mfile, meta_file = vdir / "merged.json", vdir / "meta.json"
                if mfile.exists() and meta_file.exists():
                    merged = json.loads(mfile.read_text(encoding="utf-8"))
                    names = voices.display_names(self.cfg, vdir,
                                                 merged.get("speakers", []))
                    items.append((merged,
                                  json.loads(meta_file.read_text(encoding="utf-8")),
                                  names))
            if items:
                export.export_combined(items, self.cfg,
                                       self.queue_out_dir(queue_id),
                                       queue.get("name") or "queue")
                self.cb.on_log(f"Combined transcript for queue "
                               f"'{queue.get('name', '?')}' written "
                               f"({len(items)} videos).")

        if self.cfg.cache_policy == "delete_after_queue":
            for j in done:
                self._delete_audio(self.cfg.cache_path / j["video_id"])

        self.store.set_queue_active(queue_id, False)
        self.cb.on_log(f"Queue '{queue.get('name', '?')}' finished.")

    def _finish_run(self, completed: int, failed: int, wall: float) -> dict:
        return {"completed": completed, "failed": failed,
                "wall_seconds": wall, "combined": [],
                "avg_rtf": self._rtf_ema or 0}

    def unload_models(self) -> None:
        if self._transcriber:
            self._transcriber.unload()
        if self._diarizer:
            self._diarizer.unload()

    def close(self):
        """Drain owned threads before releasing models or closing the app."""
        self.stop_event.set()
        for slot in self._prefetch.values():
            slot['thread'].join()
        self._prefetch.clear()
        self.unload_models()
        if self._voice_db:
            self._voice_db.close()
            self._voice_db = None
