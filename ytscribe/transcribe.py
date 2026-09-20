"""Transcription stage: faster-whisper (CTranslate2) with word-level timestamps.

The model is loaded once and reused across the whole queue; loading large-v3
takes ~30 s and would dominate runtime if repeated per video.

Output artifact (transcript.json):
    {"language": "en", "language_probability": 0.99,
     "segments": [{"start", "end", "text", "no_speech_prob",
                   "words": [{"start", "end", "word", "probability"}]}]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .cuda_setup import pick_compute_type, pick_device, register_cuda_dlls
from .inference import guarded_batch, is_oom, release_unused
from .telemetry import span, event


class Transcriber:
    def __init__(self, cfg, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.log = log
        self._model = None
        self.device = pick_device(cfg.device)
        self.compute_type = pick_compute_type(cfg.compute_type, self.device)

    def _ensure_model(self):
        if self._model is not None:
            return
        register_cuda_dlls()
        from faster_whisper import WhisperModel
        # A previous diarization stage may own a large *unused* torch pool.
        # CTranslate2 cannot reuse that pool, especially after a diarization-first
        # restart. Retain model tensors but release cached scratch allocations.
        release_unused()
        self.log(f"Loading Whisper '{self.cfg.whisper_model}' "
                 f"({self.device}, {self.compute_type})...")
        with span("asr.model_load", cold=True, model=self.cfg.whisper_model,
                  device=self.device, compute_type=self.compute_type):
            self._model = WhisperModel(self.cfg.whisper_model, device=self.device,
                                       compute_type=self.compute_type,
                                       cpu_threads=self.cfg.asr_cpu_threads,
                                       num_workers=self.cfg.asr_num_workers)
        self.log("Whisper model ready.")

    def transcribe(self, wav: Path, duration: float,
                   progress_cb: Callable[[float, str], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> dict:
        if self._model is not None:
            release_unused()
        self._ensure_model()
        batch = guarded_batch(self.cfg.asr_batch_size, self.cfg.vram_margin_mb,
                              self.device, self.log)
        # Keep batched decoding semantics when backing off to batch 1.
        batched = self.cfg.asr_batch_size > 1
        for attempt in range(self.cfg.oom_retries + 1):
            try:
                return self._transcribe_once(wav, duration, batch, batched,
                                             progress_cb, cancelled)
            except RuntimeError as exc:
                if not is_oom(exc) or batch <= 1 or attempt >= self.cfg.oom_retries:
                    raise
                self.log(f"ASR out of memory at batch {batch}; retrying at {max(1, batch // 2)}.")
                event("asr.oom", batch=batch, attempt=attempt)
            # Outside except: failed generator frames and their tensors can die.
            release_unused()
            batch = max(1, batch // 2)

    def _transcribe_once(self, wav, duration, batch, batched, progress_cb, cancelled):
        from faster_whisper.audio import decode_audio
        from faster_whisper import BatchedInferencePipeline
        with span("asr.audio_load", audio_duration=duration):
            audio = decode_audio(str(wav), sampling_rate=16000)
        language = None if self.cfg.language == "auto" else self.cfg.language
        options = dict(
            language=language,
            beam_size=self.cfg.beam_size,
            word_timestamps=self.cfg.word_timestamps,
            vad_filter=self.cfg.vad_filter,
            condition_on_previous_text=False,
        )
        decoder = self._model
        if batched:
            decoder = BatchedInferencePipeline(self._model)
            options.update(batch_size=batch, chunk_length=self.cfg.asr_chunk_length,
                           vad_parameters={"min_silence_duration_ms": 2000})
            if not self.cfg.vad_filter:
                options['clip_timestamps'] = [
                    {'start': s, 'end': min(s + self.cfg.asr_chunk_length, len(audio)/16000)}
                    for s in range(0, int(len(audio)/16000) + 1, self.cfg.asr_chunk_length)
                    if s < len(audio)/16000]
        # faster-whisper performs VAD, feature extraction and (when auto)
        # language detection here; deferred generation is timed separately.
        with span("asr.preprocessing", audio_duration=duration, batch=batch):
            segments_iter, info = decoder.transcribe(audio, **options)
        segments = []
        with span("asr.inference", audio_duration=duration, batch=batch):
            try:
                for seg in segments_iter:
                    if cancelled and cancelled():
                        raise InterruptedError("transcription cancelled")
                    words = [{"start": w.start, "end": w.end, "word": w.word,
                              "probability": w.probability} for w in (seg.words or [])]
                    segments.append({
                        "start": seg.start, "end": seg.end, "text": seg.text,
                        "no_speech_prob": seg.no_speech_prob, "words": words,
                    })
                    if progress_cb and duration > 0:
                        progress_cb(min(seg.end / duration, 1.0),
                                    f"{seg.end / 60:.1f}/{duration / 60:.1f} min")
            finally:
                segments_iter.close()
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments,
            "inference": {"model": self.cfg.whisper_model, "compute_type": self.compute_type,
                          "batch_size": batch, "batched": batched,
                          "beam_size": self.cfg.beam_size},
        }

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            release_unused()
            event("asr.unloaded")


def run_stage(wav: Path, out_file: Path, transcriber: Transcriber, duration: float,
              progress_cb=None, cancelled=None) -> dict:
    """Cached stage wrapper: skip work if the artifact already exists."""
    if out_file.exists():
        event("asr.cache_hit", path=str(out_file))
        return json.loads(out_file.read_text(encoding="utf-8"))
    result = transcriber.transcribe(wav, duration, progress_cb, cancelled)
    tmp = out_file.with_suffix(".tmp")
    with span("asr.serialization"):
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_file)
    return result
