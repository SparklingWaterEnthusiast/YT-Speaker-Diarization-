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
        self.log(f"Loading Whisper '{self.cfg.whisper_model}' "
                 f"({self.device}, {self.compute_type})...")
        try:
            self._model = WhisperModel(self.cfg.whisper_model, device=self.device,
                                       compute_type=self.compute_type)
        except (RuntimeError, ValueError) as exc:
            if self.device == "cuda":
                self.log(f"WARNING: CUDA init failed ({exc}); falling back to CPU. "
                         "Processing will be much slower.")
                self.device, self.compute_type = "cpu", "int8"
                self._model = WhisperModel(self.cfg.whisper_model, device="cpu",
                                           compute_type="int8")
            else:
                raise
        self.log("Whisper model ready.")

    def transcribe(self, wav: Path, duration: float,
                   progress_cb: Callable[[float, str], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> dict:
        self._ensure_model()
        language = None if self.cfg.language == "auto" else self.cfg.language
        segments_iter, info = self._model.transcribe(
            str(wav),
            language=language,
            beam_size=self.cfg.beam_size,
            word_timestamps=True,
            vad_filter=self.cfg.vad_filter,
            # avoids repetition/hallucination loops on long recordings
            condition_on_previous_text=False,
        )
        segments = []
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
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments,
        }

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None


def run_stage(wav: Path, out_file: Path, transcriber: Transcriber, duration: float,
              progress_cb=None, cancelled=None) -> dict:
    """Cached stage wrapper: skip work if the artifact already exists."""
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    result = transcriber.transcribe(wav, duration, progress_cb, cancelled)
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_file)
    return result
