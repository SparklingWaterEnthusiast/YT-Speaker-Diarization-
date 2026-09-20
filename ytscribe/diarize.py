"""Speaker diarization stage: pyannote.audio 4.x community-1 pipeline.

Uses the ungated pyannote-community mirror by default (no HuggingFace token
required); a token is honored if configured, which also unlocks the gated
official repos.

Output artifact (diarization.json):
    {"speakers": ["SPEAKER_00", ...],
     "turns":     [{"start", "end", "speaker"}],       # raw, may overlap
     "exclusive": [{"start", "end", "speaker"}]}       # non-overlapping timeline

The exclusive timeline is pyannote 4's reconciliation output, designed for
merging with ASR timestamps; the merge stage consumes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .cuda_setup import pick_device
from .inference import guarded_batch, is_oom, release_unused, torch_budget
from .telemetry import span, event


class Diarizer:
    def __init__(self, cfg, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.log = log
        self._pipeline = None
        self.device = pick_device(cfg.device)

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        import torch
        from pyannote.audio import Pipeline
        self.log(f"Loading diarization pipeline '{self.cfg.diarization_model}'...")
        token = self.cfg.hf_token or None
        with span("diarization.model_load", cold=True, model=self.cfg.diarization_model):
            pipeline = Pipeline.from_pretrained(self.cfg.diarization_model, token=token)
            if self.device == "cuda":
                pipeline.to(torch.device("cuda"))
            self._pipeline = pipeline
        self.log("Diarization pipeline ready.")

    def diarize(self, wav: Path,
                progress_cb: Callable[[float, str], None] | None = None) -> dict:
        self._ensure_pipeline()
        kwargs = {}
        if self.cfg.min_speakers:
            kwargs["min_speakers"] = self.cfg.min_speakers
        if self.cfg.max_speakers:
            kwargs["max_speakers"] = self.cfg.max_speakers

        hook = _ProgressHook(progress_cb) if progress_cb else None
        # Audio is preloaded in memory: pyannote's built-in decoder (torchcodec)
        # needs FFmpeg *shared* DLLs, which static Windows FFmpeg builds lack.
        with span("diarization.audio_load"):
            audio = _load_wav(wav)
        batch = guarded_batch(self.cfg.diarization_batch_size, self.cfg.vram_margin_mb,
                              self.device, self.log)
        with torch_budget(self.cfg.vram_margin_mb, self.device, self.log):
            for attempt in range(self.cfg.oom_retries + 1):
                self._pipeline.segmentation_batch_size = batch
                self._pipeline.embedding_batch_size = batch
                try:
                    with span("diarization.inference", batch=batch,
                              audio_duration=audio['waveform'].shape[1]/audio['sample_rate']):
                        # Each attempt gets a fresh mapping, not failed cached tensors.
                        output = self._pipeline(dict(audio), hook=hook, **kwargs)
                    break
                except RuntimeError as exc:
                    if not is_oom(exc) or batch <= 1 or attempt >= self.cfg.oom_retries:
                        raise
                    self.log(f"Diarization out of memory at batch {batch}; retrying at {max(1, batch // 2)}.")
                    event("diarization.oom", batch=batch, attempt=attempt)
                release_unused()
                batch = max(1, batch // 2)

        # pyannote 4 returns an object with .speaker_diarization (Annotation)
        # and .exclusive_speaker_diarization; older versions return Annotation.
        raw_ann = getattr(output, "speaker_diarization", output)
        excl_ann = getattr(output, "exclusive_speaker_diarization", raw_ann)

        def to_turns(annotation):
            return [{"start": round(turn.start, 3), "end": round(turn.end, 3),
                     "speaker": label}
                    for turn, _, label in annotation.itertracks(yield_label=True)]

        raw = to_turns(raw_ann)
        exclusive = to_turns(excl_ann)
        speakers = sorted({t["speaker"] for t in raw})
        embeddings = self._embeddings_by_label(output, raw_ann, speakers)
        return {"speakers": speakers, "turns": raw, "exclusive": exclusive,
                "embeddings": embeddings}

    @staticmethod
    def _embeddings_by_label(output, raw_ann, speakers: list[str]) -> dict:
        """Map speaker label -> 256-d centroid list (or None).

        pyannote 4 orders centroid rows to match labels(); rows may be
        zero-padded when a speaker has no clean frames — stored as None.
        """
        centroids = getattr(output, "speaker_embeddings", None)
        result: dict[str, list | None] = {s: None for s in speakers}
        if centroids is None:
            return result
        labels = list(raw_ann.labels())
        for i, label in enumerate(labels):
            if i >= len(centroids):
                break
            row = centroids[i]
            norm = float((row ** 2).sum()) ** 0.5
            if norm > 0 and norm == norm:  # nonzero and not NaN
                result[label] = [round(float(x), 6) for x in row]
        return result

    def unload(self):
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            release_unused()
            event("diarization.unloaded")


def _load_wav(path: Path) -> dict:
    """Read a 16-bit PCM WAV into pyannote's in-memory audio format."""
    import wave

    import numpy as np
    import torch

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM WAV, got {w.getsampwidth() * 8}-bit")
        rate = w.getframerate()
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(data.reshape(-1, channels).T.copy())
    return {"waveform": waveform, "sample_rate": rate}


class _ProgressHook:
    """Adapts pyannote's hook protocol to a (fraction, label) callback."""

    def __init__(self, cb):
        self.cb = cb

    def __call__(self, step_name, step_artifact, file=None,
                 total=None, completed=None):
        if total and completed is not None:
            self.cb(completed / total, step_name)

    # pyannote may use the hook as a context manager
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_stage(wav: Path, out_file: Path, diarizer: Diarizer,
              progress_cb=None) -> dict:
    """Cached stage wrapper: skip work if the artifact already exists."""
    if out_file.exists():
        cached = json.loads(out_file.read_text(encoding="utf-8"))
        # pre-v0.2 artifacts lack embeddings -> recompute for recognition
        if "embeddings" in cached:
            event("diarization.cache_hit", path=str(out_file))
            return cached
    result = diarizer.diarize(wav, progress_cb)
    tmp = out_file.with_suffix(".tmp")
    with span("diarization.serialization"):
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_file)
    return result
