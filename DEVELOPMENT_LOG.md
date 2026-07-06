# DEVELOPMENT_LOG.md

Chronological record of what was built, tested, and verified. July 7, 2026.

## 1. Environment reconnaissance

- Verified target GPU: RTX 3080 Laptop, 8 GB VRAM, driver 581.80 (nvidia-smi).
- Found no Python/FFmpeg on the machine; **Windows Smart App Control = On**,
  which blocks unsigned executables → ruled out frozen-exe packaging, chose
  winget (signed) installers + venv distribution.
- ~440 GB free disk.

## 2. Research (see DESIGN.md for full rationale)

Key findings that shaped the stack:

- `pyannote-community/speaker-diarization-community-1` mirror is **ungated**
  (verified via HF API `gated: false`) → best open diarization without
  requiring the user to create a HuggingFace account.
- WhisperX (initial favorite) is currently uninstallable from PyPI: it pins
  `pyannote-audio 3.3.2` whose `lightning` dependency was quarantined on PyPI
  (April 2026), and conflicts with pyannote 4.x → replaced by
  faster-whisper's native word timestamps + pyannote 4's exclusive
  diarization output.
- faster-whisper 1.2.1 (Oct 2025) with Silero-VAD v6 is current and healthy.
- yt-dlp 2026.07.04 current; PO-token pressure from YouTube exists for
  flagged IPs → courtesy delays + cookies + optional bgutil plugin documented.

## 3. Environment setup

- winget: Python 3.12.10, FFmpeg 8.1.2 (Gyan full build).
- venv with torch 2.12.1+cu126, ctranslate2 4.8.1, faster-whisper 1.2.1,
  pyannote.audio 4.0.7, yt-dlp 2026.07.04, PySide6, psutil, nvidia-ml-py.
- Verified `torch.cuda.is_available() == True` and
  `ctranslate2.get_cuda_device_count() == 1`.

### Issues found & fixed during setup

1. **torchcodec cannot load FFmpeg DLLs** (pyannote 4's default audio
   decoder needs FFmpeg *shared* DLLs; the winget build is static).
   Fix: bypass decoding entirely — the app feeds pyannote pre-loaded
   in-memory waveforms (`{"waveform": tensor, "sample_rate": 16000}`) read
   from its own 16 kHz WAV with the stdlib `wave` module. (diarize.py)
2. **CTranslate2 needs cuBLAS/cuDNN DLLs** which ship inside torch's wheel;
   registered `torch/lib` via `os.add_dll_directory` before model load.
   (cuda_setup.py)

## 4. Implementation

Modules as designed (see DESIGN.md §3): config, db (SQLite job store), media
(yt-dlp + ffmpeg), transcribe (faster-whisper), diarize (pyannote 4),
merge (word-level speaker assignment), export (5 formats + combined),
pipeline (queue orchestration, prefetch, pause/cancel/stop, cache policies),
resources (NVML/psutil sampling), PySide6 UI (queue table, progress, resource
strip, log, settings dialog), CLI mode (`--cli`) sharing the identical
pipeline for headless/automated runs.

## 5. Testing

- **Unit tests** (tests/test_units.py, 22 tests): URL classification, merge
  logic (word-level speaker switching inside one ASR segment, gap snapping,
  hallucination removal only-in-silence, no-word-timing fallback, empty
  diarization), export writers (all 5 formats, filename sanitization,
  speaker renaming, JSON round-trip, SRT timestamps, combined output),
  job store (dedup, lifecycle, retry, crash recovery), config validation.
  **All pass.**
- **GUI smoke test** (offscreen QPA): MainWindow + SettingsDialog construct,
  queue refresh and resource sampling run. **Pass.**
- **End-to-end smoke test** (19 s video, `--cli`): first run failed with
  `ffmpeg not found` — the failure was handled exactly as designed (job →
  failed, queue continued, nonzero exit). Root cause: stale PATH in spawned
  process + winget shim dir not containing ffmpeg.exe. Fixed ffmpeg
  resolution to scan winget package dirs; `--retry-failed` then reused the
  already-downloaded audio (cache hit) and completed. Verified retry-queue +
  cache-reuse behavior with a real failure, unplanned but valuable.

(continued in §6 with benchmark results)
