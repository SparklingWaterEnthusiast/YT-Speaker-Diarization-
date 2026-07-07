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

- **Channel detection**: `@givemeananswer` resolved to 866 videos (Videos
  tab; Shorts/live streams are separate tabs and can be queued by their tab
  URLs). Titles/durations extracted flat with no downloads.
- **Playlist**: channel's "Studio" playlist resolved to 75 videos.
- **Invalid URL / dead video**: raises a clean error, surfaced in UI/CLI.
- **Corrupted audio**: garbage `audio.m4a` → clear ffmpeg error, no temp
  files left behind, job goes to retry queue.
- **Interruption/resume**: killed the process mid-transcription of a 2-video
  queue. Restart recovered the job (`running`→`queued`), the acquire stage was
  a cache hit (no re-download), transcription redid its incomplete stage, both
  videos completed, combined transcript written. **Pass.**
- **Cache policy**: `delete_after_video` verified — audio gone after
  completion, JSON artifacts retained; re-export from artifacts alone works
  (used it to regenerate benchmark files after the filename fix).

### Bugs found & fixed during testing

3. `run.ps1`/`setup.ps1` used em-dashes; PowerShell 5.1 reads BOM-less .ps1
   as ANSI → parse error. Scripts are now pure ASCII.
4. CLI output was block-buffered when piped; progress now flushes live.
5. **Filename bug**: `Path.with_suffix()` ate everything after the last dot
   in dotted titles ("…the Same Thing. Jesus…"), dropping the `[video_id]`
   suffix and inviting collisions. Fixed with plain concatenation +
   regression test.
6. ffmpeg failure messages showed the build banner instead of the error;
   now the last stderr lines.

## 6. Benchmark (fZZXVNt1gk0, 28.5 min, street Q&A, 6 speakers)

- Processed end-to-end in **5.4 min ≈ 5.3× realtime** (including one-time
  model loading; steady-state is faster). Peak VRAM ~3.8 GB of 8 GB.
- Diarization: 6 speakers, 104 turns.
- Sanity check against the provided professional sample (19:36–27:52
  window): turn structure matches nearly 1:1 —
  SPEAKER_02=Stuart Knechtle, SPEAKER_03=Cliffe Knechtle,
  SPEAKER_01/04=audience questioners; boundaries within ~1–2 s of the
  reference; interjections ("But then why give original sin…", "Does that
  make any sense? … Yes, sir.") attributed to the correct speakers.
- Known imperfections observed: (a) occasional 1-word turns at rapid
  handoffs assigned to the interlocutor (e.g. a stray "I" at 23:19);
  (b) some rapid Q&A passages come out of Whisper lowercase/unpunctuated;
  (c) turn boundaries can shift a word relative to the reference.
  Formal WER/DER comparison awaits the full reference transcript.

## 7. v0.1 final state

All success criteria verified this session: fresh-machine setup path
(`setup.ps1` — winget + venv, Smart App Control safe), GUI launches (Qt
offscreen smoke test + live construction), URL → queue → five-stage pipeline
→ five export formats + combined transcript, 25/25 unit tests passing,
benchmark video processed successfully.

---

# Version 0.2 — speaker identification & output simplification (July 7, 2026)

## 8. Research

Key finding (verified by inspecting the installed pyannote 4.0.7 source):
the community-1 pipeline's `DiarizeOutput` **always includes
`speaker_embeddings`** — one 256-dim WeSpeaker ResNet34 centroid per
detected speaker, row-aligned with `SPEAKER_XX` labels (zero-padded rows for
speakers without clean frames). Cross-video recognition therefore reuses
what diarization already computes: no new models, no extra GPU passes.
Alternatives (speechbrain ECAPA, standalone wespeaker, Resemblyzer, NeMo
TitaNet) all rejected — see DESIGN.md §5.1.

## 9. Implemented

- **Markdown-only default** (`export_formats: ["md"]`) with config-version
  migration: an untouched v0.1 default (all five formats) migrates to
  `["md"]`; a deliberately customized subset is preserved.
- **voices.py**: `VoiceDB` (voices.sqlite3 — speakers + confirmed embedding
  samples, per-source dedup, created/updated timestamps), cosine matching
  (L2-normalized, best-sample-per-profile, dimension-mismatch guard),
  per-video `speakers.json` name map, ≤5 s playback sample extraction
  (stdlib `wave`, longest exclusive turn, cut while audio still exists),
  `apply_rename` (map update → DB update → re-export of configured + already
  present formats), reference-transcript parsing and overlap alignment.
- **Pipeline**: after diarization — samples extracted, embeddings
  auto-matched against the DB (manual entries never overwritten; unmatched
  speakers keep `SPEAKER_XX`); stale pre-v0.2 diarization artifacts (no
  embeddings) recomputed together with their merged artifact.
- **UI**: clicking a completed video opens a dropdown per speaker with
  current label/name (+ auto-match score), ▶ sample playback (winsound —
  no new dependencies), and a rename field (Enter applies). Settings gained
  recognition toggle + threshold.
- **CLI**: `--seed <url> --reference <file>` and `--profiles`.
- **Housekeeping fixes**: leftover `*.tmp*` artifacts from crashes are
  removed at job start; `delete_after_queue` now cleans audio of *all*
  finished jobs, not only the current session's.

## 10. v0.2 testing

- 44/44 unit tests (19 new: profile creation/persistence across reopen,
  source dedup, matching thresholds/guards, auto-match semantics including
  never-overwrite-manual and stale-auto cleanup, name resolution order,
  sample extraction cap, full rename flow against real artifact files,
  reference parsing/alignment, md-only default, optional formats,
  re-export of existing formats, per-video name precedence).
- GUI offscreen smoke test including speaker-menu code paths.
- **Housekeeping audit** of live data: completed videos hold only JSON
  artifacts + samples (~1.5 MB per 30-min video, kept deliberately for
  GPU-free renames/re-exports); no stray `.tmp`/`.part` files; audio
  correctly deleted by `delete_after_video`. Found one undocumented cache
  dir (ZkvsKPdXRaA) — a video the user processed with v0.1 between
  sessions; artifacts consistent, audio correctly cleaned up.
- **Seeding**: reference transcript (144 turns) aligned to the reprocessed
  benchmark video at 98 % purity for both hosts (Cliffe 747 s overlap,
  Stuart 207 s); profiles created; benchmark transcripts re-exported with
  real names.

(§11: recognition benchmark results)
