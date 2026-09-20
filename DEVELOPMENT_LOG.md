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

## 11. Recognition benchmark (against seeded profiles)

Reprocessed three videos never seen by the profiles (one from 2016, ten
years before the seed video):

| Video | Speaker | vs Cliffe | vs Stuart | Result |
|---|---|---|---|---|
| #1016 (2016, ZkvsKPdXRaA) | host | **0.909** | 0.480 | ✔ named Cliffe Knechtle |
| #1016 (2016) | questioner | 0.199 | 0.417 | ✔ left as SPEAKER_00 |
| Everyone Worships (Mu_DIW7bX70) | host | 0.457 | **0.895** | ✔ named Stuart Knechtle |
| Pluralism (1e-iolWove4) | host | 0.580 | **0.900** | ✔ named Stuart Knechtle |

**3/3 correct recognitions, 0 false positives**, names verified in the
exported Markdown. Calibration insight: the hardest impostor pair is Stuart
vs his father Cliffe's profile (0.58 — related voices, same acoustics);
genuine matches score ~0.90; unrelated speakers ≤0.48. Default threshold
raised 0.6 → **0.7** to sit mid-gap. Stuart's profile achieved 0.90 matches
from just 207 s of seeded speech.

Additional verifications: manual rename on a pre-v0.2 video without
embeddings degrades gracefully (transcripts renamed, DB untouched, note
logged); playback samples present for every v0.2-processed video, play
button disabled where none exist; `run_single` applies the cache policy
(0 audio files left after the benchmark runs).

## 12. v0.2 final state

All v0.2 objectives verified: Markdown-only default with opt-in formats,
researched and documented recognition design, persistent voice DB
(name/embeddings/dates/sample counts), automatic post-diarization matching
that never invents names, click-to-edit speaker dropdown with ≤5 s samples
and instant transcript rewrite, Cliffe & Stuart profiles seeded from the
professional reference at 98 % alignment purity, housekeeping audited with
two real fixes (crash-orphaned .tmp files; delete_after_queue across
sessions). 44/44 tests passing.

---

# Version 0.2.1 — queue tabs & file-manager UX (September 2, 2026)

## 13. Implemented

Quality-of-life release modeled on File Explorer conventions; the
processing pipeline is untouched.

- **Queue tabs**: `queues` table added (schema v2, automatic in-place
  migration that rebuilds the jobs table — the constraint changed from
  UNIQUE(video_id) to UNIQUE(video_id, queue_id) so one video can sit in
  several tabs; its second run is nearly free via the artifact cache).
  Each tab owns an output subfolder (default name = creation date/time;
  the pre-existing "Main" queue keeps writing to the output root).
  Double-click renames tab + folder together; tabs are draggable
  (order persisted); right-click: rename / open folder / stop / delete.
- **Priority-stack scheduling**: active queues are processed most recently
  started first. Starting a small tab mid-bulk-run preempts at the next
  video boundary; the bulk queue resumes automatically when the tab
  drains. Queues deactivate themselves and write their own combined
  transcript (into their subfolder) when they finish.
- **Queue table as file manager**: extended multi-select (Ctrl/Shift,
  Ctrl+A), right-click context menu — edit speakers, open transcript,
  open output folder, retry failed *and cancelled* (the gap noted by the
  user), reprocess completed, move up/down/top/bottom, remove selection
  (Del key too; removal never touches transcripts or cache). Speaker
  editor moved from single-click to double-click / context menu so plain
  clicks select rows.
- Renames now re-export into **every** queue folder containing the video.

## 14. v0.2.1 testing

- 57/57 unit tests (13 new): queue CRUD + cascade, same-video-in-two-
  queues, per-queue status isolation, tab-order persistence, priority
  ordering (newest queue first, fallback after drain), inactive queues
  skipped, manual reorder respected by the scheduler, multi-select move
  semantics, remove-skips-running, requeue of cancelled, and a real
  v1-database migration test.
- GUI offscreen test: real production DB (868 jobs) migrated in place;
  tab create/switch/add/select-all/delete/tab-delete all exercised.
- Live end-to-end: a "Priority Test" tab processed a video into its own
  subfolder while the 866-video Main queue (inactive) was untouched;
  cached stages reused (no model load for transcription); the queue
  deactivated itself on completion; a previously renamed speaker's name
  carried into the new folder automatically.

---

# Version 0.2.2 — YouTube 403 fix & download diagnostics (September 2, 2026)

## 15. Incident

User reported repeated `HTTP Error 403: Forbidden` ("unable to download
video data") during the channel run; metadata resolved fine, only media
downloads failed.

**Diagnosis (reproduced locally on the reported video IDs):** the installed
yt-dlp was 2026.07.04 while 2026.08.19 was current. The verbose log showed
yt-dlp falling back to the `android_vr` client and requesting format 140 —
the exact combination YouTube began rejecting (yt-dlp issues #14680,
#17456). Nothing about YTScribe's own code was at fault: rate limiting,
cookies and PO tokens were red herrings.

**Fix:** `pip install -U yt-dlp` (→ 2026.08.19). Both failing videos then
downloaded successfully (24.6 MiB and 26.1 MiB). A second warning surfaced
after the update — "No supported JavaScript runtime could be found;
YouTube extraction without a JS runtime has been deprecated" — so Deno
2.9.6 was installed via winget; the warning cleared and yt-dlp now uses
the `visionos` client cleanly.

## 16. Hardening (so this self-diagnoses next time)

- `media._explain()` rewrites known failure modes into actionable errors:
  403 → "run update-deps.ps1"; bot-check → "set Cookies from browser";
  unavailable video → stated plainly. The error text is what lands in the
  UI log and the job's Error column.
- `media.environment_report()` runs at startup (GUI log + CLI stdout):
  warns when yt-dlp is >60 days old (CalVer parsed from the version) or no
  JS runtime (deno/node/bun/qjs) is on PATH.
- `update-deps.ps1` added: updates yt-dlp, installs Deno if absent.
- INSTALL.md gained a "Routine maintenance" section and two troubleshooting
  rows; this class of breakage is expected maintenance, not a defect.
- 5 new unit tests (62 total, all passing).

---

# Version 0.3.0 — Optimization Overhaul (September 19, 2026)

## 17. Evidence-led changes

Started from clean v0.2.2 commit `d66665c`; preserved its package in an isolated
benchmark directory before editing. No production queues, profiles, outputs or
audio were processed or deleted. Only the requested benchmark source was acquired.
Research compared upstream CT2/faster-whisper modes and hardware capabilities;
the existing acquisition/ASR/diarization/merge/export architecture remains intact.

- P0: process-owned leases cover both cache and job database directories;
  asynchronous Qt shutdown drains workers/prefetch and requeues the interrupted
  stage; startup recovery cannot reset another worker's running jobs.
- P0: reproduced a fresh-process diarization-first memory pathology. PyTorch
  reserved 11.375 GB on the physical 8 GB GPU. Added a temporary physical-memory
  allocator budget and release of unused torch scratch memory before CT2 work.
  No driver policy changes. The exact workspace algorithm remains unverified.
- P0: bounded OOM retries halve batches without preserving partial inference;
  non-memory errors propagate. A failed model transfer cannot leave a silently
  CPU-resident pipeline marked ready. Invalid processing settings fail early.
- P1: optional batched ASR reuses weights but uses a new decoder per job/retry.
  Sequential large-v3/beam 5/word timestamps remain defaults. Cached complete
  artifacts can re-export without audio, FFmpeg or model construction.
- P1: prefetch owns at most one thread across priority changes, reuses cached
  WAVs and drains at shutdown. Cache-only exports do not inflate throughput/ETA.
- P2/P3: explicit asynchronous capability probe; Safe/Balanced/Performance
  presets; advanced controls, temperature/clock/power and sustained thermal
  warnings; clearer retention labels and cache-folder actions.
- P3: opt-in JSONL stage/resource traces; high-resolution monotonic host timing;
  separate PyTorch and NVML counters, process/thread identity, cache/cold flags,
  real-time ratios and explicit missing values. Sampling does not initialize CUDA.
- Housekeeping: delete only recognized application-owned audio and unfinished
  artifacts; retain stage JSON, short voice samples and user-named adjacent files.

## 18. Actual verification

The original 62-test suite passed before changes; the expanded suite has **143
passing tests**. Includes settings migration/round trips,
Qt shutdown/startup ownership, bounded retry behavior, non-OOM propagation,
cache policies, cache-only export, immutable run configuration, telemetry and
thermal sampling. `compileall` and `git diff --check` passed. Real Qt settings
were rendered and inspected; opening the dialog did not import torch.

Full-source initial ASR: 230.208 s cold-inference / 210.785 s warm; optional
batch 4: 83.750 / 90.448 s. Final memory-safe batch 4: 67.580 / 63.928 s.
Different laptop thermal/power observations prohibit attributing every later
gain to code. Cold model initialization is separately measured. Thirteen
180-second configuration experiments each ran three repetitions. A later
frozen-source recheck and final evidence are recorded in the linked report.

Reference evaluation: 315/4794 normalized word errors originally versus
306/4794 with batch 4; this imperfect single-video reference is not a general
accuracy guarantee. Raw/exclusive speaker turns were identical and stored voice
profiles remained compatible. Real cached exports generated Markdown only by
default, then all optional formats; manual renaming updated each format without
audio or model loading. Profile persistence was tested in an isolated database.

Real Qt restart PASS: process 8516 transcribed/paused/closed; process 18760
reused its cached ASR and completed three 300/480/600-second excerpts in
105.717 s. All jobs done, cached hash unchanged, workers finished. The initial
memory-pathology diagnostic was intentionally terminated, not counted as a pass.
Fixed first-item diarization: 103.344 → 20.922 s; identical speaker boundaries.
Driver VRAM returned to ~740 MiB desktop/background occupancy after final exit.

Final checks: cold-model full-source Qt pipeline completed in 136.271 s from
cached WAV through export/worker shutdown. A real GPU test restricted the torch
budget to 1173 MiB: batch 32 raised OOM, batch 16 succeeded with identical speaker
turns. The frozen-source recheck measured ASR 200.482/205.401 s and diarization
44.961/44.124 s. Final warm ASR therefore took 68.88% less time than that recheck.
Live acquisition retest passed (14.566 s including resolution/download; FFmpeg
resolution/conversion 1.740 s), with transfer/preparation telemetry populated.

A final test-process audit caught a Windows Qt teardown abort **after** unittest
printed OK. It was traced to test-owned hidden widgets surviving QApplication;
explicit module teardown destroys widgets while Qt remains alive. Both focused
Qt tests and the full suite then exited with code 0; repeated full runs confirmed
clean exits. An OK line alone was not accepted as a passing process result.

Full methodology, raw evidence locations, hardware tiers and limitations:
[v0.3.0 report](benchmark/V0.3.0_REPORT.md). Other GPUs, a fresh-machine install,
and a full 866-video/multi-day queue were not tested during this engagement.

## 19. Final handoff and agreed verification scope

The user clarified that the 866 videos are the production workload, not a test
suite, and no GPU other than the RTX 3080 Laptop is available or required.
The implementation and bounded local verification are complete. No further
benchmark runs are scheduled as release gates. Final review corrected ownership
lease wording and distinguished early batch-memory results from the final
memory-safe measurements; these were documentation-only edits.

Launch the updated checkout using `run.ps1`. Existing saved settings are
preserved; Performance is explicitly opt-in through Settings → Optimization →
Apply preset → OK and takes effect on the next run. Sequential large-v3 remains
the default. Changes are local and uncommitted; no new standalone installer or
fresh-machine certification is claimed. The measured report and user guide are
the handoff; production queues, voice profiles and output files were untouched.
