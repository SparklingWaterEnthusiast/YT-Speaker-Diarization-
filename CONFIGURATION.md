# CONFIGURATION.md — v0.3.0

All settings live in `%USERPROFILE%\Documents\YTScribe\config.json`, created
with defaults on first launch. The Settings dialog edits the common ones; the
file is the complete reference. A running queue uses its startup snapshot;
changes apply to the next run. Invalid values are reported at save/start.
Schema `config_version=3` adds optimization fields without applying a preset
to existing inference settings; the earlier Markdown-default migration remains.

## Paths

| Key | Default | Meaning |
|---|---|---|
| `output_dir` | `Documents\YTScribe\transcripts` | Root for transcript files. Since v0.2.1 every queue tab writes into its own subfolder here (named after the tab); the original "Main" tab writes to the root |
| `cache_dir` | `Documents\YTScribe\cache` | Per-video working data (`cache/<video_id>/`) |
| `ffmpeg_path` | `""` | Explicit ffmpeg.exe path; empty = auto-detect (PATH, then winget dirs) |

## Cache policy — `cache_policy`

Applies to the **audio files** only; the small JSON stage artifacts
(transcript, diarization, merged) and short speaker samples are kept so
transcripts can be re-exported without re-downloading or re-transcribing.
Both the downloaded original and converted `audio.wav` follow this policy.

| Value | Behavior |
|---|---|
| `delete_after_video` *(default)* | Application-owned audio removed after the video's successful export |
| `delete_after_queue` | Audio for completed videos removed when their queue has no queued work left |
| `keep_forever` | Retains downloaded/processed audio in the existing cache until manually deleted |

Failed/cancelled items may retain audio for retry. Pause/close does not apply a
blanket audio deletion. Cleanup targets the app's `audio.*` names and download
temporary files, not arbitrary neighboring files. The UI exposes retention in
Settings → Downloads and folder access through **Open Audio / Cache**.

Caches are shared by video ID across queue tabs. Changing models, precision,
batch size or other settings does **not** invalidate existing stage JSON;
**Reprocess** requeues the item and also reuses caches. For a fresh comparison,
use an isolated benchmark cache/output. Missing/unreadable required artifacts
are recomputed; older diarization without embeddings also invalidates merge.

## Transcription

| Key | Default | Notes |
|---|---|---|
| `whisper_model` | `large-v3` | Accuracy baseline. Alternatives include `large-v3-turbo`, English-focused `distil-large-v3`, `medium`, `small`, or a compatible CTranslate2 model ID; quality/speed depend on workload |
| `compute_type` | `auto` | CPU: `int8`. CUDA: first reported supported type among `int8_float16`, `int8_float32`, `float32`, otherwise CT2 `default`. Explicit unsupported types fail; FP16 has no universal 10 GB requirement |
| `beam_size` | `5` | Beam 1 uses greedy search; changing beam can change words, speed and memory |
| `language` | `auto` | ISO code (`en`) skips per-video language detection |
| `vad_filter` | `true` | Filters non-speech; batched mode supplies fixed clips if disabled. Can affect quiet speech/context |
| `asr_batch_size` | `1` | 1 selects standard sequential transcription; >1 enables batched decoding with different fallback/timestamp semantics. Optional Performance preset uses 4 |
| `asr_cpu_threads` | `4` | CT2 threads per worker; positive integer. More threads are not necessarily faster |
| `asr_num_workers` | `1` | Config-file-only CT2 worker count; does not enable multiple queue GPU jobs or automatically parallelize one generator |
| `asr_chunk_length` | `30` | 1–30 seconds; applies to the batched path. Smaller windows may alter context/accuracy |
| `word_timestamps` | `true` | Keep enabled for word-to-speaker assignment; disabling is a benchmark option, not a recommended speed preset |

## Diarization

| Key | Default | Notes |
|---|---|---|
| `diarization_model` | `pyannote-community/speaker-diarization-community-1` | Ungated mirror — no token needed. The official `pyannote/…` repos work too but require `hf_token` |
| `min_speakers` / `max_speakers` | `0` / `0` | `0` = unconstrained. Set only when known; incorrect bounds can harm attribution |
| `hf_token` | `""` | Only for gated model repos |
| `diarization_batch_size` | `32` | Sets both segmentation and embedding batch sizes, matching the cached community-1 configuration used in this work; higher values use more VRAM |

## Downloads

| Key | Default | Notes |
|---|---|---|
| `sleep_between_downloads_min`/`_max` | `8` / `15` | Randomized courtesy delay (seconds) between YouTube downloads |
| `max_retries` | `3` | Stored/UI control; currently not wired into the downloader. The inspected wrapper uses 3 download retries and 5 fragment retries; queue retries are manual |
| `cookies_from_browser` | `""` | e.g. `chrome`, `edge`, `firefox` — fixes age-restricted videos and most bot-checks by reusing your logged-in session |
| `rate_limit` | `""` | e.g. `2M` = cap download bandwidth at 2 MiB/s |

### If YouTube challenges large runs (PO tokens)

For very large channel runs from flagged networks, yt-dlp supports Proof-of-
Origin token providers. Install the plugin into the app's venv and it is
picked up automatically — no code changes:

```powershell
.venv\Scripts\python.exe -m pip install bgutil-ytdlp-pot-provider
```

(Requires Node.js/Deno per that project's README. Usually unnecessary on
residential connections with the default courtesy delays.)

## Output

| Key | Default | Notes |
|---|---|---|
| `export_formats` | `["md"]` | Markdown only by default (v0.2). Add any of `json`, `txt`, `srt`, `vtt` in Settings for extra formats |
| `combined_transcript` | `true` | One combined transcript per finished queue (multi-video runs), in the configured formats (md/json) |
| `speaker_names` | `{}` | Global fallback relabeling, e.g. `{"SPEAKER_00": "Cliffe Knechtle"}`. Per-video names from recognition/renaming take precedence |
| `remove_hallucinations` | `true` | Drops known Whisper spam ("Thanks for watching!") **only** when it occurs in diarized silence |
| `paragraph_gap_seconds` | `3.0` | Pause length that starts a new paragraph within a speaker turn |

## Speaker recognition (v0.2)

| Key | Default | Notes |
|---|---|---|
| `recognition_enabled` | `true` | Match detected voices against the voice database after diarization and name confident matches automatically |
| `recognition_threshold` | `0.7` | Minimum cosine similarity for an automatic match. Calibrated on the target channel: genuine matches ~0.90, unrelated speakers ≤0.48, hardest impostor (father/son voices) 0.58. Lower with care — 0.7 already has wide margins |

How it works (details in DESIGN.md §5): diarization already computes one
voice embedding per detected speaker; these are compared against
`Documents\YTScribe\voices.sqlite3`. Matches at or above the threshold get
the stored name (shown as `auto` with the score in the speaker editor);
everything else keeps its `SPEAKER_XX` label. Only *confirmed* voices enter
the database — via the rename dropdown on a completed video, or via seeding:

```powershell
# create/refresh profiles from a professionally diarized transcript
.\run.ps1 --seed "https://www.youtube.com/watch?v=fZZXVNt1gk0" --reference benchmark\reference_fZZXVNt1gk0.txt
# inspect the database
.\run.ps1 --profiles
```

The reference file needs lines of the form `Name (M:SS-M:SS): text`;
placeholder labels (`speaker_2`, …) are ignored. Renaming a speaker rewrites
that video's transcript files in place (all formats present on disk plus the
configured ones) — no re-download, no GPU work.

## Hardware and optimization

| Key | Default | Notes |
|---|---|---|
| `device` | `auto` | Selects CUDA when PyTorch reports it available, otherwise CPU. Explicit `cuda`/`cpu` is honored; runtime failures are reported |
| `optimization_profile` | `custom` | Records `custom`, `safe`, `balanced`, or `performance`; editing this label alone does not apply settings. Use **Apply preset** |
| `model_residency` | `keep` | `keep`: lazily load and retain models during the run. `stage`: unload the other model before an uncached inference stage. Both release at run completion |
| `vram_margin_mb` | `1024` | MiB (binary units, retaining the configuration key): reduces requested batch >1 to 1 if NVML free memory is below this margin; also informs the temporary diarization allocator budget. Not a fit guarantee |
| `oom_retries` | `2` | Maximum additional retries for recognized runtime OOM errors; halve the batch, floor 1. Batch 1, model-load failures and non-OOM errors propagate |
| `profiling_enabled` | `false` | Writes per-run JSONL timing/resource diagnostics under `<cache_dir>\diagnostics\` |
| `profiling_interval` | `1.0` | Positive seconds between diagnostic samples; shorter intervals increase overhead and file size |
| `config_version` | `3` | Internal migration version; distinct from application version 0.3.0 |

**Apply preset** sets large-v3, beam 5, VAD/words on, 30-second chunks,
one CT2 worker, two OOM retries and at least 1024 MiB margin. Safe uses batch 1
and stage residency; Balanced uses batch 1/keep; Performance uses batch 4/keep.
A probed GPU with ≤6 GB changes residency to stage. CPU-thread and diarization-
batch controls retain existing values. Precisions use the probe's supported
types when available, otherwise `auto`. Custom makes no preset changes.

OOM backoff preserves batched semantics even when its effective batch becomes
1, restarts that stage without committing partial results, and does not change
the saved requested batch or model. It is not a guarantee against OOM. No
profile changes device after an error or enables concurrent GPU stages.

For CUDA diarization, the app releases unused PyTorch cache and temporarily
caps its allocator using NVML free memory plus PyTorch reserved memory minus
the margin, with live allocations as a floor. A pre-existing stricter fraction
is preserved, and the previous fraction is restored afterward. The budget does
not cover external allocations or model loading. If unavailable, a warning is
logged and bounded OOM recovery remains. ASR releases unused PyTorch cache
before loading/reusing CT2, including with `model_residency=keep`; live models
remain resident. See [DESIGN.md](DESIGN.md#71-memory-lifecycle-and-restart).

Diagnostic spans cover acquisition/download, conversion, model/audio loading,
ASR preprocessing and generator consumption, diarization, serialization,
speaker matching, merge/export, cleanup and queue transitions. Sampling adds
available CPU/per-core/process, RAM, disk I/O, GPU/VRAM, clocks, temperature,
power/limits, P-state and driver slowdown information. Missing fields are null.
PyTorch counters are read only after CUDA is initialized and exclude CT2;
their peaks are lifetime counters, not automatically reset per stage. Spans
measure host elapsed time without extra GPU synchronization; nested spans
must not be summed as independent work. See the [benchmark report](benchmark/V0.3.0_REPORT.md).

## Where things live

| Data | Location |
|---|---|
| Config | `Documents\YTScribe\config.json` |
| Job queue/state | `Documents\YTScribe\jobs.sqlite3` |
| Voice profiles (v0.2) | `Documents\YTScribe\voices.sqlite3` |
| Per-video cache | `<cache_dir>\<video_id>\` |
| Opt-in traces | `<cache_dir>\diagnostics\<timestamp>-<pid>.jsonl` (not removed by audio cleanup) |
| Processing leases | `<cache_dir>\.worker.lock` and the job database directory's `.worker.lock`; OS locks release when the worker/process exits, even after a crash |
| ML models (one-time download) | `%USERPROFILE%\.cache\huggingface` |

## Housekeeping (what stays on disk and why)

Per completed video, the cache keeps only small JSON artifacts plus voice
samples — audio is removed according to `cache_policy`:

| File | ~Size (30-min video) | Purpose |
|---|---|---|
| `meta.json` | 2 KB | title/date/URL for exports |
| `transcript.json`, `merged.json` | ~0.5 MB each | re-export & renames without re-transcribing |
| `diarization.json` | ~50 KB | speaker turns + voice embeddings |
| `speakers.json` | <1 KB | per-video name assignments |
| `samples\SPEAKER_XX.wav` | ~160 KB each | ≤5 s playback in the speaker editor |

That is roughly 1.5 MB per 30-minute video (≈2 GB for a 1,400-video
channel) — deliberately kept so speakers can be renamed and transcripts
regenerated without inference when artifacts are usable. These sizes are
estimates, not storage limits. Deleting a video's cache also deletes retained
audio, samples and per-video names; back up what you want to keep. Known
half-written stage `.tmp` files and `audio.tmp.wav` are discarded when that
video is next processed. Successful WAV conversion and inference-stage JSON
writes use temporary files followed by replacement.
