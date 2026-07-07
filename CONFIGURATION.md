# CONFIGURATION.md

All settings live in `%USERPROFILE%\Documents\YTScribe\config.json`, created
with defaults on first launch. The Settings dialog edits the common ones; the
file is the complete reference. Changes apply to the next queue run.

## Paths

| Key | Default | Meaning |
|---|---|---|
| `output_dir` | `Documents\YTScribe\transcripts` | Where transcript files are written |
| `cache_dir` | `Documents\YTScribe\cache` | Per-video working data (`cache/<video_id>/`) |
| `ffmpeg_path` | `""` | Explicit ffmpeg.exe path; empty = auto-detect (PATH, then winget dirs) |

## Cache policy — `cache_policy`

Applies to the **audio files** only; the small JSON stage artifacts
(transcript, diarization, merged) are always kept so transcripts can be
regenerated or re-exported without re-downloading or re-transcribing.

| Value | Behavior |
|---|---|
| `delete_after_video` *(default)* | Audio deleted as soon as a video completes — flat disk usage |
| `delete_after_queue` | Audio kept until the whole queue finishes, then deleted |
| `keep_forever` | Nothing deleted |

## Transcription

| Key | Default | Notes |
|---|---|---|
| `whisper_model` | `large-v3` | Best accuracy. Alternatives: `large-v3-turbo`, `distil-large-v3` (~5× faster, slightly less accurate), `medium`, `small`, or any CTranslate2 model id |
| `compute_type` | `auto` | `auto` = `int8_float16` on GPU (fits 8 GB), `int8` on CPU. Override with `float16` on ≥10 GB GPUs |
| `beam_size` | `5` | Higher = marginally better, slower |
| `language` | `auto` | ISO code (`en`) skips per-video language detection |
| `vad_filter` | `true` | Silero-VAD v6 removes non-speech before ASR — keep on |

## Diarization

| Key | Default | Notes |
|---|---|---|
| `diarization_model` | `pyannote-community/speaker-diarization-community-1` | Ungated mirror — no token needed. The official `pyannote/…` repos work too but require `hf_token` |
| `min_speakers` / `max_speakers` | `0` / `0` | `0` = automatic. Set both to a known count for better accuracy on fixed-format shows |
| `hf_token` | `""` | Only for gated model repos |

## Downloads

| Key | Default | Notes |
|---|---|---|
| `sleep_between_downloads_min`/`_max` | `8` / `15` | Randomized courtesy delay (seconds) between YouTube downloads |
| `max_retries` | `3` | Per-video retry budget when using Retry Failed |
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
| `recognition_threshold` | `0.6` | Minimum cosine similarity for an automatic match. Calibrated on the target channel: same speaker across videos scored ≥0.7, different speakers ≤0.4. Raise toward 0.7 to be stricter, lower with care |

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

## Hardware

| Key | Default | Notes |
|---|---|---|
| `device` | `auto` | `auto` = CUDA if available, else CPU (with a log warning) |

## Where things live

| Data | Location |
|---|---|
| Config | `Documents\YTScribe\config.json` |
| Job queue/state | `Documents\YTScribe\jobs.sqlite3` |
| Voice profiles (v0.2) | `Documents\YTScribe\voices.sqlite3` |
| Per-video cache | `<cache_dir>\<video_id>\` |
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
regenerated at any time without any re-processing. Deleting a video's cache
folder is safe; it just forfeits those abilities until reprocessed.
Half-written `.tmp` files from crashes are removed automatically the next
time the video is processed.
