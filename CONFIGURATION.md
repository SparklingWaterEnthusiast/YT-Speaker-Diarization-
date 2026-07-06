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
| `export_formats` | `["md","json","txt","srt","vtt"]` | Any subset |
| `combined_transcript` | `true` | One combined MD+JSON per finished queue (multi-video runs) |
| `speaker_names` | `{}` | Relabel speakers at export, e.g. `{"SPEAKER_00": "Cliffe Knechtle", "SPEAKER_01": "Stuart Knechtle"}` |
| `remove_hallucinations` | `true` | Drops known Whisper spam ("Thanks for watching!") **only** when it occurs in diarized silence |
| `paragraph_gap_seconds` | `3.0` | Pause length that starts a new paragraph within a speaker turn |

## Hardware

| Key | Default | Notes |
|---|---|---|
| `device` | `auto` | `auto` = CUDA if available, else CPU (with a log warning) |

## Where things live

| Data | Location |
|---|---|
| Config | `Documents\YTScribe\config.json` |
| Job queue/state | `Documents\YTScribe\jobs.sqlite3` |
| Per-video cache | `<cache_dir>\<video_id>\` |
| ML models (one-time download) | `%USERPROFILE%\.cache\huggingface` |
