# DESIGN.md — YTScribe

Architecture and technology decisions for YTScribe, a Windows desktop application
that downloads audio from YouTube (videos, playlists, channels) and produces
speaker-diarized, timestamped transcripts.

Decisions were researched in July 2026 against the current state of the
ecosystem, not assumed from prior knowledge. Accuracy was weighted above speed
throughout, per project requirements.

---

## 1. Target hardware and its consequences

Development/deployment target: Razer Blade Pro 17 (2021), RTX 3080 Laptop GPU
with **8 GB VRAM**, Windows 11, CUDA available.

Consequences baked into the design:

- Whisper **large-v3** in `float16` needs ~10 GB VRAM and does not fit. In
  **`int8_float16`** (CTranslate2 quantization) it needs ~4.5 GB and fits with
  headroom for the diarization models (~1.5 GB). Quality loss from int8_float16
  is negligible (documented by CTranslate2/SYSTRAN benchmarks).
- ASR and diarization models are both kept resident on the GPU across the whole
  queue (loading large-v3 takes ~30 s; reloading per video would dominate
  runtime on a 1,400-video channel). Stages within one video run sequentially,
  so peak VRAM is bounded.
- One video is processed at a time. On 8 GB there is no VRAM budget for
  parallel GPU work, and downloads are overlapped with GPU work instead
  (the downloader prefetches the next item while the GPU processes the
  current one).

**Windows Smart App Control (SAC) is enabled on the target machine.** SAC
blocks unsigned executables. This was verified on this machine (locally
compiled exes are blocked). Consequences:

- **No PyInstaller/Nuitka frozen exe.** A frozen exe is unsigned and would be
  blocked. Instead the app ships as a repository with a `setup.ps1` that
  creates a virtual environment; everything executes through the signed
  `python.exe` from the official Python installer.
- Python and FFmpeg are installed through **winget** (signed installers with
  established reputation), not ad-hoc downloaded zips.

## 2. Technology choices

### 2.1 Audio acquisition — **yt-dlp** (selected)

| Candidate | Verdict |
|---|---|
| **yt-dlp** | **Selected.** De-facto standard, weekly releases, largest extractor community, first to receive fixes when YouTube changes. Supports audio-only format selection, playlists, channels, download archives (resume), rate limiting, retries, metadata extraction (title, upload date, duration, chapters) in one tool. Python API and CLI. Unlicense. |
| youtube-dl | Rejected: effectively unmaintained; broken against current YouTube. |
| pytube / pytubefix | Rejected: single-maintainer projects, historically break for weeks when YouTube changes; no download archive, weaker playlist/channel handling. |
| Invidious/Piped APIs | Rejected: third-party instances are unreliable and rate-limited; adds a network dependency outside our control. |

Implementation details:

- Audio-only format selection: `bestaudio[ext=m4a]/bestaudio` — YouTube's
  ~128 kbps AAC/Opus audio streams. Whisper and pyannote both downsample to
  16 kHz mono internally, so the source stream is already beyond the quality
  ceiling that matters; downloading video would waste ~10× bandwidth for zero
  accuracy gain.
- Downloaded audio is converted once with FFmpeg to **16 kHz mono WAV** for the
  ML stages (both models want this; converting once avoids two on-the-fly
  resamples), while the compact original (m4a/webm) is what the cache policy
  manages.
- `download_archive` file + SQLite job state give two independent layers of
  "never repeat completed work".
- **YouTube bot-checks / PO tokens (2026 reality):** YouTube increasingly
  challenges datacenter/flagged IPs ("Sign in to confirm you're not a bot").
  Residential IPs at human-ish rates are normally fine. For the 1,400-video
  channel run, the app supports: configurable sleep between downloads
  (default 8–15 s randomized), limited retries with exponential backoff, a
  retry queue, and optional `cookies-from-browser`/`bgutil-ytdlp-pot-provider`
  plugin passthrough (documented in CONFIGURATION.md) if YouTube starts
  challenging. These are configuration, not code changes.

### 2.2 Transcription — **faster-whisper (CTranslate2) + Whisper large-v3** (selected)

| Candidate | Verdict |
|---|---|
| **faster-whisper 1.2.x** | **Selected.** CTranslate2 backend, 4× faster than openai/whisper at equal accuracy, int8_float16 fits large-v3 in 8 GB, built-in Silero-VAD v6 filtering, built-in word-level timestamps, MIT, actively maintained (v1.2.1, Oct 2025). |
| openai/whisper | Rejected: same models, ~4× slower, higher VRAM, no VAD integration. |
| whisper.cpp | Rejected: CUDA build on Windows must be compiled locally → blocked by Smart App Control; word timestamps weaker; Python bindings thin. |
| **WhisperX** | Rejected **for now** (was the leading candidate): it is currently broken on PyPI — it pins `pyannote-audio 3.3.2`, whose `lightning` dependency was quarantined on PyPI in April 2026, and it conflicts with pyannote.audio 4.x. Its two advantages (batched inference, wav2vec2 forced alignment) are speed-oriented; faster-whisper's native DTW word timestamps plus pyannote 4's *exclusive diarization* output (§2.3) close the accuracy gap for speaker attribution. |
| NVIDIA NeMo (Parakeet/Canary) | Rejected: best English WER on some benchmarks, but NeMo is heavy and poorly supported on native Windows (no official support, frequent build issues); English-only would limit future use. |
| Cloud APIs (AssemblyAI, Deepgram, Rev) | Rejected: per-minute cost at 1,400×30 min scale (hundreds of dollars per pass), no offline capability — both explicit project requirements. |

Model choice: **large-v3** (default). `large-v3-turbo` and `distil-large-v3`
are 4–6× faster with a small but real accuracy loss, and are exposed in
configuration for users who prefer throughput; the default follows the
project's "accuracy over speed" rule. Settings: `beam_size=5`,
`word_timestamps=True`, `vad_filter=True`, language auto-detected then pinned
per video, `condition_on_previous_text=False` (avoids hallucination loops on
long recordings — a known large-v3 failure mode).

### 2.3 Speaker diarization — **pyannote.audio 4.x, `speaker-diarization-community-1`** (selected)

| Candidate | Verdict |
|---|---|
| **pyannote community-1** | **Selected.** Successor to pyannote 3.1; best open-source DER in 2026 third-party benchmarks (pyannoteAI's paid model leads; community-1 is the best self-hostable). Handles unknown speaker counts, overlap-aware. Crucially, pyannote 4 adds an **`exclusive_speaker_diarization`** output: a non-overlapping segmentation designed specifically to reconcile diarization with "sometimes not so precise" transcription timestamps — exactly the ASR-merge problem this app has. CC-BY-4.0 model, MIT code. The official repo is token-gated, but the **`pyannote-community/` mirror is ungated** (verified via HF API: `gated: false`) — no HuggingFace account needed. |
| pyannote 3.1 | Rejected: superseded; its `lightning` dependency chain is the one quarantined on PyPI. |
| NVIDIA Sortformer v2 | Rejected: excellent DER on some sets and native overlap handling, but hard 4-speaker limit (street-interview videos routinely exceed this) and NeMo-on-Windows pain. |
| DiariZen | Considered: competitive DER (13.3% vs community-1's ~11–13% in 2026 benchmarks), but much smaller community, less battle-tested, no exclusive-mode output. |
| Reverb diarization | Rejected: non-commercial license restrictions; based on older pyannote anyway. |

Speaker count is left unconstrained by default (`min_speakers`/`max_speakers`
configurable per run) — street-interview content has unpredictable
questioners.

### 2.4 Timestamp alignment & transcript assembly — custom merge (selected)

The merge stage assigns each ASR **word** to a speaker using maximal temporal
overlap against the **exclusive** diarization timeline (midpoint fallback for
zero-overlap words), then rebuilds speaker turns, sentences, and paragraphs.

- Word-level assignment (not segment-level) is what makes rapid interjections
  ("Yes." — "Right.") attach to the correct speaker; Whisper segments often
  span speaker changes.
- Rejected: wav2vec2 forced alignment (WhisperX-style) as an extra stage —
  unavailable via WhisperX right now (§2.2) and faster-whisper's DTW word
  timestamps are within tens of ms, sufficient once exclusive diarization
  provides clean boundaries.
- Punctuation restoration: **not needed as a separate stage** — Whisper
  natively emits punctuation and casing. A dedicated model
  (deepmultilingualpunctuation etc.) would only matter for punctuation-free
  ASR like classic NeMo CTC models. Cleanup is instead rule-based: whitespace
  normalization, dangling-fragment merging, configurable removal of Whisper's
  known hallucination strings ("Thanks for watching!", subtitle-credit spam)
  when they appear in silence-adjacent segments.

### 2.5 Export — hand-rolled writers (selected)

Markdown (readable, video header + `**Speaker** (H:MM:SS):` turns), JSON (full
structure: words, timings, confidences, speakers, metadata), TXT, SRT, VTT.
All are simple line formats; a library (e.g. `srt`, `webvtt-py`) would add
dependencies for ~40 lines of code each. Combined-queue transcript is a
concatenation with per-video headers plus a table of contents.

### 2.6 Desktop UI — **PySide6** (selected)

| Candidate | Verdict |
|---|---|
| **PySide6 (Qt 6)** | **Selected.** The pipeline is necessarily Python (all ML deps are Python). Qt gives native Windows widgets, QThread workers for a responsive UI, mature signals/slots for progress reporting, LGPL licensing (vs PyQt's GPL/commercial). |
| Tkinter | Rejected: in stdlib, but visibly dated widgets, weak threading ergonomics, poor table/log widgets. |
| Tauri/Electron + Python sidecar | Rejected: two runtimes, IPC layer, Node toolchain — massive complexity for a single-window tool; Electron alone ~200 MB. |
| Web UI (FastAPI + browser) | Rejected: user asked for a desktop application; also complicates single-instance queue semantics. |

Threading model: UI thread + one **pipeline worker thread** (owns the GPU) +
one **downloader thread** (prefetch next item) + a 1 Hz **resource monitor**
(GPU/VRAM via `pynvml`, CPU/RAM via `psutil`, disk via `shutil.disk_usage`).
Communication is exclusively Qt signals; no shared mutable state.

### 2.7 State & configuration

- **SQLite** (stdlib `sqlite3`): jobs table = queue, per-stage status
  timestamps, error text, retry count. Restart-safe: on launch, `running`
  states are demoted to `pending` for their incomplete stages. Rejected:
  JSON state file (no atomic partial updates, corrupts on crash), a server DB
  (absurd for a desktop app).
- **Config**: single `config.json` in the app directory, dataclass-backed,
  every tunable (models, compute type, cache policy, output dir, export
  formats, sleep intervals…) — "prefer configuration over code changes".

### 2.8 GPU stack

PyTorch CUDA wheels (cu126) for pyannote + `ctranslate2` (bundles its own
cuBLAS/cuDNN via pip `nvidia-*` wheels) for faster-whisper. Both verified
against driver 581.80. CPU fallback is automatic (with a UI warning) if CUDA
init fails.

## 3. Architecture

```
ytscribe/
  config.py        Config dataclass ⇄ config.json
  db.py            SQLite job store (queue, stage status, retries)
  media.py         yt-dlp wrapper: URL classification (video/playlist/channel),
                   metadata extraction, audio download, WAV conversion
  transcribe.py    faster-whisper wrapper → Word/Segment dataclasses
  diarize.py       pyannote community-1 wrapper → speaker turns (exclusive+raw)
  merge.py         word↔speaker assignment, turn building, paragraphing, cleanup
  export.py        md/json/txt/srt/vtt writers + combined transcript
  pipeline.py      per-video stage runner with caching + queue orchestration
  resources.py     GPU/CPU/RAM/disk sampling
  ui/              PySide6 MainWindow, queue table, settings dialog, log pane
  app.py           entry point
```

Every stage writes its output to the video's cache directory
(`cache/<video_id>/`): `audio.m4a`, `audio.wav`, `meta.json`,
`transcript.json` (raw ASR), `diarization.json`, `merged.json`. A stage runs
only if its output artifact is missing or its input changed — re-running a
queue never repeats completed work, and any module can be replaced as long as
it honors the artifact contract.

Cache policies (configurable): `delete_after_video` (default), `delete_after_queue`,
`keep_forever` — applied to the audio files only; the small JSON artifacts are
always kept so transcripts can be re-exported without re-downloading.

## 4. Pipeline

```
URL → classify → enumerate videos → per video:
  [acquire audio] → [transcribe] → [diarize] → [merge] → [export]
```

Failures at any stage mark the job `failed` with the error recorded, move it
to the retry queue, and processing continues with the next item. Nothing a
single bad video does can stop a 1,400-video run.

## 5. Version 0.2 — speaker identification (July 2026)

### 5.1 Embedding source — pipeline centroids (selected)

Cross-video speaker recognition needs one voice embedding per detected
speaker. Candidates evaluated:

| Candidate | Verdict |
|---|---|
| **community-1 pipeline centroids** | **Selected.** pyannote 4's `DiarizeOutput.speaker_embeddings` already returns one 256-dim WeSpeaker ResNet34 centroid per speaker — the very vectors the diarizer clustered on, computed on overlap-excluded (clean, single-speaker) frames. Verified in pipeline source: centroid rows are re-ordered to match `labels()` order (`SPEAKER_00`…), zero-padded when a speaker has no centroid (guarded in code). **Zero new models, zero extra GPU passes, and recognition lives in the same embedding space as diarization.** |
| speechbrain ECAPA-TDNN | Rejected: adds the speechbrain dependency chain and a second inference pass per video; VoxCeleb-grade accuracy comparable to WeSpeaker ResNet34; no benefit that justifies a parallel embedding space. |
| wespeaker (standalone) | Rejected: same model family as what the pipeline already runs — pure redundancy. |
| Resemblyzer (GE2E) | Rejected: 2019-era accuracy, clearly below current models. |
| NVIDIA TitaNet | Rejected: NeMo on native Windows again. |

Matching is **cosine similarity** (embeddings L2-normalized first) against
every stored sample of every profile, taking each profile's best score. A
match requires `score >= recognition_threshold` (default **0.7**). Empirical
calibration on the target channel (July 2026): genuine cross-video matches
scored 0.895–0.909 (including Cliffe across a 10-year-old video); audience
questioners scored ≤0.48 against the hosts; the hardest impostor pair —
Stuart against his father Cliffe's profile, similar voices in identical
acoustic conditions — scored 0.58. The 0.7 default sits mid-gap. Below
threshold, labels stay exactly `SPEAKER_XX` — names are never invented.

### 5.2 Voice database

`Documents\YTScribe\voices.sqlite3` (separate file from the job queue, so
clearing one never touches the other):

    speakers(id, name UNIQUE, created_at, updated_at)
    samples(id, speaker_id, embedding BLOB float32, dim, source, created_at)

A profile holds N confirmed embeddings (one per confirmed video appearance;
`source` = "video_id:SPEAKER_XX" is unique so re-confirming replaces rather
than duplicates). Only *confirmed* embeddings enter the DB — from manual
renames or reference-transcript seeding, never from automatic matches, so a
borderline auto-match can never poison a profile (no drift).

### 5.3 Per-video name resolution

Diarization saves per-speaker embeddings into `diarization.json`; right after
that stage (while `audio.wav` still exists) the pipeline extracts one ≤5 s
playback sample per speaker (`cache/<id>/samples/SPEAKER_XX.wav`, longest
exclusive turn, center window, stdlib `wave` — no ffmpeg) and runs
auto-matching, writing `cache/<id>/speakers.json`:

    {"SPEAKER_00": {"name": "Cliffe Knechtle", "source": "auto", "score": 0.81}}

Display names resolve as: speakers.json (manual > auto) → config
`speaker_names` → raw label. Manual renames (UI dropdown on any completed
video, or after auto-match correction) update speakers.json, re-export every
transcript format present on disk plus the configured ones, and add the
video's embedding to the profile. Exports are pure functions of cached
artifacts, so renaming needs no GPU and no re-download.

### 5.4 Seeding from a reference transcript

`--seed <url> --reference <file>` parses a professionally diarized transcript
("Name (M:SS-M:SS): text" lines), aligns each detected speaker to reference
speakers by temporal overlap (real names only — `speaker_N` placeholders are
ignored), and creates profiles for alignments with ≥60 % purity and ≥10 s of
overlap. Used to seed Cliffe & Stuart Knechtle from the benchmark video.

## 6. Known limitations / future work
- Age-restricted/member-only videos need `cookies_from_browser` (documented).
- Whisper hallucinations on long music/silence stretches are mitigated by VAD
  + `condition_on_previous_text=False` + cleanup rules, not eliminated.
