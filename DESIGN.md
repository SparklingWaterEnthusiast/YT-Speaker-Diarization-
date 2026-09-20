# DESIGN.md — YTScribe v0.3.0

Architecture and technology decisions for YTScribe, a Windows desktop application
that downloads audio from YouTube (videos, playlists, channels) and produces
speaker-diarized, timestamped transcripts.

Initial technology comparisons were recorded in July 2026; third-party
availability/ranking claims in those historical comparisons are not current
compatibility guarantees. The v0.3.0 updates below reflect inspected code and
September 2026 measurements/research. Accuracy remains the priority. Historical
release records are preserved in [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md);
the detailed experiment and regression report is [V0.3.0_REPORT.md](benchmark/V0.3.0_REPORT.md).

---

## 1. Target hardware and its consequences

Development/deployment target: Razer Blade Pro 17 (2021), RTX 3080 Laptop GPU
with **8 GB VRAM**, Windows 11, CUDA available.

Current consequences:

- **MEASURED:** large-v3 FP16 ran on this 8 GB GPU in the 180-second ASR-only
  sweep, with sampled driver VRAM peaking near **5700 MiB**. This refutes the
  old absolute “FP16 requires 10 GB” claim; it does not establish full-video
  FP16 feasibility with pyannote resident. Initial baseline/early batch-4 peaks
  were **5651/6876 MiB**; final memory-safe batch 4 peaked at **5117 MiB**,
  including driver-wide allocations, not just model weights.
- Models load lazily only for missing inference stages. `keep` retains them
  during the run; `stage` unloads the other model before uncached inference.
  Both are released when the runner closes. Reload cost and quantization
  quality are workload-dependent, not fixed 30-second/negligible-loss guarantees.
- One video uses inference at a time; one prefetch thread overlaps the next
  download. This is the implemented policy, not proof that all GPU concurrency
  is impossible. OS leases on both the cache and job database directories reject
  competing application workers sharing either resource.

Evidence: [measured aggregate](benchmark/results/v0.3.0-summary.json).
Memory includes context, temporary workspaces, allocator caches, display and
other applications; total VRAM alone cannot predict a safe batch.

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

- Format preference is `bestaudio[ext=m4a]/bestaudio/best`; audio-only streams
  are preferred, with a general-format fallback if unavailable.
- Downloaded audio is converted once with FFmpeg to **16 kHz mono WAV** for the
  ML stages (both models want this; converting once avoids two on-the-fly
  resamples). The cache policy covers both original media and `audio.wav`.
  Conversion writes `audio.tmp.wav` and replaces the final WAV on success.
- SQLite state, existing media/metadata, and stage artifacts govern reuse;
  the inspected downloader does not configure a separate `download_archive`.
- **YouTube bot-checks / PO tokens (2026 reality):** YouTube increasingly
  challenges datacenter/flagged IPs ("Sign in to confirm you're not a bot").
  Residential IPs at human-ish rates are normally fine. For the 1,400-video
  channel run, the app supports: configurable sleep between downloads
  (default 8–15 s randomized), yt-dlp retries, a
  retry queue, and optional `cookies-from-browser`/`bgutil-ytdlp-pot-provider`
  plugin passthrough (documented in CONFIGURATION.md) if YouTube starts
  challenging. These are configuration, not code changes.

### 2.2 Transcription — **faster-whisper (CTranslate2) + Whisper large-v3** (selected)

| Candidate | Verdict |
|---|---|
| **faster-whisper 1.2.x** | **Selected.** CTranslate2 backend, quantization, Silero VAD and native word timestamps. Version 1.2.1 was used in v0.3 measurements; published speed ratios are not universal. |
| openai/whisper | Not selected; retain the existing CTranslate2 integration. Compare equal models, beam sizes, output quality and timing boundaries before claiming a runtime advantage. |
| whisper.cpp | Rejected: CUDA build on Windows must be compiled locally → blocked by Smart App Control; word timestamps weaker; Python bindings thin. |
| **WhisperX** | Not selected. Historical dependency concerns were recorded in July; v0.3 does not depend on their continued validity. faster-whisper now supplies the optional batched path directly. Forced alignment is an additional accuracy/runtime tradeoff, not proven redundant by exclusive diarization. |
| NVIDIA NeMo (Parakeet/Canary) | Rejected: best English WER on some benchmarks, but NeMo is heavy and poorly supported on native Windows (no official support, frequent build issues); English-only would limit future use. |
| Cloud APIs (AssemblyAI, Deepgram, Rev) | Rejected: per-minute cost at 1,400×30 min scale (hundreds of dollars per pass), no offline capability — both explicit project requirements. |

Model choice: **large-v3** (default). `large-v3-turbo` and English-focused
`distil-large-v3` are explicit alternatives; neither receives a universal
speed/quality guarantee. Settings: sequential `asr_batch_size=1`, `beam_size=5`,
`word_timestamps=True`, `vad_filter=True`, language auto-detected then pinned
per video, `condition_on_previous_text=False` (reduces repetition-loop risk;
does not eliminate hallucination).

**RESEARCH-BACKED:** [faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py)
batched decoding uses only the first temperature, omits sequential fallback/
no-speech rejection heuristics and defaults to suppressing timestamp tokens.
Word alignment remains available. YTScribe supplies 2000 ms VAD silence to
reduce one default difference, but chunking and decoding still differ. Each
attempt gets a new batched wrapper; OOM backoff to batch 1 stays batched. The
separate benchmark-only batched-one case must not be confused with the app's
sequential setting 1. Words stay enabled because merge requires their timing.

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
- Additional forced alignment is not implemented. The old “within tens of ms”
  claim was not validated. Native alignment has real cost and imperfect
  boundaries; exclusive diarization simplifies reconciliation without proving
  timestamp accuracy. [Official community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1).
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
one **downloader thread** (prefetch next item) + a 1 Hz UI resource monitor.
UI updates use Qt signals; runner control uses threading events and the store
uses locks. Opt-in telemetry adds a sampling thread. Runner shutdown joins
prefetch before releasing models; the GUI waits for worker completion before
exit, including when a model call cannot stop immediately.

### 2.7 State & configuration

- **SQLite** (stdlib `sqlite3`): jobs table = queue, per-stage status
  timestamps, error text, retry count. Restart-safe: on launch, `running`
  states are demoted to `queued` for their incomplete stages. Rejected:
  JSON state file (no atomic partial updates, corrupts on crash), a server DB
  (absurd for a desktop app).
- **Config**: `%USERPROFILE%\Documents\YTScribe\config.json`, dataclass-backed,
  every tunable (models, compute type, cache policy, output dir, export
  formats, sleep intervals…). Schema v3 adds fields without applying a preset
  to old settings. Validation precedes save/start; each runner copies settings.

### 2.8 GPU stack

The measured environment used PyTorch 2.12.1+cu126, CT2 4.8.1, faster-whisper
1.2.1 and pyannote.audio 4.0.7, driver 616.92. `cuda_setup` registers
PyTorch's DLL directory and optional `nvidia-*` directories, retaining the
registration handles. PyTorch and CT2 compatibility must be checked separately.
Auto device selection uses CPU if CUDA is unavailable; inference/load failures
are reported, not silently converted into CPU work. See [INSTALL.md](INSTALL.md)
for legacy/Blackwell differences.

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
  optimization.py  explicit presets and supported-precision policy
  inference.py     VRAM pressure check and OOM helpers
  telemetry.py     opt-in JSONL spans/events/resource samples
  worker_lock.py   OS processing leases for cache and job database directories
  ui/              PySide6 MainWindow, queue table, settings dialog, log pane
  app.py           entry point
```

Every stage writes its output to the video's cache directory
(`cache/<video_id>/`): `audio.m4a`, `audio.wav`, `meta.json`,
`transcript.json` (raw ASR), `diarization.json`, `merged.json`. Required ASR/
diarization artifacts are checked for readable dictionaries with their expected
top-level key. Missing/invalid ones trigger inference and merge invalidation;
pre-v0.2 diarization without embeddings is stale. Valid caches can bypass audio
acquisition and model creation. There is no configuration/content fingerprint:
settings changes and Reprocess alone do not invalidate cached results.

Cache policies (configurable): `delete_after_video` (default), `delete_after_queue`,
`keep_forever` — applied to app-owned audio, including the WAV. JSON and speaker
samples remain. Queue cleanup covers completed items; failed/cancelled audio
may remain for retry. Retention and both folder paths are exposed in the UI.

## 4. Pipeline

```
URL → classify → enumerate videos → per video:
  [acquire audio] → [transcribe] → [diarize] → [merge] → [export]
```

Handled per-item stage failures mark that job `failed` and processing continues.
Pause waits at stage checkpoints without unloading models. Closing requests
cooperative stop/cancel and drains workers; incomplete work is queued for
restart. Cache hits do not contribute to the queue's inference-speed estimate.
These policies do not guarantee recovery from process/driver failure; regression
status and endurance limits belong in the [benchmark report](benchmark/V0.3.0_REPORT.md).

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

## 7. v0.3.0 optimization evidence and policy

Labels: **MEASURED** = actual local experiment; **RESEARCH-BACKED** = primary
source; **INFERRED** = interpretation of code/observations; **ESTIMATED** =
unvalidated tuning candidate; **EXPERIMENTAL** = requires further validation;
**REPORTED** = historical/user account not reproduced by that evidence.
The [research notes](benchmark/research_notes.md) retain the independent
pre-experiment audit; current implementation and subsequent measurements here
supersede its tuning suggestions.

### 7.1 Memory lifecycle and restart

**MEASURED, diagnostic observation:** testing reproduced a fresh-process,
diarization-first restart on a 300-second input: **103.344 s including load**,
PyTorch peak allocated **9,432,108,032 bytes**, reserved **11,374,952,448 bytes**
on the physical 8 GB GPU. The following ASR slowed with unused torch cache
retained. That diagnostic run was deliberately terminated after collecting
evidence; it was not a passed queue test.

**INFERRED:** Windows shared-system-memory fallback plausibly explains
allocations exceeding physical VRAM and severe slowdown. NVIDIA documents this
fallback and its performance cost in [System Memory Fallback](https://nvidia.custhelp.com/app/answers/detail/a_id/5490/kw/support).
The allocation counters are measured; the spill mechanism and exact cuDNN
workspace selection were not traced. Objects did not survive the old process;
the fresh process created these allocations itself.

**Implemented:** around CUDA diarization inference, `torch_budget` clears unused
torch cache and calculates `max(live_allocated, NVML_free + torch_reserved -
margin)`. It applies a fraction of device total memory, capped by the caller's
existing stricter fraction, and restores the previous setting afterward.
PyTorch's [allocator-fraction API](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.set_per_process_memory_fraction.html)
limits its caching allocator and can raise OOM; the existing bounded retry
halves diarization batch size. This does not cap CT2/external allocations,
retroactively shrink live tensors, or budget model loading. Failure to set the
budget is logged. **No NVIDIA driver setting was changed.**

Before both new and reused ASR model execution, unused PyTorch cache is released
even with `keep` residency. Live diarization weights remain. PyTorch allocated/
reserved counters do not include CT2; NVML samples the whole GPU. Emptying
PyTorch's pool does not empty CT2's separate pool or remove context overhead.
[PyTorch memory semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html#memory-management),
[CT2 allocators](https://opennmt.net/CTranslate2/environment_variables.html#ct2-cuda-allocator).

**MEASURED, fixed restart PASS:** the 300-second diarization-first stage completed
in **20.922 s**. Peak counters across the resumed queue were **1.708 GB allocated /
2.108 GB reserved** (decimal GB). The fresh Qt process then processed 480- and
600-second excerpts; all three finished in **105.717 s**, with unchanged cached
ASR, one model load per runtime, no worker accumulation and clean process exit.
These are three excerpts of the benchmark, not three independent videos or a
multi-day endurance claim. The restart excerpt's raw/exclusive speaker boundaries
were identical before/after; embedding cosine similarity exceeded 0.9999999999999.
The [main report](benchmark/V0.3.0_REPORT.md) contains final timing boundaries.

### 7.2 Settings, threads and telemetry

Presets require an explicit Apply action. Safe and Balanced use sequential
large-v3/beam 5; Performance opts into batch 4, keeping words and beam 5.
Default profile is Custom. One worker and sequential GPU stages remain the
policy; CPU threads and diarization batch retain saved values when applying a
preset. A detailed capability probe is explicit; opening Settings and routine
resource sampling do not initialize CUDA.

**RESEARCH-BACKED:** [CT2 precision support](https://opennmt.net/CTranslate2/quantization.html)
depends on capability and build. Pascal CC 6.1 favors `int8_float32`; Ampere/Ada
support FP16 and mixed INT8/FP16. Query the installed supported-type set instead
of assuming every CUDA GPU accepts the same precision. Blackwell INT8 was
re-enabled in [CT2 PR #1982](https://github.com/OpenNMT/CTranslate2/pull/1982),
included in 4.7.0 and the measured 4.8.1 build. Neither FP8 nor FP4 hardware
marketing implies a faster-whisper preset.

CT2 `cpu_threads` maps to intra-op threads; `num_workers` enables concurrent
submissions, not automatic batching of one generator. Avoid oversubscribing
physical CPU cores; more workers/threads are experimental for this pipeline.
[CT2 parallelism](https://opennmt.net/CTranslate2/parallel.html). Pyannote exposes
separate segmentation/embedding batch attributes; YTScribe's single control
sets both, default **32**, matching the cached model configuration used here.
Voice centroids and label mapping remain intact. [pyannote source](https://github.com/pyannote/pyannote-audio/blob/4.0.7/src/pyannote/audio/pipelines/speaker_diarization.py).

Opt-in JSONL traces identify PID/thread, stages, cache hits and host elapsed
time. They sample NVML and process/system counters, plus already-initialized
PyTorch allocator counters without forcing CUDA initialization or resetting
peaks. Missing values remain null; a single sample or average utilization is
not a bottleneck diagnosis. The UI shows temperature/clock/power and warns on
repeated thermal slowdown flags without changing processing settings. Full
field descriptions and storage paths are in [CONFIGURATION.md](CONFIGURATION.md).

### 7.3 Hardware configuration matrix

This matrix is guidance, **not an automatic per-product tuning table**.
All rows keep large-v3, beam 5, words on, one CT2 worker, four CPU threads as a
starting value, and sequential GPU stages. Batch values above 1 use the optional
batched semantics. Only the specified RTX 3080 Laptop workload was measured.
Other batch/residency suggestions are **ESTIMATED**, not guaranteed fit or speed.

| GPU / capability tier | VRAM | Evidence | ASR precision / batch candidates | Residency / diarization |
|---|---:|---|---|---|
| GTX 1060 Laptop, Pascal CC 6.1 | 6 GB | RESEARCH-BACKED capability; ESTIMATED tuning | Supported `int8_float32`; sequential 1 first | Stage residency; consider smaller diarization batches, e.g. 4–8; full model fit unverified |
| RTX 3080 Laptop, Ampere CC 8.6 | 8 GB | MEASURED on one source | `int8_float16`: sequential default or opt-in batch 4; batch 8 and FP16 only prefix-tested | Final batch-4 peak 5117 MiB, baseline recheck 5423 MiB (early batch-4 peak 6876); diarization 32 with allocator budget/handoff cleanup |
| Other Ampere/Ada CC 8.6/8.9 (e.g. RTX 4060/4070 Laptop) | 8 GB | RESEARCH-BACKED / ESTIMATED | Supported mixed INT8/FP16; explore batch 2–4 | Validate dual residency; stage residency under pressure; begin diarization conservatively |
| RTX 3060 desktop 12 GB, RTX 4070 desktop, RTX 4080 Laptop | 12 GB | RESEARCH-BACKED / ESTIMATED | Compare FP16/mixed INT8; explore batch 4–8 | Dual residency candidate; validate diarization 16–32 |
| RTX 4080 desktop, RTX 4090 Laptop, RTX 3080 Laptop 16 GB | 16 GB | RESEARCH-BACKED / ESTIMATED | Compare FP16/mixed INT8; explore batch 8–16 | Dual residency candidate; diarization 32 is a starting point |
| RTX 3090 / RTX 4090 desktop | 24 GB | RESEARCH-BACKED / ESTIMATED | Compare FP16/mixed INT8; explore batch 8–16, then 32 experimentally | Keep models if measured peaks fit; concurrency not enabled |
| Blackwell CC 12.0: RTX 5060/5070/5080-class variants | 8/12/16 GB | RESEARCH-BACKED capability; EXPERIMENTAL local compatibility | Verify torch/CT2 build first; FP16 or supported INT8; corresponding capacity row above | No presumption that the cu126 installer works; conservative memory budget |
| RTX 5090 Laptop / RTX 5090 desktop | 24/32 GB | RESEARCH-BACKED / ESTIMATED | Verify runtime; compare FP16/supported INT8; explore batch 8–16/32 | Dual residency candidate; no invented throughput or laptop/desktop equivalence |

Primary specifications: [NVIDIA compute capability](https://developer.nvidia.com/cuda/gpus),
[legacy capability](https://developer.nvidia.com/cuda/gpus/legacy),
[desktop memory](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/),
[laptop memory](https://www.nvidia.com/en-us/geforce/laptops/compare/),
[30-series laptops](https://www.nvidia.com/en-us/geforce/laptops/30-series/).
RTX 3060 Laptop is **6 GB**, not 8; RTX 4070 Laptop is **8 GB**; 4090 Laptop is
**16 GB** versus desktop **24 GB**; 5090 Laptop is **24 GB** versus desktop
**32 GB**. Detect capacity/capability rather than guessing from product names.

The shipped margin is 1024 MiB. For untested tiers, an **ESTIMATED** starting
target of unused peak VRAM ≥`max(1 GiB, 15% of total)` adds room for display and
other applications; it is not the implemented automatic margin formula or an
OOM guarantee. Start laptops conservatively and assess sustained power/clocks/
thermals. The small WER difference on the imperfect reference does not establish
general accuracy equivalence, and the no-word-timestamps comparison has a
different time-crop/alignment basis. Keep words; see the [main report](benchmark/V0.3.0_REPORT.md).
