# YTScribe v0.3.0

A Windows desktop application that turns YouTube videos, playlists, and entire
channels into **speaker-diarized, timestamped transcripts** you can read
instead of watching.

- Downloads **audio only** (no video) via yt-dlp — minimal bandwidth and storage
- Transcribes with **Whisper large-v3** (faster-whisper / CTranslate2, GPU-accelerated)
- Identifies **who spoke when** with pyannote `speaker-diarization-community-1`
- **Recognizes recurring voices** (v0.2): a persistent voice database learns
  speakers over time and names them automatically in future videos —
  unknown voices keep their neutral `SPEAKER_00` labels
- Click any completed video to **play speaker samples and rename speakers**;
  transcripts update automatically and the voice is remembered
- Merges ASR and diarization at the **word level** for accurate attribution
- Exports **Markdown** by default; JSON, TXT, SRT, VTT are optional, plus a
  combined transcript per queue
- **Queue tabs** (v0.2.1): File-Explorer-style tabs, each an independent
  queue with its own output subfolder — start a priority tab mid-run and the
  bulk queue steps aside, then resumes automatically
- File-manager queue control: multi-select, right-click retry/reprocess/
  reorder/remove, drag tabs, rename tabs (renames the folder)
- Caches completed stages for resume and re-export; models load only for missing inference stages
- **Optimization controls** (v0.3.0): explicit hardware probe, optional batch-4
  ASR, model residency, bounded memory retries, and opt-in performance traces
- **Open Audio / Cache** and audio-retention settings make saved audio easy to find
- Designed for large jobs: per-video failures go to a retry queue while the
  remaining items continue; v0.3's local endurance test is shorter than a full channel

Defaults remain **sequential large-v3, beam 5, word timestamps enabled**.
The optional Performance preset was faster on one measured full video on an
RTX 3080 Laptop 8 GB; batching can increase memory demand and changes decoding
semantics. The final memory safeguards reduced the measured whole-run VRAM peak.
See the [v0.3.0 benchmark report](benchmark/V0.3.0_REPORT.md) for measurements, quality limitations, and
regression status. Automatic device selection uses CPU when CUDA is unavailable;
runtime model/DLL errors are reported rather than silently changing devices.

## Quick start

```powershell
git clone <this-repository>
cd "Speaker Diariztion v0.2.2"                          # or your checkout folder
powershell -ExecutionPolicy Bypass -File setup.ps1   # one time, installs everything
.\run.ps1                                            # launches the app
```

Paste a YouTube URL (video, playlist, or channel), click **Add to Queue**,
then **Start**. Transcripts appear in `Documents\YTScribe\transcripts`.

Headless operation (same pipeline, no window):

```powershell
.\run.ps1 --cli "https://www.youtube.com/watch?v=fZZXVNt1gk0"
```

## Documentation

| File | Contents |
|---|---|
| [INSTALL.md](INSTALL.md) | Fresh-machine installation, requirements, troubleshooting |
| [USER_GUIDE.md](USER_GUIDE.md) | Using the app: queue, progress, pause/resume, retries, outputs |
| [CONFIGURATION.md](CONFIGURATION.md) | Every setting in `config.json`, cache policies, speaker renaming |
| [DESIGN.md](DESIGN.md) | Architecture and technology decisions with rationale |
| [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) | What was built, tested, and verified, in order |
| [Benchmark report](benchmark/V0.3.0_REPORT.md) | v0.3.0 measurements, methodology, lifecycle checks, and limitations |

## Sample output (Markdown)

```markdown
**Cliffe Knechtle** (23:33–23:39):

Good point. Very good point. So maybe Jesus did sin at some point.
But I have to follow the evidence.
```

## Teaching it voices

Rename a speaker once (click the completed video → play the sample → type
the name) and YTScribe stores that person's voice fingerprint. Future videos
attempt automatic recognition; only confident matches receive names. Profiles can also be seeded from a
professionally diarized transcript:

```powershell
.\run.ps1 --seed "<video-url>" --reference benchmark\reference_fZZXVNt1gk0.txt
.\run.ps1 --profiles     # list stored voice profiles
```

## License notes

YTScribe's dependencies: yt-dlp (Unlicense), faster-whisper (MIT),
pyannote.audio (MIT, model CC-BY-4.0), PySide6 (LGPL), FFmpeg (GPL/LGPL
build via gyan.dev). Respect YouTube's Terms of Service and copyright law for
the content you process.
