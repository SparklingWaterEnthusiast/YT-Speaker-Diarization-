# YTScribe

A Windows desktop application that turns YouTube videos, playlists, and entire
channels into **speaker-diarized, timestamped transcripts** you can read
instead of watching.

- Downloads **audio only** (no video) via yt-dlp — minimal bandwidth and storage
- Transcribes with **Whisper large-v3** (faster-whisper / CTranslate2, GPU-accelerated)
- Identifies **who spoke when** with pyannote `speaker-diarization-community-1`
- Merges both at the **word level** for accurate speaker attribution
- Exports **Markdown, JSON, TXT, SRT, VTT** per video, plus a combined
  transcript per queue
- Caches every stage — interrupted runs resume without repeating work
- Designed for large jobs: a 1,000+-video channel can run unattended; failures
  go to a retry queue and never stop the run

Sized for an 8 GB VRAM GPU (RTX 3080 Laptop) out of the box; falls back to CPU
automatically.

## Quick start

```powershell
git clone <this-repository>
cd "Speedch Diariztion v0.1"
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

## Sample output (Markdown)

```markdown
**Cliffe Knechtle** (23:33–23:39):

Good point. Very good point. So maybe Jesus did sin at some point.
But I have to follow the evidence.
```

## License notes

YTScribe's dependencies: yt-dlp (Unlicense), faster-whisper (MIT),
pyannote.audio (MIT, model CC-BY-4.0), PySide6 (LGPL), FFmpeg (GPL/LGPL
build via gyan.dev). Respect YouTube's Terms of Service and copyright law for
the content you process.
