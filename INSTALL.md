# INSTALL.md

## Requirements

- Windows 10/11, 64-bit
- ~15 GB free disk (dependencies ~7 GB, models ~4 GB, working space)
- NVIDIA GPU with ≥6 GB VRAM strongly recommended (8 GB targeted); CPU-only
  works but is roughly 10× slower
- A current NVIDIA driver (the app uses the driver's CUDA — no CUDA Toolkit
  install needed)
- Internet connection for setup and for downloading audio (processing itself
  is fully offline once models are cached)

## Automatic install (recommended)

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The script:

1. Installs **Python 3.12** via winget (skipped if present)
2. Installs **FFmpeg** via winget (skipped if present)
3. Creates a local virtual environment `.venv`
4. Installs PyTorch (CUDA 12.6 build) and all Python dependencies
5. Verifies CUDA visibility

Then launch with `.\run.ps1`.

> **Smart App Control / SmartScreen:** everything installed here is a signed
> installer (Python, FFmpeg via winget) or runs through the signed Python
> interpreter, so the app works on machines with Smart App Control enabled.
> This is also why YTScribe ships as a repository + venv instead of a
> single frozen .exe (unsigned frozen executables are blocked by SAC).

## Manual install

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
# new terminal so PATH refreshes
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Routine maintenance

YouTube changes how it serves media every few months, which breaks older
yt-dlp releases (typically a blanket `HTTP Error 403` on downloads while
metadata still resolves). Keep it current:

```powershell
powershell -ExecutionPolicy Bypass -File update-deps.ps1
```

This updates yt-dlp and installs Deno (the JavaScript runtime yt-dlp now
requires for YouTube) if missing. Restart YTScribe afterwards — a running
instance keeps the old version loaded in memory. YTScribe warns in its log
at startup when yt-dlp is over 60 days old or no JS runtime is present.

## First run

The first processed video additionally downloads (once, to the HuggingFace
cache in `%USERPROFILE%\.cache\huggingface`):

- Whisper `large-v3` (~3 GB)
- pyannote `speaker-diarization-community-1` (~30 MB, plus embedding model)

No HuggingFace account or token is required (the app uses the ungated
community mirror of the diarization pipeline).

## Verifying the install

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # True
.venv\Scripts\python.exe -m unittest discover tests                            # all pass
.\run.ps1 --cli "https://www.youtube.com/watch?v=jNQXAC9IVRw"                   # 19 s test video
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| **`HTTP Error 403: Forbidden` / "unable to download video data"** | **yt-dlp is out of date** — YouTube changed its streaming protocol. Run `update-deps.ps1`, restart YTScribe, then Retry Failed. This recurs every few months; it is normal maintenance, not a bug in YTScribe |
| `No supported JavaScript runtime could be found` | Install Deno: `winget install DenoLand.Deno`, then restart YTScribe. yt-dlp has deprecated YouTube extraction without a JS runtime |
| `ffmpeg not found` | Open a new terminal (PATH refresh) or set `ffmpeg_path` in Settings/config.json |
| `CUDA init failed; falling back to CPU` in the log | Update the NVIDIA driver; the app still works on CPU meanwhile |
| `Sign in to confirm you're not a bot` on downloads | Set **Cookies from browser** in Settings (e.g. `chrome`), lower download rates, or see CONFIGURATION.md → PO tokens |
| Age-restricted video fails | Set **Cookies from browser** in Settings |
| Antivirus flags model downloads | Files land in `%USERPROFILE%\.cache\huggingface` — allow-list that folder |
