# INSTALL.md — v0.3.0

## Requirements

- Windows 10/11, 64-bit
- ~15 GB free disk (dependencies ~7 GB, models ~4 GB, working space)
- NVIDIA GPU recommended; the measured target is **RTX 3080 Laptop 8 GB**.
  A 6 GB GPU requires a compatible runtime and conservative settings; capacity
  alone does not guarantee model fit. CPU execution has no guaranteed speed ratio
- A compatible NVIDIA driver and CUDA runtime libraries supplied by the Python
  packages; a separate CUDA Toolkit is normally unnecessary for this setup
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
4. Installs PyTorch (CUDA 12.6 build) and all Python dependencies; this is the
   existing target-machine setup, **not a universal Blackwell installer**
5. Verifies CUDA visibility

Then launch with `.\run.ps1`. Run `update-deps.ps1` if the startup diagnostics
report a missing YouTube JavaScript runtime (Deno is not installed by `setup.ps1`).

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

## GPU compatibility and upgrading to v0.3.0

Existing configuration migrates additively to schema 3. Application version
0.3.0 keeps sequential large-v3/beam 5 as the default; upgrading does not apply
the Performance preset. Preserve `Documents\YTScribe` and the model cache when
updating the checkout. Stop the old process before starting the new version.

The measured environment used Python 3.12.10, torch 2.12.1+cu126, CT2 4.8.1,
faster-whisper 1.2.1, pyannote.audio 4.0.7 and driver 616.92. Requirements use
version ranges, so a new install may differ; record versions when comparing runs.

- **Pascal (GTX 1060):** retain a wheel with legacy architecture support;
  CT2 commonly uses `int8_float32`, not efficient FP16. Probe before selecting.
- **Ampere/Ada:** compare supported `int8_float16` and FP16 with memory
  headroom; 8 GB does not categorically exclude FP16. Defaults stay conservative.
- **Blackwell (RTX 50):** validate PyTorch and CT2 independently. For PyTorch
  2.12, [official release guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/)
  recommends CUDA 13.0+ wheels and Windows driver ≥580.88. CT2's CUDA libraries
  must also match its build; a newer torch wheel is not proof its DLLs meet CT2's
  requirements. [CT2 4.7.0](https://github.com/OpenNMT/CTranslate2/blob/v4.8.1/CHANGELOG.md)
  includes re-enabled Blackwell INT8 support after older releases disabled it.
  This repository's cu126 setup has not been validated on Blackwell.

Use **Settings → Optimization → Probe hardware capabilities** for supported
precisions. Detection does not prove a model will load or fit. Explicit CUDA,
precision, or DLL failures are shown as errors; choose CPU deliberately if
needed. [DESIGN.md](DESIGN.md#73-hardware-configuration-matrix) gives research
tiers; [the benchmark report](benchmark/V0.3.0_REPORT.md) gives measured scope.

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
.venv\Scripts\python.exe -m unittest discover tests                            # run regression checks
.\run.ps1 --cli "https://www.youtube.com/watch?v=jNQXAC9IVRw"                   # 19 s test video
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| **`HTTP Error 403: Forbidden` / "unable to download video data"** | **yt-dlp is out of date** — YouTube changed its streaming protocol. Run `update-deps.ps1`, restart YTScribe, then Retry Failed. This recurs every few months; it is normal maintenance, not a bug in YTScribe |
| `No supported JavaScript runtime could be found` | Install Deno: `winget install DenoLand.Deno`, then restart YTScribe. yt-dlp has deprecated YouTube extraction without a JS runtime |
| `ffmpeg not found` | Open a new terminal (PATH refresh) or set `ffmpeg_path` in Settings/config.json |
| CUDA/DLL/unsupported compute-type error | Check driver, torch/CT2 builds and supported types. Auto selects CPU only when CUDA is unavailable; runtime failures are not silently retried on CPU. Explicitly select CPU if desired |
| Out of memory, or slowdown after diarization | Try Safe/stage residency or a smaller batch. v0.3 applies a temporary PyTorch budget and frees unused torch cache before ASR; neither guarantees every workload fits. Enable diagnostic traces; do not equate reserved memory with a leak |
| Another worker is processing this cache | Finish/close the previous GUI or CLI worker. The OS releases the lease on exit; the remaining `.worker.lock` file alone is not a live lock |
| `Sign in to confirm you're not a bot` on downloads | Set **Cookies from browser** in Settings (e.g. `chrome`), lower download rates, or see CONFIGURATION.md → PO tokens |
| Age-restricted video fails | Set **Cookies from browser** in Settings |
| Antivirus flags model downloads | Files land in `%USERPROFILE%\.cache\huggingface` — allow-list that folder |
