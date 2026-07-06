"""Application configuration: a dataclass persisted to config.json.

Every tunable lives here so behavior changes never require code changes.
Paths default to a per-user data directory, never hardcoded absolutes.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path.home() / "Documents" / "YTScribe"

CACHE_POLICIES = ("delete_after_video", "delete_after_queue", "keep_forever")
EXPORT_FORMATS = ("md", "json", "txt", "srt", "vtt")


@dataclass
class Config:
    # --- paths ---
    output_dir: str = str(APP_DIR / "transcripts")
    cache_dir: str = str(APP_DIR / "cache")
    ffmpeg_path: str = ""  # empty = resolve from PATH at startup

    # --- audio cache policy: delete_after_video | delete_after_queue | keep_forever ---
    cache_policy: str = "delete_after_video"

    # --- transcription ---
    whisper_model: str = "large-v3"
    compute_type: str = "auto"  # auto -> int8_float16 on CUDA, int8 on CPU
    beam_size: int = 5
    language: str = "auto"  # ISO code or "auto"
    vad_filter: bool = True

    # --- diarization ---
    diarization_model: str = "pyannote-community/speaker-diarization-community-1"
    min_speakers: int = 0  # 0 = unconstrained
    max_speakers: int = 0  # 0 = unconstrained
    hf_token: str = ""  # only needed for gated model repos

    # --- hardware ---
    device: str = "auto"  # auto | cuda | cpu

    # --- downloads ---
    sleep_between_downloads_min: float = 8.0
    sleep_between_downloads_max: float = 15.0
    max_retries: int = 3
    cookies_from_browser: str = ""  # e.g. "chrome" for age-restricted videos
    rate_limit: str = ""  # e.g. "2M" to cap download bandwidth

    # --- output ---
    export_formats: list[str] = field(default_factory=lambda: list(EXPORT_FORMATS))
    combined_transcript: bool = True
    # optional relabeling applied at export time, e.g. {"SPEAKER_00": "Cliffe"}
    speaker_names: dict[str, str] = field(default_factory=dict)

    # --- transcript cleanup ---
    remove_hallucinations: bool = True
    paragraph_gap_seconds: float = 3.0

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    def resolve_ffmpeg(self) -> str:
        """Return a usable ffmpeg executable path or raise RuntimeError."""
        if self.ffmpeg_path and Path(self.ffmpeg_path).exists():
            return self.ffmpeg_path
        found = shutil.which("ffmpeg")
        if found:
            return found
        # winget installs land here; the user's PATH may not be refreshed yet
        winget_dirs = (
            Path.home() / "AppData/Local/Microsoft/WinGet/Links",
            Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
        )
        for base in winget_dirs:
            if base.is_dir():
                for hit in base.glob("**/ffmpeg.exe"):
                    return str(hit)
        raise RuntimeError(
            "ffmpeg not found. Install it (winget install Gyan.FFmpeg) or set "
            "ffmpeg_path in Settings."
        )

    def validate(self) -> list[str]:
        problems = []
        if self.cache_policy not in CACHE_POLICIES:
            problems.append(f"cache_policy must be one of {CACHE_POLICIES}")
        bad = [f for f in self.export_formats if f not in EXPORT_FORMATS]
        if bad:
            problems.append(f"unknown export formats: {bad}")
        if self.min_speakers and self.max_speakers and self.min_speakers > self.max_speakers:
            problems.append("min_speakers cannot exceed max_speakers")
        if self.sleep_between_downloads_min > self.sleep_between_downloads_max:
            problems.append("sleep_between_downloads_min cannot exceed max")
        return problems


def config_file() -> Path:
    return APP_DIR / "config.json"


def load_config() -> Config:
    path = config_file()
    cfg = Config()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {f.name for f in dataclasses.fields(Config)}
            for key, value in data.items():
                if key in known:
                    setattr(cfg, key, value)
        except (json.JSONDecodeError, OSError):
            # corrupt config falls back to defaults; will be rewritten on save
            pass
    return cfg


def save_config(cfg: Config) -> None:
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")
    tmp.replace(path)
