"""Application configuration: a dataclass persisted to config.json.

Every tunable lives here so behavior changes never require code changes.
Paths default to a per-user data directory, never hardcoded absolutes.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path.home() / "Documents" / "YTScribe"

CACHE_POLICIES = ("delete_after_video", "delete_after_queue", "keep_forever")
EXPORT_FORMATS = ("md", "json", "txt", "srt", "vtt")
COMPUTE_TYPES = ("auto", "default", "int8_float16", "float16", "int8",
                 "int8_float32", "float32", "bfloat16", "int8_bfloat16", "int16")
OPTIMIZATION_PROFILES = ("custom", "safe", "balanced", "performance")


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
    asr_batch_size: int = 1  # 1 = standard; >1 = experimental batched ASR
    asr_cpu_threads: int = 4
    asr_num_workers: int = 1  # internal worker count, not GPU job concurrency
    asr_chunk_length: int = 30
    word_timestamps: bool = True  # disable only for explicit benchmark experiments

    # --- diarization ---
    diarization_model: str = "pyannote-community/speaker-diarization-community-1"
    min_speakers: int = 0  # 0 = unconstrained
    max_speakers: int = 0  # 0 = unconstrained
    hf_token: str = ""  # only needed for gated model repos
    # Cached community-1 config.yaml sets both segmentation/embedding to 32.
    diarization_batch_size: int = 32

    # --- hardware ---
    device: str = "auto"  # auto | cuda | cpu
    model_residency: str = "keep"  # keep both models | release between stages
    vram_margin_mb: int = 1024
    oom_retries: int = 2
    profiling_enabled: bool = False
    profiling_interval: float = 1.0
    optimization_profile: str = "custom"

    # --- downloads ---
    sleep_between_downloads_min: float = 8.0
    sleep_between_downloads_max: float = 15.0
    max_retries: int = 3
    cookies_from_browser: str = ""  # e.g. "chrome" for age-restricted videos
    rate_limit: str = ""  # e.g. "2M" to cap download bandwidth

    # --- output ---
    # Markdown only by default (v0.2); other formats are opt-in via Settings
    export_formats: list[str] = field(default_factory=lambda: ["md"])
    combined_transcript: bool = True
    # optional relabeling applied at export time, e.g. {"SPEAKER_00": "Cliffe"}
    speaker_names: dict[str, str] = field(default_factory=dict)

    # --- transcript cleanup ---
    remove_hallucinations: bool = True
    paragraph_gap_seconds: float = 3.0

    # --- speaker recognition (v0.2) ---
    recognition_enabled: bool = True
    # min cosine similarity for an automatic name match. Calibrated on the
    # target channel: genuine cross-video matches ~0.90, hardest impostor
    # (Stuart vs his father Cliffe's profile) 0.58 (see DESIGN.md §5.1)
    recognition_threshold: float = 0.7

    # bumped when defaults change in a way that needs migration on load
    config_version: int = 3

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
        """Report invalid values without coercing or replacing custom settings."""
        problems = []
        for name, choices in (
            ("cache_policy", CACHE_POLICIES), ("compute_type", COMPUTE_TYPES),
            ("device", ("auto", "cuda", "cpu")),
            ("model_residency", ("keep", "stage")),
            ("optimization_profile", OPTIMIZATION_PROFILES),
        ):
            if getattr(self, name) not in choices:
                problems.append(f"{name} must be one of {choices}")

        def integer(name, minimum, maximum=None):
            value = getattr(self, name)
            valid = (type(value) is int and value >= minimum
                     and (maximum is None or value <= maximum))
            if not valid:
                bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
                problems.append(f"{name} must be an integer {bounds}")
            return valid

        for name in ("beam_size", "asr_batch_size", "asr_cpu_threads",
                     "asr_num_workers", "diarization_batch_size", "config_version"):
            integer(name, 1)
        integer("asr_chunk_length", 1, 30)
        for name in ("vram_margin_mb", "oom_retries", "max_retries"):
            integer(name, 0)
        min_valid = integer("min_speakers", 0)
        max_valid = integer("max_speakers", 0)
        if (min_valid and max_valid and self.min_speakers and self.max_speakers
                and self.min_speakers > self.max_speakers):
            problems.append("min_speakers cannot exceed max_speakers")

        def number(name, minimum, maximum=None, exclusive=False):
            value = getattr(self, name)
            valid = (type(value) in (int, float) and math.isfinite(value)
                     and (value > minimum if exclusive else value >= minimum)
                     and (maximum is None or value <= maximum))
            if not valid:
                problems.append(f"{name} must be a finite number "
                                f"{'>' if exclusive else '>='} {minimum}"
                                + (f" and <= {maximum}" if maximum is not None else ""))
            return valid

        lo_valid = number("sleep_between_downloads_min", 0)
        hi_valid = number("sleep_between_downloads_max", 0)
        if lo_valid and hi_valid and self.sleep_between_downloads_min > self.sleep_between_downloads_max:
            problems.append("sleep_between_downloads_min cannot exceed max")
        number("profiling_interval", 0, exclusive=True)
        number("paragraph_gap_seconds", 0)
        number("recognition_threshold", 0, 1)
        for name in ("vad_filter", "word_timestamps", "profiling_enabled",
                     "combined_transcript", "remove_hallucinations", "recognition_enabled"):
            if type(getattr(self, name)) is not bool:
                problems.append(f"{name} must be true or false")
        for name in ("output_dir", "cache_dir", "whisper_model", "diarization_model", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{name} must be a nonempty string")
        for name in ("ffmpeg_path", "hf_token", "cookies_from_browser", "rate_limit"):
            if not isinstance(getattr(self, name), str):
                problems.append(f"{name} must be a string")
        if not isinstance(self.export_formats, list) or not self.export_formats:
            problems.append("export_formats must be a nonempty list")
        elif any(f not in EXPORT_FORMATS for f in self.export_formats):
            problems.append("unknown export formats")
        if (not isinstance(self.speaker_names, dict)
                or any(not isinstance(k, str) or not isinstance(v, str)
                       for k, v in self.speaker_names.items())):
            problems.append("speaker_names must map strings to strings")
        return problems


def config_file() -> Path:
    return APP_DIR / "config.json"


def load_config() -> Config:
    path = config_file()
    cfg = Config()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cfg
            known = {f.name for f in dataclasses.fields(Config)}
            for key, value in data.items():
                if key in known:
                    setattr(cfg, key, value)
            # v1 -> v2: default output became Markdown-only. Only migrate the
            # untouched v1 default (all five formats); a deliberate subset is
            # a user choice and is preserved.
            version = data.get("config_version", 1)
            if type(version) is int and version < 2:
                if (isinstance(cfg.export_formats, list)
                        and all(isinstance(f, str) for f in cfg.export_formats)
                        and sorted(cfg.export_formats) == sorted(EXPORT_FORMATS)):
                    cfg.export_formats = ["md"]
            if type(version) is int and version < 3:
                # v3 is additive: never apply a preset to migrated inference settings.
                cfg.config_version = 3
                if not cfg.validate():
                    save_config(cfg)
        except (json.JSONDecodeError, OSError):
            # corrupt config falls back to defaults; will be rewritten on save
            pass
    return cfg


def save_config(cfg: Config) -> None:
    problems = cfg.validate()
    if problems:
        raise ValueError("Invalid configuration: " + "; ".join(problems))
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")
    tmp.replace(path)
