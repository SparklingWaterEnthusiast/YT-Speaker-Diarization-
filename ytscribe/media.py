"""Audio acquisition & metadata via yt-dlp, plus WAV conversion via ffmpeg.

Only audio streams are downloaded (bestaudio, typically ~128 kbps AAC/Opus) —
both ML models resample to 16 kHz mono anyway, so video would be wasted
bandwidth. Each video gets a cache directory cache/<video_id>/ containing:

    audio.<ext>   original downloaded stream (subject to cache policy)
    audio.wav     16 kHz mono PCM used by the ML stages
    meta.json     full metadata snapshot
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import yt_dlp

WATCH_RE = re.compile(r"(?:youtube\.com/(?:watch|shorts|live)|youtu\.be/)")
PLAYLIST_RE = re.compile(r"youtube\.com/playlist\?|[?&]list=")
CHANNEL_RE = re.compile(r"youtube\.com/(?:@[\w.\-]+|channel/|c/|user/)")


class DownloadError(Exception):
    pass


class CancelledError(Exception):
    pass


def classify_url(url: str) -> str:
    """Return 'video' | 'playlist' | 'channel' | 'unknown'."""
    if WATCH_RE.search(url):
        return "video"
    if PLAYLIST_RE.search(url):
        return "playlist"
    if CHANNEL_RE.search(url):
        return "channel"
    return "unknown"


def _base_opts(cfg) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 5,
        "socket_timeout": 30,
    }
    if cfg.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.cookies_from_browser,)
    if cfg.rate_limit:
        opts["ratelimit"] = _parse_rate(cfg.rate_limit)
    return opts


def _parse_rate(text: str) -> int | None:
    m = re.fullmatch(r"([\d.]+)\s*([KMG]?)", text.strip(), re.IGNORECASE)
    if not m:
        return None
    mult = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def enumerate_videos(url: str, cfg, log: Callable[[str], None] = print) -> list[dict]:
    """Expand a URL into a flat list of {video_id, url, title, duration} dicts.

    Single videos return a one-element list. Channels are expanded through
    their /videos tab. Unavailable entries (private/deleted) are skipped.
    """
    kind = classify_url(url)
    if kind == "channel" and not re.search(r"/(videos|streams|shorts|playlists)([/?]|$)", url):
        url = url.rstrip("/") + "/videos"

    opts = _base_opts(cfg) | {"extract_flat": "in_playlist", "ignoreerrors": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        raise DownloadError(f"Could not resolve URL: {url}")

    entries = []
    if info.get("_type") == "playlist":
        for e in info.get("entries") or []:
            if not e or not e.get("id"):
                continue
            entries.append({
                "video_id": e["id"],
                "url": e.get("url") or f"https://www.youtube.com/watch?v={e['id']}",
                "title": e.get("title") or "",
                "duration": float(e.get("duration") or 0),
                "channel": e.get("channel") or info.get("channel") or info.get("uploader") or "",
            })
        log(f"Resolved {kind}: '{info.get('title', url)}' -> {len(entries)} videos")
    else:
        entries.append({
            "video_id": info["id"],
            "url": info.get("webpage_url") or url,
            "title": info.get("title") or "",
            "duration": float(info.get("duration") or 0),
            "channel": info.get("channel") or info.get("uploader") or "",
        })
    return entries


def download_audio(url: str, video_dir: Path, cfg,
                   progress_cb: Callable[[float, str], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> Path:
    """Download the best audio-only stream into video_dir/audio.<ext>.

    Saves meta.json alongside. Returns the audio file path.
    Raises CancelledError if the cancel flag fires mid-download.
    """
    video_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_audio(video_dir)
    if existing and (video_dir / "meta.json").exists():
        return existing  # cache hit — never repeat completed work

    def hook(d):
        if cancelled and cancelled():
            raise CancelledError("download cancelled")
        if progress_cb and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                speed = d.get("speed")
                label = f"{done / 2**20:.1f}/{total / 2**20:.1f} MiB"
                if speed:
                    label += f" @ {speed / 2**20:.2f} MiB/s"
                progress_cb(done / total, label)

    opts = _base_opts(cfg) | {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(video_dir / "audio.%(ext)s"),
        "progress_hooks": [hook],
        "continuedl": True,          # resume partial downloads
        "overwrites": False,
        "concurrent_fragment_downloads": 4,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except CancelledError:
        raise
    except Exception as exc:  # yt-dlp raises many exception types
        raise DownloadError(str(exc)) from exc

    _write_meta(video_dir, info)
    audio = _find_audio(video_dir)
    if not audio:
        raise DownloadError("yt-dlp reported success but no audio file was produced")
    return audio


def _find_audio(video_dir: Path) -> Path | None:
    for ext in ("m4a", "webm", "opus", "mp4", "mp3", "ogg", "aac"):
        p = video_dir / f"audio.{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def _write_meta(video_dir: Path, info: dict) -> None:
    meta = {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "url": info.get("webpage_url"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": _iso_date(info.get("upload_date")),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "description": (info.get("description") or "")[:5000],
        "chapters": info.get("chapters") or [],
    }
    (video_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _iso_date(yyyymmdd: str | None) -> str:
    if yyyymmdd and len(yyyymmdd) == 8:
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    return yyyymmdd or ""


def load_meta(video_dir: Path) -> dict:
    p = video_dir / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def to_wav(audio_path: Path, ffmpeg: str) -> Path:
    """Convert to 16 kHz mono PCM WAV (what both ML models consume)."""
    wav = audio_path.parent / "audio.wav"
    if wav.exists() and wav.stat().st_size > 44:
        return wav
    tmp = wav.with_suffix(".tmp.wav")
    cmd = [ffmpeg, "-y", "-i", str(audio_path), "-vn",
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp)]
    creation = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=creation)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        # last non-empty stderr lines carry the actual error, not the build banner
        lines = [l for l in (proc.stderr or "").splitlines() if l.strip()]
        raise DownloadError("ffmpeg conversion failed: " + " | ".join(lines[-3:]))
    tmp.replace(wav)
    return wav


def wav_duration(wav: Path) -> float:
    """Duration in seconds from the WAV header (16-bit mono 16 kHz)."""
    import wave
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / w.getframerate()
