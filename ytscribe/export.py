"""Export stage: write merged transcripts as Markdown, JSON, TXT, SRT, VTT.

Per-video files are named "<upload_date> <title> [<video_id>].<ext>" in the
output directory; a combined transcript for the whole batch can be produced
at the end of a queue run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_CUE_CHARS = 90   # subtitle cue budget
MAX_CUE_SECONDS = 6.0


def speaker_label(speaker: str, cfg) -> str:
    return cfg.speaker_names.get(speaker, speaker)


def fmt_ts(seconds: float, force_hours: bool = False) -> str:
    seconds = max(seconds, 0.0)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if (h or force_hours) else f"{m}:{s:02d}"


def safe_filename(meta: dict) -> str:
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", meta.get("title") or "untitled")
    title = re.sub(r"\s+", " ", title).strip()[:120]
    date = meta.get("upload_date") or ""
    vid = meta.get("video_id") or "unknown"
    return f"{date} {title} [{vid}]".strip()


def export_video(merged: dict, meta: dict, cfg, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / safe_filename(meta)
    written = []
    writers = {"md": _write_md, "json": _write_json, "txt": _write_txt,
               "srt": _write_srt, "vtt": _write_vtt}
    for fmt in cfg.export_formats:
        writer = writers.get(fmt)
        if writer:
            # NOT with_suffix(): titles may contain dots, which it would eat
            path = base.parent / f"{base.name}.{fmt}"
            writer(merged, meta, cfg, path)
            written.append(path)
    return written


# --- per-format writers ----------------------------------------------------

def _header_lines(meta: dict) -> list[str]:
    dur = fmt_ts(float(meta.get("duration") or 0))
    return [
        f"# {meta.get('title', 'Untitled')}",
        "",
        f"**Channel:** {meta.get('channel', 'unknown')}  ",
        f"**Published:** {meta.get('upload_date', 'unknown')}  ",
        f"**Duration:** {dur}  ",
        f"**Source:** {meta.get('url', '')}",
        "",
        "---",
        "",
    ]


def _video_markdown_body(merged: dict, meta: dict, cfg) -> list[str]:
    force_h = float(meta.get("duration") or 0) >= 3600
    lines = []
    for turn in merged["turns"]:
        name = speaker_label(turn["speaker"], cfg)
        span = f"{fmt_ts(turn['start'], force_h)}–{fmt_ts(turn['end'], force_h)}"
        lines.append(f"**{name}** ({span}):")
        lines.append("")
        for para in turn["paragraphs"]:
            lines.append(para)
            lines.append("")
    return lines


def _write_md(merged: dict, meta: dict, cfg, path: Path) -> None:
    lines = _header_lines(meta) + _video_markdown_body(merged, meta, cfg)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_txt(merged: dict, meta: dict, cfg, path: Path) -> None:
    force_h = float(meta.get("duration") or 0) >= 3600
    lines = [meta.get("title", ""), meta.get("url", ""),
             f"{meta.get('channel', '')} — {meta.get('upload_date', '')}", ""]
    for turn in merged["turns"]:
        name = speaker_label(turn["speaker"], cfg)
        lines.append(f"[{fmt_ts(turn['start'], force_h)}] {name}:")
        lines.extend(turn["paragraphs"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(merged: dict, meta: dict, cfg, path: Path) -> None:
    doc = {
        "video": meta,
        "language": merged.get("language", ""),
        "speakers": [{"id": s, "label": speaker_label(s, cfg)}
                     for s in merged.get("speakers", [])],
        "turns": [{**t, "speaker_label": speaker_label(t["speaker"], cfg)}
                  for t in merged["turns"]],
    }
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


def _cues(merged: dict, cfg):
    """Split turns into subtitle-sized cues along word boundaries."""
    for turn in merged["turns"]:
        name = speaker_label(turn["speaker"], cfg)
        words = turn["words"]
        if not words:
            continue
        buf, start = [], words[0]["start"]
        chars = 0
        for w in words:
            if buf and (chars + len(w["word"]) > MAX_CUE_CHARS
                        or w["end"] - start > MAX_CUE_SECONDS):
                yield start, buf[-1][1], name, "".join(t for t, _ in buf).strip()
                buf, start, chars = [], w["start"], 0
            buf.append((w["word"], w["end"]))
            chars += len(w["word"])
        if buf:
            yield start, buf[-1][1], name, "".join(t for t, _ in buf).strip()


def _srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(merged: dict, meta: dict, cfg, path: Path) -> None:
    blocks = []
    for i, (start, end, name, text) in enumerate(_cues(merged, cfg), 1):
        blocks.append(f"{i}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{name}: {text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def _write_vtt(merged: dict, meta: dict, cfg, path: Path) -> None:
    lines = ["WEBVTT", ""]
    for start, end, name, text in _cues(merged, cfg):
        ts = f"{_srt_ts(start).replace(',', '.')} --> {_srt_ts(end).replace(',', '.')}"
        lines.append(ts)
        lines.append(f"<v {name}>{text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --- combined batch transcript ----------------------------------------------

def export_combined(items: list[tuple[dict, dict]], cfg, out_dir: Path,
                    batch_name: str) -> list[Path]:
    """items = [(merged, meta), ...] in queue order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    safe_batch = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", batch_name)[:80]

    if "md" in cfg.export_formats:
        lines = [f"# Combined transcript — {batch_name}", "",
                 f"{len(items)} videos", "", "## Contents", ""]
        for i, (_, meta) in enumerate(items, 1):
            lines.append(f"{i}. [{meta.get('title', 'Untitled')}](#video-{i}) "
                         f"({meta.get('upload_date', '')})")
        lines.append("")
        for i, (merged, meta) in enumerate(items, 1):
            lines.append(f'<a id="video-{i}"></a>')
            lines.append("")
            lines.extend(_header_lines(meta))
            lines.extend(_video_markdown_body(merged, meta, cfg))
            lines.append("---")
            lines.append("")
        path = out_dir / f"combined {safe_batch}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        written.append(path)

    if "json" in cfg.export_formats:
        docs = [{"video": meta, "language": merged.get("language", ""),
                 "turns": merged["turns"]} for merged, meta in items]
        path = out_dir / f"combined {safe_batch}.json"
        path.write_text(json.dumps(docs, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        written.append(path)
    return written
