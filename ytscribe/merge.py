"""Merge stage: attach a speaker to every transcribed word, build turns.

Strategy (see DESIGN.md §2.4):
  1. Flatten ASR segments into words (segments lacking word timings become a
     single pseudo-word so nothing is lost).
  2. Optionally drop Whisper hallucinations: segments that match known spam
     phrases AND fall in diarized silence / high no-speech probability.
  3. Assign each word to the exclusive-diarization turn with maximal temporal
     overlap; words in gaps snap to the nearest turn within TOLERANCE, else
     inherit their neighbor's speaker. Word-level (not segment-level)
     assignment is what keeps rapid interjections on the correct speaker.
  4. Group consecutive same-speaker words into turns; split long turns into
     paragraphs at sentence boundaries following long pauses.

Output artifact (merged.json):
    {"language", "speakers", "turns": [{"speaker", "start", "end",
                                        "paragraphs": [str], "words": [...]}]}
"""

from __future__ import annotations

import bisect
import json
import re
from pathlib import Path

TOLERANCE = 2.0          # s: max snap distance for words outside any turn
MAX_PARAGRAPH_CHARS = 600
SENTENCE_END = re.compile(r"[.!?…][\"')\]]?$")

HALLUCINATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"^\s*thanks? for watching\W*$",
        r"^\s*please (like and )?subscribe\W*$",
        r"^\s*subtitles? by\b",
        r"^\s*transcri(bed|ption) by\b",
        r"^\s*copyright\b.{0,40}$",
        r"^\s*\.{2,}\s*$",
    )
]


def merge(asr: dict, diar: dict, cfg) -> dict:
    exclusive = sorted(diar.get("exclusive") or diar.get("turns") or [],
                       key=lambda t: t["start"])
    words = _flatten_words(asr, cfg, exclusive)
    if not words:
        return {"language": asr.get("language", ""), "speakers": [], "turns": []}

    starts = [t["start"] for t in exclusive]
    for w in words:
        w["speaker"] = _assign(w, exclusive, starts)
    _fill_unknowns(words)

    turns = _build_turns(words, cfg)
    speakers = sorted({t["speaker"] for t in turns})
    return {"language": asr.get("language", ""), "speakers": speakers, "turns": turns}


def run_stage(asr: dict, diar: dict, out_file: Path, cfg) -> dict:
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))
    result = merge(asr, diar, cfg)
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_file)
    return result


# --------------------------------------------------------------------------

def _flatten_words(asr: dict, cfg, exclusive: list[dict]) -> list[dict]:
    words: list[dict] = []
    for seg in asr.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if cfg.remove_hallucinations and _is_hallucination(seg, text, exclusive):
            continue
        if seg.get("words"):
            for w in seg["words"]:
                token = w.get("word", "")
                if token.strip():
                    words.append({"start": w["start"], "end": w["end"],
                                  "word": token,
                                  "probability": w.get("probability", 1.0)})
        else:  # no word timings — keep the whole segment as one unit
            words.append({"start": seg["start"], "end": seg["end"],
                          "word": " " + text, "probability": 1.0})
    words.sort(key=lambda w: w["start"])
    return words


def _is_hallucination(seg: dict, text: str, exclusive: list[dict]) -> bool:
    if not any(p.search(text) for p in HALLUCINATION_PATTERNS):
        return False
    if seg.get("no_speech_prob", 0.0) > 0.5:
        return True
    # spam phrase entirely outside diarized speech -> hallucinated in silence
    return _overlap_with_timeline(seg["start"], seg["end"], exclusive) < 0.2


def _overlap_with_timeline(start: float, end: float, timeline: list[dict]) -> float:
    total = 0.0
    for t in timeline:
        if t["end"] <= start:
            continue
        if t["start"] >= end:
            break
        total += min(end, t["end"]) - max(start, t["start"])
    return total / max(end - start, 1e-6)


def _assign(word: dict, exclusive: list[dict], starts: list[float]) -> str | None:
    """Speaker whose exclusive turn maximally overlaps the word."""
    if not exclusive:
        return "SPEAKER_00"
    ws, we = word["start"], word["end"]
    idx = bisect.bisect_right(starts, ws) - 1
    best, best_ov = None, 0.0
    for i in range(max(idx, 0), len(exclusive)):
        t = exclusive[i]
        if t["start"] >= we and best is not None:
            break
        ov = min(we, t["end"]) - max(ws, t["start"])
        if ov > best_ov:
            best, best_ov = t["speaker"], ov
        if t["start"] >= we:
            break
    if best:
        return best
    # word in a diarization gap: snap to nearest turn within tolerance
    mid = (ws + we) / 2
    nearest, dist = None, TOLERANCE
    for i in (max(idx, 0), min(idx + 1, len(exclusive) - 1)):
        t = exclusive[i]
        d = max(t["start"] - mid, mid - t["end"], 0.0)
        if d < dist:
            nearest, dist = t["speaker"], d
    return nearest


def _fill_unknowns(words: list[dict]) -> None:
    """Words too far from any turn inherit the previous (else next) speaker."""
    last = None
    for w in words:
        if w["speaker"] is None:
            w["speaker"] = last
        else:
            last = w["speaker"]
    nxt = None
    for w in reversed(words):
        if w["speaker"] is None:
            w["speaker"] = nxt or "SPEAKER_00"
        else:
            nxt = w["speaker"]


def _build_turns(words: list[dict], cfg) -> list[dict]:
    turns: list[dict] = []
    current: list[dict] = []
    for w in words:
        if current and w["speaker"] != current[-1]["speaker"]:
            turns.append(_finalize_turn(current, cfg))
            current = []
        current.append(w)
    if current:
        turns.append(_finalize_turn(current, cfg))
    return turns


def _finalize_turn(words: list[dict], cfg) -> dict:
    paragraphs: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    prev_end = None
    for w in words:
        token = w["word"]
        long_pause = (prev_end is not None
                      and w["start"] - prev_end >= cfg.paragraph_gap_seconds)
        at_sentence = buf and SENTENCE_END.search(buf[-1].strip())
        if buf and at_sentence and (long_pause or buf_chars > MAX_PARAGRAPH_CHARS):
            paragraphs.append(_clean("".join(buf)))
            buf, buf_chars = [], 0
        buf.append(token)
        buf_chars += len(token)
        prev_end = w["end"]
    if buf:
        paragraphs.append(_clean("".join(buf)))
    return {
        "speaker": words[0]["speaker"],
        "start": round(words[0]["start"], 3),
        "end": round(words[-1]["end"], 3),
        "paragraphs": [p for p in paragraphs if p],
        "words": [{"start": w["start"], "end": w["end"], "word": w["word"],
                   "probability": w["probability"]} for w in words],
    }


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
