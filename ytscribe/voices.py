"""Speaker identification: persistent voice database, matching, samples.

See DESIGN.md §5. Embeddings are the 256-dim WeSpeaker centroids the
diarization pipeline already produces — recognition adds no models and no
GPU passes. Only *confirmed* embeddings (manual rename / reference seeding)
are stored, so automatic matches can never poison a profile.

Per-video artifacts managed here:
    cache/<id>/speakers.json      label -> {"name", "source", "score"}
    cache/<id>/samples/<label>.wav  <=5 s playback sample per speaker
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_SECONDS = 5.0
MIN_TURN_SECONDS = 1.0  # don't build playback samples from blips

_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(speaker_id, source)
);
"""


class VoiceDB:
    """Persistent speaker profiles: name + confirmed voice embeddings."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def add_sample(self, name: str, embedding: list[float] | np.ndarray,
                   source: str) -> None:
        """Add (or refresh) one confirmed embedding for a speaker profile.

        `source` identifies where it came from ("video_id:SPEAKER_XX");
        re-confirming the same source replaces instead of duplicating.
        """
        name = name.strip()
        if not name:
            raise ValueError("speaker name must not be empty")
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim != 1 or not np.isfinite(vec).all() or np.linalg.norm(vec) == 0:
            raise ValueError("invalid embedding")
        now = time.time()
        with self._lock:
            cur = self._conn.execute("SELECT id FROM speakers WHERE name=?", (name,))
            row = cur.fetchone()
            if row:
                sid = row["id"]
                self._conn.execute("UPDATE speakers SET updated_at=? WHERE id=?",
                                   (now, sid))
            else:
                sid = self._conn.execute(
                    "INSERT INTO speakers (name, created_at, updated_at) VALUES (?,?,?)",
                    (name, now, now)).lastrowid
            self._conn.execute(
                "INSERT INTO samples (speaker_id, embedding, dim, source, created_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(speaker_id, source) "
                "DO UPDATE SET embedding=excluded.embedding, created_at=excluded.created_at",
                (sid, vec.tobytes(), vec.shape[0], source, now))
            self._conn.commit()

    def profiles(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.name, s.created_at, s.updated_at, COUNT(m.id) AS n "
                "FROM speakers s LEFT JOIN samples m ON m.speaker_id = s.id "
                "GROUP BY s.id ORDER BY s.name").fetchall()
        return [{"name": r["name"], "created_at": r["created_at"],
                 "updated_at": r["updated_at"], "confirmed_samples": r["n"]}
                for r in rows]

    def delete_profile(self, name: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM speakers WHERE name=?", (name,))
            self._conn.commit()
            return cur.rowcount > 0

    def match(self, embedding, threshold: float) -> tuple[str, float] | None:
        """Best-matching profile for an embedding, or None below threshold.

        Score = max cosine similarity over the profile's confirmed samples.
        """
        vec = _normalize(np.asarray(embedding, dtype=np.float32))
        if vec is None:
            return None
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.name, m.embedding, m.dim FROM samples m "
                "JOIN speakers s ON s.id = m.speaker_id").fetchall()
        best_name, best_score = None, -1.0
        for r in rows:
            stored = np.frombuffer(r["embedding"], dtype=np.float32)
            if stored.shape[0] != vec.shape[0]:
                continue  # embedding from a different/older model
            stored = _normalize(stored)
            if stored is None:
                continue
            score = float(np.dot(vec, stored))
            if score > best_score:
                best_name, best_score = r["name"], score
        if best_name is not None and best_score >= threshold:
            return best_name, best_score
        return None

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm == 0 or not np.isfinite(norm):
        return None
    return vec / norm


# --- per-video speaker-name map (speakers.json) -----------------------------

def load_speaker_map(video_dir: Path) -> dict:
    p = video_dir / "speakers.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_speaker_map(video_dir: Path, mapping: dict) -> None:
    p = video_dir / "speakers.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(mapping, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(p)


def display_names(cfg, video_dir: Path, labels: list[str]) -> dict[str, str]:
    """Resolve every label: speakers.json -> config speaker_names -> label."""
    mapping = load_speaker_map(video_dir)
    out = {}
    for label in labels:
        entry = mapping.get(label)
        if entry and entry.get("name"):
            out[label] = entry["name"]
        else:
            out[label] = cfg.speaker_names.get(label, label)
    return out


def auto_match(video_dir: Path, diar: dict, vdb: VoiceDB, threshold: float,
               log=print) -> dict:
    """Match detected speakers against the voice DB; write speakers.json.

    Never overwrites manual entries; unmatched speakers keep their labels
    (no entry written — names are never invented).
    """
    mapping = load_speaker_map(video_dir)
    embeddings = diar.get("embeddings") or {}
    for label, vec in embeddings.items():
        if vec is None:
            continue
        existing = mapping.get(label)
        if existing and existing.get("source") == "manual":
            continue
        hit = vdb.match(vec, threshold)
        if hit:
            name, score = hit
            mapping[label] = {"name": name, "source": "auto",
                              "score": round(score, 4)}
            log(f"Recognized {label} as '{name}' (similarity {score:.2f})")
        elif existing and existing.get("source") == "auto":
            del mapping[label]  # stale auto-match from an earlier DB state
    save_speaker_map(video_dir, mapping)
    return mapping


def apply_rename(cfg, video_dir: Path, label: str, new_name: str,
                 vdb: VoiceDB | None, log=print,
                 out_dirs: list[Path] | None = None) -> list[Path]:
    """Rename one speaker of a completed video (see DESIGN.md §5.3).

    Updates speakers.json, stores the confirmed embedding in the voice DB,
    and re-exports every transcript format configured or already on disk —
    into every output folder the video was exported to (`out_dirs`; queues
    have their own subfolders since v0.2.1; defaults to the output root).
    Returns the rewritten files.
    """
    from . import export  # local import: export is a leaf module

    new_name = new_name.strip()
    if not new_name:
        return []
    mapping = load_speaker_map(video_dir)
    mapping[label] = {"name": new_name, "source": "manual"}
    save_speaker_map(video_dir, mapping)

    diar_file = video_dir / "diarization.json"
    if vdb is not None and diar_file.exists():
        diar = json.loads(diar_file.read_text(encoding="utf-8"))
        vec = (diar.get("embeddings") or {}).get(label)
        if vec:
            vdb.add_sample(new_name, vec, source=f"{video_dir.name}:{label}")
            log(f"Voice profile '{new_name}' updated "
                f"(+1 confirmed sample from {video_dir.name}).")
        else:
            log(f"NOTE: no voice embedding stored for {label} in this video "
                "(processed before v0.2?) — name applied to transcripts only.")

    merged_file = video_dir / "merged.json"
    meta_file = video_dir / "meta.json"
    if not (merged_file.exists() and meta_file.exists()):
        log("Transcript artifacts missing; nothing to re-export.")
        return []
    merged = json.loads(merged_file.read_text(encoding="utf-8"))
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    names = display_names(cfg, video_dir, merged.get("speakers", []))
    files: list[Path] = []
    for out_dir in out_dirs or [Path(cfg.output_dir)]:
        files += export.export_video(
            merged, meta, cfg, out_dir, names=names,
            extra_formats=export.existing_formats(meta, out_dir))
    log(f"Re-exported {len(files)} file(s) with '{label}' -> '{new_name}'.")
    return files


# --- playback samples --------------------------------------------------------

def extract_samples(wav_path: Path, diar: dict, samples_dir: Path) -> int:
    """Save one <=5 s WAV per speaker (center of their longest clean turn).

    Runs while audio.wav still exists (right after diarization) so the
    speaker editor can play voices even after the cache policy deletes audio.
    """
    turns = diar.get("exclusive") or diar.get("turns") or []
    best: dict[str, dict] = {}
    for t in turns:
        dur = t["end"] - t["start"]
        if dur < MIN_TURN_SECONDS:
            continue
        cur = best.get(t["speaker"])
        if cur is None or dur > cur["end"] - cur["start"]:
            best[t["speaker"]] = t
    if not best:
        return 0
    samples_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with wave.open(str(wav_path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        total = w.getnframes()
        for speaker, turn in best.items():
            out = samples_dir / f"{speaker}.wav"
            if out.exists():
                written += 1
                continue
            dur = min(turn["end"] - turn["start"], SAMPLE_SECONDS)
            mid = (turn["start"] + turn["end"]) / 2
            start = max(turn["start"], mid - dur / 2)
            first = min(int(start * rate), max(total - 1, 0))
            count = min(int(dur * rate), total - first)
            if count <= 0:
                continue
            w.setpos(first)
            frames = w.readframes(count)
            with wave.open(str(out), "wb") as o:
                o.setnchannels(channels)
                o.setsampwidth(width)
                o.setframerate(rate)
                o.writeframes(frames)
            written += 1
    return written


def sample_path(video_dir: Path, label: str) -> Path | None:
    p = video_dir / "samples" / f"{label}.wav"
    return p if p.exists() else None


# --- reference-transcript seeding (DESIGN.md §5.4) ---------------------------

REF_LINE = re.compile(r"^(.+?)\s*\((\d+):(\d{2})-(\d+):(\d{2})\):")
PLACEHOLDER = re.compile(r"^speaker[_ ]?\d+$", re.IGNORECASE)

MIN_OVERLAP_SECONDS = 10.0
MIN_PURITY = 0.6


def parse_reference(path: Path) -> list[dict]:
    """Parse 'Name (M:SS-M:SS): text' lines into turns."""
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = REF_LINE.match(line.strip())
        if not m:
            continue
        name = m.group(1).strip()
        start = int(m.group(2)) * 60 + int(m.group(3))
        end = int(m.group(4)) * 60 + int(m.group(5))
        if end >= start:
            turns.append({"speaker": name, "start": float(start), "end": float(end)})
    return turns


def align_to_reference(diar: dict, ref_turns: list[dict]) -> dict[str, dict]:
    """Map detected labels to real reference names by temporal overlap.

    Returns {label: {"name", "overlap", "purity"}} for confident alignments
    only. Reference placeholders (speaker_2 ...) are ignored.
    """
    our_turns = diar.get("exclusive") or diar.get("turns") or []
    overlap: dict[str, dict[str, float]] = {}
    for ot in our_turns:
        for rt in ref_turns:
            ov = min(ot["end"], rt["end"]) - max(ot["start"], rt["start"])
            if ov > 0:
                overlap.setdefault(ot["speaker"], {})
                overlap[ot["speaker"]][rt["speaker"]] = (
                    overlap[ot["speaker"]].get(rt["speaker"], 0.0) + ov)
    result = {}
    for label, by_ref in overlap.items():
        total = sum(by_ref.values())
        ref_name, ov = max(by_ref.items(), key=lambda kv: kv[1])
        if PLACEHOLDER.match(ref_name):
            continue
        purity = ov / total if total else 0.0
        if ov >= MIN_OVERLAP_SECONDS and purity >= MIN_PURITY:
            result[label] = {"name": ref_name, "overlap": round(ov, 1),
                             "purity": round(purity, 3)}
    return result
