"""Unit tests for speaker identification (voices.py) — no network, no GPU."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from ytscribe import voices
from ytscribe.config import Config
from ytscribe.voices import VoiceDB


def vec(seed: int, dim: int = 16) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


class TestVoiceDB(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.db_path = Path(self.td.name) / "voices.sqlite3"
        self.db = VoiceDB(self.db_path)

    def tearDown(self):
        self.db.close()
        self.td.cleanup()

    def test_profile_creation_and_fields(self):
        self.db.add_sample("Cliffe Knechtle", vec(1), "vid1:SPEAKER_00")
        profiles = self.db.profiles()
        self.assertEqual(len(profiles), 1)
        p = profiles[0]
        self.assertEqual(p["name"], "Cliffe Knechtle")
        self.assertEqual(p["confirmed_samples"], 1)
        self.assertGreater(p["created_at"], 0)
        self.assertGreaterEqual(p["updated_at"], p["created_at"])

    def test_persistence_across_reopen(self):
        self.db.add_sample("Stuart Knechtle", vec(2), "vid1:SPEAKER_01")
        self.db.close()
        reopened = VoiceDB(self.db_path)
        try:
            self.assertEqual(reopened.profiles()[0]["name"], "Stuart Knechtle")
            self.assertIsNotNone(reopened.match(vec(2), threshold=0.99))
        finally:
            reopened.close()

    def test_same_source_replaces_not_duplicates(self):
        self.db.add_sample("Cliffe", vec(1), "vid1:SPEAKER_00")
        self.db.add_sample("Cliffe", vec(3), "vid1:SPEAKER_00")
        self.assertEqual(self.db.profiles()[0]["confirmed_samples"], 1)
        self.db.add_sample("Cliffe", vec(4), "vid2:SPEAKER_02")
        self.assertEqual(self.db.profiles()[0]["confirmed_samples"], 2)

    def test_match_picks_best_profile(self):
        self.db.add_sample("Cliffe", vec(1), "a")
        self.db.add_sample("Stuart", vec(2), "b")
        hit = self.db.match(vec(1), threshold=0.9)
        self.assertEqual(hit[0], "Cliffe")
        self.assertGreater(hit[1], 0.99)

    def test_no_match_below_threshold(self):
        self.db.add_sample("Cliffe", vec(1), "a")
        self.assertIsNone(self.db.match(vec(99), threshold=0.9))

    def test_rejects_bad_embeddings(self):
        with self.assertRaises(ValueError):
            self.db.add_sample("X", [0.0] * 16, "s")
        with self.assertRaises(ValueError):
            self.db.add_sample("  ", vec(1), "s")
        self.assertIsNone(self.db.match([0.0] * 16, 0.5))

    def test_dimension_mismatch_skipped(self):
        self.db.add_sample("Old", vec(1, dim=8), "old-model")
        self.assertIsNone(self.db.match(vec(1, dim=16), threshold=0.1))

    def test_delete_profile(self):
        self.db.add_sample("X", vec(1), "s")
        self.assertTrue(self.db.delete_profile("X"))
        self.assertEqual(self.db.profiles(), [])


class TestAutoMatch(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.vdir = Path(self.td.name)
        self.db = VoiceDB(self.vdir / "voices.sqlite3")

    def tearDown(self):
        self.db.close()
        self.td.cleanup()

    def test_match_writes_name_unknown_keeps_label(self):
        self.db.add_sample("Cliffe Knechtle", vec(1), "seed")
        diar = {"embeddings": {"SPEAKER_00": vec(1), "SPEAKER_01": vec(50)}}
        mapping = voices.auto_match(self.vdir, diar, self.db, 0.9, lambda m: None)
        self.assertEqual(mapping["SPEAKER_00"]["name"], "Cliffe Knechtle")
        self.assertEqual(mapping["SPEAKER_00"]["source"], "auto")
        self.assertNotIn("SPEAKER_01", mapping)  # unknown: label untouched

    def test_manual_entry_never_overwritten(self):
        voices.save_speaker_map(self.vdir, {"SPEAKER_00": {
            "name": "Manually Set", "source": "manual"}})
        self.db.add_sample("Somebody Else", vec(1), "seed")
        mapping = voices.auto_match(self.vdir, {"embeddings": {
            "SPEAKER_00": vec(1)}}, self.db, 0.5, lambda m: None)
        self.assertEqual(mapping["SPEAKER_00"]["name"], "Manually Set")

    def test_stale_auto_match_removed(self):
        voices.save_speaker_map(self.vdir, {"SPEAKER_00": {
            "name": "Ghost", "source": "auto", "score": 0.7}})
        mapping = voices.auto_match(self.vdir, {"embeddings": {
            "SPEAKER_00": vec(1)}}, self.db, 0.5, lambda m: None)
        self.assertNotIn("SPEAKER_00", mapping)

    def test_display_names_resolution_order(self):
        cfg = Config()
        cfg.speaker_names = {"SPEAKER_00": "FromConfig", "SPEAKER_01": "AlsoConfig"}
        voices.save_speaker_map(self.vdir, {"SPEAKER_00": {
            "name": "FromVideo", "source": "manual"}})
        names = voices.display_names(cfg, self.vdir,
                                     ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
        self.assertEqual(names, {"SPEAKER_00": "FromVideo",
                                 "SPEAKER_01": "AlsoConfig",
                                 "SPEAKER_02": "SPEAKER_02"})


class TestSamples(unittest.TestCase):
    def _make_wav(self, path: Path, seconds: float, rate: int = 16000):
        t = np.arange(int(seconds * rate))
        data = (np.sin(2 * math.pi * 440 * t / rate) * 20000).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data.tobytes())

    def test_extract_capped_at_five_seconds(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio.wav"
            self._make_wav(wav, 30.0)
            diar = {"exclusive": [
                {"start": 1.0, "end": 21.0, "speaker": "SPEAKER_00"},
                {"start": 22.0, "end": 24.0, "speaker": "SPEAKER_01"},
                {"start": 24.0, "end": 24.3, "speaker": "SPEAKER_02"},  # blip
            ]}
            n = voices.extract_samples(wav, diar, Path(td) / "samples")
            self.assertEqual(n, 2)
            with wave.open(str(Path(td) / "samples" / "SPEAKER_00.wav")) as w:
                dur = w.getnframes() / w.getframerate()
            self.assertLessEqual(dur, 5.01)
            self.assertGreater(dur, 4.5)
            with wave.open(str(Path(td) / "samples" / "SPEAKER_01.wav")) as w:
                dur = w.getnframes() / w.getframerate()
            self.assertAlmostEqual(dur, 2.0, delta=0.1)
            # blip speaker skipped entirely
            self.assertFalse((Path(td) / "samples" / "SPEAKER_02.wav").exists())


class TestRename(unittest.TestCase):
    def test_rename_updates_map_db_and_transcripts(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            vdir = td / "cache" / "vid01"
            vdir.mkdir(parents=True)
            cfg = Config()
            cfg.output_dir = str(td / "out")
            cfg.cache_dir = str(td / "cache")
            meta = {"video_id": "vid01", "title": "T", "url": "u",
                    "channel": "c", "upload_date": "2026-07-07", "duration": 10.0}
            merged = {"language": "en", "speakers": ["SPEAKER_00"],
                      "turns": [{"speaker": "SPEAKER_00", "start": 0, "end": 2,
                                 "paragraphs": ["Hello."],
                                 "words": [{"start": 0, "end": 2,
                                            "word": " Hello.", "probability": 1}]}]}
            (vdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (vdir / "merged.json").write_text(json.dumps(merged), encoding="utf-8")
            (vdir / "diarization.json").write_text(json.dumps(
                {"embeddings": {"SPEAKER_00": vec(7)}}), encoding="utf-8")

            db = VoiceDB(td / "voices.sqlite3")
            try:
                files = voices.apply_rename(cfg, vdir, "SPEAKER_00",
                                            "Cliffe Knechtle", db,
                                            lambda m: None)
                # transcript rewritten with the new name
                md = [f for f in files if f.suffix == ".md"][0]
                text = md.read_text(encoding="utf-8")
                self.assertIn("**Cliffe Knechtle**", text)
                self.assertNotIn("SPEAKER_00", text)
                # speaker map updated as manual
                mapping = voices.load_speaker_map(vdir)
                self.assertEqual(mapping["SPEAKER_00"]["source"], "manual")
                # voice DB gained a confirmed sample
                self.assertEqual(db.profiles()[0]["confirmed_samples"], 1)
                # future recognition works from it
                self.assertEqual(db.match(vec(7), 0.9)[0], "Cliffe Knechtle")
            finally:
                db.close()


class TestReferenceAlignment(unittest.TestCase):
    def test_parse_reference(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.txt"
            ref.write_text(
                "junk header\n"
                "Cliffe Knechtle (0:08-0:24): Good. There can be.\n"
                "speaker_4 (0:26-0:26): I mean\n"
                "Stuart Knechtle (19:36-19:38): Well yeah.\n",
                encoding="utf-8")
            turns = voices.parse_reference(ref)
            self.assertEqual(len(turns), 3)
            self.assertEqual(turns[0]["speaker"], "Cliffe Knechtle")
            self.assertEqual(turns[2]["start"], 19 * 60 + 36)

    def test_alignment_ignores_placeholders_and_weak_overlap(self):
        ref = [
            {"speaker": "Cliffe Knechtle", "start": 0, "end": 100},
            {"speaker": "Stuart Knechtle", "start": 100, "end": 200},
            {"speaker": "speaker_4", "start": 200, "end": 300},
        ]
        diar = {"exclusive": [
            {"speaker": "SPEAKER_00", "start": 0, "end": 95},      # Cliffe
            {"speaker": "SPEAKER_01", "start": 105, "end": 190},   # Stuart
            {"speaker": "SPEAKER_02", "start": 210, "end": 290},   # placeholder
            {"speaker": "SPEAKER_03", "start": 98, "end": 103},    # too little
        ]}
        result = voices.align_to_reference(diar, ref)
        self.assertEqual(result["SPEAKER_00"]["name"], "Cliffe Knechtle")
        self.assertEqual(result["SPEAKER_01"]["name"], "Stuart Knechtle")
        self.assertNotIn("SPEAKER_02", result)  # placeholder ignored
        self.assertNotIn("SPEAKER_03", result)  # < 10 s overlap


if __name__ == "__main__":
    unittest.main()
