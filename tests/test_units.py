"""Unit tests for the pure-logic modules (no network, no GPU).

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
  or: .venv\\Scripts\\python.exe -m unittest discover tests
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ytscribe import export, media, merge
from ytscribe.config import Config
from ytscribe.db import JobStore


def make_asr(segments):
    return {"language": "en", "language_probability": 0.99, "segments": segments}


def seg(start, end, text, words=None, nsp=0.01):
    if words is None:
        # evenly spread words across the segment
        toks = text.split()
        step = (end - start) / max(len(toks), 1)
        words = [{"start": start + i * step, "end": start + (i + 1) * step,
                  "word": " " + t, "probability": 0.9}
                 for i, t in enumerate(toks)]
    return {"start": start, "end": end, "text": text,
            "no_speech_prob": nsp, "words": words}


class TestClassifyUrl(unittest.TestCase):
    def test_video(self):
        self.assertEqual(media.classify_url(
            "https://www.youtube.com/watch?v=fZZXVNt1gk0"), "video")
        self.assertEqual(media.classify_url("https://youtu.be/fZZXVNt1gk0"), "video")

    def test_playlist(self):
        self.assertEqual(media.classify_url(
            "https://www.youtube.com/playlist?list=PL123"), "playlist")

    def test_channel(self):
        self.assertEqual(media.classify_url(
            "https://www.youtube.com/@givemeananswer"), "channel")
        self.assertEqual(media.classify_url(
            "https://www.youtube.com/channel/UCxyz"), "channel")

    def test_watch_with_list_is_video(self):
        # a watch URL inside a playlist should process the single video
        self.assertEqual(media.classify_url(
            "https://www.youtube.com/watch?v=abc&list=PL1"), "video")

    def test_unknown(self):
        self.assertEqual(media.classify_url("https://example.com/x"), "unknown")


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_word_level_speaker_switch(self):
        """A single ASR segment spanning two speakers splits at the boundary."""
        asr = make_asr([seg(0.0, 4.0, "hello there yes indeed")])
        diar = {"speakers": ["SPEAKER_00", "SPEAKER_01"],
                "turns": [{"start": 0, "end": 2, "speaker": "SPEAKER_00"},
                          {"start": 2, "end": 4, "speaker": "SPEAKER_01"}],
                "exclusive": [{"start": 0, "end": 2, "speaker": "SPEAKER_00"},
                              {"start": 2, "end": 4, "speaker": "SPEAKER_01"}]}
        out = merge.merge(asr, diar, self.cfg)
        self.assertEqual(len(out["turns"]), 2)
        self.assertEqual(out["turns"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(out["turns"][0]["paragraphs"], ["hello there"])
        self.assertEqual(out["turns"][1]["paragraphs"], ["yes indeed"])

    def test_gap_word_snaps_to_nearest(self):
        asr = make_asr([seg(0.0, 1.0, "hi"), seg(5.0, 5.4, "ok")])
        diar = {"speakers": ["SPEAKER_00"],
                "turns": [{"start": 0, "end": 1.2, "speaker": "SPEAKER_00"}],
                "exclusive": [{"start": 0, "end": 1.2, "speaker": "SPEAKER_00"}]}
        out = merge.merge(asr, diar, self.cfg)
        # both words end up with the only speaker (second via inherit)
        self.assertTrue(all(t["speaker"] == "SPEAKER_00" for t in out["turns"]))

    def test_hallucination_removed_in_silence(self):
        asr = make_asr([
            seg(0.0, 2.0, "real speech here"),
            seg(50.0, 52.0, "Thanks for watching!", nsp=0.9),
        ])
        diar = {"speakers": ["SPEAKER_00"],
                "turns": [{"start": 0, "end": 2.2, "speaker": "SPEAKER_00"}],
                "exclusive": [{"start": 0, "end": 2.2, "speaker": "SPEAKER_00"}]}
        out = merge.merge(asr, diar, self.cfg)
        text = " ".join(p for t in out["turns"] for p in t["paragraphs"])
        self.assertNotIn("Thanks for watching", text)
        self.assertIn("real speech", text)

    def test_hallucination_kept_when_actually_spoken(self):
        """The same phrase during diarized speech must NOT be removed."""
        asr = make_asr([seg(0.0, 2.0, "Thanks for watching!", nsp=0.01)])
        diar = {"speakers": ["SPEAKER_00"],
                "turns": [{"start": 0, "end": 2.0, "speaker": "SPEAKER_00"}],
                "exclusive": [{"start": 0, "end": 2.0, "speaker": "SPEAKER_00"}]}
        out = merge.merge(asr, diar, self.cfg)
        self.assertIn("Thanks for watching", out["turns"][0]["paragraphs"][0])

    def test_empty_diarization_defaults_speaker(self):
        asr = make_asr([seg(0.0, 1.0, "hello")])
        out = merge.merge(asr, {"speakers": [], "turns": [], "exclusive": []}, self.cfg)
        self.assertEqual(out["turns"][0]["speaker"], "SPEAKER_00")

    def test_segment_without_words_survives(self):
        asr = make_asr([{"start": 0, "end": 2, "text": "no word timings",
                         "no_speech_prob": 0.0, "words": []}])
        diar = {"speakers": ["SPEAKER_01"],
                "turns": [{"start": 0, "end": 2, "speaker": "SPEAKER_01"}],
                "exclusive": [{"start": 0, "end": 2, "speaker": "SPEAKER_01"}]}
        out = merge.merge(asr, diar, self.cfg)
        self.assertEqual(out["turns"][0]["paragraphs"], ["no word timings"])


class TestExport(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.meta = {"video_id": "abc123", "title": 'A "Test" Video: Part 1/2',
                     "url": "https://youtu.be/abc123", "channel": "TestChan",
                     "upload_date": "2026-07-01", "duration": 125.0}
        self.merged = {"language": "en", "speakers": ["SPEAKER_00"],
                       "turns": [{"speaker": "SPEAKER_00", "start": 0.5, "end": 4.0,
                                  "paragraphs": ["Hello world."],
                                  "words": [{"start": 0.5, "end": 1.0, "word": " Hello",
                                             "probability": 0.9},
                                            {"start": 1.1, "end": 4.0, "word": " world.",
                                             "probability": 0.95}]}]}

    def test_markdown_only_by_default(self):
        self.assertEqual(Config().export_formats, ["md"])
        with tempfile.TemporaryDirectory() as td:
            files = export.export_video(self.merged, self.meta, self.cfg, Path(td))
            self.assertEqual([f.suffix for f in files], [".md"])

    def test_all_formats_written_when_configured(self):
        self.cfg.export_formats = ["md", "json", "txt", "srt", "vtt"]
        with tempfile.TemporaryDirectory() as td:
            files = export.export_video(self.merged, self.meta, self.cfg, Path(td))
            exts = sorted(f.suffix for f in files)
            self.assertEqual(exts, [".json", ".md", ".srt", ".txt", ".vtt"])
            for f in files:
                self.assertGreater(f.stat().st_size, 20)

    def test_extra_formats_reexported(self):
        """Re-export covers formats already on disk even if deconfigured."""
        with tempfile.TemporaryDirectory() as td:
            self.cfg.export_formats = ["md", "srt"]
            export.export_video(self.merged, self.meta, self.cfg, Path(td))
            self.cfg.export_formats = ["md"]
            extra = export.existing_formats(self.meta, Path(td))
            self.assertIn("srt", extra)
            files = export.export_video(self.merged, self.meta, self.cfg,
                                        Path(td), extra_formats=extra)
            self.assertEqual(sorted(f.suffix for f in files), [".md", ".srt"])

    def test_dotted_title_keeps_video_id(self):
        """Titles ending in '.' must not swallow the [video_id] suffix."""
        self.meta["title"] = "All Religions Do Not Say the Same Thing."
        with tempfile.TemporaryDirectory() as td:
            files = export.export_video(self.merged, self.meta, self.cfg, Path(td))
            for f in files:
                self.assertIn("[abc123]", f.name)
                self.assertIn(f.suffix.lstrip("."), self.cfg.export_formats)

    def test_filename_sanitized(self):
        name = export.safe_filename(self.meta)
        for ch in '<>:"/\\|?*':
            self.assertNotIn(ch, name)
        self.assertIn("abc123", name)

    def test_speaker_rename(self):
        self.cfg.speaker_names = {"SPEAKER_00": "Cliffe"}
        with tempfile.TemporaryDirectory() as td:
            export.export_video(self.merged, self.meta, self.cfg, Path(td))
            md = next(Path(td).glob("*.md")).read_text(encoding="utf-8")
            self.assertIn("**Cliffe**", md)
            self.assertNotIn("SPEAKER_00", md)

    def test_json_roundtrip(self):
        self.cfg.export_formats = ["json"]
        with tempfile.TemporaryDirectory() as td:
            export.export_video(self.merged, self.meta, self.cfg, Path(td))
            doc = json.loads(next(Path(td).glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(doc["video"]["video_id"], "abc123")
            self.assertEqual(doc["turns"][0]["words"][0]["word"], " Hello")

    def test_srt_timestamps(self):
        self.cfg.export_formats = ["srt"]
        with tempfile.TemporaryDirectory() as td:
            export.export_video(self.merged, self.meta, self.cfg, Path(td))
            srt = next(Path(td).glob("*.srt")).read_text(encoding="utf-8")
            self.assertIn("00:00:00,500 --> ", srt)
            self.assertIn("SPEAKER_00:", srt)

    def test_combined(self):
        self.cfg.export_formats = ["md", "json"]
        with tempfile.TemporaryDirectory() as td:
            files = export.export_combined([(self.merged, self.meta, None)] * 2,
                                           self.cfg, Path(td), "TestChan")
            self.assertEqual(len(files), 2)  # md + json
            md = next(Path(td).glob("combined*.md")).read_text(encoding="utf-8")
            self.assertIn("## Contents", md)

    def test_per_video_names_override_config(self):
        self.cfg.speaker_names = {"SPEAKER_00": "ConfigName"}
        with tempfile.TemporaryDirectory() as td:
            export.export_video(self.merged, self.meta, self.cfg, Path(td),
                                names={"SPEAKER_00": "VideoName"})
            md = next(Path(td).glob("*.md")).read_text(encoding="utf-8")
            self.assertIn("**VideoName**", md)
            self.assertNotIn("ConfigName", md)


class TestJobStore(unittest.TestCase):
    def make_store(self):
        self.td = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.td.name) / "jobs.sqlite3")
        return self.store

    def tearDown(self):
        if hasattr(self, "store"):
            self.store.close()
        if hasattr(self, "td"):
            self.td.cleanup()

    def test_add_dedup(self):
        s = self.make_store()
        self.assertTrue(s.add("v1", "http://u/1"))
        self.assertFalse(s.add("v1", "http://u/1"))
        self.assertEqual(len(s.pending()), 1)

    def test_lifecycle(self):
        s = self.make_store()
        s.add("v1", "u")
        s.set_stage("v1", "transcribe")
        self.assertEqual(s.get("v1")["status"], "running")
        s.mark_done("v1", 12.5)
        self.assertEqual(s.get("v1")["status"], "done")
        self.assertEqual(s.pending(), [])

    def test_retry_flow(self):
        s = self.make_store()
        s.add("v1", "u")
        s.mark_failed("v1", "boom")
        self.assertEqual(len(s.failed()), 1)
        self.assertEqual(s.requeue_failed(), 1)
        self.assertEqual(s.get("v1")["status"], "queued")
        self.assertEqual(s.get("v1")["retries"], 1)

    def test_crash_recovery(self):
        s = self.make_store()
        s.add("v1", "u")
        s.set_stage("v1", "diarize")  # simulates crash mid-stage
        self.assertEqual(s.recover_interrupted(), 1)
        self.assertEqual(s.get("v1")["status"], "queued")


class TestConfig(unittest.TestCase):
    def test_validation(self):
        cfg = Config()
        self.assertEqual(cfg.validate(), [])
        cfg.cache_policy = "bogus"
        cfg.min_speakers, cfg.max_speakers = 5, 2
        self.assertEqual(len(cfg.validate()), 2)

    def test_save_load_roundtrip(self):
        import dataclasses
        from unittest import mock

        from ytscribe import config as cfgmod
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(cfgmod, "config_file",
                                   return_value=Path(td) / "config.json"):
                cfg = Config()
                cfg.whisper_model = "large-v3-turbo"
                cfg.speaker_names = {"SPEAKER_00": "Cliffe"}
                cfg.export_formats = ["md", "json"]
                cfgmod.save_config(cfg)
                loaded = cfgmod.load_config()
                self.assertEqual(dataclasses.asdict(loaded), dataclasses.asdict(cfg))

    def test_corrupt_config_falls_back_to_defaults(self):
        from unittest import mock

        from ytscribe import config as cfgmod
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "config.json"
            bad.write_text("{not json", encoding="utf-8")
            with mock.patch.object(cfgmod, "config_file", return_value=bad):
                loaded = cfgmod.load_config()
                self.assertEqual(loaded.whisper_model, Config().whisper_model)


if __name__ == "__main__":
    unittest.main()


class TestDownloadDiagnostics(unittest.TestCase):
    """v0.2.2: actionable messages for the recurring YouTube failures."""

    def test_403_suggests_update(self):
        msg = media._explain("ERROR: unable to download video data: "
                             "HTTP Error 403: Forbidden")
        self.assertIn("update-deps.ps1", msg)
        self.assertIn("out-of-date yt-dlp", msg)

    def test_bot_check_suggests_cookies(self):
        msg = media._explain("Sign in to confirm you're not a bot")
        self.assertIn("Cookies from browser", msg)

    def test_unrelated_error_unchanged(self):
        self.assertEqual(media._explain("some other failure"),
                         "some other failure")

    def test_release_age_parsing(self):
        self.assertIsNone(media._release_age_days("unknown"))
        self.assertGreater(media._release_age_days("2020.01.01"), 2000)

    def test_environment_report_returns_strings(self):
        for w in media.environment_report():
            self.assertIsInstance(w, str)
