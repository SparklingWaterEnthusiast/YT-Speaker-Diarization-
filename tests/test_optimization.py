"""Optimization/config/UI regressions, with temporary state and no GPU work."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread, QCoreApplication, QEvent
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QScrollArea

from ytscribe import config as config_module
from ytscribe.config import Config
from ytscribe.optimization import apply_profile, precision_options, recommend_compute_type
from ytscribe.ui import settings_dialog


def tearDownModule():
    """Destroy Qt widgets while QApplication still exists, not at Python exit.

    Hidden dialogs with signal/closure cycles otherwise outlive the class-owned
    QApplication in unittest and can crash Windows during interpreter teardown.
    """
    import gc
    import shiboken6
    app=QApplication.instance()
    if app is not None:
        for widget in app.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        app.processEvents()
        gc.collect()
        shiboken6.delete(app)


class TestOptimizationConfig(unittest.TestCase):
    def test_v2_migration_preserves_custom_inference(self):
        old = Config(whisper_model="custom/model", compute_type="int8_float32",
                     beam_size=17, language="hi", vad_filter=False,
                     min_speakers=2, max_speakers=7, config_version=2)
        data = dataclasses.asdict(old)
        for key in ("asr_batch_size", "asr_cpu_threads", "asr_num_workers", "asr_chunk_length",
                    "word_timestamps", "diarization_batch_size", "model_residency",
                    "vram_margin_mb", "oom_retries", "profiling_enabled",
                    "profiling_interval", "optimization_profile"):
            data.pop(key)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with mock.patch.object(config_module, "config_file", return_value=path):
                cfg = config_module.load_config()
                self.assertEqual(cfg.config_version, 3)
                for key, value in data.items():
                    if key != "config_version":
                        self.assertEqual(getattr(cfg, key), value, key)
                self.assertEqual(cfg.optimization_profile, "custom")
                self.assertEqual(cfg.asr_batch_size, 1)
                self.assertEqual(cfg.diarization_batch_size, 32)
                self.assertTrue(cfg.word_timestamps)
                self.assertEqual(dataclasses.asdict(config_module.load_config()),
                                 dataclasses.asdict(cfg))

    def test_v1_output_migration_retains_deliberate_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            with mock.patch.object(config_module, "config_file", return_value=path):
                for before, after in ((["md", "srt"], ["md", "srt"]),
                                      (list(config_module.EXPORT_FORMATS), ["md"])):
                    path.write_text(json.dumps({"export_formats": before, "beam_size": 11}),
                                    encoding="utf-8")
                    cfg = config_module.load_config()
                    self.assertEqual(cfg.export_formats, after)
                    self.assertEqual(cfg.beam_size, 11)

    def test_invalid_values_report_without_mutation_or_type_errors(self):
        invalid = {
            "asr_batch_size": [0, -1, True, 1.5, "4"],
            "asr_cpu_threads": [0, None], "asr_num_workers": [0, False],
            "asr_chunk_length": [0, 31], "diarization_batch_size": [0, -2],
            "vram_margin_mb": [-1, 0.5], "oom_retries": [-1, True],
            "profiling_enabled": [1, "false"], "word_timestamps": [None, "true"],
            "profiling_interval": [0, -1, float("inf"), float("nan")],
            "model_residency": ["unload", None], "optimization_profile": ["fast"],
            "beam_size": [0, True], "min_speakers": [-1, "two"],
            "recognition_threshold": [-0.1, 1.1], "export_formats": [[], None, ["pdf"]],
            "sleep_between_downloads_min": [-1, "fast"],
            "device": ["mps"], "compute_type": ["bogus"], "cache_dir": ["", None],
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    cfg = Config()
                    setattr(cfg, field, value)
                    self.assertTrue(cfg.validate())
                    self.assertIs(getattr(cfg, field), value)

    def test_invalid_save_does_not_overwrite_valid_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            with mock.patch.object(config_module, "config_file", return_value=path):
                config_module.save_config(Config())
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    config_module.save_config(Config(asr_batch_size=0))
                self.assertEqual(path.read_bytes(), before)

    def test_unusual_valid_custom_values_survive_validation(self):
        cfg = Config(whisper_model="organization/my-whisper", beam_size=42,
                     min_speakers=23, max_speakers=50, asr_cpu_threads=96,
                     recognition_threshold=0.99, asr_num_workers=3,
                     compute_type="int8_float32", profiling_interval=0.25)
        before = dataclasses.asdict(cfg)
        self.assertEqual(cfg.validate(), [])
        self.assertEqual(dataclasses.asdict(cfg), before)


class TestOptimizationPolicy(unittest.TestCase):
    def test_presets_are_explicit_independent_candidates(self):
        cfg = Config(whisper_model="custom/model", beam_size=19,
                     language="hi", speaker_names={"SPEAKER_00": "Name"})
        before = dataclasses.asdict(cfg)
        for profile, batch, residency in (("safe", 1, "stage"),
                                           ("balanced", 1, "keep"),
                                           ("performance", 4, "keep")):
            with self.subTest(profile=profile):
                result = apply_profile(cfg, profile)
                self.assertEqual(result.whisper_model, "large-v3")
                self.assertEqual(result.beam_size, 5)
                self.assertEqual(result.asr_batch_size, batch)
                self.assertEqual(result.model_residency, residency)
                self.assertEqual(result.asr_num_workers, 1)
                self.assertTrue(result.word_timestamps)
                self.assertEqual(result.language, "hi")
                self.assertEqual(result.validate(), [])
                result.speaker_names.clear()
                self.assertEqual(dataclasses.asdict(cfg), before)

    def test_custom_is_a_noop_copy_and_invalid_preset_fails(self):
        cfg = Config(whisper_model="small", asr_batch_size=3, compute_type="float32")
        result = apply_profile(cfg, "custom")
        self.assertEqual(dataclasses.asdict(result), dataclasses.asdict(cfg))
        self.assertIsNot(result, cfg)
        with self.assertRaises(ValueError):
            apply_profile(cfg, "fastest")

    def test_precision_follows_capability_not_gpu_name(self):
        self.assertEqual(recommend_compute_type(None), "auto")
        self.assertEqual(recommend_compute_type([]), "auto")
        self.assertEqual(recommend_compute_type(["int8", "int8_float32", "float32"], "cuda"),
                         "int8_float32")
        self.assertEqual(recommend_compute_type(["float16", "int8_float16"], "cuda"),
                         "int8_float16")
        self.assertEqual(recommend_compute_type(["int8_float32", "int8"], "cpu"), "int8")
        choices = precision_options(["int8", "float32"], current="float16")
        self.assertIn("float16", choices)  # retain saved choice, don't silently rewrite it
        self.assertIn("auto", choices)
        self.assertNotIn("int8_float16", choices)

    def test_low_vram_presets_release_models_without_raising_batch(self):
        result = apply_profile(Config(), "balanced", vram_total_gb=6,
                               supported_compute_types=["int8_float32"], device="cuda")
        self.assertEqual(result.model_residency, "stage")
        self.assertEqual(result.asr_batch_size, 1)
        self.assertEqual(result.compute_type, "int8_float32")


class QtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def spin_until(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(predicate(), "Qt condition did not complete")


class TestRunWorker(QtTestCase):
    def test_fatal_error_reports_failure_not_success(self):
        from ytscribe.ui import workers
        with mock.patch.object(workers.pipeline, "QueueRunner") as runner:
            runner.return_value.run.side_effect = RuntimeError("simulated failure")
            worker = workers.RunWorker(Config(), mock.Mock())
            summaries, logs = [], []
            worker.queue_done.connect(summaries.append)
            worker.log.connect(logs.append)
            worker.run()  # synchronous mocked runner; no thread, model, or database
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["failed"], 1)
        self.assertEqual(summaries[0]["completed"], 0)
        self.assertIn("FATAL queue error: RuntimeError: simulated failure", logs)


class TestSettingsDialog(QtTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config.json"
        self.config_patch = mock.patch.object(config_module, "config_file", return_value=self.path)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def dialog(self, cfg=None):
        dlg = settings_dialog.SettingsDialog(cfg or Config())
        self.addCleanup(dlg.close)
        return dlg

    def test_custom_values_round_trip_without_opening_gpu_or_database(self):
        cfg = Config(whisper_model="organization/custom", compute_type="float32", beam_size=29,
                     min_speakers=21, max_speakers=25, recognition_threshold=0.9991234567,
                     sleep_between_downloads_min=700.12345678,
                     sleep_between_downloads_max=900.012345678,
                     cookies_from_browser="chrome:CustomProfile", asr_num_workers=3,
                     word_timestamps=False, profiling_interval=0.00000001,
                     export_formats=["srt", "md"])
        # A tiny valid sampling interval should also survive a no-op UI save.
        before = dataclasses.asdict(cfg)
        with mock.patch("ytscribe.resources.hardware_inventory") as probe:
            dlg = self.dialog(cfg)
            self.assertIsNotNone(dlg.findChild(QScrollArea))
            dlg._save()
            probe.assert_not_called()
        self.assertEqual(dataclasses.asdict(cfg), before)
        self.assertEqual(dataclasses.asdict(config_module.load_config()), before)

    def test_invalid_candidate_never_mutates_or_persists(self):
        cfg = Config()
        before = dataclasses.asdict(cfg)
        dlg = self.dialog(cfg)
        dlg.output_dir.setText("changed output")
        dlg.min_spk.setValue(8)
        dlg.max_spk.setValue(2)
        with mock.patch.object(settings_dialog.QMessageBox, "warning") as warning:
            dlg._save()
        warning.assert_called_once()
        self.assertFalse(self.path.exists())
        self.assertEqual(dataclasses.asdict(cfg), before)

    def test_download_bounds_are_rejected_not_silently_fixed(self):
        dlg = self.dialog()
        dlg.sleep_min.setValue(20)
        dlg.sleep_max.setValue(10)
        with mock.patch.object(settings_dialog.QMessageBox, "warning") as warning:
            dlg._save()
        warning.assert_called_once()
        self.assertFalse(self.path.exists())
        self.assertEqual(dlg.sleep_max.value(), 10)

    def test_failed_write_keeps_live_config_unchanged(self):
        cfg = Config()
        before = dataclasses.asdict(cfg)
        dlg = self.dialog(cfg)
        dlg.asr_batch.setValue(4)
        with mock.patch.object(settings_dialog, "save_config", side_effect=OSError("disk full")), \
                mock.patch.object(settings_dialog.QMessageBox, "warning") as warning:
            dlg._save()
        warning.assert_called_once()
        self.assertEqual(dataclasses.asdict(cfg), before)

    def test_friendly_retention_label_saves_original_enum(self):
        cfg = Config()
        dlg = self.dialog(cfg)
        dlg.cache_policy.setCurrentIndex(dlg.cache_policy.findData("keep_forever"))
        self.assertIn("Retain downloaded", dlg.cache_policy.currentText())
        dlg._save()
        self.assertEqual(cfg.cache_policy, "keep_forever")

    def test_preset_selection_requires_apply_and_manual_edit_becomes_custom(self):
        cfg = Config(whisper_model="custom/model", beam_size=15)
        dlg = self.dialog(cfg)
        dlg.profile.setCurrentIndex(dlg.profile.findData("performance"))
        self.assertEqual(dlg._candidate().whisper_model, "custom/model")
        self.assertEqual(dlg._candidate().asr_batch_size, 1)
        self.assertEqual(dlg._candidate().optimization_profile, "custom")
        dlg._apply_preset()
        self.assertEqual(dlg._candidate().whisper_model, "large-v3")
        self.assertEqual(dlg._candidate().asr_batch_size, 4)
        self.assertEqual(dlg._candidate().optimization_profile, "performance")
        dlg.beam.setValue(6)
        self.assertEqual(dlg._candidate().optimization_profile, "custom")
        self.assertEqual(cfg.whisper_model, "custom/model")

    def test_inventory_filters_precision_for_selected_device_without_changing_setting(self):
        dlg = self.dialog(Config(compute_type="float16"))
        dlg._hardware_ready({"gpu_name": "GPU", "gpu_vram_total_bytes": 6 * 2**30,
                             "ram_total_bytes": 32 * 2**30, "cpu_logical_count": 16,
                             "cpu_physical_count": 8,
                             "ctranslate2_supported_types": {
                                 "cpu": ["int8", "float32"],
                                 "cuda": ["int8_float32", "float32"]}})
        self.assertEqual(dlg.compute.currentText(), "float16")
        dlg.profile.setCurrentIndex(dlg.profile.findData("safe"))
        dlg._apply_preset()
        self.assertEqual(dlg.compute.currentText(), "int8_float32")
        dlg.device.setCurrentText("cpu")
        dlg.profile.setCurrentIndex(dlg.profile.findData("balanced"))
        dlg._apply_preset()
        self.assertEqual(dlg.compute.currentText(), "int8")

    def test_cancel_waits_for_real_hardware_probe_completion(self):
        dlg = self.dialog()
        gate = threading.Event()
        entered = threading.Event()

        def probe(**kwargs):
            self.assertTrue(kwargs["detailed"])
            entered.set()
            gate.wait(5)
            return {}

        with mock.patch("ytscribe.resources.hardware_inventory", side_effect=probe):
            dlg.show()
            dlg._probe_hardware()
            try:
                self.spin_until(entered.is_set)
                dlg.reject()
                self.assertTrue(dlg.isVisible())
                self.assertTrue(dlg._probe.isRunning())
                gate.set()
                self.spin_until(lambda: not dlg.isVisible())
                self.assertEqual(dlg.result(), QDialog.DialogCode.Rejected)
            finally:
                gate.set()
                dlg._probe.wait(2000)


class TestWindowLifecycle(QtTestCase):
    def setUp(self):
        from ytscribe.ui import main_window
        self.mod = main_window
        self.temp = tempfile.TemporaryDirectory()
        folder = Path(self.temp.name)
        self.cfg = Config(output_dir=str(folder / "output"), cache_dir=str(folder / "cache"))
        with mock.patch.object(main_window, "APP_DIR", folder), \
                mock.patch.object(main_window, "load_config", return_value=self.cfg), \
                mock.patch.object(main_window, "save_config"), \
                mock.patch("ytscribe.media.environment_report", return_value=[]):
            self.window = main_window.MainWindow()
        self.window.res_timer.stop()  # never sample the real GPU in these tests
        self.window.clock_timer.stop()

    def tearDown(self):
        self.window.run_worker = None
        self.window.enqueue_worker = None
        self.window.close()
        self.window.store.close()
        self.window.voice_db.close()
        self.temp.cleanup()

    def _second_window(self):
        with mock.patch.object(self.mod, "APP_DIR", Path(self.temp.name)), \
                mock.patch.object(self.mod, "load_config", return_value=self.cfg), \
                mock.patch.object(self.mod, "save_config"), \
                mock.patch("ytscribe.media.environment_report", return_value=[]):
            window = self.mod.MainWindow()
        window.res_timer.stop()
        window.clock_timer.stop()
        return window

    def test_second_gui_does_not_recover_a_live_workers_jobs(self):
        from ytscribe.worker_lock import processing_lease
        self.window.store.add("live", "https://example.test/live")
        self.window.store.set_stage("live", "transcribe")
        with processing_lease(self.cfg.cache_path):
            second = self._second_window()
            try:
                self.assertEqual(self.window.store.get("live")["status"], "running")
                self.assertIn("Startup recovery skipped", second.log.toPlainText())
            finally:
                second.close()
                second.store.close()
                second.voice_db.close()

    def test_startup_recovers_interrupted_jobs_when_lease_is_available(self):
        self.window.store.add("interrupted", "https://example.test/interrupted")
        self.window.store.set_stage("interrupted", "diarize")
        second = self._second_window()
        try:
            self.assertEqual(self.window.store.get("interrupted")["status"], "queued")
            self.assertIn("Recovered 1 interrupted job", second.log.toPlainText())
        finally:
            second.close()
            second.store.close()
            second.voice_db.close()

    def test_start_rejects_invalid_saved_config_before_activation(self):
        self.window.store.add("pending", "https://example.test/pending")
        before = dataclasses.asdict(self.window.cfg)
        with mock.patch.object(self.mod, "load_config", return_value=Config(asr_batch_size=0)), \
                mock.patch.object(self.mod, "RunWorker") as worker, \
                mock.patch.object(QMessageBox, "warning") as warning, \
                mock.patch.object(self.window.store, "set_queue_active") as activate:
            self.window.start_queue()
        worker.assert_not_called()
        activate.assert_not_called()
        warning.assert_called_once()
        self.assertIn("asr_batch_size", warning.call_args.args[2])
        self.assertEqual(dataclasses.asdict(self.window.cfg), before)
        self.assertIsNone(self.window.run_worker)

    def test_close_waits_for_both_workers_and_keeps_event_loop_responsive(self):
        class HeldWorker(QThread):
            def __init__(self, parent):
                super().__init__(parent)
                self.gate = threading.Event()
                self.runner = mock.Mock()

            def run(self):
                self.gate.wait(5)

        workers = [HeldWorker(self.window), HeldWorker(self.window)]
        self.window.run_worker, self.window.enqueue_worker = workers
        for worker in workers:
            worker.start()
        self.window.show()
        try:
            with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
                event = QCloseEvent()
                before = time.monotonic()
                self.window.closeEvent(event)
                self.assertLess(time.monotonic() - before, 0.5)
                self.assertFalse(event.isAccepted())
                workers[0].runner.stop.assert_called_once()
                workers[0].runner.cancel_current_job.assert_called_once()
                self.assertTrue(workers[1].isInterruptionRequested())
                workers[0].gate.set()
                self.spin_until(lambda: not workers[0].isRunning())
                QTest.qWait(150)
                self.assertTrue(self.window.isVisible())
                self.assertTrue(workers[1].isRunning())
                workers[1].gate.set()
                self.spin_until(lambda: not self.window.isVisible())
                self.assertFalse(self.window._close_timer.isActive())
        finally:
            for worker in workers:
                worker.gate.set()
                worker.wait(2000)

    def test_enqueue_only_blocks_close_and_decline_does_not_stop_work(self):
        worker = mock.Mock()
        worker.isRunning.return_value = True
        self.window.enqueue_worker = worker
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            event = QCloseEvent()
            self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertFalse(self.window._closing)
        worker.requestInterruption.assert_not_called()
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertTrue(self.window._closing)
        worker.requestInterruption.assert_called_once()

    def test_resource_strip_handles_missing_optional_metrics_and_real_zero(self):
        from types import SimpleNamespace
        snap = SimpleNamespace(gpu_available=False, gpu_percent=0, vram_used_gb=0,
                               vram_total_gb=0, cpu_percent=0, ram_used_gb=0,
                               ram_total_gb=0, disk_free_gb=0)
        with mock.patch.object(self.mod.resources, "sample", return_value=snap):
            self.window._update_resources()
            self.assertEqual(self.window.res_labels["temperature"].text(), "Temp: n/a")
            snap.temperature_c, snap.sm_clock_mhz, snap.power_w = 0, 1234, 0
            self.window._update_resources()
            self.assertEqual(self.window.res_labels["temperature"].text(), "Temp: 0°C")
            self.assertEqual(self.window.res_labels["clock"].text(), "Clock: 1234 MHz")
            self.assertEqual(self.window.res_labels["power"].text(), "Power: 0 W")

    def test_thermal_warning_requires_five_observations_and_limits_log_frequency(self):
        worker = mock.Mock()
        worker.isRunning.return_value = True
        self.window.run_worker = worker
        snap = self.mod.resources.Snapshot(
            throttle_reasons=["sw_thermal_slowdown", "hw_thermal_slowdown"])
        with mock.patch.object(self.mod.resources, "sample", return_value=snap), \
                mock.patch.object(self.window, "log_line") as log, \
                mock.patch.object(self.mod.time, "monotonic", return_value=0) as clock:
            for _ in range(4):
                self.window._update_resources()
            log.assert_not_called()  # two flags still count as just one observation
            self.window._update_resources()
            log.assert_called_once()
            self.assertIn("5 of the last 5", log.call_args.args[0])
            clock.return_value = 59.999
            for _ in range(20):
                self.window._update_resources()
            self.assertEqual(log.call_count, 1)
            self.assertEqual(len(self.window._thermal_observations), 10)
            clock.return_value = 60
            self.window._update_resources()
            self.assertEqual(log.call_count, 2)

    def test_thermal_window_ages_out_and_does_not_infer_from_temperature_or_power(self):
        from types import SimpleNamespace
        worker = mock.Mock()
        worker.isRunning.return_value = True
        self.window.run_worker = worker
        thermal = SimpleNamespace(throttle_reasons=["sw_thermal_slowdown"])
        other = SimpleNamespace(temperature_c=110, throttle_reasons=["sw_power_cap", "hw_slowdown"])
        missing = SimpleNamespace(temperature_c=110)
        with mock.patch.object(self.window, "log_line") as log:
            for _ in range(4):
                self.window._check_thermal_warning(thermal)
            for _ in range(10):
                self.window._check_thermal_warning(other)
                self.window._check_thermal_warning(missing)
            self.window._check_thermal_warning(thermal)
            log.assert_not_called()
            self.assertEqual(sum(self.window._thermal_observations), 1)

    def test_thermal_warning_ignores_idle_and_enqueue_workers(self):
        from types import SimpleNamespace
        snap = SimpleNamespace(throttle_reasons=["hw_thermal_slowdown"])
        self.window.enqueue_worker = mock.Mock()
        self.window.enqueue_worker.isRunning.return_value = True
        with mock.patch.object(self.window, "log_line") as log:
            for _ in range(10):
                self.window._check_thermal_warning(snap)
            log.assert_not_called()
            worker = mock.Mock()
            worker.isRunning.return_value = True
            self.window.run_worker = worker
            for _ in range(4):
                self.window._check_thermal_warning(snap)
            worker.isRunning.return_value = False
            self.window._check_thermal_warning(snap)
            self.assertEqual(len(self.window._thermal_observations), 0)
            worker.isRunning.return_value = True
            self.window._check_thermal_warning(snap)
            log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
