"""Telemetry contracts: no inference, downloads, or production data access."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ytscribe import resources, telemetry


class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.jsonl"

    def rows(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def recorder(self, **kwargs):
        recorder = telemetry.Recorder(self.path, **kwargs)
        self.addCleanup(recorder.close)
        return recorder

    def test_disabled_does_not_create_files_threads_or_probe(self):
        with patch.object(resources, "sample") as sample, patch.object(threading, "Thread") as thread:
            for recorder in (telemetry.Recorder(), telemetry.Recorder(self.path, enabled=False)):
                with recorder:
                    with telemetry.span("disabled", cold=True):
                        self.assertFalse(telemetry.event("ignored"))
                    self.assertFalse(recorder.sample_once())
            sample.assert_not_called()
            thread.assert_not_called()
        self.assertFalse(self.path.exists())
        self.assertIsNone(telemetry.current_recorder())
        with telemetry.span("unbound"):
            self.assertFalse(telemetry.event("unbound"))

    def test_span_event_schema_and_monotonic_timing(self):
        with patch.object(telemetry.time, "perf_counter", side_effect=[10, 11, 12, 13, 14, 15]):
            recorder = self.recorder()
            with telemetry.use_recorder(recorder):
                with telemetry.span("transcribe", cold=True, cache_hit=False, video_id="test"):
                    telemetry.event("progress", cache_hit=True, done=2)
        rows = self.rows()
        self.assertEqual([row["kind"] for row in rows], ["span_start", "event", "span_end"])
        self.assertEqual([row["elapsed_s"] for row in rows], [2, 3, 5])
        self.assertEqual(rows[-1]["duration_s"], 3)
        self.assertEqual(rows[0]["span_id"], rows[-1]["span_id"])
        self.assertEqual(rows[-1]["status"], "ok")
        self.assertTrue(rows[0]["cold"])
        self.assertFalse(rows[0]["cache_hit"])
        self.assertTrue(rows[1]["cache_hit"])
        self.assertEqual(rows[0]["fields"], {"video_id": "test"})
        for row in rows:
            self.assertEqual(row["schema_version"], 1)
            self.assertEqual(row["pid"], os.getpid())
            self.assertEqual(row["thread_id"], threading.get_ident())
            self.assertEqual(row["thread"], threading.current_thread().name)

    def test_exception_and_cancellation_propagate_and_end_span(self):
        recorder = self.recorder()
        for exception in (RuntimeError("original"), KeyboardInterrupt()):
            with self.assertRaises(type(exception)) as caught:
                with recorder.span("failed"):
                    raise exception
            self.assertIs(caught.exception, exception)
        ends = [r for r in self.rows() if r["kind"] == "span_end"]
        self.assertEqual([r["error_type"] for r in ends], ["RuntimeError", "KeyboardInterrupt"])
        self.assertTrue(all(r["status"] == "error" for r in ends))

    def test_span_realtime_ratios(self):
        recorder = self.recorder()
        with patch.object(telemetry.time, "perf_counter", side_effect=[10, 10, 12, 12]):
            with recorder.span("audio", audio_duration=8.0):
                pass
        end = self.rows()[-1]
        self.assertEqual(end["duration_s"], 2)
        self.assertEqual(end["realtime_multiplier"], 4)
        self.assertEqual(end["realtime_factor"], 0.25)
        self.assertNotIn("realtime_factor", self.rows()[0])

    def test_ratio_edge_cases_do_not_raise_or_emit_infinity(self):
        for duration in (None, 0, -1, float("inf"), float("nan"), "8", True):
            self.assertEqual(telemetry._realtime_ratios(duration, 2), {})
        self.assertEqual(telemetry._realtime_ratios(8, 0),
                         {"realtime_factor": 0, "realtime_multiplier": None})
        self.assertIsNone(telemetry._realtime_ratios(1e308, 1e-308)["realtime_multiplier"])
        recorder = self.recorder()
        with recorder.span("no_audio"):
            pass
        self.assertNotIn("realtime_factor", self.rows()[-1])

    def test_events_and_samples_include_noninitializing_runtime_flags(self):
        recorder = self.recorder()
        cuda = Mock()
        cuda.is_initialized.return_value = False
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda), "ctranslate2": None}), \
                patch.object(resources, "sample", return_value=resources.Snapshot()):
            recorder.event("device", device="cuda")
            recorder.sample_once()
        rows = self.rows()
        self.assertEqual(rows[0]["device"], "cuda")
        self.assertEqual(rows[0]["fields"]["device"], "cuda")
        for row in rows:
            self.assertTrue(row["torch_loaded"])
            self.assertFalse(row["cuda_initialized"])
            self.assertFalse(row["ctranslate2_loaded"])
        self.assertTrue(all(call[0] == "is_initialized" for call in cuda.mock_calls))

    def test_nested_contexts_restore_parent_even_on_error(self):
        outer = self.recorder()
        inner = telemetry.Recorder(Path(self.temp.name) / "inner.jsonl")
        self.addCleanup(inner.close)
        with patch.object(telemetry.Recorder, "start", autospec=True, side_effect=lambda r: r):
            with outer:
                self.assertIs(telemetry.current_recorder(), outer)
                with self.assertRaises(ValueError):
                    with inner:
                        self.assertIs(telemetry.current_recorder(), inner)
                        raise ValueError("test")
                self.assertIs(telemetry.current_recorder(), outer)
                telemetry.event("restored")
        self.assertIsNone(telemetry.current_recorder())
        self.assertEqual(self.rows()[0]["name"], "restored")

    def test_thread_propagation_and_concurrent_jsonl(self):
        recorder = self.recorder()
        unbound = []

        def work(active):
            unbound.append(telemetry.current_recorder())
            with telemetry.use_recorder(active):
                for number in range(30):
                    with telemetry.span("worker", number=number):
                        telemetry.event("item")

        with telemetry.use_recorder(recorder):
            threads = [threading.Thread(target=work, args=(telemetry.current_recorder(),))
                       for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
        rows = self.rows()
        self.assertEqual(unbound, [None] * 4)
        self.assertEqual(len(rows), 360)
        starts = [r["span_id"] for r in rows if r["kind"] == "span_start"]
        self.assertEqual(len(set(starts)), 120)
        elapsed = [r["elapsed_s"] for r in rows]
        self.assertEqual(elapsed, sorted(elapsed))

    def test_append_and_bad_metadata_do_not_break_writer(self):
        first = self.recorder()
        first.event("first")
        first.close()
        second = self.recorder()
        self.assertFalse(second.event("bad", value=object()))
        self.assertIn("TypeError", second.last_error)
        self.assertFalse(second.event("nan", value=float("nan")))
        self.assertTrue(second.event("second", text="hello \u2603"))
        self.assertEqual([row["name"] for row in self.rows()], ["first", "second"])

    def test_failed_open_and_failed_write_are_nonfatal(self):
        failed = telemetry.Recorder(self.temp.name)  # directory, not a file
        self.assertFalse(failed.enabled)
        self.assertIsNotNone(failed.last_error)
        with failed.span("safe"):
            pass
        recorder = self.recorder()
        original = recorder._file
        with patch.object(recorder, "_file", Mock(write=Mock(side_effect=OSError("full")))):
            self.assertFalse(recorder.event("fails"))
        self.assertFalse(original.closed)
        self.assertIn("full", recorder.last_error)

    def test_daemon_lifecycle_interval_and_sample(self):
        recorder = self.recorder(interval=0.02, disk_path=self.temp.name)
        sampled_twice = threading.Event()
        count = []

        def sampled(_path):
            count.append(1)
            if len(count) >= 2:
                sampled_twice.set()
            return resources.Snapshot(sm_clock_mhz=1200)

        with patch.object(resources, "sample", side_effect=sampled) as sample, \
                patch.object(telemetry, "torch_metrics", return_value={"cuda_initialized": False}):
            with recorder:
                recorder.start()  # idempotent
                self.assertTrue(recorder._thread.daemon)
                self.assertTrue(sampled_twice.wait(timeout=2))
            self.assertFalse(recorder._thread.is_alive())
            self.assertFalse(recorder.enabled)
            sample.assert_called_with(self.temp.name)
        rows = self.rows()
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0]["kind"], "sample")
        self.assertEqual(rows[0]["fields"]["resources"]["sm_clock_mhz"], 1200)
        self.assertEqual(rows[0]["thread"], "ytscribe-telemetry")
        recorder.close()
        self.assertFalse(recorder.event("closed"))
        self.assertEqual(telemetry.Recorder().interval, 1.0)

    def test_long_interval_stop_is_interruptible_and_restart_works(self):
        recorder = self.recorder(interval=3600)
        sampled = threading.Event()
        with patch.object(recorder, "sample_once", side_effect=lambda: sampled.set()):
            recorder.start()
            self.assertTrue(sampled.wait(2))
            recorder.stop()
            self.assertFalse(recorder._thread.is_alive())
            sampled.clear()
            recorder.start()
            self.assertTrue(sampled.wait(2))
            recorder.stop()
            self.assertFalse(recorder._thread.is_alive())
        self.assertTrue(recorder.event("after_stop"))

    def test_sampling_failure_is_logged_and_next_sample_recovers(self):
        recorder = self.recorder()
        with patch.object(resources, "sample", side_effect=[RuntimeError("sensor"), resources.Snapshot()]), \
                patch.object(telemetry, "torch_metrics", return_value={}):
            recorder.sample_once()
            recorder.sample_once()
        self.assertEqual([r["name"] for r in self.rows()], ["sample_error", "resources"])
        self.assertIn("sensor", recorder.last_error)

    def test_thread_start_failure_preserves_events_and_pipeline(self):
        recorder = self.recorder()
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("no threads")):
            with recorder:
                self.assertTrue(telemetry.event("still_running"))
        self.assertIn("no threads", recorder.last_error)
        self.assertIsNone(telemetry.current_recorder())
        self.assertEqual(self.rows()[0]["name"], "still_running")

    def test_invalid_intervals(self):
        for interval in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                telemetry.Recorder(interval=interval)


class TestNoCudaInitialization(unittest.TestCase):
    def test_absent_torch_is_not_imported(self):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "ctranslate2", "faster_whisper"):
                raise AssertionError("inference import attempted")
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"torch": None}), patch("builtins.__import__", side_effect=guarded):
            self.assertTrue(all(value is None for value in telemetry.torch_metrics().values()))

    def test_uninitialized_torch_only_checks_initialization(self):
        cuda = Mock()
        cuda.is_initialized.return_value = False
        torch = SimpleNamespace(cuda=cuda, get_num_threads=Mock(), get_num_interop_threads=Mock())
        with patch.dict(sys.modules, {"torch": torch}):
            metrics = telemetry.torch_metrics()
        self.assertFalse(metrics["cuda_initialized"])
        self.assertIsNone(metrics["memory_allocated_bytes"])
        self.assertEqual([call[0] for call in cuda.mock_calls], ["is_initialized"])
        torch.get_num_threads.assert_not_called()
        torch.get_num_interop_threads.assert_not_called()
        self.assertIsNone(metrics["intra_op_threads"])
        self.assertIsNone(metrics["inter_op_threads"])

    def test_initialized_torch_reads_counters_only(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        cuda.memory_allocated.return_value = 10
        cuda.memory_reserved.return_value = 20
        cuda.max_memory_allocated.side_effect = RuntimeError("unavailable")
        cuda.max_memory_reserved.return_value = 40
        torch = SimpleNamespace(cuda=cuda, get_num_threads=Mock(return_value=4),
                                get_num_interop_threads=Mock(return_value=8))
        with patch.dict(sys.modules, {"torch": torch}):
            metrics = telemetry.torch_metrics()
        self.assertEqual(metrics["intra_op_threads"], 4)
        self.assertEqual(metrics["inter_op_threads"], 8)
        torch.get_num_threads.assert_called_once_with()
        torch.get_num_interop_threads.assert_called_once_with()
        self.assertEqual(metrics["memory_allocated_bytes"], 10)
        self.assertEqual(metrics["memory_reserved_bytes"], 20)
        self.assertIsNone(metrics["max_memory_allocated_bytes"])
        self.assertEqual(metrics["max_memory_reserved_bytes"], 40)
        self.assertEqual([call[0] for call in cuda.mock_calls], ["is_initialized", "memory_allocated",
                         "memory_reserved", "max_memory_allocated", "max_memory_reserved"])

    def test_thread_counter_failure_is_independent(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        torch = SimpleNamespace(cuda=cuda, get_num_threads=Mock(side_effect=RuntimeError),
                                get_num_interop_threads=Mock(return_value=8))
        with patch.dict(sys.modules, {"torch": torch}):
            metrics = telemetry.torch_metrics()
        self.assertIsNone(metrics["intra_op_threads"])
        self.assertEqual(metrics["inter_op_threads"], 8)

    def test_fresh_import_and_real_cpu_sampling_do_not_load_runtimes(self):
        # Isolated interpreter catches imports hidden by the test runner. NVML
        # is mocked out, so this does not query or run anything on a GPU.
        script = """
import sys, tempfile
from pathlib import Path
from ytscribe import telemetry, resources
assert resources.pynvml is None
resources._NVML_TRIED = True
with tempfile.TemporaryDirectory() as tmp:
    with telemetry.Recorder(Path(tmp) / 'trace.jsonl', tmp):
        with telemetry.span('cpu_only'):
            telemetry.event('test')
assert not {'torch', 'ctranslate2', 'faster_whisper', 'pynvml'} & sys.modules.keys()
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestResources(unittest.TestCase):
    def setUp(self):
        self.local = patch.object(resources, "_LOCAL", threading.local())
        self.local.start()
        self.addCleanup(self.local.stop)
        self.handle = patch.object(resources, "_nvml_handle", return_value=None)
        self.handle.start()
        self.addCleanup(self.handle.stop)

    def test_legacy_defaults_and_missing_new_metrics(self):
        with patch.object(resources.psutil, "cpu_percent", side_effect=OSError), \
                patch.object(resources.psutil, "virtual_memory", side_effect=OSError), \
                patch.object(resources.psutil, "disk_io_counters", return_value=None), \
                patch.object(resources.psutil, "sensors_battery", return_value=None), \
                patch.object(resources, "_process", side_effect=OSError), \
                patch.object(resources, "_windows_ac_connected", return_value=None), \
                patch.object(resources.shutil, "disk_usage", side_effect=OSError):
            snap = resources.sample("missing")
        self.assertEqual(snap.cpu_percent, 0)
        self.assertEqual(snap.ram_used_gb, 0)
        self.assertEqual(snap.disk_free_gb, 0)
        self.assertFalse(snap.gpu_available)
        for field in ("temperature_c", "sm_clock_mhz", "power_w", "power_limit_w", "pstate",
                      "throttle_reasons", "process_cpu_percent", "process_rss_bytes",
                      "process_threads", "cpu_per_core_percent", "disk_read_bytes",
                      "disk_write_bytes", "battery_percent", "ac_connected"):
            self.assertIsNone(getattr(snap, field), field)

    def test_process_disk_battery_and_percore_counters(self):
        proc = Mock()
        proc.cpu_percent.side_effect = [0.0, 240.0]
        proc.memory_info.return_value = SimpleNamespace(rss=12345)
        proc.num_threads.return_value = 6
        io = SimpleNamespace(read_bytes=100, write_bytes=200, read_count=3, write_count=4)
        with patch.object(resources.psutil, "Process", return_value=proc) as process, \
                patch.object(resources.psutil, "cpu_percent", side_effect=[20, [10, 30]] * 2), \
                patch.object(resources.psutil, "disk_io_counters", return_value=io), \
                patch.object(resources.psutil, "sensors_battery",
                             return_value=SimpleNamespace(percent=55, power_plugged=False)):
            first, second = resources.sample(), resources.sample()
        process.assert_called_once()
        self.assertIsNone(first.process_cpu_percent)
        self.assertEqual(second.process_cpu_percent, 240)
        self.assertEqual(second.cpu_per_core_percent, [10, 30])
        self.assertEqual(second.process_rss_bytes, 12345)
        self.assertEqual(second.process_threads, 6)
        self.assertEqual(second.disk_read_bytes, 100)
        self.assertEqual(second.disk_write_count, 4)
        self.assertEqual(second.battery_percent, 55)
        self.assertFalse(second.ac_connected)

    def fake_nvml(self):
        return SimpleNamespace(
            NVML_TEMPERATURE_GPU=0, NVML_CLOCK_GRAPHICS=0, NVML_CLOCK_SM=1, NVML_CLOCK_MEM=2,
            nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=91),
            nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(used=2**30, total=8 * 2**30),
            nvmlDeviceGetTemperature=lambda h, sensor: 73,
            nvmlDeviceGetClockInfo=lambda h, clock: [1500, 1450, 7000][clock],
            nvmlDeviceGetPowerUsage=lambda h: 95123,
            nvmlDeviceGetPowerManagementLimit=lambda h: 120000,
            nvmlDeviceGetPerformanceState=lambda h: 2,
            nvmlDeviceGetCurrentClocksEventReasons=lambda h: 0x44,
        )

    def test_nvml_units_throttle_bits_and_partial_failure(self):
        nvml = self.fake_nvml()
        nvml.nvmlDeviceGetTemperature = Mock(side_effect=RuntimeError("unsupported"))
        with patch.object(resources, "pynvml", nvml), \
                patch.object(resources, "_nvml_handle", return_value=object()):
            snap = resources.sample()
        self.assertTrue(snap.gpu_available)
        self.assertEqual(snap.gpu_percent, 91)
        self.assertEqual(snap.vram_total_gb, 8)
        self.assertIsNone(snap.temperature_c)
        self.assertEqual(snap.sm_clock_mhz, 1450)
        self.assertEqual(snap.graphics_clock_mhz, 1500)
        self.assertEqual(snap.memory_clock_mhz, 7000)
        self.assertAlmostEqual(snap.power_w, 95.123)
        self.assertEqual(snap.power_limit_w, 120)
        self.assertEqual(snap.pstate, 2)
        self.assertEqual(snap.throttle_reasons_mask, 0x44)
        self.assertEqual(snap.throttle_reasons, ["sw_power_cap", "hw_thermal_slowdown"])

    def test_nvml_old_api_zero_mask_and_unknown_pstate(self):
        nvml = self.fake_nvml()
        del nvml.nvmlDeviceGetCurrentClocksEventReasons
        nvml.nvmlDeviceGetCurrentClocksThrottleReasons = Mock(return_value=0)
        nvml.nvmlDeviceGetPerformanceState = lambda h: 32
        with patch.object(resources, "pynvml", nvml), \
                patch.object(resources, "_nvml_handle", return_value=object()):
            snap = resources.sample()
        self.assertEqual(snap.throttle_reasons, [])
        self.assertEqual(snap.throttle_reasons_mask, 0)
        self.assertIsNone(snap.pstate)
        self.assertEqual(snap.temperature_c, 73)

    def test_throttle_unknown_1024_preserved_and_32_is_sw_thermal(self):
        nvml = self.fake_nvml()
        nvml.nvmlDeviceGetCurrentClocksEventReasons = Mock(side_effect=[1024, 32, 1056])
        with patch.object(resources, "pynvml", nvml), \
                patch.object(resources, "_nvml_handle", return_value=object()):
            snapshots = [resources.sample() for _ in range(3)]
        self.assertEqual([s.throttle_reasons_mask for s in snapshots], [1024, 32, 1056])
        self.assertEqual([s.throttle_reasons for s in snapshots],
                         [[], ["sw_thermal_slowdown"], ["sw_thermal_slowdown"]])

    def test_ac_fallback_only_when_battery_missing(self):
        with patch.object(resources.psutil, "sensors_battery", return_value=None), \
                patch.object(resources, "_windows_ac_connected", return_value=True) as fallback:
            snap = resources.sample()
        fallback.assert_called_once()
        self.assertTrue(snap.ac_connected)
        self.assertIsNone(snap.battery_percent)
        with patch.object(resources.psutil, "sensors_battery",
                          return_value=SimpleNamespace(percent=50, power_plugged=False)), \
                patch.object(resources, "_windows_ac_connected") as fallback:
            snap = resources.sample()
        fallback.assert_not_called()
        self.assertFalse(snap.ac_connected)

    def test_win32_ac_status_success_unknown_and_failure(self):
        import ctypes

        def query(pointer):
            pointer._obj.ACLineStatus = ac
            pointer._obj.BatteryFlag = 128  # no battery must not imply no AC
            return success

        api = Mock(side_effect=query)
        with patch.object(resources.sys, "platform", "win32"), \
                patch.object(ctypes, "WinDLL", create=True,
                             return_value=SimpleNamespace(GetSystemPowerStatus=api)):
            for ac, success, expected in ((1, 1, True), (0, 1, False),
                                          (255, 1, None), (1, 0, None)):
                self.assertIs(resources._windows_ac_connected(), expected)
        with patch.object(resources.sys, "platform", "linux"):
            self.assertIsNone(resources._windows_ac_connected())

    def test_basic_inventory_and_sample_never_import_ctranslate2(self):
        with patch.object(resources.importlib, "import_module") as importer:
            resources.sample()
            info = resources.hardware_inventory()
        importer.assert_not_called()
        self.assertIsNone(info["ctranslate2_supported_types"])

    def test_only_explicit_detailed_inventory_probes_compute_types(self):
        ct2 = SimpleNamespace(get_supported_compute_types=Mock(
            side_effect=[{"int8", "float32"}, RuntimeError("no CUDA")]))
        with patch.object(resources.importlib, "import_module", return_value=ct2) as importer:
            info = resources.hardware_inventory(detailed=True)
        importer.assert_called_once_with("ctranslate2")
        self.assertEqual(info["ctranslate2_supported_types"], {"cpu": ["float32", "int8"], "cuda": None})
        self.assertEqual([c.args for c in ct2.get_supported_compute_types.call_args_list],
                         [("cpu",), ("cuda",)])
        self.assertIsNone(info["cuda_available"])

    def test_nvml_inventory_reports_capability_without_claiming_runtime(self):
        nvml = self.fake_nvml()
        nvml.nvmlDeviceGetCudaComputeCapability = lambda h: (8, 6)
        nvml.nvmlSystemGetCudaDriverVersion_v2 = lambda: 12060
        nvml.nvmlDeviceGetName = lambda h: b"test GPU"
        nvml.nvmlDeviceGetUUID = lambda h: b"GPU-test"
        with patch.object(resources, "pynvml", nvml), \
                patch.object(resources, "_nvml_handle", return_value=object()), \
                patch.object(resources.importlib, "import_module") as importer:
            info = resources.hardware_inventory()
        importer.assert_not_called()
        self.assertEqual(info["compute_capability"], "8.6")
        self.assertTrue(info["cuda_capable"])
        self.assertEqual(info["cuda_driver_version"], 12060)
        self.assertEqual(info["gpu_name"], "test GPU")
        self.assertIsNone(info["cuda_available"])
        self.assertIsNone(info["ctranslate2_supported_types"])

    def test_missing_capability_and_failed_detailed_import_remain_unknown(self):
        with patch.object(resources.importlib, "import_module", side_effect=ImportError):
            info = resources.hardware_inventory(detailed=True)
        self.assertIsNone(info["compute_capability"])
        self.assertIsNone(info["cuda_capable"])
        self.assertIsNone(info["cuda_driver_version"])
        self.assertIsNone(info["cuda_available"])
        self.assertEqual(info["ctranslate2_supported_types"], {"cpu": None, "cuda": None})

    def test_successful_detailed_probe_reports_runtime_availability(self):
        ct2 = SimpleNamespace(get_supported_compute_types=lambda device: {"float32", "float16"})
        with patch.object(resources.importlib, "import_module", return_value=ct2):
            info = resources.hardware_inventory(detailed=True)
        self.assertTrue(info["cuda_available"])
        self.assertEqual(info["ctranslate2_supported_types"]["cuda"], ["float16", "float32"])


class TestNvmlInitialization(unittest.TestCase):
    def test_lazy_init_once_under_concurrent_sampling(self):
        nvml = SimpleNamespace(nvmlInit=Mock(), nvmlDeviceGetHandleByIndex=Mock(return_value="gpu0"))
        with patch.object(resources, "_NVML_TRIED", False), \
                patch.object(resources, "_NVML_HANDLE", None), \
                patch.object(resources, "pynvml", None), \
                patch.object(resources.importlib, "import_module", return_value=nvml) as importer:
            threads = [threading.Thread(target=resources._nvml_handle) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(2)
                self.assertFalse(thread.is_alive())
            self.assertEqual(resources._NVML_HANDLE, "gpu0")
        importer.assert_called_once_with("pynvml")
        nvml.nvmlInit.assert_called_once()
        nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)

    def test_absent_nvml_does_not_retry_import_on_every_tick(self):
        with patch.object(resources, "_NVML_TRIED", False), \
                patch.object(resources, "_NVML_HANDLE", None), \
                patch.object(resources, "pynvml", None), \
                patch.object(resources.importlib, "import_module", side_effect=ImportError) as importer:
            self.assertIsNone(resources._nvml_handle())
            self.assertIsNone(resources._nvml_handle())
        importer.assert_called_once_with("pynvml")


if __name__ == "__main__":
    unittest.main()
