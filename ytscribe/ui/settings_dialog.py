"""Settings dialog — edits the Config dataclass, persisted on OK."""

from __future__ import annotations

from copy import deepcopy
import dataclasses
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)

from ..config import CACHE_POLICIES, EXPORT_FORMATS, Config, save_config
from ..optimization import PROFILE_DESCRIPTIONS, apply_profile, precision_options

WHISPER_MODELS = ("large-v3", "large-v3-turbo", "distil-large-v3", "medium", "small")
BROWSERS = ("", "chrome", "edge", "firefox", "brave", "opera")
CACHE_LABELS = ("Delete audio after each completed video",
                "Delete audio when the queue finishes",
                "Retain downloaded and processed audio until I delete it")


class _HardwareProbe(QThread):
    """Inventory is explicit and off the UI thread, including any CUDA probe."""
    result = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            from .. import resources
            self.result.emit(resources.hardware_inventory(detailed=True))
        except Exception as exc:
            self.failed.emit(str(exc))


class SettingsDialog(QDialog):
    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._base = deepcopy(cfg)
        self._supported_precisions = None
        self._precision_by_device = {}
        self._hardware_device = "auto"
        self._vram_total_gb = None
        self._probe = None
        self._pending_result = None
        self.setWindowTitle("Settings")
        self.setMinimumWidth(640)
        self.resize(760, 760)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        root = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # --- paths ---
        paths = QGroupBox("Paths")
        form = QFormLayout(paths)
        self.output_dir = self._dir_row(form, "Output directory", cfg.output_dir)
        self.cache_dir = self._dir_row(form, "Audio and stage cache", cfg.cache_dir)
        storage_help = QLabel("Transcripts are stored in the output directory. Each video's "
                              "cache folder holds downloaded/processed audio and stage JSON. "
                              "The retention policy below controls audio; stage JSON and "
                              "short speaker samples are kept for resume and renaming.")
        storage_help.setWordWrap(True)
        form.addRow(storage_help)
        root.addWidget(paths)

        # --- processing ---
        proc = QGroupBox("Processing")
        form = QFormLayout(proc)
        self.model = QComboBox(); self.model.addItems(WHISPER_MODELS)
        self.model.setEditable(True)  # allow any HF model id
        self.model.setCurrentText(cfg.whisper_model)
        self.model.setToolTip("Model controls accuracy, speed and VRAM. large-v3 remains the "
                              "accuracy baseline; smaller models trade accuracy for speed/memory.")
        form.addRow("Whisper model", self.model)
        self.compute = QComboBox(); self.compute.addItems(precision_options(current=cfg.compute_type))
        self.compute.setCurrentText(cfg.compute_type)
        self.compute.setToolTip("Precision affects speed and VRAM and may change output. "
                                "Auto uses runtime capability checks. Probe hardware before "
                                "choosing a precision; unsupported saved values are preserved.")
        form.addRow("Compute type", self.compute)
        self.language = QLineEdit(cfg.language)
        self.language.setPlaceholderText('"auto" or ISO code like "en"')
        form.addRow("Language", self.language)
        self.language.setToolTip("A known language skips detection; use auto for mixed/unknown "
                                 "languages. A wrong language can reduce accuracy; little VRAM effect.")
        self.beam = QSpinBox(); self.beam.setRange(1, 2147483647); self.beam.setValue(cfg.beam_size)
        self.beam.setToolTip("Larger beams cost more time and memory and can improve decoding. "
                             "Keep 5 for the baseline; compare quality before lowering it.")
        form.addRow("Beam size", self.beam)
        self.min_spk = QSpinBox(); self.min_spk.setRange(0, 2147483647)
        self.min_spk.setValue(cfg.min_speakers)
        self.min_spk.setSpecialValueText("auto")
        form.addRow("Min speakers", self.min_spk)
        self.max_spk = QSpinBox(); self.max_spk.setRange(0, 2147483647)
        self.max_spk.setValue(cfg.max_speakers)
        self.max_spk.setSpecialValueText("auto")
        form.addRow("Max speakers", self.max_spk)
        for control in (self.min_spk, self.max_spk):
            control.setToolTip("0 leaves speaker count unconstrained. Set a known count to "
                               "help attribution; incorrect bounds can hurt accuracy. "
                               "This is not a primary speed or memory control.")
        root.addWidget(proc)

        opt = QGroupBox("Optimization")
        form = QFormLayout(opt)
        self.hardware_label = QLabel("Hardware has not been probed. Opening Settings does not initialize CUDA.")
        self.hardware_label.setWordWrap(True)
        form.addRow(self.hardware_label)
        self.probe_btn = QPushButton("Probe hardware capabilities")
        self.probe_btn.clicked.connect(self._probe_hardware)
        form.addRow(self.probe_btn)
        self.profile = QComboBox()
        for key, title in (("custom", "Custom"), ("safe", "Safe / Conservative"),
                           ("balanced", "Balanced / Baseline"),
                           ("performance", "Performance / Optional batch mode")):
            self.profile.addItem(title, key)
        self.profile.setCurrentIndex(self.profile.findData(cfg.optimization_profile))
        preset_row = QHBoxLayout()
        preset_row.addWidget(self.profile)
        self.apply_preset_btn = QPushButton("Apply preset")
        self.apply_preset_btn.clicked.connect(self._apply_preset)
        preset_row.addWidget(self.apply_preset_btn)
        form.addRow("Preset", preset_row)
        self.profile_help = QLabel()
        self.profile_help.setWordWrap(True)
        self.profile.currentIndexChanged.connect(self._profile_description)
        self._profile_description()
        form.addRow(self.profile_help)
        self.device = QComboBox(); self.device.addItems(("auto", "cuda", "cpu"))
        self.device.setCurrentText(cfg.device)
        self.device.currentTextChanged.connect(self._update_precision_options)
        self.device.setToolTip("Auto selects an available device. CPU usually uses less GPU "
                               "memory but is slower; the model and decoding settings still govern accuracy.")
        form.addRow("Device", self.device)
        self.asr_batch = self._integer(form, "ASR batch size", cfg.asr_batch_size, 1,
            "1 uses standard transcription (default). Values above 1 opt into batched "
            "decoding with different semantics and higher VRAM use. Batch 4 was faster on "
            "one measured full source in two runs; speed and quality can vary by recording.")
        self.cpu_threads = self._integer(form, "ASR CPU threads", cfg.asr_cpu_threads, 1,
            "CPU threads per ASR worker. More can improve CPU feeding but contend with other "
            "work and use more RAM; no intended accuracy change. Start with 4.")
        self.chunk_length = self._integer(form, "ASR chunk length (s)", cfg.asr_chunk_length, 1,
            "Audio per ASR chunk (1–30 seconds). Smaller chunks may reduce memory and change "
            "context/accuracy or speed. Keep 30 unless benchmarking.", maximum=30)
        self.diar_batch = self._integer(form, "Diarization batch size", cfg.diarization_batch_size, 1,
            "Segmentation/embedding batch size. Larger batches may improve throughput at "
            "higher VRAM cost; no intended accuracy change. Installed community-1 uses 32.")
        self.residency = QComboBox()
        self.residency.addItem("Keep models loaded across stages", "keep")
        self.residency.addItem("Release models between stages (lower VRAM)", "stage")
        self.residency.setCurrentIndex(self.residency.findData(cfg.model_residency))
        self.residency.setToolTip("Stage residency reduces simultaneous model VRAM, but repeated "
                                  "loads may slow processing. No intended accuracy change.")
        form.addRow("Model residency", self.residency)
        self.vram_margin = self._integer(form, "VRAM safety margin (MiB)", cfg.vram_margin_mb, 0,
            "Headroom for other GPU allocations. A larger margin may reduce effective batching "
            "and speed but lowers OOM risk; no intended accuracy change.")
        self.oom_retries = self._integer(form, "OOM retry limit", cfg.oom_retries, 0,
            "Maximum retries after a memory failure while reducing batch size. Extra retries "
            "cost time; they do not enable concurrent GPU jobs or alter the selected model.")
        self.vad = QCheckBox("Filter non-speech before transcription")
        self.vad.setChecked(cfg.vad_filter)
        self.vad.setToolTip("Usually saves compute and avoids silence hallucinations; may discard "
                            "very quiet speech. Keep enabled unless comparing a difficult recording.")
        form.addRow("Voice activity detection", self.vad)
        self.word_timestamps = QCheckBox("Keep word timestamps (recommended for speaker attribution)")
        self.word_timestamps.setChecked(cfg.word_timestamps)
        self.word_timestamps.setToolTip("Disable only for benchmark comparisons. May save compute, "
                                        "but reduces speaker-boundary precision. Presets keep this on.")
        form.addRow("Benchmark option", self.word_timestamps)
        self.profiling = QCheckBox("Record diagnostic performance traces")
        self.profiling.setChecked(cfg.profiling_enabled)
        self.profiling.setToolTip("Records timing and resource samples to diagnose performance. "
                                  "Small CPU/disk overhead; no intended accuracy effect.")
        form.addRow("Profiling", self.profiling)
        self.profiling_interval = QDoubleSpinBox()
        self.profiling_interval.setDecimals(6)
        self.profiling_interval.setRange(0.000001, max(86400, cfg.profiling_interval))
        self.profiling_interval.setValue(cfg.profiling_interval)
        self.profiling_interval.setSuffix(" s")
        self.profiling_interval.setToolTip("Time between trace samples. Short intervals produce "
                                          "more detail, overhead and disk use; 1 s is a starting point.")
        form.addRow("Profiling interval", self.profiling_interval)
        root.addWidget(opt)

        # --- downloads ---
        dl = QGroupBox("Downloads")
        form = QFormLayout(dl)
        self.sleep_min = QDoubleSpinBox(); self.sleep_min.setDecimals(6)
        self.sleep_min.setRange(0, max(86400, cfg.sleep_between_downloads_min))
        self.sleep_min.setValue(cfg.sleep_between_downloads_min); self.sleep_min.setSuffix(" s")
        form.addRow("Sleep between downloads (min)", self.sleep_min)
        self.sleep_max = QDoubleSpinBox(); self.sleep_max.setDecimals(6)
        self.sleep_max.setRange(0, max(86400, cfg.sleep_between_downloads_max))
        self.sleep_max.setValue(cfg.sleep_between_downloads_max); self.sleep_max.setSuffix(" s")
        form.addRow("Sleep between downloads (max)", self.sleep_max)
        self.retries = QSpinBox(); self.retries.setRange(0, 2147483647)
        self.retries.setValue(cfg.max_retries)
        form.addRow("Max retries per video", self.retries)
        self.cookies = QComboBox(); self.cookies.addItems(BROWSERS)
        self.cookies.setEditable(True)
        self.cookies.setCurrentText(cfg.cookies_from_browser)
        form.addRow("Cookies from browser", self.cookies)
        self.cache_policy = QComboBox()
        for label, value in zip(CACHE_LABELS, CACHE_POLICIES):
            self.cache_policy.addItem(label, value)
        self.cache_policy.setCurrentIndex(self.cache_policy.findData(cfg.cache_policy))
        form.addRow("Audio cache policy", self.cache_policy)
        root.addWidget(dl)

        # --- output ---
        out = QGroupBox("Output")
        form = QFormLayout(out)
        fmt_row = QHBoxLayout()
        self.fmt_boxes = {}
        for fmt in EXPORT_FORMATS:
            box = QCheckBox(fmt.upper())
            box.setChecked(fmt in cfg.export_formats)
            self.fmt_boxes[fmt] = box
            fmt_row.addWidget(box)
        form.addRow("Export formats", fmt_row)
        self.combined = QCheckBox("Write combined transcript for each finished queue")
        self.combined.setChecked(cfg.combined_transcript)
        form.addRow("", self.combined)
        self.hf_token = QLineEdit(cfg.hf_token)
        self.hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token.setPlaceholderText("only needed for gated diarization models")
        form.addRow("HuggingFace token", self.hf_token)
        root.addWidget(out)

        # --- speaker recognition ---
        rec = QGroupBox("Speaker recognition")
        form = QFormLayout(rec)
        self.rec_enabled = QCheckBox("Automatically name recognized voices "
                                     "from the voice database")
        self.rec_enabled.setChecked(cfg.recognition_enabled)
        form.addRow("", self.rec_enabled)
        self.rec_threshold = QDoubleSpinBox()
        self.rec_threshold.setDecimals(6)
        self.rec_threshold.setRange(0, 1)
        self.rec_threshold.setSingleStep(0.05)
        self.rec_threshold.setValue(cfg.recognition_threshold)
        self.rec_threshold.setToolTip("Minimum voice similarity for an "
                                      "automatic match (higher = stricter)")
        form.addRow("Match threshold", self.rec_threshold)
        root.addWidget(rec)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        # Merely browsing the preset selector never applies a preset on save.
        self._applied_profile = cfg.optimization_profile
        for control in (self.model, self.compute, self.device, self.residency):
            control.currentTextChanged.connect(self._mark_custom)
        for control in (self.beam, self.asr_batch, self.cpu_threads, self.chunk_length,
                        self.diar_batch, self.vram_margin, self.oom_retries):
            control.valueChanged.connect(self._mark_custom)
        self.vad.toggled.connect(self._mark_custom)
        self.word_timestamps.toggled.connect(self._mark_custom)
        self._initial_numeric_values = {
            name: control.value() for name, control in (
                ("recognition_threshold", self.rec_threshold),
                ("profiling_interval", self.profiling_interval),
                ("sleep_between_downloads_min", self.sleep_min),
                ("sleep_between_downloads_max", self.sleep_max))
        }

    def _integer(self, form, label, value, minimum, tooltip, maximum=2147483647):
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        control.setToolTip(tooltip)
        form.addRow(label, control)
        return control

    def _profile_description(self, *_):
        self.profile_help.setText(PROFILE_DESCRIPTIONS[self.profile.currentData()])

    def _mark_custom(self, *_):
        self._applied_profile = "custom"
        self.profile.setCurrentIndex(self.profile.findData("custom"))

    def _apply_preset(self):
        profile = self.profile.currentData()
        device = self.device.currentText()
        supported = self._supported_precisions
        candidate = apply_profile(self._candidate(), profile,
                                  supported_compute_types=supported,
                                  device=self._hardware_device if device == "auto" else device,
                                  vram_total_gb=self._vram_total_gb if device != "cpu" else None)
        self.model.setCurrentText(candidate.whisper_model)
        self.compute.setCurrentText(candidate.compute_type)
        self.beam.setValue(candidate.beam_size)
        self.asr_batch.setValue(candidate.asr_batch_size)
        self.chunk_length.setValue(candidate.asr_chunk_length)
        self.residency.setCurrentIndex(self.residency.findData(candidate.model_residency))
        self.vram_margin.setValue(candidate.vram_margin_mb)
        self.oom_retries.setValue(candidate.oom_retries)
        self.word_timestamps.setChecked(candidate.word_timestamps)
        self.vad.setChecked(candidate.vad_filter)
        self._base.asr_num_workers = candidate.asr_num_workers
        self._applied_profile = profile
        self.profile.setCurrentIndex(self.profile.findData(profile))

    def _probe_hardware(self):
        if self._probe and self._probe.isRunning():
            return
        self.probe_btn.setEnabled(False)
        self.hardware_label.setText("Probing hardware capabilities…")
        self._probe = _HardwareProbe(self)
        self._probe.result.connect(self._hardware_ready)
        self._probe.failed.connect(lambda text: self.hardware_label.setText(f"Hardware probe unavailable: {text}"))
        self._probe.finished.connect(lambda: self.probe_btn.setEnabled(True))
        self._probe.start()

    def _hardware_ready(self, info):
        data = info if isinstance(info, dict) else vars(info)
        self._precision_by_device = data.get("ctranslate2_supported_types") or {}
        cuda_types = self._precision_by_device.get("cuda")
        self._hardware_device = "cuda" if cuda_types else "cpu"
        vram_bytes = data.get("gpu_vram_total_bytes")
        self._vram_total_gb = vram_bytes / 2**30 if vram_bytes else None
        ram_bytes = data.get("ram_total_bytes")
        ram_text = f"{ram_bytes / 2**30:.1f}" if ram_bytes else "?"
        capability = data.get("compute_capability") or "unknown"
        cuda = data.get("cuda_available", bool(cuda_types) if cuda_types is not None else None)
        cuda_text = "unknown" if cuda is None else ("available" if cuda else "unavailable")
        precision_text = "; ".join(
            f"{device}: {', '.join(types) if types else 'unavailable/unknown'}"
            for device, types in self._precision_by_device.items()) or "not probed"
        self.hardware_label.setText(
            f"GPU: {data.get('gpu_name') or 'not detected'}; VRAM: {self._vram_total_gb or '?'} GiB; "
            f"CUDA: {cuda_text}; compute capability: {capability}.\n"
            f"CPU: {data.get('cpu_physical_count') or '?'} physical / "
            f"{data.get('cpu_logical_count') or '?'} logical; RAM: {ram_text} GiB.\n"
            f"Reported precisions: {precision_text}. Form factor: {data.get('form_factor') or 'unknown'}.")
        self._update_precision_options()

    def _update_precision_options(self, *_):
        device = self.device.currentText()
        if device == "auto":
            device = self._hardware_device
        self._supported_precisions = self._precision_by_device.get(device)
        current = self.compute.currentText()
        self.compute.blockSignals(True)
        self.compute.clear()
        self.compute.addItems(precision_options(self._supported_precisions, current))
        self.compute.setCurrentText(current)
        self.compute.blockSignals(False)

    def done(self, result):
        if self._probe and (self._probe.isRunning() or not self._probe.wait(0)):
            self._pending_result = result
            self.setEnabled(False)
            QTimer.singleShot(100, self._finish_when_ready)
            return
        super().done(result)

    def closeEvent(self, event):
        if self._probe and (self._probe.isRunning() or not self._probe.wait(0)):
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)

    def _finish_when_ready(self):
        if self._probe and (self._probe.isRunning() or not self._probe.wait(0)):
            QTimer.singleShot(100, self._finish_when_ready)
            return
        super().done(self._pending_result)

    def _dir_row(self, form: QFormLayout, label: str, value: str) -> QLineEdit:
        edit = QLineEdit(value)
        btn = QPushButton("Browse…")

        def browse():
            path = QFileDialog.getExistingDirectory(self, label, edit.text())
            if path:
                edit.setText(path)

        btn.clicked.connect(browse)
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(lambda: self._open_directory(edit.text()))
        row.addWidget(open_btn)
        form.addRow(label, row)
        return edit

    def _open_directory(self, value):
        if not value.strip():
            return
        try:
            path = Path(value).expanduser().absolute()
            path.mkdir(parents=True, exist_ok=True)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
                raise OSError("The system could not open this folder.")
        except OSError as exc:
            QMessageBox.warning(self, "Could not open folder", str(exc))

    def _candidate(self):
        c = deepcopy(self._base)
        c.output_dir = self.output_dir.text().strip()
        c.cache_dir = self.cache_dir.text().strip()
        c.whisper_model = self.model.currentText().strip()
        c.compute_type = self.compute.currentText()
        c.language = self.language.text().strip() or "auto"
        c.beam_size = self.beam.value()
        c.min_speakers = self.min_spk.value()
        c.max_speakers = self.max_spk.value()
        c.sleep_between_downloads_min = self.sleep_min.value()
        c.sleep_between_downloads_max = self.sleep_max.value()
        c.max_retries = self.retries.value()
        c.cookies_from_browser = self.cookies.currentText()
        c.cache_policy = self.cache_policy.currentData()
        selected = [f for f, b in self.fmt_boxes.items() if b.isChecked()]
        c.export_formats = [f for f in self._base.export_formats if f in selected]
        c.export_formats.extend(f for f in selected if f not in c.export_formats)
        c.combined_transcript = self.combined.isChecked()
        c.hf_token = self.hf_token.text().strip()
        c.recognition_enabled = self.rec_enabled.isChecked()
        c.recognition_threshold = self.rec_threshold.value()
        c.device = self.device.currentText()
        c.asr_batch_size = self.asr_batch.value()
        c.asr_cpu_threads = self.cpu_threads.value()
        c.asr_chunk_length = self.chunk_length.value()
        c.diarization_batch_size = self.diar_batch.value()
        c.model_residency = self.residency.currentData()
        c.vram_margin_mb = self.vram_margin.value()
        c.oom_retries = self.oom_retries.value()
        c.vad_filter = self.vad.isChecked()
        c.word_timestamps = self.word_timestamps.isChecked()
        c.profiling_enabled = self.profiling.isChecked()
        c.profiling_interval = self.profiling_interval.value()
        c.optimization_profile = self._applied_profile
        # Spin boxes have finite display precision. Preserve untouched exact values.
        for name, control in (("recognition_threshold", self.rec_threshold),
                              ("profiling_interval", self.profiling_interval),
                              ("sleep_between_downloads_min", self.sleep_min),
                              ("sleep_between_downloads_max", self.sleep_max)):
            original = getattr(self._base, name)
            if control.value() == self._initial_numeric_values[name]:
                setattr(c, name, original)
        return c

    def _save(self):
        c = self._candidate()
        problems = c.validate()
        if problems:
            QMessageBox.warning(self, "Invalid settings", "\n".join(problems))
            return
        try:
            save_config(c)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save settings", str(exc))
            return
        # Persist the validated candidate before updating the caller's object.
        for field in dataclasses.fields(c):
            setattr(self.cfg, field.name, deepcopy(getattr(c, field.name)))
        self.accept()
