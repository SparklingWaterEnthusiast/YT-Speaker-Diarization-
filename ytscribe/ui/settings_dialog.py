"""Settings dialog — edits the Config dataclass, persisted on OK."""

from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLineEdit, QPushButton,
                               QSpinBox, QVBoxLayout)

from ..config import CACHE_POLICIES, EXPORT_FORMATS, Config, save_config

WHISPER_MODELS = ("large-v3", "large-v3-turbo", "distil-large-v3", "medium", "small")
COMPUTE_TYPES = ("auto", "int8_float16", "float16", "int8")
BROWSERS = ("", "chrome", "edge", "firefox", "brave", "opera")


class SettingsDialog(QDialog):
    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        root = QVBoxLayout(self)

        # --- paths ---
        paths = QGroupBox("Paths")
        form = QFormLayout(paths)
        self.output_dir = self._dir_row(form, "Output directory", cfg.output_dir)
        self.cache_dir = self._dir_row(form, "Audio cache directory", cfg.cache_dir)
        root.addWidget(paths)

        # --- processing ---
        proc = QGroupBox("Processing")
        form = QFormLayout(proc)
        self.model = QComboBox(); self.model.addItems(WHISPER_MODELS)
        self.model.setCurrentText(cfg.whisper_model)
        self.model.setEditable(True)  # allow any HF model id
        form.addRow("Whisper model", self.model)
        self.compute = QComboBox(); self.compute.addItems(COMPUTE_TYPES)
        self.compute.setCurrentText(cfg.compute_type)
        form.addRow("Compute type", self.compute)
        self.language = QLineEdit(cfg.language)
        self.language.setPlaceholderText('"auto" or ISO code like "en"')
        form.addRow("Language", self.language)
        self.beam = QSpinBox(); self.beam.setRange(1, 10); self.beam.setValue(cfg.beam_size)
        form.addRow("Beam size", self.beam)
        self.min_spk = QSpinBox(); self.min_spk.setRange(0, 20)
        self.min_spk.setValue(cfg.min_speakers)
        self.min_spk.setSpecialValueText("auto")
        form.addRow("Min speakers", self.min_spk)
        self.max_spk = QSpinBox(); self.max_spk.setRange(0, 20)
        self.max_spk.setValue(cfg.max_speakers)
        self.max_spk.setSpecialValueText("auto")
        form.addRow("Max speakers", self.max_spk)
        root.addWidget(proc)

        # --- downloads ---
        dl = QGroupBox("Downloads")
        form = QFormLayout(dl)
        self.sleep_min = QDoubleSpinBox(); self.sleep_min.setRange(0, 600)
        self.sleep_min.setValue(cfg.sleep_between_downloads_min); self.sleep_min.setSuffix(" s")
        form.addRow("Sleep between downloads (min)", self.sleep_min)
        self.sleep_max = QDoubleSpinBox(); self.sleep_max.setRange(0, 600)
        self.sleep_max.setValue(cfg.sleep_between_downloads_max); self.sleep_max.setSuffix(" s")
        form.addRow("Sleep between downloads (max)", self.sleep_max)
        self.retries = QSpinBox(); self.retries.setRange(0, 10)
        self.retries.setValue(cfg.max_retries)
        form.addRow("Max retries per video", self.retries)
        self.cookies = QComboBox(); self.cookies.addItems(BROWSERS)
        self.cookies.setCurrentText(cfg.cookies_from_browser)
        form.addRow("Cookies from browser", self.cookies)
        self.cache_policy = QComboBox(); self.cache_policy.addItems(CACHE_POLICIES)
        self.cache_policy.setCurrentText(cfg.cache_policy)
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

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

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
        form.addRow(label, row)
        return edit

    def _save(self):
        c = self.cfg
        c.output_dir = self.output_dir.text().strip()
        c.cache_dir = self.cache_dir.text().strip()
        c.whisper_model = self.model.currentText().strip()
        c.compute_type = self.compute.currentText()
        c.language = self.language.text().strip() or "auto"
        c.beam_size = self.beam.value()
        c.min_speakers = self.min_spk.value()
        c.max_speakers = self.max_spk.value()
        c.sleep_between_downloads_min = self.sleep_min.value()
        c.sleep_between_downloads_max = max(self.sleep_max.value(), self.sleep_min.value())
        c.max_retries = self.retries.value()
        c.cookies_from_browser = self.cookies.currentText()
        c.cache_policy = self.cache_policy.currentText()
        c.export_formats = [f for f, b in self.fmt_boxes.items() if b.isChecked()]
        c.combined_transcript = self.combined.isChecked()
        c.hf_token = self.hf_token.text().strip()
        save_config(c)
        self.accept()
