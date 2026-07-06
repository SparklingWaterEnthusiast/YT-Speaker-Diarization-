"""System resource sampling for the UI status bar (1 Hz polling)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # no NVIDIA GPU / driver — degrade gracefully
    _NVML_HANDLE = None


@dataclass
class Snapshot:
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_percent: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    gpu_available: bool = False


def sample(disk_path: str) -> Snapshot:
    snap = Snapshot()
    snap.cpu_percent = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    snap.ram_used_gb = (vm.total - vm.available) / 2**30
    snap.ram_total_gb = vm.total / 2**30
    try:
        usage = shutil.disk_usage(disk_path)
        snap.disk_free_gb = usage.free / 2**30
    except OSError:
        pass
    if _NVML_HANDLE is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
            snap.gpu_percent = util.gpu
            snap.vram_used_gb = mem.used / 2**30
            snap.vram_total_gb = mem.total / 2**30
            snap.gpu_available = True
        except Exception:
            pass
    return snap
