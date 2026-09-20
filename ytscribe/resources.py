"""Nonblocking UI/trace counters; never imports torch or a CUDA runtime.

Legacy UI fields keep numeric defaults. New metrics are None when unavailable.
NVML is independent of torch and is initialized lazily on the first probe.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
import threading
from dataclasses import dataclass

import psutil

_NVML_HANDLE = None
_NVML_TRIED = False
_NVML_LOCK = threading.Lock()
pynvml = None
_LOCAL = threading.local()


def _windows_ac_connected() -> bool | None:
    """Read AC status independently of battery presence; unknown stays None.

    https://learn.microsoft.com/windows/win32/api/winbase/ns-winbase-system_power_status
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte),
                        ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte),
                        ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_uint32),
                        ("BatteryFullLifeTime", ctypes.c_uint32)]

        # A local function object avoids shared argtypes mutation across the
        # UI/sampler threads. The OS reuses the loaded system DLL.
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetSystemPowerStatus
        query.argtypes = [ctypes.POINTER(PowerStatus)]
        query.restype = ctypes.c_int
        status = PowerStatus()
        if query(ctypes.byref(status)) and status.ACLineStatus in (0, 1):
            return bool(status.ACLineStatus)
    except Exception:
        pass
    return None


def _optional(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except Exception:
        return None


def _nvml_handle():
    global pynvml, _NVML_HANDLE, _NVML_TRIED
    with _NVML_LOCK:
        if not _NVML_TRIED:
            _NVML_TRIED = True
            try:
                pynvml = importlib.import_module("pynvml")
                pynvml.nvmlInit()
                _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                _NVML_HANDLE = None
        return _NVML_HANDLE


def _nvml(name, *args):
    return _optional(getattr(pynvml, name, None), *args)


def _process():
    # Separate psutil CPU baselines for the UI and sampler threads.
    proc = getattr(_LOCAL, "process", None)
    if proc is None:
        proc = psutil.Process()
        _LOCAL.process = proc
        _LOCAL.cpu_primed = False
    return proc


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
    temperature_c: float | None = None
    graphics_clock_mhz: int | None = None
    sm_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    pstate: int | None = None
    throttle_reasons_mask: int | None = None
    throttle_reasons: list[str] | None = None
    process_cpu_percent: float | None = None
    process_rss_bytes: int | None = None
    process_threads: int | None = None
    cpu_per_core_percent: list[float] | None = None
    disk_read_bytes: int | None = None
    disk_write_bytes: int | None = None
    disk_read_count: int | None = None
    disk_write_count: int | None = None
    battery_percent: float | None = None
    ac_connected: bool | None = None


# NVML clocks-event bits: idle/application caps are not necessarily harmful.
# Preserve the raw mask, including unknown future bits.
# https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlClocksEventReasons.html
_CLOCK_REASONS = {
    0x001: "gpu_idle", 0x002: "applications_clocks_setting",
    0x004: "sw_power_cap", 0x008: "hw_slowdown", 0x010: "sync_boost",
    0x020: "sw_thermal_slowdown", 0x040: "hw_thermal_slowdown",
    0x080: "hw_power_brake_slowdown", 0x100: "display_clock_setting",
}


def sample(disk_path: str = ".") -> Snapshot:
    """Sample GPU 0 and this process without initializing a CUDA context.

    CPU percentages are since this thread's last call; process CPU may exceed
    100% (psutil semantics), and its first reading is None while priming.
    Disk I/O values are cumulative system counters, not rates/path-specific.
    NVML power can be a 1 s average. Historical *_gb fields use GiB (2**30).
    """
    snap = Snapshot()
    cpu = _optional(psutil.cpu_percent, interval=None)
    if cpu is not None:
        snap.cpu_percent = cpu
    snap.cpu_per_core_percent = _optional(psutil.cpu_percent, interval=None, percpu=True)
    vm = _optional(psutil.virtual_memory)
    if vm is not None:
        snap.ram_used_gb = (vm.total - vm.available) / 2**30
        snap.ram_total_gb = vm.total / 2**30
    usage = _optional(shutil.disk_usage, disk_path)
    if usage is not None:
        snap.disk_free_gb = usage.free / 2**30
    proc = _optional(_process)
    if proc is not None:
        cpu = _optional(proc.cpu_percent, interval=None)
        if getattr(_LOCAL, "cpu_primed", False):
            snap.process_cpu_percent = cpu
        _LOCAL.cpu_primed = cpu is not None
        mem = _optional(proc.memory_info)
        if mem is not None:
            snap.process_rss_bytes = mem.rss
        snap.process_threads = _optional(proc.num_threads)
    io = _optional(psutil.disk_io_counters)
    if io is not None:
        for field in ("read_bytes", "write_bytes", "read_count", "write_count"):
            setattr(snap, "disk_" + field, getattr(io, field, None))
    battery = _optional(getattr(psutil, "sensors_battery", None))
    if battery is not None:
        snap.battery_percent = battery.percent
        snap.ac_connected = battery.power_plugged
    else:
        snap.ac_connected = _windows_ac_connected()

    handle = _nvml_handle()
    if handle is None:
        return snap
    util = _nvml("nvmlDeviceGetUtilizationRates", handle)
    mem = _nvml("nvmlDeviceGetMemoryInfo", handle)
    if util is not None:
        snap.gpu_percent = util.gpu
    if mem is not None:
        snap.vram_used_gb = mem.used / 2**30
        snap.vram_total_gb = mem.total / 2**30
    snap.gpu_available = util is not None and mem is not None
    snap.temperature_c = _nvml(
        "nvmlDeviceGetTemperature", handle, getattr(pynvml, "NVML_TEMPERATURE_GPU", 0))
    for field, constant in (("graphics", "NVML_CLOCK_GRAPHICS"),
                            ("sm", "NVML_CLOCK_SM"), ("memory", "NVML_CLOCK_MEM")):
        clock_type = getattr(pynvml, constant, None)
        if clock_type is not None:
            setattr(snap, f"{field}_clock_mhz",
                    _nvml("nvmlDeviceGetClockInfo", handle, clock_type))
    # NVML returns milliwatts. Each query can fail independently.
    for field, query in (("power_w", "nvmlDeviceGetPowerUsage"),
                         ("power_limit_w", "nvmlDeviceGetPowerManagementLimit")):
        power = _nvml(query, handle)
        if power is not None:
            setattr(snap, field, power / 1000.0)
    state = _nvml("nvmlDeviceGetPerformanceState", handle)
    if state is not None and 0 <= state <= 15:
        snap.pstate = state  # 0 fastest, 15 slowest; unknown sentinel -> None
    mask = _nvml("nvmlDeviceGetCurrentClocksEventReasons", handle)
    if mask is None:  # older bindings/drivers
        mask = _nvml("nvmlDeviceGetCurrentClocksThrottleReasons", handle)
    snap.throttle_reasons_mask = mask
    if mask is not None:
        snap.throttle_reasons = [name for bit, name in _CLOCK_REASONS.items() if mask & bit]
    return snap


def _text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def hardware_inventory(*, detailed: bool = False) -> dict:
    """Explicit probe, never called by sample() or the trace sampler.

    Basic inventory uses OS/psutil/NVML only. detailed=True opts into importing
    CTranslate2 and querying CPU/CUDA compute types, which may initialize CUDA.
    No model is loaded/downloaded. Unavailable or unrequested values are None.
    compute_capability is a 'major.minor' string from NVML. cuda_capable describes
    hardware, not runtime usability: cuda_available stays None until a detailed
    CTranslate2 query succeeds. cuda_driver_version is CUDA's encoded integer
    (e.g. 12060), not the installed toolkit/runtime version.
    """
    vm = _optional(psutil.virtual_memory)
    info = {
        "platform": platform.platform(), "python_version": platform.python_version(),
        "cpu_name": platform.processor() or None,
        "cpu_logical_count": _optional(psutil.cpu_count),
        "cpu_physical_count": _optional(psutil.cpu_count, logical=False),
        "ram_total_bytes": vm.total if vm is not None else None,
        "gpu_name": None, "gpu_uuid": None, "gpu_driver_version": None,
        "gpu_vram_total_bytes": None, "ctranslate2_supported_types": None,
        "compute_capability": None, "cuda_capable": None,
        "cuda_driver_version": None, "cuda_available": None,
    }
    handle = _nvml_handle()
    if handle is not None:
        info["gpu_name"] = _text(_nvml("nvmlDeviceGetName", handle))
        info["gpu_uuid"] = _text(_nvml("nvmlDeviceGetUUID", handle))
        info["gpu_driver_version"] = _text(_nvml("nvmlSystemGetDriverVersion"))
        mem = _nvml("nvmlDeviceGetMemoryInfo", handle)
        info["gpu_vram_total_bytes"] = mem.total if mem is not None else None
        capability = _nvml("nvmlDeviceGetCudaComputeCapability", handle)
        if capability is not None and len(capability) == 2:
            major, minor = capability
            if major > 0 and minor >= 0:
                info["compute_capability"] = f"{major}.{minor}"
                info["cuda_capable"] = True
        # v2 queries the actual driver library; the legacy function may return
        # an assumed version when that library is absent. Neither proves that
        # this process can run a CUDA workload (visibility/DLLs may differ).
        version = _nvml("nvmlSystemGetCudaDriverVersion_v2")
        if version is not None and version > 0:
            info["cuda_driver_version"] = version
    if detailed:
        ct2 = _optional(importlib.import_module, "ctranslate2")
        supported = {}
        for device in ("cpu", "cuda"):
            types = _optional(getattr(ct2, "get_supported_compute_types", None), device)
            supported[device] = sorted(types) if types is not None else None
        info["ctranslate2_supported_types"] = supported
        if supported["cuda"] is not None:
            info["cuda_available"] = bool(supported["cuda"])
    return info
