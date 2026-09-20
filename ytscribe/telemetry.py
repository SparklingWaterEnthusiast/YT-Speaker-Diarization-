"""Opt-in JSONL performance traces; standard-library-only at import.

    with Recorder(path, disk_path=cache_dir, interval=1.0):
        with span("transcribe", cold=True, cache_hit=False, video_id=vid):
            ...
        event("model_reused", cold=False)

Module-level calls are no-ops unless a Recorder context is active. Recorder()
and Recorder(path, enabled=False) create no file/thread and perform no probes.
For a worker thread, capture current_recorder() in its parent and wrap work in
use_recorder(recorder). That context binds only; it does not close the recorder.
One recorder supports multiple threads; use separate files across processes.
Metadata must be JSON-serializable; do not supply transcripts or credentials.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import math
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path


_ACTIVE = contextvars.ContextVar("ytscribe_recorder", default=None)


def current_recorder():
    """Return the active Recorder or None, for explicit thread propagation."""
    return _ACTIVE.get()


@contextmanager
def use_recorder(recorder):
    """Bind a recorder in this context without starting/stopping/owning it."""
    token = _ACTIVE.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE.reset(token)


def event(name: str, **metadata) -> bool:
    recorder = current_recorder()
    return recorder.event(name, **metadata) if recorder is not None else False


@contextmanager
def span(name: str, **metadata):
    recorder = current_recorder()
    if recorder is None:
        yield
    else:
        with recorder.span(name, **metadata):
            yield


def runtime_flags() -> dict:
    """Observe loaded runtimes without importing them or probing devices."""
    torch = sys.modules.get("torch")
    initialized = None
    if torch is not None:
        try:
            initialized = bool(torch.cuda.is_initialized())
        except Exception:
            pass
    return {"torch_loaded": torch is not None,
            "ctranslate2_loaded": sys.modules.get("ctranslate2") is not None,
            "cuda_initialized": initialized}


def _realtime_ratios(audio_duration, duration_s):
    """Finite ratios only; zero wall duration has no defined multiplier."""
    try:
        if (isinstance(audio_duration, bool) or audio_duration <= 0
                or not math.isfinite(audio_duration)):
            return {}
        factor = duration_s / audio_duration
        multiplier = audio_duration / duration_s if duration_s > 0 else None
        return {
            "realtime_factor": factor if math.isfinite(factor) else None,
            "realtime_multiplier": multiplier if multiplier is not None
            and math.isfinite(multiplier) else None,
        }
    except (TypeError, ValueError, OverflowError):
        return {}


def torch_metrics() -> dict:
    """Read only an already-loaded, already-initialized torch CUDA allocator.

    No torch import, is_available(), synchronize(), or peak-counter reset.
    These current-device counters exclude CTranslate2 allocations; resources'
    independent NVML counters measure the whole GPU. Thread counts are observed
    with get_num_threads/get_num_interop_threads under the same initialization
    guard; no thread settings are changed.
    """
    metrics = dict.fromkeys(("cuda_initialized", "memory_allocated_bytes",
                             "memory_reserved_bytes", "max_memory_allocated_bytes",
                             "max_memory_reserved_bytes", "intra_op_threads",
                             "inter_op_threads"))
    torch = sys.modules.get("torch")
    if torch is None:
        return metrics
    try:
        cuda = torch.cuda
        initialized = bool(cuda.is_initialized())
        metrics["cuda_initialized"] = initialized
        if not initialized:
            return metrics
    except Exception:
        return metrics
    for field, getter in (("intra_op_threads", "get_num_threads"),
                          ("inter_op_threads", "get_num_interop_threads")):
        try:
            metrics[field] = getattr(torch, getter)()
        except Exception:
            pass
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated",
                 "max_memory_reserved"):
        try:
            metrics[name + "_bytes"] = getattr(cuda, name)()
        except Exception:
            pass
    return metrics


class Recorder:
    """Thread-safe writer; context entry activates it and starts a daemon.

    Every row has schema_version, trace_id, kind, name, elapsed_s (monotonic
    since construction), pid, thread (name), thread_id, cold, cache_hit, fields.
    Flags default to null. span_start/span_end share span_id; the end includes
    duration_s, status and error_type. For finite positive audio_duration metadata,
    ends also include realtime_factor (wall/audio) and realtime_multiplier
    (audio/wall, null for zero wall time). Exceptions propagate unchanged.
    Every row includes runtime_flags() and caller-supplied device (otherwise
    null). Samples have kind='sample' with fields.resources and fields.torch.

    Construction opens an append-only file, but sampling starts only on start()
    or context entry. I/O/sampling failures don't abort work; see last_error.
    A Recorder context is single-use; use_recorder() supports nested bindings.
    """

    def __init__(self, path: str | Path | None = None, disk_path: str | Path = ".",
                 interval: float = 1.0, *, enabled: bool = True):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and greater than zero")
        self.interval = interval
        self.disk_path = str(disk_path)
        self.trace_id = uuid.uuid4().hex
        self.last_error: str | None = None
        self._origin = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._file = None
        self._span_ids = itertools.count(1)
        self._token = None
        if enabled and path is not None:
            try:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._file = target.open("a", encoding="utf-8", buffering=1)
            except (OSError, ValueError) as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def _emit(self, kind, name, fields, cold=None, cache_hit=None, **extra):
        if not self.enabled:
            return False
        with self._lock:
            if self._file is None:
                return False
            record = {
                "schema_version": 1, "trace_id": self.trace_id,
                "kind": kind, "name": name,
                "elapsed_s": time.perf_counter() - self._origin,
                "pid": os.getpid(), "thread": threading.current_thread().name,
                "thread_id": threading.get_ident(),
                "device": fields.get("device"), **runtime_flags(),
                "cold": cold, "cache_hit": cache_hit, "fields": fields, **extra,
            }
            try:
                # Never stringify arbitrary objects/tensors: repr may be costly
                # or reveal content. Bad metadata drops only this record.
                line = json.dumps(record, ensure_ascii=False, allow_nan=False)
                self._file.write(line + "\n")
                return True
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                return False

    def event(self, name: str, *, cold: bool | None = None,
              cache_hit: bool | None = None, **metadata) -> bool:
        return self._emit("event", name, metadata, cold, cache_hit)

    @contextmanager
    def span(self, name: str, *, cold: bool | None = None,
             cache_hit: bool | None = None, **metadata):
        """Host elapsed time only; never synchronizes GPU work."""
        if not self.enabled:
            yield
            return
        span_id = next(self._span_ids)
        started = time.perf_counter()
        self._emit("span_start", name, metadata, cold, cache_hit, span_id=span_id)
        status, error_type = "ok", None
        try:
            yield
        except BaseException as exc:
            status, error_type = "error", type(exc).__name__
            raise
        finally:
            duration_s = time.perf_counter() - started
            self._emit("span_end", name, metadata, cold, cache_hit,
                       span_id=span_id, duration_s=duration_s,
                       status=status, error_type=error_type,
                       **_realtime_ratios(metadata.get("audio_duration"), duration_s))

    def sample_once(self) -> bool:
        if not self.enabled:
            return False
        try:
            from . import resources
            return self._emit("sample", "resources", {
                "resources": asdict(resources.sample(self.disk_path)),
                "torch": torch_metrics(),
            })
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.event("sample_error", error_type=type(exc).__name__)

    def _run(self, stop):
        while not stop.is_set():
            started = time.perf_counter()
            self.sample_once()
            remaining = self.interval - (time.perf_counter() - started)
            stop.wait(remaining if remaining > 0 else self.interval)

    def start(self):
        """Start at most one daemon, sampling immediately; return self."""
        with self._lock:
            if self._file is None or (self._thread and self._thread.is_alive()):
                return self
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(self._stop,),
                                            name="ytscribe-telemetry", daemon=True)
            try:
                self._thread.start()
            except RuntimeError as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._stop.set()
                self._thread = None
        return self

    def stop(self):
        """Stop sampling; keep the file open for final events or restart."""
        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def close(self):
        # Detach the file and stop atomically so a concurrent start() cannot
        # create a new daemon between stop() and closing the writer.
        with self._lock:
            self._stop.set()
            thread = self._thread
            stream, self._file = self._file, None
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("Use use_recorder() to rebind an active Recorder")
        self._token = _ACTIVE.set(self)
        try:
            return self.start()
        except BaseException:
            _ACTIVE.reset(self._token)
            self._token = None
            self.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        finally:
            _ACTIVE.reset(self._token)
            self._token = None
