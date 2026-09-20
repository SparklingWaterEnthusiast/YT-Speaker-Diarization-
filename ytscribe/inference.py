"""Resource safeguards; budget changes apply only to an initialized CUDA stage."""
from __future__ import annotations

import gc
import sys
from contextlib import contextmanager


def is_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        'out of memory', 'cuda_error_out_of_memory', 'cublas_status_alloc_failed'))


def release_unused():
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def guarded_batch(requested: int, margin_mb: int, device: str, log) -> int:
    """A pressure guard, not an estimator of model working-set size.

    NVML includes other applications and CTranslate2; torch does not. A
    minimum batch can still OOM, which is surfaced after bounded retries.
    """
    if requested <= 1 or device != 'cuda':
        return requested
    try:
        import pynvml as nv
        nv.nvmlInit()
        try:
            mem = nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByIndex(0))
            free_mb = mem.free / 2**20
        finally:
            nv.nvmlShutdown()
        if free_mb < margin_mb:
            log(f'VRAM guard: {free_mb:.0f} MiB free is below the {margin_mb} MiB '
                'margin; reducing work batch to 1.')
            return 1
    except Exception:
        pass
    return requested


@contextmanager
def torch_budget(margin_mb: int, device: str, log):
    """Cap torch scratch allocations to physically available VRAM.

    WDDM may otherwise spill CUDA allocations into system RAM instead of
    raising OOM. This is an allocator limit, not a guarantee against every
    external CUDA allocation. Preserve the caller's stricter limit and restore
    it afterward. No model is unloaded and no driver setting is changed.
    """
    torch = sys.modules.get('torch')
    previous = None
    if device == 'cuda' and torch is not None and torch.cuda.is_initialized():
        release_unused()
        try:
            import pynvml as nv
            nv.nvmlInit()
            try:
                mem = nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByIndex(0))
            finally:
                nv.nvmlShutdown()
            cuda = torch.cuda
            previous = cuda.get_per_process_memory_fraction()
            total = cuda.get_device_properties(cuda.current_device()).total_memory
            # Reserved bytes are already subtracted from NVML free, so add
            # them back once when calculating a total torch allocator budget.
            budget = max(cuda.memory_allocated(),
                         mem.free + cuda.memory_reserved() - margin_mb * 2**20)
            fraction = min(previous, max(0.001, budget / total))
            cuda.set_per_process_memory_fraction(fraction)
            log(f'Diarization allocator budget: {fraction*total/2**20:.0f} MiB '
                f'(physical VRAM margin {margin_mb} MiB).')
        except Exception as exc:
            log(f'Diarization allocator budget unavailable: {type(exc).__name__}; '
                'bounded OOM recovery remains enabled.')
    try:
        yield
    finally:
        if previous is not None:
            try:
                torch.cuda.set_per_process_memory_fraction(previous)
            except Exception as exc:
                log(f'Could not restore allocator budget: {type(exc).__name__}')
