"""Make CUDA libraries visible to CTranslate2 on Windows.

PyTorch pip wheels bundle cuBLAS/cuDNN DLLs in torch/lib; CTranslate2
(faster-whisper's backend) loads the same DLLs by name but does not know
torch's directory. Registering it with os.add_dll_directory lets one set of
CUDA libraries serve both frameworks — no system-wide CUDA install needed.

Must be called before the first WhisperModel(...) construction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES = {}


def register_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    candidates = []
    try:
        import torch  # noqa: F401
        candidates.append(Path(torch.__file__).parent / "lib")
    except ImportError:
        pass
    # nvidia-* pip wheels (used when torch is absent)
    for pkg in ("cublas", "cudnn"):
        try:
            mod = __import__(f"nvidia.{pkg}", fromlist=["__file__"])
            candidates.append(Path(mod.__file__).parent / "bin")
        except ImportError:
            pass
    for path in candidates:
        if path.is_dir() and str(path) not in _DLL_HANDLES:
            # The handle controls registration lifetime; discarding it closes
            # the DLL directory immediately on CPython.
            _DLL_HANDLES[str(path)] = os.add_dll_directory(str(path))
            # some loaders still consult PATH
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def pick_device(preference: str = "auto") -> str:
    """Resolve 'auto' to cuda when available, else cpu."""
    if preference in ("cuda", "cpu"):
        return preference
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def pick_compute_type(preference: str, device: str) -> str:
    if preference != "auto":
        return preference
    if device == "cpu":
        return "int8"
    import ctranslate2
    supported = ctranslate2.get_supported_compute_types("cuda")
    for candidate in ("int8_float16", "int8_float32", "float32"):
        if candidate in supported:
            return candidate
    return "default"
