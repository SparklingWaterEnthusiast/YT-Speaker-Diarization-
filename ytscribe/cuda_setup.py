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
        if path.is_dir():
            os.add_dll_directory(str(path))
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
    # int8_float16 fits Whisper large-v3 in 8 GB VRAM with negligible quality loss
    return "int8_float16" if device == "cuda" else "int8"
