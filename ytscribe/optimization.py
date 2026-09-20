"""Pure optimization policy; importing this module never probes CUDA.

Hardware discovery belongs to resources. Pass its supported compute types to
these helpers after an explicit probe; unknown capabilities retain ``auto``.
Measurements describe the tested workload, not general speed or quality guarantees.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterable

from .config import COMPUTE_TYPES, Config

PROFILE_DESCRIPTIONS = {
    "custom": "Your saved settings. Presets are applied only when you click Apply preset.",
    "safe": "Conservative: large-v3, beam 5, standard batch 1; release models between "
            "stages to reduce resident VRAM. Reloading may slow queues.",
    "balanced": "Baseline: large-v3, beam 5, standard batch 1. Keep models resident "
                "when memory allows. Standard sequential transcription remains the default.",
    "performance": "Optional batch mode: large-v3, beam 5, ASR batch 4. Measured faster "
                   "on one full source in two RTX 3080 Laptop runs: ASR cold 67.58 s / "
                   "warm 63.93 s versus 200.48 s / 205.40 s in the baseline recheck. "
                   "WER was 6.38% versus 6.57%; driver VRAM reached 5.0 GiB peak on "
                   "the tested system (includes desktop/other owners). "
                   "Batched decoding semantics differ. These results do not guarantee "
                   "speed or quality on other recordings. One GPU job at a time.",
}


def precision_options(supported: Iterable[str] | None = None,
                      current: str = "auto") -> tuple[str, ...]:
    """Offer reported precisions, preserving a configured value for round trips.

    An absent report is unknown (not proof of CUDA support). A known empty
    report offers only automatic selection and the existing custom value.
    """
    choices = list(COMPUTE_TYPES if supported is None else
                   ("auto", *sorted(set(supported) - {"auto"})))
    if current not in choices:
        choices.append(current)
    return tuple(choices)


def recommend_compute_type(supported: Iterable[str] | None,
                           device: str = "auto") -> str:
    """Choose only an actually reported precision; otherwise defer to runtime."""
    if supported is None:
        return "auto"
    available = set(supported)
    preferred = (("int8", "int8_float32", "float32") if device == "cpu" else
                 ("int8_float16", "int8_float32", "int8", "float16", "float32"))
    return next((value for value in preferred if value in available), "auto")


def apply_profile(cfg: Config, profile: str, *,
                  supported_compute_types: Iterable[str] | None = None,
                  device: str = "auto", vram_total_gb: float | None = None) -> Config:
    """Return an independent candidate; never mutate cfg or persist anything.

    Custom is a no-op copy. Explicit presets reset their advertised inference
    controls, while paths, language, speaker constraints and other preferences
    are retained. No preset enables concurrent GPU jobs.
    """
    if profile not in PROFILE_DESCRIPTIONS:
        raise ValueError(f"Unknown optimization profile: {profile}")
    candidate = deepcopy(cfg)
    candidate.optimization_profile = profile
    if profile == "custom":
        return candidate
    candidate.whisper_model = "large-v3"
    candidate.beam_size = 5
    candidate.asr_batch_size = 4 if profile == "performance" else 1
    candidate.asr_chunk_length = 30
    candidate.asr_num_workers = 1
    candidate.word_timestamps = True
    candidate.vad_filter = True
    candidate.compute_type = recommend_compute_type(supported_compute_types, device)
    constrained = vram_total_gb is not None and 0 < vram_total_gb <= 6
    candidate.model_residency = "stage" if profile == "safe" or constrained else "keep"
    candidate.vram_margin_mb = max(1024, cfg.vram_margin_mb)
    candidate.oom_retries = 2
    return candidate
