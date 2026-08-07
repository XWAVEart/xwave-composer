"""OUTPUT quality modes: Fast vs Quality presets for live refine.

Fast keeps the classic snappy knobs. Quality uses higher denoise/steps so
placed objects integrate into the scene (more real denoising steps).

Live interaction always uses a cheap preview pass; these presets mainly
control the settle / full refine knobs (and preview step count).
"""

from __future__ import annotations

from typing import Mapping

from xwave_composer.config import AppConfig

# Built-in packs; config.yaml may override per-key values.
DEFAULT_QUALITY_MODES: dict[str, dict[str, float | int]] = {
    "fast": {
        "denoise": 0.30,
        "steps": 6,
        "preview_steps": 4,
    },
    "quality": {
        "denoise": 0.55,
        "steps": 10,
        "preview_steps": 6,
    },
}

QUALITY_MODE_ORDER: tuple[str, ...] = ("fast", "quality")
QUALITY_MODE_LABELS: dict[str, str] = {
    "fast": "Fast",
    "quality": "Quality",
}


def normalize_quality_mode(value: object | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in DEFAULT_QUALITY_MODES:
        return raw
    if raw in ("hi", "high", "hq"):
        return "quality"
    return "quality"


def quality_mode_pack(config: AppConfig, mode: str | None = None) -> dict[str, float | int]:
    """Return denoise/steps/preview_steps for ``mode`` (config overrides builtins)."""
    key = normalize_quality_mode(mode or config.get("sdxl_hyper", "default_quality_mode", default="quality"))
    base = dict(DEFAULT_QUALITY_MODES[key])
    overrides = config.get("sdxl_hyper", "quality_modes", key, default=None)
    if isinstance(overrides, Mapping):
        if "denoise" in overrides:
            base["denoise"] = float(overrides["denoise"])
        if "steps" in overrides:
            base["steps"] = int(overrides["steps"])
        if "preview_steps" in overrides:
            base["preview_steps"] = int(overrides["preview_steps"])
    return base


def quality_mode_choices() -> list[tuple[str, str]]:
    """Gradio-friendly (label, value) pairs."""
    return [(QUALITY_MODE_LABELS[k], k) for k in QUALITY_MODE_ORDER]
