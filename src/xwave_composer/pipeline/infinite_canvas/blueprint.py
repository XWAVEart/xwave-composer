"""Low-res global blueprint for Infinite Canvas priming / first paint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from xwave_composer.models.sdxl_hyper import SDXLHyperPipeline

BLUEPRINT_LONG_EDGE = 1152


def blueprint_init_image(width: int, height: int, seed: int | None = None) -> Image.Image:
    """Return a non-semantic, low-frequency init plate.

    A flat grey rectangle is interpreted by SDXL as a literal panel/frame during
    outpainting. Smooth colour noise gives img2img a defined ``z0`` without
    introducing a rectangular object into the scene.
    """
    rng = np.random.default_rng(None if seed is None or seed < 0 else int(seed))
    small_w = max(8, width // 64)
    small_h = max(8, height // 64)
    low = rng.integers(48, 208, size=(small_h, small_w, 3), dtype=np.uint8)
    plate = Image.fromarray(low, mode="RGB").resize(
        (width, height), Image.Resampling.BICUBIC
    )
    return plate


def blueprint_size(canvas_w: int, canvas_h: int, long_edge: int = BLUEPRINT_LONG_EDGE) -> tuple[int, int]:
    """Fit canvas aspect into ≤ long_edge, snapped to multiples of 64."""
    long_edge = max(512, int(long_edge))
    long_edge = (long_edge // 64) * 64
    aspect = max(1, canvas_w) / max(1, canvas_h)
    if canvas_w >= canvas_h:
        w = long_edge
        h = max(64, (int(round(w / aspect)) // 64) * 64)
    else:
        h = long_edge
        w = max(64, (int(round(h * aspect)) // 64) * 64)
    return w, h


def make_blueprint(
    sdxl: "SDXLHyperPipeline",
    canvas_w: int,
    canvas_h: int,
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int = 8,
    cfg: float = 1.0,
    eta: float = 0.0,
    seed: int | None = None,
) -> Image.Image:
    """One Hyper img2img pass at blueprint resolution from non-semantic noise."""
    bw, bh = blueprint_size(canvas_w, canvas_h)
    init = blueprint_init_image(bw, bh, seed)
    pos = (prompt or "").strip()
    if not pos:
        raise ValueError("Enter a region prompt before creating a blueprint.")
    neg = (negative_prompt or "").strip()
    return sdxl.refine(
        init_image=init,
        prompt=pos,
        negative_prompt=neg,
        denoise=1.0,
        steps=max(4, int(steps)),
        seed=seed,
        guidance_scale=cfg,
        eta=eta,
        change_map=None,
    ).resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
