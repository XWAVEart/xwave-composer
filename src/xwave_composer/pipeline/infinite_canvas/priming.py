"""Infill priming for blank canvas regions — never hand SDXL flat grey alone."""

from __future__ import annotations

import numpy as np
from PIL import Image

_patchmatch_disabled = False


def _nonsemantic_prime(width: int, height: int) -> np.ndarray:
    """Smooth deterministic colour noise for a completely blank read window."""
    seed = ((int(width) * 73856093) ^ (int(height) * 19349663)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    sw = max(8, width // 64)
    sh = max(8, height // 64)
    low = rng.integers(56, 200, size=(sh, sw, 3), dtype=np.uint8)
    return np.asarray(
        Image.fromarray(low, mode="RGB").resize(
            (width, height), Image.Resampling.BICUBIC
        )
    ).copy()


def _mirror_extend(rgb: np.ndarray, committed: np.ndarray) -> np.ndarray:
    """Extend real boundary pixels through every hole.

    The old implementation propagated only 64 pixels, then ran Telea over the
    still-grey backing buffer. Large outpaint windows therefore retained broad
    RGB(128,128,128) regions. Differential Diffusion deliberately preserves
    some initialization near the source boundary, so that grey became visible
    output regardless of the prompt.

    OpenCV's labelled distance transform gives every hole pixel the identity
    of its nearest committed pixel in one pass. This is a deterministic,
    full-window edge extension: no hole can retain the backing colour. It is a
    safe fallback when PatchMatch is unavailable and supplies meaningful
    low-frequency boundary structure for the latent transition band.
    """
    import cv2

    out = rgb.copy()
    hole = (~committed).astype(np.uint8)
    if int(hole.sum()) == 0:
        return out
    if int(committed.sum()) == 0:
        return _nonsemantic_prime(rgb.shape[1], rgb.shape[0])

    _distance, labels = cv2.distanceTransformWithLabels(
        hole,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    # DIST_LABEL_PIXEL numbers zero-valued source pixels (our committed
    # pixels) from 1 in row-major order.
    palette = np.ascontiguousarray(out[committed])
    nearest = palette[np.clip(labels - 1, 0, len(palette) - 1)]
    out[~committed] = nearest[~committed]
    return out


def _patchmatch(rgb: np.ndarray, committed: np.ndarray) -> np.ndarray | None:
    """PatchMatch content-aware fill — the strongest non-blueprint prime
    (research §4.4 rank #2, InvokeAI's preferred infill). Optional dependency
    (``pypatchmatch``); returns None when unavailable so callers fall back."""
    global _patchmatch_disabled
    if _patchmatch_disabled:
        return None
    try:
        from patchmatch import patch_match
    except Exception:  # noqa: BLE001
        _patchmatch_disabled = True
        return None
    try:
        hole = (~committed).astype(np.uint8)
        out = patch_match.inpaint(
            np.ascontiguousarray(rgb), hole, patch_size=5
        )
        return np.asarray(out, dtype=np.uint8)
    except Exception:  # noqa: BLE001
        # Avoid retrying a missing/broken native extension on every patch.
        _patchmatch_disabled = True
        return None


def _telea(rgb: np.ndarray, committed: np.ndarray) -> np.ndarray:
    import cv2

    hole = (~committed).astype(np.uint8)
    if int(hole.sum()) < 16:
        return rgb
    kernel = np.ones((3, 3), np.uint8)
    hole = cv2.dilate(hole, kernel, iterations=1)
    return cv2.inpaint(rgb, hole * 255, 3, cv2.INPAINT_TELEA)


def prime_canvas(
    init_rgb: Image.Image,
    occupied: Image.Image,
    *,
    method: str = "mirror",
    blueprint: Image.Image | None = None,
) -> Image.Image:
    """Fill unpainted pixels so DiffDiff has a coherent init everywhere.

    Methods: ``blueprint`` (crop already aligned), ``mirror``, ``telea``.
    """
    rgb = np.asarray(init_rgb.convert("RGB"))
    committed = np.asarray(occupied.convert("L")) >= 128
    if blueprint is not None:
        bp = blueprint.convert("RGB")
        if bp.size != init_rgb.size:
            bp = bp.resize(init_rgb.size, Image.Resampling.LANCZOS)
        bp_arr = np.asarray(bp)
        out = rgb.copy()
        hole = ~committed
        out[hole] = bp_arr[hole]
        # Still soft-fill any leftover if blueprint had black voids.
        still = hole & (out.sum(axis=-1) < 3)
        if int(still.sum()) > 0:
            tmp = Image.fromarray(out, mode="RGB")
            occ2 = Image.fromarray((~still).astype(np.uint8) * 255, mode="L")
            return prime_canvas(tmp, occ2, method="mirror")
        return Image.fromarray(out, mode="RGB")

    if method == "telea":
        filled = _telea(rgb, committed)
    else:
        # Priming quality order (research §4.4): blueprint > patchmatch >
        # mirror > telea. Try PatchMatch when there is committed content to
        # sample from; fall back to mirror-extend when unavailable.
        filled = None
        if int(committed.sum()) > 0 and not committed.all():
            filled = _patchmatch(rgb, committed)
        if filled is None:
            filled = _mirror_extend(rgb, committed)
    return Image.fromarray(filled, mode="RGB")
