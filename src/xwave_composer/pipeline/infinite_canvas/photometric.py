"""Photometric reconciliation and seam metrics for Infinite Canvas."""

from __future__ import annotations

import numpy as np
from PIL import Image


def match_stats_in_band(
    new_rgb: Image.Image | np.ndarray,
    ref_rgb: Image.Image | np.ndarray,
    band_mask: Image.Image | np.ndarray,
    *,
    ref_mask: Image.Image | np.ndarray | None = None,
    apply_weight: np.ndarray | None = None,
    strength: float = 1.0,
) -> Image.Image:
    """Match new image mean/std to ref statistics (LAB space).

    ``band_mask`` selects the pixels of ``new_rgb`` used for the *new* stats.
    ``ref_mask`` selects the pixels of ``ref_rgb`` used for the *reference*
    stats (defaults to ``band_mask``). Keep the reference restricted to
    committed content — grey-primed or blank pixels poison the estimate and
    wash the patch out. ``apply_weight`` is an optional per-pixel [0,1] map so
    the correction is applied to new content only.
    """
    import cv2

    def _arr(im: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(im, Image.Image):
            return np.asarray(im.convert("RGB"))
        return np.asarray(im)

    def _mask(m: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(m, Image.Image):
            a = np.asarray(m.convert("L"))
        else:
            a = np.asarray(m)
        if a.dtype == np.bool_ or a.max() <= 1:
            return a.astype(bool)
        return a >= 128

    new_a = _arr(new_rgb)
    ref_a = _arr(ref_rgb)
    m = _mask(band_mask)
    if m.shape[:2] != new_a.shape[:2]:
        raise ValueError("band_mask shape must match images")
    mr = _mask(ref_mask) if ref_mask is not None else m
    if int(m.sum()) < 64 or int(mr.sum()) < 64:
        return Image.fromarray(new_a, mode="RGB")

    a = cv2.cvtColor(new_a, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(ref_a, cv2.COLOR_RGB2LAB).astype(np.float32)
    s = max(0.0, min(1.0, float(strength)))
    w = None
    if apply_weight is not None:
        w = np.clip(np.asarray(apply_weight, dtype=np.float32), 0.0, 1.0)
        if w.shape != m.shape:
            raise ValueError("apply_weight shape must match images")
    for c in range(3):
        va = a[..., c][m]
        vb = b[..., c][mr]
        # Trim outliers so occluders / genuine content differences don't skew
        # the estimate (standard panorama-stitching practice).
        lo_a, hi_a = np.percentile(va, [2.0, 98.0])
        lo_b, hi_b = np.percentile(vb, [2.0, 98.0])
        va = va[(va >= lo_a) & (va <= hi_a)]
        vb = vb[(vb >= lo_b) & (vb <= hi_b)]
        if va.size < 16 or vb.size < 16:
            continue
        sa = float(va.std()) + 1e-6
        sb = float(vb.std()) + 1e-6
        ma = float(va.mean())
        mb = float(vb.mean())
        adj = (a[..., c] - ma) * (sb / sa) + mb
        delta = s * (adj - a[..., c])
        if w is not None:
            delta = delta * w
        a[..., c] = a[..., c] + delta
    out = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out, mode="RGB")


def reconcile_seam_tone(
    new_rgb: Image.Image,
    ref_rgb: Image.Image,
    committed: np.ndarray,
    *,
    band_px: float = 128.0,
) -> Image.Image:
    """Wavelet-style low-frequency seam reconciliation (research §6.3b).

    Replaces the low-frequency component of the generated image with the low
    frequency of the *committed* canvas near seams, fading out over
    ``band_px`` into the new content. High-frequency detail is untouched.
    This forces the local tone at every seam to equal the existing canvas
    exactly, killing tone-step lines that survive statistical matching.
    """
    import cv2

    occ = np.asarray(committed).astype(bool)
    if occ.ndim != 2 or not occ.any():
        return new_rgb.convert("RGB")

    gen = np.asarray(new_rgb.convert("RGB"), dtype=np.float32)
    base = np.asarray(ref_rgb.convert("RGB"), dtype=np.float32)
    if base.shape != gen.shape or occ.shape != gen.shape[:2]:
        raise ValueError("shapes must match")

    sigma = float(min(96.0, max(8.0, band_px / 2.5)))
    occf = occ.astype(np.float32)

    def blur(a: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(a, (0, 0), sigma)

    # Normalized convolution: committed-only low-pass, extended into blank
    # areas so grey/black uncommitted pixels never contaminate the estimate.
    occ_b = blur(occf)
    c_low = blur(base * occf[..., None]) / np.maximum(occ_b, 1e-4)[..., None]
    g_low = blur(gen)

    d = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5)
    u = np.clip(1.0 - d / float(max(8.0, band_px)), 0.0, 1.0)
    w = u * u * (3.0 - 2.0 * u)  # smoothstep: no visible kink at either end
    # No committed tone within blur reach → no correction to apply.
    w = w * np.clip(occ_b * 4.0, 0.0, 1.0)

    out = gen + (c_low - g_low) * w[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def seam_energy(img_gray: np.ndarray, boundary_x: int, band: int = 24) -> float:
    """Ratio of cross-boundary gradient energy to local baseline (~1.0 = invisible)."""
    g = np.asarray(img_gray, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("img_gray must be 2D")
    w = g.shape[1]
    if boundary_x < 2 or boundary_x >= w - 2:
        return 1.0
    gx = np.abs(np.diff(g, axis=1))
    at = float(gx[:, boundary_x - 1 : boundary_x + 1].mean())
    left0 = max(0, boundary_x - band)
    left1 = max(0, boundary_x - 2)
    right0 = min(w - 1, boundary_x + 2)
    right1 = min(w - 1, boundary_x + band)
    parts = []
    if left1 > left0:
        parts.append(gx[:, left0:left1])
    if right1 > right0:
        parts.append(gx[:, right0:right1])
    if not parts:
        return 1.0
    near = float(np.concatenate(parts, axis=1).mean())
    return float(at / (near + 1e-6))
