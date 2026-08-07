"""SDF-based continuous strength maps for Differential Diffusion soft-inpaint."""

from __future__ import annotations

import numpy as np
from PIL import Image

# Research defaults (§4.3): keep a soft floor on committed content, never
# give the pixel immediately outside committed full freedom.
S_KEEP = 0.05
S_BAND_MAX = 0.92
S_NEW = 1.00
DEFAULT_GAMMA = 1.4
DEFAULT_FEATHER = 96


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_stamp_mask(
    height: int,
    width: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    feather: float,
) -> np.ndarray:
    """Smoothstep falloff from stamp exterior → interior (float32 [0,1]).

    Sides flush with the window are not feathered (no exterior context).
    """
    feather = max(8.0, float(feather))
    ys, xs = np.mgrid[0:height, 0:width]
    big = feather + 1.0
    left = (xs - x0).astype(np.float32) if x0 > 0 else np.full((height, width), big, np.float32)
    right = (
        (x1 - 1 - xs).astype(np.float32)
        if x1 < width
        else np.full((height, width), big, np.float32)
    )
    top = (ys - y0).astype(np.float32) if y0 > 0 else np.full((height, width), big, np.float32)
    bottom = (
        (y1 - 1 - ys).astype(np.float32)
        if y1 < height
        else np.full((height, width), big, np.float32)
    )
    dist_in = np.minimum(np.minimum(left, right), np.minimum(top, bottom))
    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    dist_in = np.where(inside, dist_in, -1.0)
    return _smoothstep(dist_in / feather)


def build_strength_map(
    committed: np.ndarray,
    *,
    feather_px: float = DEFAULT_FEATHER,
    s_keep: float = S_KEEP,
    s_band_max: float = S_BAND_MAX,
    s_new: float = S_NEW,
    gamma: float = DEFAULT_GAMMA,
    stamp_mask: np.ndarray | None = None,
    overpaint: float = 0.0,
    s_blank_min: float | None = None,
) -> np.ndarray:
    """SDF strength map in [0, 1] at pixel resolution.

    ``committed`` — bool or 0/1 array, True where paint already exists.
    ``stamp_mask`` — optional [0,1] soft stamp footprint. It gates rewrite of
    committed content only.
    ``overpaint`` — [0,1] deliberate-repaint strength for committed content
    inside the stamp footprint. This must come from the caller (the stamp's
    own occupancy), NOT be inferred from the window: a window almost always
    contains some blank or off-canvas pixels, and deriving intent from it
    silently disables repainting.
    ``s_blank_min`` — optional lower bound for uncommitted pixels. Few-step
    Hyper schedules otherwise give boundary pixels only 2–3 effective UNet
    steps, preserving the low-detail infill prime as a flat grey band. The
    ordinary SDF profile remains intact above this floor.
    """
    import cv2

    feather_px = max(8.0, float(feather_px))
    ov = max(0.0, min(1.0, float(overpaint)))
    blank_min = (
        None
        if s_blank_min is None
        else max(0.0, min(float(s_new), float(s_blank_min)))
    )
    c = (np.asarray(committed) > 0).astype(np.uint8)
    if c.ndim != 2:
        raise ValueError("committed must be 2D")
    is_committed = c.astype(bool)

    sm = None
    if stamp_mask is not None:
        sm = np.asarray(stamp_mask, dtype=np.float32)
        if sm.shape != c.shape:
            raise ValueError("stamp_mask shape must match committed")

    def _stamp_tail(reach_px: float) -> np.ndarray | None:
        """Decaying influence outside the stamp footprint ("feather out").

        The stamp must reach past its own rectangle: when a stamp fills blank
        space flush against committed content, the blend band lives on
        committed pixels *outside* the stamp. Gating strictly by the stamp
        mask pins those pixels to the keep floor and produces a hard edge
        along the entire stamp boundary.
        """
        assert sm is not None
        # The soft mask is exactly 0 at the stamp boundary and outside; any
        # positive value marks the true footprint (> 0.5 would shrink it by
        # the feather width and erase it entirely for small stamps).
        hard = (sm > 1e-4).astype(np.uint8)
        if not hard.any():
            return None
        d_stamp = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)
        return 1.0 - _smoothstep(d_stamp / max(8.0, float(reach_px)))

    if int(c.sum()) == 0:
        # Fully blank window: generate freely everywhere.
        return np.full(c.shape, float(s_new), dtype=np.float32)
    if int((1 - c).sum()) == 0:
        # Fully committed window: deliberate overpaint. Full strength inside
        # the stamp, then a band that decays from s_band_max at the stamp
        # boundary to the harmonization floor outside it — feather in AND out,
        # so the box edge is invisible in both diffusion and pixel space.
        if sm is None:
            return np.full(c.shape, float(s_keep), dtype=np.float32)
        tail = _stamp_tail(0.5 * feather_px)
        s = np.maximum(float(s_keep), float(s_new) * sm)
        if tail is not None:
            s = np.maximum(s, float(s_band_max) * tail)
        return np.clip(s, 0.0, 1.0).astype(np.float32)

    d_out = cv2.distanceTransform((1 - c), cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform(c, cv2.DIST_L2, 5)
    sdf = d_out - d_in  # >0 blank, <0 committed
    u = np.clip((sdf + feather_px) / (2.0 * feather_px), 0.0, 1.0)
    u = np.power(u, float(gamma))
    s = (s_keep + (s_band_max - s_keep) * u).astype(np.float32)
    s[sdf > feather_px] = float(s_new)
    if blank_min is not None:
        s[~is_committed] = np.maximum(s[~is_committed], blank_min)

    if sm is not None:
        # Committed content: blend band near blank frontiers, gated by the
        # stamp's *extended* influence (footprint + outward tail), plus the
        # deliberate-overpaint term inside the footprint itself. Far committed
        # content keeps the harmonization floor. Uncommitted stays fully free.
        tail = _stamp_tail(1.5 * feather_px)
        gate = sm if tail is None else np.maximum(sm, tail)
        rewrite = np.maximum(s * gate, ov * float(s_new) * sm)
        s = np.where(
            is_committed, np.maximum(float(s_keep), rewrite), s
        ).astype(np.float32)

    return np.clip(s, 0.0, 1.0).astype(np.float32)


def strength_to_change_map(strength: np.ndarray) -> Image.Image:
    """Convert float strength [0,1] to L-mode change map for DiffDiff."""
    arr = np.clip(np.asarray(strength, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def to_latent_strength(
    strength_px: np.ndarray,
    *,
    scale: int = 8,
) -> np.ndarray:
    """Downsample pixel strength to latent resolution with area averaging."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(strength_px, dtype=np.float32))[None, None]
    h, w = int(t.shape[-2]), int(t.shape[-1])
    lh = max(1, h // scale)
    lw = max(1, w // scale)
    out = F.interpolate(t, size=(lh, lw), mode="area")
    return out[0, 0].numpy().astype(np.float32)
