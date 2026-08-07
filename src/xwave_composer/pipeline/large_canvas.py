"""Infinite Canvas Mode — flat RGB session state (independent of Compose WORK layers)."""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# Region / canvas sizes must stay multiples of 64 for SDXL latent alignment.
SIZE_STEP = 64
MIN_STAMP = 512
DEFAULT_CANVAS = 2048
DEFAULT_STAMP = 1024
LATENT_SCALE = 8
LATENT_CHANNELS = 4

CANVAS_PRESETS: dict[str, tuple[int, int]] = {
    "2048×2048": (2048, 2048),
    "3072×3072": (3072, 3072),
    "4096×4096": (4096, 4096),
    "2048×3072": (2048, 3072),
    "3072×2048": (3072, 2048),
}

STAMP_ASPECTS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:2": (3, 2),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "3:4": (3, 4),
    "2:3": (2, 3),
}


def snap64(value: int, minimum: int = SIZE_STEP) -> int:
    v = max(minimum, int(value))
    return max(minimum, (v // SIZE_STEP) * SIZE_STEP)


def _snap64_up(value: int) -> int:
    """Ceil to a multiple of 64; 0 stays 0 (used for canvas expansion deltas)."""
    v = max(0, int(value))
    if v == 0:
        return 0
    return ((v + SIZE_STEP - 1) // SIZE_STEP) * SIZE_STEP


def stamp_size_for_aspect(aspect: str, short_side: int) -> tuple[int, int]:
    """Return (w, h) multiples of 64 for the given aspect and short-side target."""
    rw, rh = STAMP_ASPECTS.get(aspect, (1, 1))
    short = snap64(short_side, SIZE_STEP)
    if rw >= rh:
        h = short
        w = snap64(int(round(h * rw / rh)))
    else:
        w = short
        h = snap64(int(round(w * rh / rw)))
    return w, h


@dataclass
class StampRect:
    """Generation region in canvas coordinates. May extend outside the canvas."""

    x: int = 0
    y: int = 0
    w: int = DEFAULT_STAMP
    h: int = DEFAULT_STAMP

    def normalized(self) -> "StampRect":
        """Snap size to ×64; keep position (may be off-canvas)."""
        w = snap64(self.w, SIZE_STEP)
        h = snap64(self.h, SIZE_STEP)
        x = (int(self.x) // SIZE_STEP) * SIZE_STEP
        y = (int(self.y) // SIZE_STEP) * SIZE_STEP
        return StampRect(x=x, y=y, w=w, h=h)

    def intersection(self, canvas_w: int, canvas_h: int) -> "StampRect | None":
        """Overlap of this rect with the canvas, or None if disjoint."""
        x0 = max(0, self.x)
        y0 = max(0, self.y)
        x1 = min(canvas_w, self.x + self.w)
        y1 = min(canvas_h, self.y + self.h)
        if x0 >= x1 or y0 >= y1:
            return None
        return StampRect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class LatentCanvas:
    """Persistent canvas-wide latent store — the generation source of truth.

    The pixel canvas is a *view*; repeated VAE encode/decode of committed
    content destroys detail and creates seams (latent README §4/§8). Committed
    regions enter each new generation as their ORIGINAL latents via
    :meth:`crop`, and results are written back with the soft commit mask in
    latent space via :meth:`write`. All coordinates are in latent units
    (pixels // LATENT_SCALE); canvas sizes are 64-multiples so every stamp,
    window, and expansion is exactly representable.
    """

    z: np.ndarray  # (LATENT_CHANNELS, lh, lw) float16, scaled SDXL latents
    valid: np.ndarray  # (lh, lw) bool — True where z is authoritative

    @classmethod
    def empty(cls, width_px: int, height_px: int) -> "LatentCanvas":
        lh = int(height_px) // LATENT_SCALE
        lw = int(width_px) // LATENT_SCALE
        return cls(
            z=np.zeros((LATENT_CHANNELS, lh, lw), dtype=np.float16),
            valid=np.zeros((lh, lw), dtype=bool),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.z.shape[1]), int(self.z.shape[2])

    def copy(self) -> "LatentCanvas":
        return LatentCanvas(z=self.z.copy(), valid=self.valid.copy())

    def crop(self, lx: int, ly: int, lw: int, lh: int) -> tuple[np.ndarray, np.ndarray]:
        """Zero-padded crop that may extend outside the store (off-canvas)."""
        z = np.zeros((self.z.shape[0], lh, lw), dtype=np.float16)
        v = np.zeros((lh, lw), dtype=bool)
        H, W = self.shape
        x0, y0 = max(0, lx), max(0, ly)
        x1, y1 = min(W, lx + lw), min(H, ly + lh)
        if x1 > x0 and y1 > y0:
            z[:, y0 - ly : y1 - ly, x0 - lx : x1 - lx] = self.z[:, y0:y1, x0:x1]
            v[y0 - ly : y1 - ly, x0 - lx : x1 - lx] = self.valid[y0:y1, x0:x1]
        return z, v

    def write(
        self,
        lx: int,
        ly: int,
        z_new: np.ndarray,
        alpha: np.ndarray,
        seed_mask: np.ndarray | None = None,
    ) -> None:
        """Mask-directed write-back without averaging two final latents.

        The denoising loop already fused old and new content into ``z_new``.
        Averaging that result with the old final latent a second time
        low-pass-filters/desaturates overlaps (research §4.5: do not average
        committed latents). ``alpha`` therefore selects ownership: generated
        pixels use ``z_new`` and protected pixels keep the exact old latent.
        The narrow pixel feather may straddle the 0.5 ownership boundary, but
        the next read creates a context-aware latent transition before
        denoising; no stored latent is an arbitrary mixture of two worlds.

        Where the store has no valid latents yet, any positive alpha takes the
        new value fully. ``seed_mask`` marks
        pixels whose z_new is trustworthy even at alpha 0 — e.g. committed
        content that was pinned during generation: seeding it here means the
        one VAE encode it just went through is the LAST one it ever needs.
        """
        H, W = self.shape
        h, w = alpha.shape
        x0, y0 = max(0, lx), max(0, ly)
        x1, y1 = min(W, lx + w), min(H, ly + h)
        if x1 <= x0 or y1 <= y0:
            return
        sl = np.s_[y0 - ly : y1 - ly, x0 - lx : x1 - lx]
        a = np.clip(alpha[sl].astype(np.float32), 0.0, 1.0)
        zn = z_new[:, y0 - ly : y1 - ly, x0 - lx : x1 - lx].astype(np.float32)
        seed = (
            seed_mask[sl].astype(bool)
            if seed_mask is not None
            else np.zeros_like(a, dtype=bool)
        )
        oldv = self.valid[y0:y1, x0:x1]
        touched = (a > 1e-3) | seed
        old = self.z[:, y0:y1, x0:x1].astype(np.float32)
        take_new = (~oldv & touched) | (a >= 0.5)
        self.z[:, y0:y1, x0:x1] = np.where(
            take_new[None], zn, old
        ).astype(np.float16)
        self.valid[y0:y1, x0:x1] = oldv | touched

    def invalidate(self, lx: int, ly: int, mask: np.ndarray) -> None:
        """Mark a region stale (pixels changed outside the latent path)."""
        H, W = self.shape
        h, w = mask.shape
        x0, y0 = max(0, lx), max(0, ly)
        x1, y1 = min(W, lx + w), min(H, ly + h)
        if x1 <= x0 or y1 <= y0:
            return
        m = mask[y0 - ly : y1 - ly, x0 - lx : x1 - lx].astype(bool)
        self.valid[y0:y1, x0:x1] &= ~m

    def expand(self, left_px: int, right_px: int, top_px: int, bottom_px: int) -> None:
        l = int(left_px) // LATENT_SCALE
        r = int(right_px) // LATENT_SCALE
        t = int(top_px) // LATENT_SCALE
        b = int(bottom_px) // LATENT_SCALE
        self.z = np.pad(self.z, ((0, 0), (t, b), (l, r)))
        self.valid = np.pad(self.valid, ((t, b), (l, r)))


@dataclass
class LargeCanvasSession:
    """Authoritative full-res canvas for the Infinite Canvas tab."""

    width: int = DEFAULT_CANVAS
    height: int = DEFAULT_CANVAS
    image: Image.Image = field(init=False)
    occupied: Image.Image = field(init=False)
    latent: LatentCanvas = field(init=False, repr=False)
    undo_image: Image.Image | None = None
    undo_occupied: Image.Image | None = None
    undo_latent: LatentCanvas | None = field(default=None, repr=False)
    stamp: StampRect = field(default_factory=StampRect)
    stamp_aspect: str = "1:1"
    style_name: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    concat_order: str = "pcs"
    manual_style: bool = False
    manual_prefix: str = ""
    manual_suffix: str = ""
    seed: int = -1
    context_pad: int = 128
    overlap: int = 256
    feather: float = 96.0
    falloff: float = 0.35
    # Global img2img schedule used only when the selected stamp is already
    # fully painted. Spatial edge blending remains controlled by the SDF map.
    overpaint_denoise: float = 0.85
    steps: int = 8
    cfg: float = 1.0
    eta: float = 0.0
    fit_view: bool = False
    refine_denoise: float = 0.25
    refine_tile: int = 1024
    refine_overlap: int = 256
    refine_mode: str = "Selected style"
    refine_prompt: str = ""
    blueprint: Image.Image | None = None
    # World-anchored noise: same canvas location → same diffusion noise, so
    # overlapping patches agree at boundaries. Offsets track canvas expansion.
    world_seed: int = field(default_factory=lambda: random.randrange(2**31))
    world_ox: int = 0
    world_oy: int = 0
    status: str = "Infinite Canvas ready."
    rev: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        self.width = snap64(self.width)
        self.height = snap64(self.height)
        self.image = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        self.occupied = Image.new("L", (self.width, self.height), 0)
        self.latent = LatentCanvas.empty(self.width, self.height)
        self.stamp = StampRect(
            x=snap64((self.width - DEFAULT_STAMP) // 2),
            y=snap64((self.height - DEFAULT_STAMP) // 2),
            w=DEFAULT_STAMP,
            h=DEFAULT_STAMP,
        ).normalized()

    def create(self, width: int, height: int) -> None:
        with self._lock:
            self.width = snap64(width)
            self.height = snap64(height)
            self.image = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            self.occupied = Image.new("L", (self.width, self.height), 0)
            self.latent = LatentCanvas.empty(self.width, self.height)
            self.undo_image = None
            self.undo_occupied = None
            self.undo_latent = None
            self.blueprint = None
            self.world_seed = random.randrange(2**31)
            self.world_ox = 0
            self.world_oy = 0
            sw, sh = stamp_size_for_aspect(self.stamp_aspect, DEFAULT_STAMP)
            self.stamp = StampRect(
                x=snap64((self.width - sw) // 2),
                y=snap64((self.height - sh) // 2),
                w=sw,
                h=sh,
            ).normalized()
            self.fit_view = True
            self.rev += 1
            self.status = f"Canvas created {self.width}×{self.height}."

    def reset(self) -> None:
        """Clear pixels and occupancy; keep current canvas size and region."""
        with self._lock:
            self.image = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            self.occupied = Image.new("L", (self.width, self.height), 0)
            self.latent = LatentCanvas.empty(self.width, self.height)
            self.undo_image = None
            self.undo_occupied = None
            self.undo_latent = None
            self.blueprint = None
            self.world_seed = random.randrange(2**31)
            self.world_ox = 0
            self.world_oy = 0
            self.fit_view = True
            self.rev += 1
            self.status = f"Canvas reset ({self.width}×{self.height})."

    def expand(self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> None:
        left = _snap64_up(int(left or 0))
        right = _snap64_up(int(right or 0))
        top = _snap64_up(int(top or 0))
        bottom = _snap64_up(int(bottom or 0))
        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return
        with self._lock:
            new_w = self.width + left + right
            new_h = self.height + top + bottom
            canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
            occ = Image.new("L", (new_w, new_h), 0)
            canvas.paste(self.image, (left, top))
            occ.paste(self.occupied, (left, top))
            self.image = canvas
            self.occupied = occ
            self.latent.expand(left, right, top, bottom)
            self.width = new_w
            self.height = new_h
            self.stamp = StampRect(
                x=self.stamp.x + left,
                y=self.stamp.y + top,
                w=self.stamp.w,
                h=self.stamp.h,
            ).normalized()
            # Content shifted by (left, top); keep the noise field anchored.
            self.world_ox += left
            self.world_oy += top
            self.undo_image = None
            self.undo_occupied = None
            self.undo_latent = None
            # Invalidate blueprint geometry after expand; regenerate lazily.
            self.blueprint = None
            self.fit_view = True
            self.rev += 1
            self.status = f"Canvas expanded to {self.width}×{self.height}."

    def set_stamp(
        self,
        x: int | None = None,
        y: int | None = None,
        w: int | None = None,
        h: int | None = None,
    ) -> StampRect:
        with self._lock:
            stamp = StampRect(
                x=self.stamp.x if x is None else int(x),
                y=self.stamp.y if y is None else int(y),
                w=self.stamp.w if w is None else int(w),
                h=self.stamp.h if h is None else int(h),
            ).normalized()
            # Regions may straddle an edge for outpainting, but never allow
            # the entire drag target to disappear beyond the document. Keep
            # one 64px grid cell visible and therefore recoverable.
            self.stamp = StampRect(
                x=min(max(stamp.x, SIZE_STEP - stamp.w), self.width - SIZE_STEP),
                y=min(max(stamp.y, SIZE_STEP - stamp.h), self.height - SIZE_STEP),
                w=stamp.w,
                h=stamp.h,
            )
            return self.stamp

    def set_stamp_aspect(self, aspect: str, short_side: int | None = None) -> StampRect:
        with self._lock:
            self.stamp_aspect = aspect if aspect in STAMP_ASPECTS else "1:1"
            short = short_side if short_side is not None else min(self.stamp.w, self.stamp.h)
            w, h = stamp_size_for_aspect(self.stamp_aspect, short)
            cx = self.stamp.x + self.stamp.w // 2
            cy = self.stamp.y + self.stamp.h // 2
            return self.set_stamp(x=cx - w // 2, y=cy - h // 2, w=w, h=h)

    def nudge_stamp_size(self, direction: int) -> StampRect:
        """Grow/shrink stamp by SIZE_STEP on the short side, keeping aspect."""
        with self._lock:
            short = min(self.stamp.w, self.stamp.h) + int(direction) * SIZE_STEP
            short = max(SIZE_STEP, short)
            return self.set_stamp_aspect(self.stamp_aspect, short)

    def push_undo(self) -> None:
        with self._lock:
            self.undo_image = self.image.copy()
            self.undo_occupied = self.occupied.copy()
            self.undo_latent = self.latent.copy()

    def undo(self) -> bool:
        with self._lock:
            if self.undo_image is None or self.undo_occupied is None:
                self.status = "Nothing to undo."
                return False
            self.image = self.undo_image
            self.occupied = self.undo_occupied
            if self.undo_latent is not None:
                self.latent = self.undo_latent
            self.undo_image = None
            self.undo_occupied = None
            self.undo_latent = None
            self.rev += 1
            self.status = "Undid last generation."
            return True

    def apply_result(self, image: Image.Image, occupied: Image.Image) -> None:
        with self._lock:
            if image.size != (self.width, self.height):
                raise ValueError("Result size mismatch")
            self.image = image.convert("RGB")
            self.occupied = occupied.convert("L")
            self.rev += 1
