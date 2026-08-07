"""Region fill + fused refine for Infinite Canvas Mode.

Soft-inpaint on the shared Hyper SDXL img2img stack (keeps Juggernaut / NVFP4):
SDF strength map + Differential Diffusion + priming + photometric stitch.
Regions may extend outside the canvas; only the on-canvas intersection is written.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from xwave_composer.pipeline.infinite_canvas.blueprint import make_blueprint
from xwave_composer.pipeline.infinite_canvas.fuse import (
    fused_differential_fill,
    fused_tiled_refine,
)
from xwave_composer.pipeline.infinite_canvas.priming import prime_canvas
from xwave_composer.pipeline.infinite_canvas.strength import (
    S_KEEP,
    build_strength_map,
    soft_stamp_mask,
    strength_to_change_map,
)
from xwave_composer.pipeline.large_canvas import (
    LATENT_SCALE,
    MIN_STAMP,
    SIZE_STEP,
    LatentCanvas,
    StampRect,
    snap64,
)
from xwave_composer.style.style_manager import StylePreset, assemble_style_prompt

if TYPE_CHECKING:
    from xwave_composer.models.sdxl_hyper import SDXLHyperPipeline

logger = logging.getLogger(__name__)

_EMPTY_FILL = (128, 128, 128)
DEFAULT_OVERLAP = 256
# Research: dilated *read* window may be ~1728²; UNet must stay at native 1024².
MAX_CONTEXT_EDGE = 1728
MAX_UNET_EDGE = 1024

def build_region_prompt(
    user_prompt: str,
    preset: StylePreset | None,
    *,
    order: str = "pcs",
    manual_prefix: str | None = None,
    manual_suffix: str | None = None,
) -> tuple[str, str]:
    """Return only user content + selected style injection."""
    content = str(user_prompt or "").strip()
    if manual_prefix is not None or manual_suffix is not None:
        prefix = manual_prefix or ""
        suffix = manual_suffix or ""
        negative = str(preset.negative or "").strip() if preset else ""
        positive = assemble_style_prompt(content, prefix, suffix, order=order)
    elif preset is not None:
        positive = assemble_style_prompt(
            content,
            preset.prefix,
            preset.suffix,
            order=order,
        )
        negative = str(preset.negative or "").strip()
    else:
        positive = content
        negative = ""

    return positive, negative


def _sample_world_rect(
    canvas: Image.Image,
    occupied: Image.Image,
    rect: StampRect,
) -> tuple[Image.Image, Image.Image]:
    """Sample a world rect that may extend outside the canvas into (RGB, L)."""
    cw, ch = canvas.size
    init = Image.new("RGB", (rect.w, rect.h), _EMPTY_FILL)
    occ = Image.new("L", (rect.w, rect.h), 0)
    inter = rect.intersection(cw, ch)
    if inter is None:
        return init, occ
    src = canvas.crop((inter.x, inter.y, inter.x + inter.w, inter.y + inter.h))
    src_occ = occupied.crop((inter.x, inter.y, inter.x + inter.w, inter.y + inter.h))
    dx = inter.x - rect.x
    dy = inter.y - rect.y
    init.paste(src, (dx, dy), mask=src_occ)
    occ.paste(src_occ, (dx, dy))
    return init, occ


def _gen_window(stamp: StampRect, pad: int) -> StampRect:
    """Stamp expanded by pad (may grow further off-canvas)."""
    pad = max(0, (int(pad) // SIZE_STEP) * SIZE_STEP) if pad else 0
    return StampRect(
        x=stamp.x - pad,
        y=stamp.y - pad,
        w=stamp.w + 2 * pad,
        h=stamp.h + 2 * pad,
    )


def _dilated_window(
    stamp: StampRect,
    *,
    feather: float,
    context_pad: int,
    overlap: int,
) -> StampRect:
    """Context read window = stamp + max(feather, context_pad) + overlap.

    May exceed UNet native size; ``fill_region`` downscales to ``MAX_UNET_EDGE``
    for the Hyper pass (research: denoise at 1024, keep dilated context).
    """
    pad = max(int(feather), int(context_pad), 0) + max(0, int(overlap))
    pad = snap64(pad, SIZE_STEP) if pad else 0
    long_edge = max(stamp.w, stamp.h)
    if long_edge + 2 * pad > MAX_CONTEXT_EDGE and long_edge < MAX_CONTEXT_EDGE:
        pad = max(0, (MAX_CONTEXT_EDGE - long_edge) // 2)
        pad = (pad // SIZE_STEP) * SIZE_STEP
    elif long_edge >= MAX_CONTEXT_EDGE:
        pad = 0
    return _gen_window(stamp, pad)


def _occupancy_ratio_local(occ: Image.Image) -> float:
    arr = np.asarray(occ, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(arr.mean() / 255.0)


def _occupancy_ratio(occupied: Image.Image, stamp: StampRect) -> float:
    """Fraction of the stamp's on-canvas area that is already painted."""
    cw, ch = occupied.size
    inter = stamp.intersection(cw, ch)
    if inter is None:
        return 0.0
    crop = occupied.crop((inter.x, inter.y, inter.x + inter.w, inter.y + inter.h))
    return _occupancy_ratio_local(crop)


def _effective_denoise(
    occupancy: float,
    overpaint_denoise: float = 0.85,
) -> float:
    """Global schedule strength for the window.

    Full overwrite runs an img2img-truncated schedule from the existing
    latents (latent README §7: "noise the existing latent to the desired
    maximum strength") so composition, lighting and borders stay anchored to
    what's there while the prompt restyles it. Anything containing blank
    canvas needs the full schedule — blank pixels hold priming smear that a
    truncated schedule would partially preserve.
    """
    occ = max(0.0, min(1.0, occupancy))
    if occ > 0.92:
        return max(0.35, min(1.0, float(overpaint_denoise)))
    return 1.0


def _soft_inpaint_mask(
    stamp: StampRect,
    window: StampRect,
    occ_window: Image.Image,
    feather: float,
    falloff: float,
    overpaint: float = 0.0,
    steps: int = 30,
) -> Image.Image:
    """SDF change map in window space (255 = rewrite, 0 = keep)."""
    fo = max(0.0, min(1.0, float(falloff)))
    feather_px = max(64.0, float(feather) * (1.35 - 0.45 * fo))
    gamma = 1.4 + 0.4 * fo  # tighter falloff → bias ramp outward more

    sx = stamp.x - window.x
    sy = stamp.y - window.y
    stamp_a = soft_stamp_mask(
        window.h, window.w, sx, sy, sx + stamp.w, sy + stamp.h, feather_px
    )
    occ = occ_window.convert("L")
    if occ.size != (window.w, window.h):
        occ = occ.resize((window.w, window.h), Image.Resampling.BILINEAR)
    committed = np.asarray(occ, dtype=np.uint8) >= 128
    # The research SDF defaults assume a 24–40 step sampler. At eight Hyper
    # steps, its ~0.38 boundary strength releases blank pixels for only three
    # UNet evaluations, leaving the low-detail prime visible as a grey slab.
    # Guarantee roughly six effective evaluations on blank boundary pixels,
    # while converging back to the research default on ordinary schedules.
    n_steps = max(1, int(steps))
    blank_min = min(0.75, max(0.38, 6.0 / float(n_steps)))

    strength = build_strength_map(
        committed,
        feather_px=feather_px,
        gamma=gamma,
        stamp_mask=stamp_a,
        overpaint=overpaint,
        s_blank_min=blank_min,
    )
    return strength_to_change_map(strength)


# Keep old Telea helper name for tests that may import it.
def _telea_fill_empty(init_rgb: Image.Image, occ: Image.Image) -> Image.Image:
    return prime_canvas(init_rgb, occ, method="telea")


def _composite_once(
    patch: Image.Image, destination: Image.Image, alpha: Image.Image
) -> Image.Image:
    """Composite a generated patch exactly once through its commit ramp."""
    return Image.composite(
        patch.convert("RGB"), destination.convert("RGB"), alpha.convert("L")
    )


def _merge_coverage(existing: Image.Image, committed_alpha: Image.Image) -> Image.Image:
    """Union soft coverage without converting feathered edges to a rectangle."""
    old = np.asarray(existing.convert("L"), dtype=np.uint8)
    new = np.asarray(committed_alpha.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.maximum(old, new), mode="L")


def _commit_alpha(
    strength_arr: np.ndarray,
    committed: np.ndarray,
    keep_hard: np.ndarray,
    stamp_rect: np.ndarray,
    *,
    seam_feather_px: float = 20.0,
) -> Image.Image:
    """Per-pixel commit alpha for pasting a generated window onto the canvas.

    Trust the latent fusion: the window was denoised as ONE continuous latent,
    so its decode is internally seamless — old and new content already mixed
    during generation. Anything the diffusion was allowed to touch is written
    FULLY. Wide pixel-space crossfades here would blend two slightly different
    images (the fresh decode vs. the old canvas) and reintroduce exactly the
    smeared-halo seams the latent fusion eliminated (research §4.1/§6.2).

    The only partial alpha is a NARROW feather around the deep-preserved
    region (``keep_hard``, hard-pasted from the original afterwards) — its
    sole job is hiding VAE round-trip differences, and the content on both
    sides already agrees (§6.2: 8–24 px). Blank canvas outside the stamp is
    discarded: the model generated context there, but the user didn't paint.
    """
    import cv2

    keep = np.asarray(keep_hard).astype(bool)
    is_committed = np.asarray(committed).astype(bool)
    blank_write = np.asarray(stamp_rect) > 0
    if keep.any():
        d = cv2.distanceTransform((~keep).astype(np.uint8), cv2.DIST_L2, 5)
        u = np.clip(d / max(4.0, float(seam_feather_px)), 0.0, 1.0)
        ramp = (u * u * (3.0 - 2.0 * u)).astype(np.float32)
    else:
        ramp = np.ones_like(np.asarray(strength_arr), dtype=np.float32)
    alpha = np.where(
        is_committed, ramp, np.where(blank_write, 1.0, 0.0)
    ).astype(np.float32)
    alpha[keep] = 0.0
    return Image.fromarray((alpha * 255).astype(np.uint8), mode="L")


def _fit_for_unet(w: int, h: int) -> tuple[int, int, bool]:
    """Map a context window to an SDXL-native UNet size (≤ ``MAX_UNET_EDGE``).

    Returns ``(gen_w, gen_h, resized)``. Upscales tiny stamps; downscales dilated
    windows so Hyper always sees ~1024² (research §2.2).
    """
    w = max(1, int(w))
    h = max(1, int(h))
    long_edge = max(w, h)
    if long_edge < MIN_STAMP:
        scale = MIN_STAMP / long_edge
        gw = snap64(int(math.ceil(w * scale)))
        gh = snap64(int(math.ceil(h * scale)))
        return max(gw, MIN_STAMP), max(gh, MIN_STAMP), True
    if long_edge > MAX_UNET_EDGE:
        scale = MAX_UNET_EDGE / long_edge
        gw = max(SIZE_STEP, snap64(int(round(w * scale))))
        gh = max(SIZE_STEP, snap64(int(round(h * scale))))
        # Keep aspect; ensure we didn't snap back above the cap.
        while max(gw, gh) > MAX_UNET_EDGE and min(gw, gh) > SIZE_STEP:
            if gw >= gh:
                gw -= SIZE_STEP
            else:
                gh -= SIZE_STEP
        return gw, gh, True
    return w, h, False


# Back-compat alias for older call sites / tests.
def _ensure_gen_size(w: int, h: int) -> tuple[int, int, bool]:
    return _fit_for_unet(w, h)


def fill_region(
    canvas: Image.Image,
    occupied: Image.Image,
    stamp: StampRect,
    sdxl: "SDXLHyperPipeline",
    prompt: str,
    negative_prompt: str = "",
    steps: int = 8,
    cfg: float = 1.0,
    eta: float = 0.0,
    seed: int | None = None,
    context_pad: int = 128,
    feather: float = 96.0,
    falloff: float = 0.35,
    overlap: int = DEFAULT_OVERLAP,
    blueprint: Image.Image | None = None,
    world_seed: int | None = None,
    world_origin: tuple[int, int] = (0, 0),
    context_prompt: str = "",
    latent_canvas: LatentCanvas | None = None,
    overpaint_denoise: float = 0.85,
) -> tuple[Image.Image, Image.Image, str]:
    """Soft-inpaint one region; write only the on-canvas intersection.

    ``world_seed`` / ``world_origin`` anchor the diffusion noise field to
    canvas coordinates so overlapping patches share noise where they overlap.
    ``context_prompt`` is a non-subject (style-only) prompt used for UNet
    views that contain no actively generated pixels.

    ``latent_canvas`` — persistent latent store (updated IN PLACE). Committed
    content enters the diffusion as its stored latents instead of a fresh VAE
    re-encode of the pixel canvas, and the window's final latents are written
    back with the commit alpha. This is the latent README's core rule: the
    latent is the source of truth, the RGB canvas is a view. Without it every
    overlapping patch re-encodes the previous patch's decode, and the VAE
    round-trip drift itself becomes a seam source.
    """
    started_at = time.perf_counter()
    canvas = canvas.convert("RGB")
    occupied = occupied.convert("L")
    cw, ch = canvas.size
    stamp = stamp.normalized()
    inter = stamp.intersection(cw, ch)
    if inter is None:
        raise ValueError("Region does not intersect the canvas — nothing to paint.")

    # Blend band scales with the stamp so large stamps get proportionally
    # smooth transitions (a 96px band on a 1024px stamp reads as a hard edge).
    # 25% of the short side ≈ the read-overlap width, so the strength ramp,
    # the pixel crossfade, and the fused-view overlap all span the same band.
    feather_eff = float(
        min(384.0, max(float(feather), 0.25 * min(stamp.w, stamp.h)))
    )
    window = _dilated_window(
        stamp, feather=feather_eff, context_pad=context_pad, overlap=overlap
    )
    base_win, occ_win = _sample_world_rect(canvas, occupied, window)
    occ_ratio = _occupancy_ratio(occupied, stamp)
    strength = _effective_denoise(occ_ratio, overpaint_denoise)
    mostly_blank = (1.0 - occ_ratio) > 0.12
    full_overpaint = occ_ratio > 0.92

    # Deliberate-repaint intent from the STAMP's own occupancy. Ramps in as
    # the stamp becomes mostly committed; 1.0 for a stamp fully on paint.
    overpaint = float(np.clip((occ_ratio - 0.70) / 0.25, 0.0, 1.0))

    change_map = _soft_inpaint_mask(
        stamp,
        window,
        occ_win,
        feather=feather_eff,
        falloff=falloff,
        overpaint=overpaint,
        steps=max(8, int(steps)),
    )
    strength_arr = np.asarray(change_map, dtype=np.float32) / 255.0

    # Blueprint crop for this window if provided; else mirror/Telea priming.
    bp_win = None
    if blueprint is not None:
        bp = blueprint.convert("RGB")
        if bp.size != canvas.size:
            bp = bp.resize(canvas.size, Image.Resampling.LANCZOS)
        bp_win, _ = _sample_world_rect(bp, Image.new("L", bp.size, 255), window)

    init_src = prime_canvas(
        base_win,
        occ_win,
        method="blueprint" if bp_win is not None else "mirror",
        blueprint=bp_win,
    )

    pos = (prompt or "").strip()
    if not pos:
        raise ValueError("Enter a region prompt before generating.")
    neg = (negative_prompt or "").strip()

    # One shared latent/noise field over the read window; native 1024px UNet
    # views fuse epsilon before the scheduler advances (no independent worlds).
    fusion_status = ""
    resized = False
    gen_w, gen_h = window.w, window.h
    noise_seed = seed if (seed is not None and seed >= 0) else world_seed
    noise_origin = (
        window.x - int(world_origin[0]),
        window.y - int(world_origin[1]),
    )
    # Stored latents for the window (README §4): committed content joins the
    # diffusion as the latents it was GENERATED as, not a re-encode of pixels.
    latent_init = None
    if latent_canvas is not None and bool(latent_canvas.valid.any()):
        z_prev, valid_prev = latent_canvas.crop(
            window.x // LATENT_SCALE,
            window.y // LATENT_SCALE,
            window.w // LATENT_SCALE,
            window.h // LATENT_SCALE,
        )
        if bool(valid_prev.any()):
            latent_init = (z_prev, valid_prev)
    lat_out: np.ndarray | None = None
    model_started_at = time.perf_counter()
    try:
        result, fusion_status, lat_out = fused_differential_fill(
            init_src,
            change_map,
            sdxl,
            prompt=pos,
            negative_prompt=neg,
            steps=max(8, int(steps)),
            cfg=cfg,
            seed=seed,
            tile=MAX_UNET_EDGE,
            overlap=max(256, int(overlap)),
            noise_seed=noise_seed,
            noise_origin=noise_origin if noise_seed is not None else None,
            context_prompt=context_prompt,
            denoise=strength,
            latent_init=latent_init,
        )
    except Exception as exc:  # noqa: BLE001
        # Keep generation available on unusual pipeline/scheduler variants.
        logger.warning(
            "Fused differential fill failed (%s); using single-view fallback",
            exc,
        )
        from xwave_composer.device import empty_cache

        empty_cache()
        gen_w, gen_h, resized = _fit_for_unet(window.w, window.h)
        init = init_src
        cmap_run = change_map
        if resized:
            init = init.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
            cmap_run = change_map.resize(
                (gen_w, gen_h), Image.Resampling.BILINEAR
            )
        result = sdxl.refine(
            init_image=init,
            prompt=pos,
            negative_prompt=neg,
            denoise=strength,
            steps=max(8, int(steps)),
            seed=seed,
            guidance_scale=cfg,
            eta=eta,
            change_map=cmap_run,
        )
        if resized:
            result = result.resize(
                (window.w, window.h), Image.Resampling.LANCZOS
            )
        fusion_status = f"single-view fallback: {type(exc).__name__}"
    model_finished_at = time.perf_counter()
    result = result.convert("RGB")

    committed_px = np.asarray(occ_win.convert("L")) >= 128
    keep_hard = (strength_arr <= (S_KEEP + 0.02)) & committed_px

    # No post-hoc LAB/wavelet reconciliation here. Besides creating grey
    # halos when its statistics are poisoned, changing decoded pixels without
    # applying the same transform to the persistent latent makes the RGB view
    # disagree with the latent source of truth. Tone and structure agreement
    # are handled in the shared latent denoising path.

    # Hard paste-back of deeply committed pixels (s ≈ s_keep), after the
    # colour pass so the reconciliation can never shift preserved content.
    result_arr = np.asarray(result).copy()
    base_arr = np.asarray(base_win.convert("RGB"))
    result_arr[keep_hard] = base_arr[keep_hard]
    result = Image.fromarray(result_arr, mode="RGB")

    stamp_rect = np.zeros((window.h, window.w), dtype=np.uint8)
    rx0 = max(0, stamp.x - window.x)
    ry0 = max(0, stamp.y - window.y)
    rx1 = min(window.w, stamp.x + stamp.w - window.x)
    ry1 = min(window.h, stamp.y + stamp.h - window.y)
    stamp_rect[ry0:ry1, rx0:rx1] = 1
    stitch_img = _commit_alpha(strength_arr, committed_px, keep_hard, stamp_rect)
    out = canvas.copy()
    out_occ = occupied.copy()
    # Paste the WINDOW ∩ canvas, not the stamp ∩ canvas. The blend band —
    # committed content the diffusion re-harmonized — lives OUTSIDE the stamp
    # rectangle; cropping the paste to the stamp discarded it and stamped a
    # guaranteed hard rectangular edge regardless of how well the latents
    # were fused. The alpha map already restricts what gets written (zero
    # over untouched keeps and blank-outside-stamp), so the wider paste is
    # exact, not a bleed.
    winter = window.intersection(cw, ch)
    assert winter is not None  # stamp ⊂ window and stamp ∩ canvas is nonempty
    lx = winter.x - window.x
    ly = winter.y - window.y
    patch = result.crop((lx, ly, lx + winter.w, ly + winter.h))
    alpha = stitch_img.crop((lx, ly, lx + winter.w, ly + winter.h))
    dest = out.crop((winter.x, winter.y, winter.x + winter.w, winter.y + winter.h))
    # Composite exactly once. Applying this alpha twice squared the ramp and
    # turned soft stamp edges into visible rectangular borders.
    merged = _composite_once(patch, dest, alpha)
    out.paste(merged, (winter.x, winter.y))

    # Coverage follows what was actually committed. A hard rectangle here made
    # the next stamp treat transparent edge pixels as immutable source content.
    old_cov = out_occ.crop(
        (winter.x, winter.y, winter.x + winter.w, winter.y + winter.h)
    )
    out_occ.paste(
        _merge_coverage(old_cov, alpha),
        (winter.x, winter.y),
    )

    # Latent write-back (README §4: write_back(crop, soft_mask)). The commit
    # alpha chooses generated-vs-protected ownership; LatentCanvas deliberately
    # does not average the final old and new latents a second time (§4.5).
    # Deep protected pixels with no authoritative stored latent stay invalid:
    # writing the generated window latent under an alpha=0 pixel would make
    # the latent disagree with the original RGB pasted back below. They are
    # encoded once on a later read instead.
    if latent_canvas is not None:
        import cv2

        s = LATENT_SCALE
        zw, zh = winter.w // s, winter.h // s
        a_win = (
            np.asarray(alpha, dtype=np.float32) / 255.0
        )  # winter-sized crop of the commit alpha
        a_lat = cv2.resize(a_win, (zw, zh), interpolation=cv2.INTER_AREA)
        if lat_out is None:
            # Pixel-space fallback path: no latents to store; anything the
            # paste touched is now stale in the store.
            latent_canvas.invalidate(
                winter.x // s, winter.y // s, a_lat > 1e-3
            )
        else:
            zx0 = (winter.x - window.x) // s
            zy0 = (winter.y - window.y) // s
            z_crop = lat_out[:, zy0 : zy0 + zh, zx0 : zx0 + zw]
            latent_canvas.write(
                winter.x // s,
                winter.y // s,
                z_crop,
                a_lat,
            )

    warn = ""
    if resized and max(window.w, window.h) > MAX_UNET_EDGE:
        warn = f" UNet={gen_w}×{gen_h} (native cap {MAX_UNET_EDGE})."
    elif resized:
        warn = f" Region below {MIN_STAMP}px — generated at {gen_w}×{gen_h}."
    off = stamp.intersection(cw, ch) != stamp
    if full_overpaint:
        mode = "overpaint+diffdiff"
    elif mostly_blank:
        mode = "fill+diffdiff"
    else:
        mode = "soft-inpaint+diffdiff"
    edge = " (clipped to canvas)" if off else ""
    # Research rule-of-thirds (§7.3 / InvokeAI): boxes with little committed
    # context inside them produce unstable results. Surface the heuristic.
    tip = ""
    if 0.0 < occ_ratio < 0.35:
        tip = (
            " Tip: overlap ~half the box onto existing paint for steadier"
            " results (at most ⅓ empty)."
        )
    finished_at = time.perf_counter()
    prep_seconds = model_started_at - started_at
    model_seconds = model_finished_at - model_started_at
    commit_seconds = finished_at - model_finished_at
    status = (
        f"Region {stamp.w}×{stamp.h} at ({stamp.x},{stamp.y}){edge} "
        f"[{mode}] win={window.w}×{window.h} denoise={strength:.2f} "
        f"occ={occ_ratio:.2f} · {fusion_status} · "
        f"timing prep={prep_seconds:.1f}s model={model_seconds:.1f}s "
        f"commit={commit_seconds:.1f}s.{warn}{tip}"
    )
    return out, out_occ, status


def _tile_origins(length: int, tile: int, overlap: int) -> list[int]:
    tile = max(SIZE_STEP, snap64(tile))
    overlap = max(0, min(overlap, tile // 2))
    step = max(SIZE_STEP, tile - overlap)
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, step))
    last = length - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


def tiled_refine(
    canvas: Image.Image,
    sdxl: "SDXLHyperPipeline",
    prompt: str,
    negative_prompt: str = "",
    denoise: float = 0.25,
    steps: int = 16,
    cfg: float = 1.0,
    eta: float = 0.0,
    seed: int | None = None,
    tile: int = 1024,
    overlap: int = 256,
    latent_canvas: LatentCanvas | None = None,
) -> tuple[Image.Image, str]:
    """Gaussian-weighted ε-fusion refine across the full canvas."""
    del eta  # fusion path uses scheduler defaults; kept for API compat
    return fused_tiled_refine(
        canvas,
        sdxl,
        prompt=prompt,
        negative_prompt=negative_prompt,
        denoise=denoise,
        steps=steps,
        cfg=cfg,
        seed=seed,
        tile=tile,
        overlap=overlap,
        latent_canvas=latent_canvas,
    )


def ensure_blueprint(
    canvas: Image.Image,
    occupied: Image.Image,
    sdxl: "SDXLHyperPipeline",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int = 8,
    cfg: float = 1.0,
    eta: float = 0.0,
    seed: int | None = None,
    existing: Image.Image | None = None,
) -> Image.Image | None:
    """Build or reuse a blueprint when the canvas is largely blank."""
    from xwave_composer.device import empty_cache

    occ = _occupancy_ratio_local(occupied)
    if occ > 0.15:
        return existing
    if existing is not None and existing.size == canvas.size:
        return existing
    try:
        bp = make_blueprint(
            sdxl,
            canvas.size[0],
            canvas.size[1],
            prompt,
            negative_prompt,
            steps=steps,
            cfg=cfg,
            eta=eta,
            seed=seed,
        )
        # Blueprint + DiffDiff must not stack activation peaks. Avoid
        # torch.compiler.reset() here: it would recompile the UNet that the
        # region fill uses immediately afterwards.
        empty_cache()
        return bp
    except Exception:  # noqa: BLE001
        logger.exception("Blueprint generation failed")
        return existing
