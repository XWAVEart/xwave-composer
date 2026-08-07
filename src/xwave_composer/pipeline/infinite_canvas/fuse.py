"""Mixture-of-Diffusers style fused ε tiling for Infinite Canvas refine."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from xwave_composer.device import empty_cache
from xwave_composer.pipeline.large_canvas import MIN_STAMP, SIZE_STEP, snap64
from xwave_composer.pipeline.infinite_canvas.strength import S_KEEP

if TYPE_CHECKING:
    from xwave_composer.models.sdxl_hyper import SDXLHyperPipeline

logger = logging.getLogger(__name__)

# A view is context-only only when every pixel is effectively immutable.
# Using 0.5 here incorrectly routed most of the SDF transition band to the
# style-only prompt, so the subject prompt had no influence where old and new
# structure must join.
CONTEXT_ACTIVE_THRESHOLD = S_KEEP + 0.01


def _view_batch_size(sdxl: Any, do_cfg: bool) -> int:
    """Choose a UNet view batch supported by the active quantization.

    TorchAO's current NVFP4 SDXL linears do not implement the ``aten.expand``
    dispatch used when the UNet receives multiple spatial views as a batch.
    Batch-one inference uses the native NVFP4 kernel correctly. FP8/BF16 keep
    the faster grouped path.
    """
    report = getattr(sdxl, "optimization_report", None)
    applied = str(getattr(report, "applied", "")).lower()
    if applied == "nvfp4":
        return 1
    return 2 if do_cfg else 4


def tile_weight(h: int, w: int, *, sigma_frac: float = 0.30, eps: float = 1e-3) -> torch.Tensor:
    """Separable 2D Gaussian weight kernel, peak 1.0 at center."""

    def axis(n: int) -> np.ndarray:
        x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        return np.exp(-(x**2) / (2.0 * sigma_frac**2))

    wy, wx = axis(h)[:, None], axis(w)[None, :]
    k = wy * wx
    k = k / (k.max() + 1e-12)
    return torch.from_numpy((k + eps).astype(np.float32))


def _block_seed(seed: int, bx: int, by: int) -> int:
    """Stable per-block seed mix (splitmix-style)."""
    h = (int(seed) & 0xFFFFFFFFFFFF) * 0x9E3779B97F4A7C15
    h ^= (bx & 0xFFFFFFFF) * 0xBF58476D1CE4E5B9
    h ^= (by & 0xFFFFFFFF) * 0x94D049BB133111EB
    return h & 0x7FFFFFFFFFFFFFFF


def world_noise(
    lat_h: int,
    lat_w: int,
    lat_x0: int,
    lat_y0: int,
    seed: int,
    *,
    channels: int = 4,
    block: int = 64,
) -> torch.Tensor:
    """Deterministic latent noise indexed by world coordinates.

    The same canvas location always receives the same noise regardless of the
    read-window placement (BlueOut / InfiniteDiffusion shared noise map), so
    overlapping generations denoise toward agreeing content at boundaries
    instead of independent worlds. Returns float32 CPU tensor [1,C,H,W].
    """
    out = torch.empty((1, channels, lat_h, lat_w), dtype=torch.float32)
    by0 = lat_y0 // block
    by1 = (lat_y0 + lat_h - 1) // block
    bx0 = lat_x0 // block
    bx1 = (lat_x0 + lat_w - 1) // block
    for by in range(by0, by1 + 1):
        for bx in range(bx0, bx1 + 1):
            g = torch.Generator().manual_seed(_block_seed(seed, bx, by))
            blk = torch.randn((1, channels, block, block), generator=g)
            # Intersection of this world block with the window, both in
            # world latent coordinates.
            wy0 = max(lat_y0, by * block)
            wy1 = min(lat_y0 + lat_h, (by + 1) * block)
            wx0 = max(lat_x0, bx * block)
            wx1 = min(lat_x0 + lat_w, (bx + 1) * block)
            out[
                :, :, wy0 - lat_y0 : wy1 - lat_y0, wx0 - lat_x0 : wx1 - lat_x0
            ] = blk[
                :,
                :,
                wy0 - by * block : wy1 - by * block,
                wx0 - bx * block : wx1 - bx * block,
            ]
    return out


def merge_persistent_latents(
    encoded: torch.Tensor,
    stored: np.ndarray,
    valid: np.ndarray,
    *,
    transition: int = 8,
) -> torch.Tensor:
    """Merge persistent canvas latents into a freshly encoded read window.

    A hard ``torch.where(valid, stored, encoded)`` creates a latent step at
    every validity edge: stored source latents on one side and a separately
    encoded infill prime on the other. The VAE decoder turns that step into a
    broad colour/structure discontinuity. Keep exact stored latents in the
    deep source region, but use the context-aware fresh encode in a narrow
    generated boundary band. Differential Diffusion then resolves that band.

    This does not average the protected source interior (research §4.5);
    blending is confined to ``transition`` latent pixels next to the dirty
    region, where generation is explicitly allowed to harmonize structure.
    """
    import cv2

    if tuple(stored.shape[-2:]) != tuple(encoded.shape[-2:]):
        raise ValueError("persistent latent shape does not match encoded window")
    v = np.asarray(valid, dtype=bool)
    if not v.any():
        return encoded
    z_stored = torch.from_numpy(np.ascontiguousarray(stored, dtype=np.float32))
    z_stored = z_stored[None].to(device=encoded.device, dtype=encoded.dtype)
    if v.all():
        return z_stored

    dist = cv2.distanceTransform(v.astype(np.uint8), cv2.DIST_L2, 5)
    u = np.clip((dist - 1.0) / max(1.0, float(transition)), 0.0, 1.0)
    mix = u * u * (3.0 - 2.0 * u)
    m = torch.from_numpy(mix)[None, None].to(
        device=encoded.device, dtype=encoded.dtype
    )
    return m * z_stored + (1.0 - m) * encoded


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


def _jittered_origins(length: int, tile: int, overlap: int, jitter: int) -> list[int]:
    """Jitter interior grid origins without changing the view count.

    Re-rolling the grid offset each denoising step stops the weight field's
    periodic structure from imprinting a faint lattice on flat gradients
    (research §2.4). Endpoints stay fixed so no pixel loses coverage.

    The previous implementation shifted *all* origins and then added both
    endpoints back. A normal two-view axis therefore became three views after
    step zero, turning a 2×2 plan into 3×3 (9 UNet views instead of 4). With
    CFG that more than quadrupled effective work. Two-view axes have no
    interior origin to jitter and remain unchanged.
    """
    base = _tile_origins(length, tile, overlap)
    if len(base) <= 2 or jitter == 0:
        return base
    last = length - tile
    shifted = [0]
    for origin in base[1:-1]:
        shifted.append(min(max(1, origin + jitter), last - 1))
    shifted.append(last)
    return sorted(set(shifted))


def _step_jitter(seed: int, step_index: int, ov_lat: int) -> tuple[int, int]:
    """Deterministic per-step grid offset in [-ov/2, ov/2] latent px."""
    if ov_lat <= 0 or step_index == 0:
        return 0, 0
    jy = int(_block_seed(seed, step_index, 101) % (ov_lat + 1)) - ov_lat // 2
    jx = int(_block_seed(seed, step_index, 202) % (ov_lat + 1)) - ov_lat // 2
    return jy, jx


@torch.no_grad()
def decode_latents_margin(
    pipe: Any,
    latents: torch.Tensor,
    *,
    tile_latent: int = 112,
    margin_latent: int = 8,
) -> torch.Tensor:
    """Tiled VAE decode with context padding and margin discard (research §6.1).

    The VAE decoder corrupts a margin around whatever rectangle it is handed,
    so each tile is decoded with genuine neighbour context that is then thrown
    away. Strictly better than overlap-and-blend tiled decoding: no decoded
    pixel comes from the decoder's edge-artifact zone unless it is a true
    canvas boundary.
    """
    vae = pipe.vae
    scale = int(getattr(pipe, "vae_scale_factor", 8))
    b, _, h, w = latents.shape
    if max(h, w) <= tile_latent + 2 * margin_latent:
        return vae.decode(
            latents / vae.config.scaling_factor, return_dict=False
        )[0]

    # The VAE's own internal tiling blends overlaps; disable it here so our
    # padded tiles decode in one piece.
    was_tiling = bool(getattr(vae, "use_tiling", False))
    if was_tiling:
        try:
            vae.disable_tiling()
        except Exception:  # noqa: BLE001
            was_tiling = False
    try:
        out = torch.zeros((b, 3, h * scale, w * scale), dtype=torch.float32)
        for y0 in range(0, h, tile_latent):
            for x0 in range(0, w, tile_latent):
                y1 = min(y0 + tile_latent, h)
                x1 = min(x0 + tile_latent, w)
                py0 = max(y0 - margin_latent, 0)
                px0 = max(x0 - margin_latent, 0)
                py1 = min(y1 + margin_latent, h)
                px1 = min(x1 + margin_latent, w)
                dec = vae.decode(
                    latents[:, :, py0:py1, px0:px1] / vae.config.scaling_factor,
                    return_dict=False,
                )[0]
                ty0 = (y0 - py0) * scale
                tx0 = (x0 - px0) * scale
                out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = (
                    dec[
                        :,
                        :,
                        ty0 : ty0 + (y1 - y0) * scale,
                        tx0 : tx0 + (x1 - x0) * scale,
                    ]
                    .float()
                    .cpu()
                )
        return out
    finally:
        if was_tiling:
            try:
                vae.enable_tiling()
            except Exception:  # noqa: BLE001
                pass


def fused_tiled_refine(
    canvas: Image.Image,
    sdxl: "SDXLHyperPipeline",
    prompt: str,
    negative_prompt: str = "",
    denoise: float = 0.25,
    steps: int = 16,
    cfg: float = 1.0,
    seed: int | None = None,
    tile: int = 1024,
    overlap: int = 256,
    skip_residual: float = 0.3,
    latent_canvas: Any | None = None,
) -> tuple[Image.Image, str]:
    """Gaussian-weighted ε-fusion refine over the full canvas (MoD-style).

    Falls back to sequential per-tile refine if the pipe cannot expose UNet internals.

    ``latent_canvas`` — optional persistent latent store (``LatentCanvas``).
    Valid stored latents are used as the refine's z0 instead of re-encoding
    the pixel canvas, and the refined latents are written back on success.
    On the pixel-space fallback the store is invalidated instead (stale).
    """
    sdxl.ensure_loaded()
    pipe = sdxl.pipe
    assert pipe is not None

    canvas = canvas.convert("RGB")
    cw, ch = canvas.size
    tile = snap64(max(MIN_STAMP, tile))
    overlap = snap64(max(0, overlap), 0) if overlap else 0
    if overlap >= tile:
        overlap = tile // 4

    # Hyper few-step schedules need headroom for fusion quality.
    steps = max(12, int(steps))
    denoise = max(0.05, min(0.6, float(denoise)))
    pos = (prompt or "").strip()
    if not pos:
        raise ValueError("Enter a prompt before running fused refine.")
    neg = (negative_prompt or "").strip()

    scale = int(getattr(pipe, "vae_scale_factor", 8))
    latent_init = None
    store_ok = (
        latent_canvas is not None
        and latent_canvas.shape == (ch // scale, cw // scale)
    )
    if store_ok and latent_canvas.valid.any():
        latent_init = latent_canvas.crop(0, 0, cw // scale, ch // scale)

    try:
        image, status, lat_out = _fused_refine_latent(
            canvas,
            sdxl,
            prompt=pos,
            negative_prompt=neg,
            denoise=denoise,
            steps=steps,
            cfg=cfg,
            seed=seed,
            tile=tile,
            overlap=overlap,
            skip_residual=skip_residual,
            latent_init=latent_init,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fused latent refine failed (%s); falling back to sequential", exc)
        if latent_canvas is not None:
            # Pixel-space fallback changed the canvas outside the latent path.
            latent_canvas.valid[:] = False
        return _sequential_fallback(
            canvas,
            sdxl,
            prompt=pos,
            negative_prompt=neg,
            denoise=denoise,
            steps=min(steps, 10),
            cfg=cfg,
            seed=seed,
            tile=tile,
            overlap=overlap,
        )
    if store_ok:
        latent_canvas.write(
            0,
            0,
            lat_out,
            np.ones(lat_out.shape[-2:], dtype=np.float32),
        )
    return image, status


@torch.no_grad()
def fused_differential_fill(
    init_image: Image.Image,
    change_map: Image.Image,
    sdxl: "SDXLHyperPipeline",
    prompt: str,
    negative_prompt: str = "",
    *,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int | None = None,
    tile: int = 1024,
    overlap: int = 256,
    noise_seed: int | None = None,
    noise_origin: tuple[int, int] | None = None,
    context_prompt: str = "",
    denoise: float = 1.0,
    latent_init: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Image.Image, str, np.ndarray]:
    """Fused ε tiled Differential Diffusion over one dilated read window.

    The read window owns one latent and one noise field. Overlapping native-size
    UNet views vote into one epsilon canvas, followed by one scheduler step.
    Low-strength positions are then pinned to the same re-noised ``z0``.

    When ``noise_seed`` and ``noise_origin`` (window top-left in world pixels)
    are given, the noise field is sampled from a deterministic world-anchored
    map so overlapping patches share noise at shared canvas locations.

    ``context_prompt`` — non-subject (style-only) prompt used for views that
    contain no actively generated pixels, so context tiles don't each try to
    render the full subject (research §5.5, the "six castles" failure).

    ``denoise`` < 1.0 runs an img2img-style truncated schedule: the whole
    window starts from the existing content noised to that strength (latent
    README §7 — full overwrite keeps structural continuity with what's there).

    ``latent_init`` — optional ``(z_prev, valid)`` pair at window latent
    resolution (float16 numpy (4, lh, lw) and bool (lh, lw)). Where valid, the
    STORED canvas latents replace the fresh VAE encode as ``z0``, so committed
    content is never round-tripped through the VAE again (README §4/§8: the
    latent is the source of truth, RGB is a view).

    Returns ``(image, status, final_latents)`` — final latents as float16
    numpy (4, lh, lw), in the scheduler's scaled-latent space, for write-back
    into the canvas latent store.
    """
    from diffusers.utils.torch_utils import randn_tensor

    sdxl.ensure_loaded()
    pipe = sdxl.pipe
    assert pipe is not None

    init = init_image.convert("RGB")
    cmap = change_map.convert("L")
    if cmap.size != init.size:
        cmap = cmap.resize(init.size, Image.Resampling.BILINEAR)

    device = pipe._execution_device
    try:
        dtype = next(pipe.unet.parameters()).dtype
    except StopIteration:
        dtype = sdxl.dtype

    steps = max(8, int(steps))
    tile = snap64(max(MIN_STAMP, int(tile)))
    overlap = snap64(max(0, int(overlap)), 0) if overlap else 0
    overlap = min(overlap, tile // 2)
    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    denoise = max(0.05, min(1.0, float(denoise)))
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps, num_steps = pipe.get_timesteps(
        num_inference_steps=steps, strength=denoise, device=device
    )
    num_steps = int(num_steps)
    if num_steps <= 0:
        raise RuntimeError("empty Differential Diffusion timestep schedule")

    image_t = pipe.image_processor.preprocess(init).to(device=device, dtype=dtype)
    z0 = pipe.prepare_latents(
        image_t,
        timesteps[:1],
        batch_size=1,
        num_images_per_prompt=1,
        dtype=dtype,
        device=device,
        generator=generator,
        add_noise=False,
    )
    del image_t
    if latent_init is not None:
        z_prev_np, valid_np = latent_init
        if tuple(z_prev_np.shape[-2:]) == tuple(z0.shape[-2:]) and valid_np.any():
            z0 = merge_persistent_latents(z0, z_prev_np, valid_np)
    if noise_seed is not None and noise_origin is not None:
        scale = int(getattr(pipe, "vae_scale_factor", 8))
        noise = world_noise(
            z0.shape[-2],
            z0.shape[-1],
            noise_origin[0] // scale,
            noise_origin[1] // scale,
            int(noise_seed),
            channels=z0.shape[1],
        ).to(device=device, dtype=z0.dtype)
    else:
        noise = randn_tensor(
            z0.shape, generator=generator, device=device, dtype=z0.dtype
        )
    latents = pipe.scheduler.add_noise(
        z0, noise, timesteps[0].expand(z0.shape[0])
    )

    strength_px = torch.from_numpy(
        np.asarray(cmap, dtype=np.float32) / 255.0
    )[None, None]
    strength = F.interpolate(
        strength_px, size=z0.shape[-2:], mode="area"
    ).to(device=device, dtype=z0.dtype)

    prompt_embeds, neg_embeds, pooled, neg_pooled = _encode_prompt(
        pipe, prompt, negative_prompt, device, dtype
    )
    do_cfg = float(cfg) > 1.0
    ctx = (context_prompt or "").strip()
    has_ctx = bool(ctx) and ctx != (prompt or "").strip()
    ctx_embeds = ctx_pooled = None
    if has_ctx:
        ctx_embeds, _, ctx_pooled, _ = _encode_prompt(
            pipe, ctx, negative_prompt, device, dtype
        )

    vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
    lh, lw = z0.shape[-2:]
    tile_lat = max(16, tile // vae_scale)
    ov_lat = max(0, overlap // vae_scale)
    if lh < 8 or lw < 8:
        raise RuntimeError("no fused Differential Diffusion views")
    active2d = strength[0, 0] > CONTEXT_ACTIVE_THRESHOLD
    weight_cache: dict[tuple[int, int], torch.Tensor] = {}
    embed_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    jseed = int(
        noise_seed
        if noise_seed is not None
        else (seed if seed is not None and seed >= 0 else 0)
    )

    def _embeds_for(n: int, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (n, kind)
        if key not in embed_cache:
            pe = ctx_embeds if kind == "ctx" else prompt_embeds
            pp = ctx_pooled if kind == "ctx" else pooled
            if do_cfg:
                e = torch.cat(
                    [neg_embeds.repeat(n, 1, 1), pe.repeat(n, 1, 1)], dim=0
                )
                p = torch.cat(
                    [neg_pooled.repeat(n, 1), pp.repeat(n, 1)], dim=0
                )
            else:
                e = pe.repeat(n, 1, 1)
                p = pp.repeat(n, 1)
            embed_cache[key] = (e, p)
        return embed_cache[key]

    # Batch same-sized views into single UNet calls ("batching is not
    # optional" — serial views multiplied per-patch latency by view count).
    view_batch = _view_batch_size(sdxl, do_cfg)
    n_views = 0

    for step_index, t in enumerate(timesteps):
        # Per-step grid jitter breaks the weight-field lattice (§2.4).
        jy, jx = _step_jitter(jseed, step_index, ov_lat)
        groups: dict[
            tuple[int, int, str],
            list[tuple[int, int, torch.Tensor, torch.Tensor]],
        ] = {}
        for y0 in _jittered_origins(lh, tile_lat, ov_lat, jy):
            for x0 in _jittered_origins(lw, tile_lat, ov_lat, jx):
                th = min(tile_lat, lh - y0)
                tw = min(tile_lat, lw - x0)
                if th < 8 or tw < 8:
                    continue
                key = (th, tw)
                if key not in weight_cache:
                    weight_cache[key] = tile_weight(th, tw).to(
                        device=device, dtype=torch.float32
                    )
                kind = "full"
                if has_ctx and not bool(
                    active2d[y0 : y0 + th, x0 : x0 + tw].any()
                ):
                    kind = "ctx"
                time_ids = _tile_time_ids(
                    pipe,
                    init.size,
                    (x0 * vae_scale, y0 * vae_scale),
                    device,
                    dtype,
                )
                groups.setdefault((th, tw, kind), []).append(
                    (y0, x0, weight_cache[key], time_ids)
                )
        n_views = sum(len(g) for g in groups.values())
        if n_views == 0:
            raise RuntimeError("no fused Differential Diffusion views")

        eps_acc = torch.zeros_like(latents, dtype=torch.float32)
        weight_acc = torch.zeros(
            (1, 1, lh, lw), device=device, dtype=torch.float32
        )
        for (th, tw, kind), group in groups.items():
            for c0 in range(0, len(group), view_batch):
                chunk = group[c0 : c0 + view_batch]
                n = len(chunk)
                z_in = torch.cat(
                    [
                        latents[:, :, y0 : y0 + th, x0 : x0 + tw]
                        for y0, x0, _, _ in chunk
                    ],
                    dim=0,
                )
                tids = torch.cat([v[3] for v in chunk], dim=0)
                if do_cfg:
                    z_in = torch.cat([z_in, z_in], dim=0)
                    tids = torch.cat([tids, tids], dim=0)
                embeds, pool = _embeds_for(n, kind)
                z_in = pipe.scheduler.scale_model_input(z_in, t)
                noise_pred = pipe.unet(
                    z_in,
                    t,
                    encoder_hidden_states=embeds,
                    added_cond_kwargs={"text_embeds": pool, "time_ids": tids},
                    return_dict=False,
                )[0]
                if do_cfg:
                    eps_u, eps_c = noise_pred.chunk(2)
                    noise_pred = eps_u + float(cfg) * (eps_c - eps_u)
                for i, (y0, x0, weight, _) in enumerate(chunk):
                    wk = weight[None, None]
                    eps_acc[:, :, y0 : y0 + th, x0 : x0 + tw] += (
                        noise_pred[i : i + 1].float() * wk
                    )
                    weight_acc[:, :, y0 : y0 + th, x0 : x0 + tw] += wk

        eps = (eps_acc / weight_acc.clamp_min(1e-4)).to(latents.dtype)
        latents = pipe.scheduler.step(
            eps, t, latents, return_dict=False
        )[0]

        next_i = step_index + 1
        if next_i < num_steps:
            threshold = 1.0 - (next_i / float(num_steps))
            # Continuous release instead of DiffDiff's binary threshold. With
            # few-step Hyper schedules a binary flip quantises the strength
            # band into visible terraces: each ring of the band jumps from
            # fully pinned to fully free in one step. Here each pixel blends
            # from the re-noised original to the generated latent over one
            # inter-step interval (Blended-Latent-Diffusion-style latent
            # compositing), so old and new content co-denoise and mix inside
            # the loop — smooth in space AND time. A pixel's full-release
            # step is unchanged, so keep-floor pixels stay pinned throughout.
            softness = 1.0 / float(num_steps)
            w = torch.clamp((strength - threshold) / softness, 0.0, 1.0)
            w = w * w * (3.0 - 2.0 * w)
            w = w.to(latents.dtype)
            t_ref = timesteps[next_i].expand(z0.shape[0])
            z_ref = pipe.scheduler.add_noise(z0, noise, t_ref)
            latents = w * latents + (1.0 - w) * z_ref

    # Snapshot BEFORE decode: these go into the persistent latent store so the
    # next generation over this area never re-encodes pixels.
    lat_out = latents.detach()[0].to("cpu", torch.float16).numpy()

    needs_upcast = (
        getattr(pipe.vae.config, "force_upcast", False)
        and latents.dtype == torch.float16
    )
    if needs_upcast:
        pipe.upcast_vae()
        latents = latents.to(
            next(iter(pipe.vae.post_quant_conv.parameters())).dtype
        )
    decoded = decode_latents_margin(pipe, latents)
    if needs_upcast:
        pipe.vae.to(dtype=torch.float16)
    result = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    del decoded, latents, z0, noise
    empty_cache()
    return (
        result.convert("RGB"),
        (
            f"fused differential ({n_views} views, "
            f"tile={tile}, overlap={overlap}, steps={num_steps}"
            + (f", denoise={denoise:.2f}" if denoise < 1.0 else "")
            + ")"
        ),
        lat_out,
    )


def _sequential_fallback(
    canvas: Image.Image,
    sdxl: "SDXLHyperPipeline",
    *,
    prompt: str,
    negative_prompt: str,
    denoise: float,
    steps: int,
    cfg: float,
    seed: int | None,
    tile: int,
    overlap: int,
) -> tuple[Image.Image, str]:
    cw, ch = canvas.size
    xs = _tile_origins(cw, tile, overlap)
    ys = _tile_origins(ch, tile, overlap)
    acc = np.zeros((ch, cw, 3), dtype=np.float64)
    weight = np.zeros((ch, cw), dtype=np.float64)
    n = 0
    for y in ys:
        for x in xs:
            tw = min(tile, cw - x)
            th = min(tile, ch - y)
            tw = snap64(tw) if tw >= SIZE_STEP else tw
            th = snap64(th) if th >= SIZE_STEP else th
            if tw < 256 or th < 256:
                continue
            if x + tw > cw:
                x = cw - tw
            if y + th > ch:
                y = ch - th
            crop = canvas.crop((x, y, x + tw, y + th))
            refined = sdxl.refine(
                init_image=crop,
                prompt=prompt,
                negative_prompt=negative_prompt,
                denoise=denoise,
                steps=steps,
                seed=None if seed is None or seed < 0 else int(seed) + n,
                guidance_scale=cfg,
            )
            wmap = tile_weight(th, tw).numpy().astype(np.float64)
            patch = np.asarray(refined.convert("RGB"), dtype=np.float64)
            acc[y : y + th, x : x + tw] += patch * wmap[:, :, None]
            weight[y : y + th, x : x + tw] += wmap
            n += 1
    weight = np.maximum(weight, 1e-6)
    out = (acc / weight[:, :, None]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB"), (
        f"Fused refine fallback ({n} tiles, denoise={denoise:.2f}, tile={tile})."
    )


@torch.no_grad()
def _fused_refine_latent(
    canvas: Image.Image,
    sdxl: "SDXLHyperPipeline",
    *,
    prompt: str,
    negative_prompt: str,
    denoise: float,
    steps: int,
    cfg: float,
    seed: int | None,
    tile: int,
    overlap: int,
    skip_residual: float = 0.3,
    latent_init: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Image.Image, str, np.ndarray]:
    """Full-canvas Gaussian ε-fusion refine.

    Implements per-step grid jitter (§2.4), skip residual re-anchoring to the
    pre-refine canvas (§5.3, DemoFusion), centered per-tile micro-conditioning
    for refinement passes (§5.4), batched views (§3.3), and margin-discard
    tiled decode (§6.1).
    """
    from diffusers.utils.torch_utils import randn_tensor

    pipe = sdxl.pipe
    assert pipe is not None
    device = pipe._execution_device
    try:
        dtype = next(pipe.unet.parameters()).dtype
    except StopIteration:
        dtype = sdxl.dtype

    generator = None
    if seed is not None and seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    # Encode full canvas.
    image = pipe.image_processor.preprocess(canvas)
    image = image.to(device=device, dtype=dtype)
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps, num_steps = pipe.get_timesteps(
        num_inference_steps=steps, strength=denoise, device=device
    )
    num_steps = int(num_steps)
    if num_steps <= 0:
        raise RuntimeError("empty timestep schedule")

    z_ref0 = pipe.prepare_latents(
        image,
        timesteps[:1],
        batch_size=1,
        num_images_per_prompt=1,
        dtype=dtype,
        device=device,
        generator=generator,
        add_noise=False,
    )
    del image
    if latent_init is not None:
        z_prev_np, valid_np = latent_init
        if tuple(z_prev_np.shape[-2:]) == tuple(z_ref0.shape[-2:]) and valid_np.any():
            z_ref0 = merge_persistent_latents(z_ref0, z_prev_np, valid_np)
    noise = randn_tensor(
        z_ref0.shape, generator=generator, device=device, dtype=z_ref0.dtype
    )
    latents = pipe.scheduler.add_noise(
        z_ref0, noise, timesteps[0].expand(z_ref0.shape[0])
    )

    # Prompt embeds (CFG).
    prompt_embeds, neg_embeds, pooled, neg_pooled = _encode_prompt(
        pipe, prompt, negative_prompt, device, dtype
    )
    do_cfg = float(cfg) > 1.0

    vae_scale = int(getattr(pipe, "vae_scale_factor", 8))
    lh, lw = latents.shape[-2], latents.shape[-1]
    tile_lat = max(16, tile // vae_scale)
    ov_lat = max(0, overlap // vae_scale)
    if lh < 8 or lw < 8:
        raise RuntimeError("no refine tiles")

    weight_cache: dict[tuple[int, int], torch.Tensor] = {}
    tids_cache: dict[tuple[int, int], torch.Tensor] = {}
    embed_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    jseed = int(seed) if (seed is not None and seed >= 0) else 0
    view_batch = _view_batch_size(sdxl, do_cfg)
    skip_w0 = max(0.0, min(1.0, float(skip_residual)))
    n_views = 0

    def _embeds_for(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n not in embed_cache:
            if do_cfg:
                e = torch.cat(
                    [neg_embeds.repeat(n, 1, 1), prompt_embeds.repeat(n, 1, 1)],
                    dim=0,
                )
                p = torch.cat(
                    [neg_pooled.repeat(n, 1), pooled.repeat(n, 1)], dim=0
                )
            else:
                e = prompt_embeds.repeat(n, 1, 1)
                p = pooled.repeat(n, 1)
            embed_cache[n] = (e, p)
        return embed_cache[n]

    for step_index, t in enumerate(timesteps):
        # Skip residual (DemoFusion): re-anchor to the noise-inverted original
        # canvas with a cosine-decaying weight, so the refine can add detail
        # without drifting away from the established global structure.
        if skip_w0 > 0.0 and step_index > 0:
            frac = step_index / max(1, num_steps - 1)
            w_skip = skip_w0 * 0.5 * (1.0 + math.cos(math.pi * frac))
            if w_skip > 1e-4:
                z_ref_t = pipe.scheduler.add_noise(
                    z_ref0, noise, t.expand(z_ref0.shape[0])
                )
                latents = w_skip * z_ref_t + (1.0 - w_skip) * latents

        jy, jx = _step_jitter(jseed, step_index, ov_lat)
        views: list[tuple[int, int, int, int]] = []
        for y0 in _jittered_origins(lh, tile_lat, ov_lat, jy):
            for x0 in _jittered_origins(lw, tile_lat, ov_lat, jx):
                th = min(tile_lat, lh - y0)
                tw = min(tile_lat, lw - x0)
                if th < 8 or tw < 8:
                    continue
                views.append((y0, x0, th, tw))
        n_views = len(views)
        if n_views == 0:
            raise RuntimeError("no refine tiles")

        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for y0, x0, th, tw in views:
            groups.setdefault((th, tw), []).append((y0, x0))

        eps_acc = torch.zeros_like(latents, dtype=torch.float32)
        w_acc = torch.zeros((1, 1, lh, lw), device=device, dtype=torch.float32)
        for (th, tw), origins in groups.items():
            key = (th, tw)
            if key not in weight_cache:
                weight_cache[key] = tile_weight(th, tw).to(
                    device=device, dtype=torch.float32
                )
            if key not in tids_cache:
                # Centered micro-conditioning: refine tiles are detail passes
                # with structure already fixed — keep SDXL in its best regime.
                tids_cache[key] = _centered_time_ids(
                    pipe, (tw * vae_scale, th * vae_scale), device, dtype
                )
            wk = weight_cache[key][None, None]
            for c0 in range(0, len(origins), view_batch):
                chunk = origins[c0 : c0 + view_batch]
                n = len(chunk)
                z_in = torch.cat(
                    [
                        latents[:, :, y0 : y0 + th, x0 : x0 + tw]
                        for y0, x0 in chunk
                    ],
                    dim=0,
                )
                tids = tids_cache[key].repeat(n, 1)
                if do_cfg:
                    z_in = torch.cat([z_in, z_in], dim=0)
                    tids = torch.cat([tids, tids], dim=0)
                embeds, pool = _embeds_for(n)
                z_in = pipe.scheduler.scale_model_input(z_in, t)
                noise_pred = pipe.unet(
                    z_in,
                    t,
                    encoder_hidden_states=embeds,
                    added_cond_kwargs={"text_embeds": pool, "time_ids": tids},
                    return_dict=False,
                )[0]
                if do_cfg:
                    eps_u, eps_c = noise_pred.chunk(2)
                    noise_pred = eps_u + float(cfg) * (eps_c - eps_u)
                for i, (y0, x0) in enumerate(chunk):
                    eps_acc[:, :, y0 : y0 + th, x0 : x0 + tw] += (
                        noise_pred[i : i + 1].float() * wk
                    )
                    w_acc[:, :, y0 : y0 + th, x0 : x0 + tw] += wk
        eps = (eps_acc / w_acc.clamp_min(1e-4)).to(latents.dtype)
        latents = pipe.scheduler.step(eps, t, latents, return_dict=False)[0]

    lat_out = latents.detach()[0].to("cpu", torch.float16).numpy()

    # Decode with context padding + margin discard (§6.1).
    needs_upcast = getattr(pipe.vae.config, "force_upcast", False) and latents.dtype == torch.float16
    if needs_upcast:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    image = decode_latents_margin(pipe, latents)
    if needs_upcast:
        pipe.vae.to(dtype=torch.float16)
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    status = (
        f"Fused refine done ({n_views} tiles, denoise={denoise:.2f}, "
        f"steps={num_steps}, tile={tile}, skip={skip_w0:.2f})."
    )
    return image.convert("RGB"), status, lat_out


def _encode_prompt(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pos_embeds, neg_embeds, pos_pooled, neg_pooled)."""
    # Prefer the pipeline helper when available.
    if hasattr(pipe, "encode_prompt"):
        out = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt or None,
            negative_prompt_2=None,
        )
        # SDXL returns (prompt_embeds, neg_embeds, pooled, neg_pooled)
        if len(out) >= 4:
            return out[0], out[1], out[2], out[3]
    raise RuntimeError("pipe.encode_prompt unavailable")


def _default_time_ids(
    pipe: Any,
    size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    w, h = size
    # original_size, crops_top_left, target_size
    ids = [h, w, 0, 0, h, w]
    t = torch.tensor([ids], dtype=dtype, device=device)
    if hasattr(pipe, "_get_add_time_ids"):
        try:
            return pipe._get_add_time_ids(
                (h, w), (0, 0), (h, w), dtype=dtype, text_encoder_projection_dim=None
            ).to(device)
        except Exception:  # noqa: BLE001
            pass
    return t


def _centered_time_ids(
    pipe: Any,
    tile_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """'centered' micro-conditioning: the tile pretends to be a standalone
    centered image (research §5.4 — best regime for detail/refine passes)."""
    w, h = tile_size
    if hasattr(pipe, "_get_add_time_ids"):
        try:
            return pipe._get_add_time_ids(
                (h, w), (0, 0), (h, w), dtype=dtype, text_encoder_projection_dim=None
            ).to(device)
        except Exception:  # noqa: BLE001
            pass
    return torch.tensor([[h, w, 0, 0, h, w]], dtype=dtype, device=device)


def _tile_time_ids(
    pipe: Any,
    size: tuple[int, int],
    crop_xy: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """SDXL micro-conditioning with the view's real canvas position."""
    w, h = size
    x, y = crop_xy
    if hasattr(pipe, "_get_add_time_ids"):
        try:
            return pipe._get_add_time_ids(
                (h, w),
                (y, x),
                (h, w),
                dtype=dtype,
                text_encoder_projection_dim=None,
            ).to(device)
        except Exception:  # noqa: BLE001
            pass
    return torch.tensor(
        [[h, w, y, x, h, w]], dtype=dtype, device=device
    )
