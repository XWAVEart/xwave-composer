"""Unit tests for Infinite Canvas Mode (no GPU required)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from xwave_composer.pipeline.infinite_canvas.blueprint import (
    blueprint_init_image,
    blueprint_size,
)
from xwave_composer.pipeline.infinite_canvas.photometric import (
    match_stats_in_band,
    seam_energy,
)
from xwave_composer.pipeline.infinite_canvas.fuse import (
    CONTEXT_ACTIVE_THRESHOLD,
    _view_batch_size,
    merge_persistent_latents,
)
from xwave_composer.pipeline.infinite_canvas.priming import prime_canvas
from xwave_composer.pipeline.infinite_canvas.strength import (
    S_KEEP,
    S_NEW,
    build_strength_map,
    soft_stamp_mask,
    strength_to_change_map,
    to_latent_strength,
)
from xwave_composer.pipeline.large_canvas import (
    DEFAULT_CANVAS,
    LATENT_SCALE,
    SIZE_STEP,
    LargeCanvasSession,
    LatentCanvas,
    StampRect,
    snap64,
    stamp_size_for_aspect,
)
from xwave_composer.pipeline.region_fill import (
    _composite_once,
    _dilated_window,
    _effective_denoise,
    _merge_coverage,
    _occupancy_ratio,
    _sample_world_rect,
    _soft_inpaint_mask,
    _tile_origins,
    build_region_prompt,
)
from xwave_composer.style.style_manager import StylePreset
from xwave_composer.ui.large_canvas_tab import (
    REFINE_CUSTOM,
    REFINE_CUSTOM_STYLE,
    REFINE_STYLE,
    resolve_refine_prompt,
)


def test_snap64():
    assert snap64(1000) == 960
    assert snap64(1024) == 1024
    assert snap64(10) == SIZE_STEP


def test_stamp_aspect():
    w, h = stamp_size_for_aspect("16:9", 512)
    assert w % SIZE_STEP == 0 and h % SIZE_STEP == 0
    assert w > h


def test_stamp_offcanvas_intersection():
    stamp = StampRect(x=-128, y=-64, w=512, h=512).normalized()
    inter = stamp.intersection(1024, 1024)
    assert inter is not None
    assert inter.x == 0 and inter.y == 0
    assert inter.w == 384 and inter.h == 448


def test_nvfp4_uses_supported_single_view_batches():
    nvfp4 = SimpleNamespace(
        optimization_report=SimpleNamespace(applied="nvfp4")
    )
    fp8 = SimpleNamespace(
        optimization_report=SimpleNamespace(applied="fp8-weight-only")
    )
    assert _view_batch_size(nvfp4, do_cfg=False) == 1
    assert _view_batch_size(nvfp4, do_cfg=True) == 1
    assert _view_batch_size(fp8, do_cfg=False) == 4
    assert _view_batch_size(fp8, do_cfg=True) == 2


def test_session_create_expand_reset():
    lc = LargeCanvasSession(width=1024, height=1024)
    assert lc.width == 1024
    lc.create(2048, 1536)
    assert lc.width == 2048 and lc.height == 1536
    assert lc.status.startswith("Canvas created")
    assert lc.blueprint is None
    lc.expand(left=128, right=0, top=0, bottom=64)
    assert lc.width == 2176 and lc.height == 1600
    lc.reset()
    assert lc.image.getpixel((0, 0)) == (0, 0, 0)
    assert lc.occupied.getpixel((0, 0)) == 0


def test_stamp_can_straddle_edge_but_never_disappear():
    lc = LargeCanvasSession(width=2048, height=2048)
    stamp = lc.set_stamp(x=100_000, y=-100_000, w=1024, h=1024)
    assert stamp.x == lc.width - SIZE_STEP
    assert stamp.y == SIZE_STEP - stamp.h
    assert stamp.intersection(lc.width, lc.height) is not None
    assert stamp.intersection(lc.width, lc.height).w == SIZE_STEP
    assert stamp.intersection(lc.width, lc.height).h == SIZE_STEP


def test_build_region_prompt_orders():
    preset = StylePreset(
        name="Test",
        family="Test",
        prefix="watercolor of",
        suffix="on paper",
        negative="ugly",
        cfg=1.0,
        denoise=0.3,
        eta=0.0,
    )
    pos, neg = build_region_prompt("clock", preset, order="pcs")
    assert pos.startswith("watercolor")
    assert "clock" in pos
    assert neg == "ugly"
    man = build_region_prompt(
        "clock",
        preset,
        order="cps",
        manual_prefix="ink",
        manual_suffix="MSUF",
    )[0]
    assert "MSUF" in man


def test_build_region_prompt_has_no_hidden_injection():
    pos, neg = build_region_prompt("clockwork heart", None)
    assert pos == "clockwork heart"
    assert neg == ""


def test_occupancy_and_denoise():
    occ = Image.new("L", (256, 256), 0)
    stamp = StampRect(0, 0, 128, 128)
    assert _occupancy_ratio(occ, stamp) == 0.0
    assert _effective_denoise(0.0) == 1.0
    assert _effective_denoise(0.4) == 1.0
    # Full overwrite runs an img2img-truncated schedule from the existing
    # latents (latent README §7) so structure stays anchored.
    assert _effective_denoise(0.95) == 0.85
    assert _effective_denoise(0.95, 1.0) == 1.0
    assert _effective_denoise(0.95, 0.4) == 0.4
    # Blank content always needs the full schedule.
    assert _effective_denoise(0.4, 0.4) == 1.0


def test_user_prompt_text_is_preserved():
    supplied = "clockwork heart, isolated on plain black background, centered"
    pos, neg = build_region_prompt(
        supplied,
        None,
    )
    assert pos == supplied
    assert neg == ""


def test_refine_prompt_modes_are_explicit():
    preset = StylePreset(
        name="Test",
        family="Art",
        cfg=1.0,
        denoise=0.3,
        eta=0.0,
        prefix="oil painting of",
        suffix="rich impasto",
        negative="flat",
    )
    pos, neg = resolve_refine_prompt(REFINE_STYLE, "", preset)
    assert pos == "oil painting of, rich impasto"
    assert neg == "flat"

    pos, neg = resolve_refine_prompt(REFINE_CUSTOM, "restore brick detail", preset)
    assert pos == "restore brick detail"
    assert neg == ""

    pos, neg = resolve_refine_prompt(
        REFINE_CUSTOM_STYLE, "restore brick detail", preset
    )
    assert pos == "oil painting of restore brick detail, rich impasto"
    assert neg == "flat"


def test_refine_prompt_modes_validate_required_inputs():
    with pytest.raises(ValueError, match="Select a style"):
        resolve_refine_prompt(REFINE_STYLE, "", None)
    with pytest.raises(ValueError, match="custom refine prompt"):
        resolve_refine_prompt(REFINE_CUSTOM, "", None)
    with pytest.raises(ValueError, match="custom refine prompt"):
        resolve_refine_prompt(
            REFINE_CUSTOM_STYLE,
            "",
            StylePreset("T", "Art", 1.0, 0.3, 0.0, "p", "s", "n"),
        )


def test_reset_clears_pixels():
    lc = LargeCanvasSession(width=512, height=512)
    lc.apply_result(
        Image.new("RGB", (512, 512), (9, 9, 9)),
        Image.new("L", (512, 512), 255),
    )
    lc.reset()
    assert lc.image.getpixel((0, 0)) == (0, 0, 0)
    assert lc.occupied.getpixel((0, 0)) == 0
    assert lc.fit_view is True


def test_sample_world_rect_offcanvas():
    canvas = Image.new("RGB", (256, 256), (10, 20, 30))
    occupied = Image.new("L", (256, 256), 255)
    rect = StampRect(x=-64, y=-64, w=128, h=128)
    init, occ = _sample_world_rect(canvas, occupied, rect)
    assert init.size == (128, 128)
    assert init.getpixel((0, 0)) == (128, 128, 128)
    assert occ.getpixel((0, 0)) == 0
    assert init.getpixel((96, 96)) == (10, 20, 30)
    assert occ.getpixel((96, 96)) == 255


def test_sample_leaves_unpainted_as_neutral_not_black():
    canvas = Image.new("RGB", (256, 256), (0, 0, 0))
    occupied = Image.new("L", (256, 256), 0)
    occupied.paste(255, box=(0, 0, 128, 256))
    canvas.paste(Image.new("RGB", (128, 256), (200, 50, 50)), (0, 0))
    rect = StampRect(0, 0, 256, 256)
    init, occ = _sample_world_rect(canvas, occupied, rect)
    assert init.getpixel((64, 128))[0] > 150
    assert init.getpixel((192, 128)) == (128, 128, 128)
    assert occ.getpixel((192, 128)) == 0


def test_sdf_strength_blank_vs_committed():
    committed = np.zeros((128, 128), dtype=np.uint8)
    committed[:, 64:] = 1
    stamp = soft_stamp_mask(128, 128, 0, 0, 128, 128, feather=32)
    s = build_strength_map(committed, feather_px=32, stamp_mask=stamp)
    # Blank side high
    assert float(s[64, 16]) > 0.9
    # Deep committed lower than frontier
    assert float(s[64, 120]) < float(s[64, 72])
    assert float(s[64, 120]) >= S_KEEP - 1e-5


def test_sdf_full_overpaint_uses_stamp():
    committed = np.ones((128, 128), dtype=np.uint8)
    stamp = soft_stamp_mask(128, 128, 16, 16, 112, 112, feather=24)
    s = build_strength_map(committed, feather_px=24, stamp_mask=stamp)
    assert float(s[64, 64]) > 0.9  # center near S_NEW
    assert float(s[4, 4]) < 0.1  # outside stamp


def test_latent_strength_area_downsample():
    s = np.ones((64, 64), dtype=np.float32)
    s[:, :32] = 0.0
    lat = to_latent_strength(s, scale=8)
    assert lat.shape == (8, 8)
    assert float(lat[4, 1]) < 0.25
    assert float(lat[4, 6]) > 0.75


def test_soft_inpaint_mask_distance_fade():
    stamp = StampRect(0, 0, 256, 256)
    window = StampRect(0, 0, 256, 256)
    occ = Image.new("L", (256, 256), 0)
    occ.paste(255, box=(128, 0, 256, 256))
    mask = _soft_inpaint_mask(stamp, window, occ, feather=64.0, falloff=0.35)
    blank = mask.getpixel((32, 128))
    frontier = mask.getpixel((128, 128))
    near = mask.getpixel((140, 128))
    deep = mask.getpixel((240, 128))
    assert blank > 200
    assert blank >= frontier >= near >= deep
    assert deep >= int(S_KEEP * 255) - 5
    assert near < 120  # ~12 px into committed under ~76 px feather


def test_soft_inpaint_full_overpaint_soft_edge():
    stamp = StampRect(32, 32, 192, 192)
    window = StampRect(0, 0, 256, 256)
    full = Image.new("L", (256, 256), 255)
    mask = _soft_inpaint_mask(stamp, window, full, feather=48.0, falloff=0.35)
    assert mask.getpixel((128, 128)) > 200
    # Outside the stamp: harmonization floor only, never a hard rewrite.
    assert mask.getpixel((8, 8)) <= int(S_KEEP * 255) + 3
    # Soft edge: a few px inside the stamp, still below center.
    edge = mask.getpixel((48, 128))
    assert 10 < edge < 250
    assert edge < mask.getpixel((128, 128))


def test_overpaint_works_when_window_touches_blank():
    # Stamp fully on committed paint, but the read window includes blank
    # pixels (canvas edge / unpainted sliver). Repainting must still work:
    # inferring overpaint intent from the window pinned everything to the
    # keep floor and made overpaint generations silently do nothing.
    stamp = StampRect(128, 32, 256, 192)
    window = StampRect(0, 0, 512, 256)
    occ = Image.new("L", (512, 256), 255)
    occ.paste(0, box=(488, 0, 512, 256))  # blank strip at the window edge
    mask = _soft_inpaint_mask(
        stamp, window, occ, feather=32.0, falloff=0.35, overpaint=1.0
    )
    assert mask.getpixel((256, 128)) > 230  # stamp centre: full repaint
    assert mask.getpixel((32, 128)) <= int(S_KEEP * 255) + 3  # outside stamp

    # Without overpaint intent, the same stamp preserves committed content.
    mask0 = _soft_inpaint_mask(
        stamp, window, occ, feather=32.0, falloff=0.35, overpaint=0.0
    )
    assert mask0.getpixel((256, 128)) <= int(S_KEEP * 255) + 3


def test_strip_fill_blends_into_committed_outside_stamp():
    # The "hard edge across the whole canvas" bug: canvas full, expanded down,
    # stamp covering exactly the new blank strip. The committed content above
    # the frontier lies OUTSIDE the stamp; it must still receive the blend
    # band ("feather out"), not be pinned to the keep floor.
    window = StampRect(0, 0, 256, 256)
    stamp = StampRect(0, 128, 256, 128)  # bottom half = new blank strip
    occ = Image.new("L", (256, 256), 0)
    occ.paste(255, box=(0, 0, 256, 128))  # top half committed
    mask = _soft_inpaint_mask(stamp, window, occ, feather=48.0, falloff=0.35)
    frontier_committed = mask.getpixel((128, 120))  # 8 px above frontier
    mid_band = mask.getpixel((128, 100))  # ~28 px above frontier
    deep = mask.getpixel((128, 16))  # far above: protected
    blank = mask.getpixel((128, 220))  # deep in the strip: free
    assert frontier_committed > 60  # was ~13 (hard step) before the fix
    assert frontier_committed > mid_band > deep
    assert deep <= int(S_KEEP * 255) + 3
    assert blank > 240
    # No step across the frontier itself.
    just_above = mask.getpixel((128, 126))
    just_below = mask.getpixel((128, 130))
    assert abs(just_above - just_below) < 40


def test_full_overpaint_feathers_out_of_the_box():
    # Deliberate overpaint on a fully committed window: the strength must
    # decay from the band max at the stamp boundary to the floor outside,
    # instead of dropping to the floor immediately at the box edge.
    stamp = StampRect(96, 96, 64, 64)
    window = StampRect(0, 0, 256, 256)
    full = Image.new("L", (256, 256), 255)
    mask = _soft_inpaint_mask(stamp, window, full, feather=48.0, falloff=0.35)
    inside = mask.getpixel((128, 128))
    at_edge_out = mask.getpixel((88, 128))  # 8 px outside the stamp
    far_out = mask.getpixel((16, 128))  # 80 px outside
    assert inside > 200
    assert at_edge_out > 100  # band continues outside the box
    assert at_edge_out < inside
    assert far_out <= int(S_KEEP * 255) + 3


def test_blank_outside_stamp_is_never_pinned():
    # Window: committed strip on the left, stamp in the middle, blank right.
    # Blank pixels OUTSIDE the stamp must stay fully free. Pinning them to
    # the primed grey/black made the model see a dark mat around the live
    # region and paint literal picture frames.
    stamp = StampRect(96, 0, 96, 128)
    window = StampRect(0, 0, 256, 128)
    occ = Image.new("L", (256, 128), 0)
    occ.paste(255, box=(0, 0, 64, 128))
    mask = _soft_inpaint_mask(stamp, window, occ, feather=32.0, falloff=0.35)
    assert mask.getpixel((240, 64)) > 240  # blank, right of the stamp
    # Blank gap between committed content and the stamp sits mid-band —
    # harmonized, but far from pinned (the old stamp gating forced it to 0).
    assert mask.getpixel((80, 64)) > 90
    assert mask.getpixel((8, 64)) <= int(S_KEEP * 255) + 3  # deep committed


def test_dilated_window_includes_overlap():
    stamp = StampRect(512, 512, 1024, 1024)
    win = _dilated_window(stamp, feather=96, context_pad=128, overlap=128)
    # pad = max(96,128)+128 = 256 → context window 1536 (UNet still caps at 1024)
    assert win.w == 1024 + 2 * 256
    assert win.x == 512 - 256


def test_dilated_window_context_cap():
    stamp = StampRect(0, 0, 1024, 1024)
    win = _dilated_window(stamp, feather=128, context_pad=256, overlap=512)
    assert max(win.w, win.h) <= 1728


def test_fit_for_unet_downscales_dilated():
    from xwave_composer.pipeline.region_fill import _fit_for_unet

    gw, gh, resized = _fit_for_unet(1536, 1536)
    assert resized
    assert max(gw, gh) <= 1024
    assert gw % 64 == 0 and gh % 64 == 0


def test_prime_mirror_fills_holes():
    rgb = Image.new("RGB", (64, 64), (10, 20, 30))
    rgb.paste(Image.new("RGB", (32, 64), (200, 100, 50)), (0, 0))
    occ = Image.new("L", (64, 64), 0)
    occ.paste(255, box=(0, 0, 32, 64))
    filled = prime_canvas(rgb, occ, method="mirror")
    # Right half should no longer be the empty-fill gray from sampling.
    px = filled.getpixel((48, 32))
    assert px != (128, 128, 128)
    assert sum(px) > 50


def test_prime_large_hole_never_retains_grey_backing():
    rgb = np.full((512, 512, 3), 128, dtype=np.uint8)
    # Real, non-grey source structure occupies a relatively small top strip.
    rgb[32:160, 64:448] = (210, 70, 35)
    occ = np.zeros((512, 512), dtype=np.uint8)
    occ[32:160, 64:448] = 255
    filled = np.asarray(
        prime_canvas(Image.fromarray(rgb), Image.fromarray(occ), method="mirror")
    )
    hole = occ == 0
    retained_grey = np.all(filled == 128, axis=2) & hole
    assert not retained_grey.any()
    assert np.all(filled[400, 256] == (210, 70, 35))


def test_few_step_mask_frees_blank_boundary_early_enough():
    stamp = StampRect(0, 128, 256, 128)
    window = StampRect(0, 0, 256, 256)
    occ = Image.new("L", (256, 256), 0)
    occ.paste(255, box=(0, 0, 256, 128))
    mask = _soft_inpaint_mask(
        stamp,
        window,
        occ,
        feather=96.0,
        falloff=0.35,
        steps=8,
    )
    # At least six of eight Hyper evaluations: low-detail priming must not
    # survive as a flat band on the blank side of the source boundary.
    assert mask.getpixel((128, 128)) >= int(0.75 * 255) - 1


def test_context_prompt_is_reserved_for_deep_keep_views():
    # The old 0.5 threshold excluded most of the SDF transition from the user
    # prompt. Any harmonizing pixel above the keep floor must make the whole
    # view use the full subject prompt.
    assert S_KEEP < CONTEXT_ACTIVE_THRESHOLD < 0.1


def test_match_stats_in_band():
    new = Image.new("RGB", (64, 64), (200, 200, 200))
    ref = Image.new("RGB", (64, 64), (50, 50, 50))
    band = Image.new("L", (64, 64), 0)
    band.paste(255, box=(16, 16, 48, 48))
    out = match_stats_in_band(new, ref, band, strength=1.0)
    # Mean should move toward the darker ref in the band.
    assert np.asarray(out)[32, 32].mean() < 180


def test_match_stats_ref_mask_excludes_grey_priming():
    # New patch is rich/dark; ref is half committed rich content, half
    # grey-primed blank. Reference stats must come only from ref_mask so
    # grey pixels cannot wash the patch out.
    new = Image.new("RGB", (64, 64), (40, 60, 120))
    ref = Image.new("RGB", (64, 64), (128, 128, 128))  # grey priming
    ref.paste((45, 65, 125), box=(0, 0, 32, 64))  # committed neighbours
    band = Image.new("L", (64, 64), 255)
    ref_mask = Image.new("L", (64, 64), 0)
    ref_mask.paste(255, box=(0, 0, 32, 64))
    out = match_stats_in_band(new, ref, band, ref_mask=ref_mask, strength=1.0)
    px = np.asarray(out)[32, 48].astype(int)
    # Matched toward committed colour, not dragged toward flat grey.
    assert abs(int(px.mean()) - int(np.mean([45, 65, 125]))) < 25
    assert px.mean() < 100


def test_match_stats_apply_weight_zero_leaves_pixels_unchanged():
    new = Image.new("RGB", (64, 64), (200, 200, 200))
    ref = Image.new("RGB", (64, 64), (50, 50, 50))
    band = Image.new("L", (64, 64), 255)
    weight = np.zeros((64, 64), dtype=np.float32)
    weight[:, 32:] = 1.0
    out = np.asarray(
        match_stats_in_band(new, ref, band, apply_weight=weight, strength=1.0)
    )
    assert int(out[32, 8].mean()) >= 195  # weight 0: untouched
    assert int(out[32, 48].mean()) < 120  # weight 1: matched


def test_commit_alpha_full_write_over_blank_canvas():
    from xwave_composer.pipeline.region_fill import _commit_alpha

    strength = np.ones((32, 32), dtype=np.float32)  # blank window: all free
    committed = np.zeros((32, 32), dtype=bool)
    keep_hard = np.zeros((32, 32), dtype=bool)
    stamp_rect = np.zeros((32, 32), dtype=np.uint8)
    stamp_rect[:, 16:] = 1
    alpha = np.asarray(_commit_alpha(strength, committed, keep_hard, stamp_rect))
    # The whole stamp rectangle is written fully over blank canvas — feathered
    # contours there darken edges into black and leave gutters between stamps.
    assert alpha[16, 24] == 255
    assert alpha[0, 16] == 255  # hard to the rect corner, no rounding
    assert alpha[16, 4] == 0  # outside the stamp: discarded


def test_commit_alpha_trusts_latent_fusion_over_committed_content():
    from xwave_composer.pipeline.region_fill import _commit_alpha

    # The window decode is ONE continuous generation: everything diffusion
    # was allowed to touch is written fully. Wide pixel crossfades would
    # re-blend two different images and reintroduce smeared-halo seams.
    strength = np.full((64, 64), 0.5, dtype=np.float32)
    committed = np.ones((64, 64), dtype=bool)
    keep_hard = np.zeros((64, 64), dtype=bool)
    keep_hard[:, :16] = True  # deep-preserved region on the left
    stamp_rect = np.ones((64, 64), dtype=np.uint8)
    alpha = np.asarray(
        _commit_alpha(strength, committed, keep_hard, stamp_rect)
    )
    assert alpha[32, 48] == 255  # rewrite zone: full write, no crossfade
    assert alpha[32, 8] == 0  # deep keep: never written
    # Narrow VAE-hiding feather (~20 px) at the keep boundary only.
    assert 0 < alpha[32, 24] < 255
    assert alpha[32, 40] == 255  # feather is over by ~24 px out


def test_world_noise_overlap_agreement():
    from xwave_composer.pipeline.infinite_canvas.fuse import world_noise

    # Two windows overlapping in world space must receive identical noise in
    # the overlap (the shared-noise-map property that prevents boundary
    # discontinuities between adjacent patches).
    a = world_noise(64, 64, 0, 0, seed=1234)
    b = world_noise(64, 64, 32, 16, seed=1234)
    # a[..., y, x] is world (x, y); overlap region: x in [32,64), y in [16,64)
    assert (a[:, :, 16:, 32:] == b[:, :, :48, :32]).all()
    # Deterministic across calls.
    assert (a == world_noise(64, 64, 0, 0, seed=1234)).all()
    # Different seed → different field.
    assert not (a == world_noise(64, 64, 0, 0, seed=99)).all()
    # Negative world coordinates (windows beyond canvas origin) work.
    c = world_noise(32, 32, -16, -16, seed=1234)
    assert (c[:, :, 16:, 16:] == a[:, :, :16, :16]).all()


def test_expand_keeps_noise_field_anchored():
    lc = LargeCanvasSession(width=512, height=512)
    seed = lc.world_seed
    lc.expand(left=128, top=64)
    # Content moved by (+128, +64); offsets must compensate so the same
    # painted pixel keeps the same world-noise coordinate.
    assert lc.world_ox == 128 and lc.world_oy == 64
    assert lc.world_seed == seed  # expansion must not reroll the field


def test_reconcile_seam_tone_matches_committed_low_freq():
    from xwave_composer.pipeline.infinite_canvas.photometric import (
        reconcile_seam_tone,
    )

    # Committed left half is bright; generated content is uniformly darker.
    base = Image.new("RGB", (256, 128), (128, 128, 128))
    base.paste((200, 180, 160), box=(0, 0, 128, 128))
    gen = Image.new("RGB", (256, 128), (90, 110, 140))
    committed = np.zeros((128, 256), dtype=bool)
    committed[:, :128] = True
    out = np.asarray(
        reconcile_seam_tone(gen, base, committed, band_px=64.0)
    ).astype(int)
    # Just inside the new region at the seam: tone pulled to committed.
    seam_px = out[64, 132]
    assert abs(seam_px[0] - 200) < 30
    assert seam_px[0] > 150
    # Far into the new region: untouched.
    far_px = out[64, 250]
    assert abs(far_px[0] - 90) < 12


def test_jittered_origins_cover_and_stay_in_bounds():
    from xwave_composer.pipeline.infinite_canvas.fuse import (
        _jittered_origins,
        _tile_origins,
    )

    length, tile, ov = 320, 128, 32
    base = _tile_origins(length, tile, ov)
    for jitter in (-16, -7, 0, 5, 16):
        origins = _jittered_origins(length, tile, ov, jitter)
        assert len(origins) == len(base)
        assert origins[0] == 0 and origins[-1] == length - tile
        assert all(0 <= o <= length - tile for o in origins)
        covered = np.zeros(length, dtype=bool)
        for o in origins:
            covered[o : o + tile] = True
        assert covered.all()
    # No jitter → identical to the base grid.
    assert _jittered_origins(length, tile, ov, 0) == _tile_origins(length, tile, ov)
    # Window smaller than the tile: single view regardless of jitter.
    assert _jittered_origins(96, 128, 32, 9) == [0]
    # Common 1664px window = 208 latent px: always two origins per axis,
    # never the old accidental three (4 total views, not 9).
    assert len(_jittered_origins(208, 128, 32, 12)) == 2


def test_step_jitter_deterministic_and_zero_at_step_zero():
    from xwave_composer.pipeline.infinite_canvas.fuse import _step_jitter

    assert _step_jitter(1234, 0, 32) == (0, 0)  # deterministic starting grid
    assert _step_jitter(1234, 3, 0) == (0, 0)  # no overlap → no jitter
    a = _step_jitter(1234, 3, 32)
    assert a == _step_jitter(1234, 3, 32)  # reproducible
    assert all(-16 <= v <= 16 for v in a)
    assert _step_jitter(1234, 3, 32) != _step_jitter(4321, 3, 32) or a != (0, 0)


def test_decode_latents_margin_matches_full_decode():
    import torch

    from xwave_composer.pipeline.infinite_canvas.fuse import decode_latents_margin

    class _Cfg:
        scaling_factor = 1.0

    class _StubVAE:
        config = _Cfg()
        use_tiling = False

        def decode(self, z, return_dict=False):
            # Artifact-free stand-in decoder: nearest ×8 upsample of 3 chans.
            up = torch.nn.functional.interpolate(z[:, :3], scale_factor=8)
            return (up,)

    class _StubPipe:
        vae = _StubVAE()
        vae_scale_factor = 8

    z = torch.arange(4 * 40 * 48, dtype=torch.float32).reshape(1, 4, 40, 48) / 100.0
    full = _StubVAE().decode(z)[0]
    tiled = decode_latents_margin(_StubPipe(), z, tile_latent=16, margin_latent=4)
    # With an artifact-free decoder, margin-discard must reproduce the full
    # decode exactly — this validates the padded-window indexing arithmetic.
    assert torch.allclose(tiled, full)


def test_patchmatch_prime_falls_back_gracefully():
    from xwave_composer.pipeline.infinite_canvas.priming import _patchmatch

    rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
    committed = np.zeros((32, 32), dtype=bool)
    committed[:, :16] = True
    out = _patchmatch(rgb, committed)
    # Either pypatchmatch is installed and returns a filled image, or it is
    # missing and the helper signals fallback with None — never raises.
    assert out is None or out.shape == rgb.shape


def test_seam_energy_spike():
    img = np.zeros((64, 64), dtype=np.float32)
    img[:, 32:] = 255
    e = seam_energy(img, 32, band=16)
    assert e > 1.3


def test_blueprint_size_aspect():
    w, h = blueprint_size(4096, 2048, long_edge=1152)
    assert w == 1152
    assert h % 64 == 0
    assert w >= h


def test_blueprint_init_is_not_flat_grey():
    init = blueprint_init_image(256, 128, seed=42)
    arr = np.asarray(init)
    assert init.size == (256, 128)
    assert float(arr.std()) > 5.0
    assert not np.all(arr == 128)


def test_commit_alpha_is_applied_once_and_kept_as_soft_coverage():
    patch = Image.new("RGB", (8, 8), (255, 255, 255))
    dest = Image.new("RGB", (8, 8), (0, 0, 0))
    alpha = Image.new("L", (8, 8), 128)
    merged = _composite_once(patch, dest, alpha)
    # One 50% composite is ~128; applying the mask twice would be ~64.
    assert 120 <= merged.getpixel((4, 4))[0] <= 136

    coverage = _merge_coverage(Image.new("L", (8, 8), 0), alpha)
    assert coverage.getpixel((4, 4)) == 128


def test_tile_origins_cover_span():
    xs = _tile_origins(2048, 1024, 128)
    assert xs[0] == 0
    assert xs[-1] == 2048 - 1024


def test_strength_to_change_map_roundtrip():
    s = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    m = strength_to_change_map(s)
    assert m.mode == "L"
    assert m.size == (16, 16)
    assert m.getpixel((15, 15)) > 240


def test_default_canvas_constant():
    assert DEFAULT_CANVAS % SIZE_STEP == 0
    assert S_NEW == 1.0
    assert LargeCanvasSession(width=512, height=512).overlap == 256


# ---------------------------------------------------------------------------
# Persistent latent store (latent README §4: latent is the source of truth)


def test_latent_canvas_crop_pads_off_canvas():
    lc = LatentCanvas.empty(512, 512)  # 64×64 latent
    lc.z[:] = 1.0
    lc.valid[:] = True
    z, v = lc.crop(-8, -8, 32, 32)
    assert z.shape == (4, 32, 32)
    # Off-canvas quadrant is zero/invalid; on-canvas part is intact.
    assert not v[:8, :8].any()
    assert v[8:, 8:].all()
    assert float(z[:, :8, :8].max()) == 0.0
    assert float(z[:, 8:, 8:].min()) == 1.0


def test_latent_canvas_write_selects_one_final_latent_and_seeds():
    lc = LatentCanvas.empty(512, 512)
    # Pre-seed left half as valid with value 2.
    lc.z[:, :, :32] = 2.0
    lc.valid[:, :32] = True

    z_new = np.full((4, 64, 64), 4.0, dtype=np.float16)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[:, 16:48] = 0.5  # straddles the valid/invalid boundary
    seed = np.zeros((64, 64), dtype=bool)
    seed[:, 52:56] = True  # committed content the store had no latents for

    lc.write(0, 0, z_new, alpha, seed_mask=seed)

    # Generated ownership selects the already-fused new latent. Averaging final
    # old/new latents again would desaturate/blur the overlap.
    assert float(lc.z[0, 0, 20]) == 4.0
    # Valid + alpha 0 → old latents kept exactly.
    assert float(lc.z[0, 0, 4]) == 2.0
    assert lc.valid[0, 4]
    # Invalid + positive alpha → new value fully (never blend against zeros).
    assert float(lc.z[0, 0, 40]) == 4.0
    assert lc.valid[0, 40]
    # Invalid + alpha 0 + seed → seeded fully.
    assert float(lc.z[0, 0, 54]) == 4.0
    assert lc.valid[0, 54]
    # Invalid + alpha 0 + no seed → untouched and still invalid.
    assert not lc.valid[0, 60]
    assert float(lc.z[0, 0, 60]) == 0.0


def test_persistent_latent_merge_has_no_hard_validity_step():
    encoded = torch.zeros((1, 4, 32, 32), dtype=torch.float32)
    stored = np.full((4, 32, 32), 4.0, dtype=np.float16)
    valid = np.zeros((32, 32), dtype=bool)
    valid[:, :16] = True
    merged = merge_persistent_latents(encoded, stored, valid, transition=8)
    row = merged[0, 0, 16].numpy()
    # Deep source is exact, dirty side is fresh encoding, and the validity
    # boundary is a transition rather than a hard 4→0 latent splice.
    assert row[2] == 4.0
    assert row[16] == 0.0
    assert 0.0 < row[10] < 4.0
    assert np.max(np.abs(np.diff(row))) < 2.0


def test_latent_canvas_expand_and_invalidate():
    lc = LatentCanvas.empty(512, 512)
    lc.valid[:] = True
    lc.expand(64, 0, 128, 0)  # +8 latent px left, +16 top
    assert lc.shape == (64 + 16, 64 + 8)
    assert not lc.valid[:16, :].any()
    assert lc.valid[16:, 8:].all()
    lc.invalidate(8, 16, np.ones((4, 4), dtype=bool))
    assert not lc.valid[16:20, 8:12].any()


def test_session_latent_store_tracks_lifecycle():
    s = LargeCanvasSession(width=512, height=512)
    assert s.latent.shape == (512 // LATENT_SCALE, 512 // LATENT_SCALE)
    s.latent.valid[:] = True
    s.push_undo()
    s.latent.valid[:] = False
    assert s.undo()
    assert s.latent.valid.all()
    s.expand(left=64, top=64)
    assert s.latent.shape == ((512 + 64) // LATENT_SCALE, (512 + 64) // LATENT_SCALE)
    s.reset()
    assert not s.latent.valid.any()
