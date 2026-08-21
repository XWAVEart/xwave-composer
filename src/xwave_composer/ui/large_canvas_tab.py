"""Gradio UI for Infinite Canvas Mode — isolated from Compose WORK/OUTPUT."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, gpu_summary
from xwave_composer.models.sdxl_hyper import BASE_MODEL_PRESETS, SDXLHyperPipeline
from xwave_composer.models.upscaler import ImageUpscaler
from xwave_composer.optimization import (
    PROFILE_CHOICES,
    configured_profile,
    normalize_profile,
    profile_label,
)
from xwave_composer.pipeline.large_canvas import (
    CANVAS_PRESETS,
    DEFAULT_CANVAS,
    STAMP_ASPECTS,
    LargeCanvasSession,
)
from xwave_composer.pipeline.region_fill import (
    build_region_prompt,
    ensure_blueprint,
    fill_region,
    tiled_refine,
)
from xwave_composer.style.style_manager import (
    CONCAT_ORDERS,
    STYLE_FAMILIES,
    StyleManager,
    StylePreset,
)
from xwave_composer.library.render import pack_library_views
from xwave_composer.library.store import ImageLibrary
from xwave_composer.ui.library_tab import IC_IMPORT_PLACEMENTS, placement_key

logger = logging.getLogger(__name__)

NO_STYLE = "— none —"
FAMILY_ALL = "All families"
_PREVIEW_MAX = 1536
REFINE_STYLE = "Selected style"
REFINE_CUSTOM = "Custom prompt"
REFINE_CUSTOM_STYLE = "Custom prompt + selected style"
REFINE_MODES = (REFINE_STYLE, REFINE_CUSTOM, REFINE_CUSTOM_STYLE)


def resolve_refine_prompt(
    mode: str,
    custom_prompt: str,
    preset: StylePreset | None,
    *,
    order: str = "pcs",
    manual_prefix: str | None = None,
    manual_suffix: str | None = None,
) -> tuple[str, str]:
    """Resolve the explicit refine prompt mode into positive/negative text."""
    selected = str(mode or REFINE_STYLE)
    custom = str(custom_prompt or "").strip()
    has_style = preset is not None or bool(
        str(manual_prefix or "").strip() or str(manual_suffix or "").strip()
    )

    if selected == REFINE_CUSTOM:
        if not custom:
            raise ValueError("Enter a custom refine prompt.")
        # Exact custom means exact custom: no selected-style positive or
        # negative injection.
        return custom, ""

    if not has_style:
        raise ValueError("Select a style (or enable a manual prefix/suffix) for refine.")

    content = custom if selected == REFINE_CUSTOM_STYLE else ""
    if selected == REFINE_CUSTOM_STYLE and not content:
        raise ValueError("Enter a custom refine prompt.")
    return build_region_prompt(
        content,
        preset,
        order=order,
        manual_prefix=manual_prefix,
        manual_suffix=manual_suffix,
    )


def _cache_dir(config: AppConfig) -> Path:
    d = config.path("paths", "workspace_dir", default="workspace") / "large_canvas_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _preview_url(config: AppConfig, image: Image.Image) -> str:
    rgb = image.convert("RGB")
    w, h = rgb.size
    max_side = max(w, h)
    if max_side > _PREVIEW_MAX:
        scale = _PREVIEW_MAX / max_side
        rgb = rgb.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
    digest = hashlib.sha1()
    digest.update(str(rgb.size).encode())
    digest.update(rgb.tobytes())
    path = _cache_dir(config) / f"lc_{digest.hexdigest()[:20]}.jpg"
    if not path.exists():
        rgb.save(path, format="JPEG", quality=88, optimize=True)
    # Gradio serves via /gradio_api/file=
    return f"/gradio_api/file={path.resolve()}"


def _scene_html(config: AppConfig, lc: LargeCanvasSession) -> str:
    with lc._lock:
        stamp = lc.stamp
        fit = bool(lc.fit_view)
        # One-shot: UI recenters after create / reset / expand.
        lc.fit_view = False
        scene = {
            "width": lc.width,
            "height": lc.height,
            "preview_url": _preview_url(config, lc.image),
            "stamp": {"x": stamp.x, "y": stamp.y, "w": stamp.w, "h": stamp.h},
            "rev": lc.rev,
            "fit": fit,
        }
    b64 = base64.b64encode(json.dumps(scene, separators=(",", ":")).encode()).decode()
    return (
        f'<div id="xwave-lc-root" class="xwave-lc-root" data-scene="{b64}">'
        f'<div id="xwave-lc-wrap" class="xwave-lc-wrap">'
        f'<canvas id="xwave-lc-canvas" class="xwave-lc-canvas"></canvas>'
        f"</div></div>"
    )


def _export_path(config: AppConfig, image: Image.Image, fmt: str) -> Path:
    out_dir = config.path("export", "output_dir", default="exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(image.tobytes()).hexdigest()[:10]
    ext = fmt.lower()
    path = out_dir / f"large_canvas_{lc_size_tag(image)}_{digest}.{ext}"
    if ext == "png":
        image.save(path, format="PNG", optimize=True)
    elif ext == "webp":
        image.save(path, format="WEBP", quality=92, method=4)
    else:
        image.convert("RGB").save(path, format="JPEG", quality=95, subsampling=0)
    return path


def lc_size_tag(image: Image.Image) -> str:
    return f"{image.size[0]}x{image.size[1]}"


def build_large_canvas_tab(
    *,
    sdxl: SDXLHyperPipeline,
    styles: StyleManager | None,
    config: AppConfig,
    infer_lock: threading.Lock,
    upscaler: ImageUpscaler | None = None,
    free_vram_fn: Any | None = None,
    library: ImageLibrary | None = None,
) -> dict[str, Any]:
    """Build Infinite Canvas controls inside the current Tab context."""
    lc = LargeCanvasSession(width=DEFAULT_CANVAS, height=DEFAULT_CANVAS)

    style_family_choices = [FAMILY_ALL] + (
        styles.families_present() if styles else list(STYLE_FAMILIES)
    )
    style_names = [NO_STYLE] + (styles.names() if styles else [])
    base_names = list(BASE_MODEL_PRESETS.keys())
    active_profile = configured_profile(config)
    cfg_base = str(
        config.get("sdxl_hyper", "base_model_id", default=BASE_MODEL_PRESETS[base_names[0]])
    )
    default_base = next(
        (name for name, ref in BASE_MODEL_PRESETS.items() if ref == cfg_base),
        "Juggernaut XL v9" if "Juggernaut XL v9" in BASE_MODEL_PRESETS else base_names[0],
    )

    def pack() -> tuple:
        return (
            _scene_html(config, lc),
            lc.status,
            lc.stamp.x,
            lc.stamp.y,
            lc.stamp.w,
            lc.stamp.h,
            None,  # export file placeholder unless set
        )

    # equal_height=False so the side panel can scroll past the canvas height.
    # Canvas column takes most of the full-bleed Infinite Canvas tab width.
    with gr.Row(elem_classes=["xwave-row", "xwave-lc-top"], equal_height=False):
        with gr.Column(scale=5, elem_classes=["xwave-col", "xwave-lc-viewport"]):
            lc_html = gr.HTML(value=_scene_html(config, lc), elem_classes=["xwave-lc-html"])
            lc_action = gr.Textbox(
                value="",
                elem_id="xwave-lc-action",
                elem_classes=["xwave-hidden"],
                container=False,
                show_label=False,
            )
        with gr.Column(scale=2, min_width=280, elem_classes=["xwave-col", "xwave-lc-side"]):
            # Scroll body + fixed action bar (sticky inside flex columns is unreliable).
            with gr.Column(elem_classes=["xwave-lc-side-scroll"]):
                lc_status = gr.Textbox(
                    label="Status",
                    value=lc.status,
                    interactive=False,
                    lines=1,
                    max_lines=2,
                    elem_classes=["xwave-status"],
                )

                # Setup stays collapsed — drag region + prompt are the main loop.
                with gr.Accordion("Model", open=False):
                    base_dd = gr.Dropdown(
                        choices=base_names,
                        value=default_base,
                        label="Base",
                    )
                    base_custom = gr.Textbox(
                        label="Custom base",
                        placeholder="HF repo / CivitAI link",
                        lines=1,
                    )
                    performance_dd = gr.Dropdown(
                        label="Performance",
                        choices=PROFILE_CHOICES,
                        value=profile_label(active_profile),
                    )
                    load_btn = gr.Button("Load SDXL", variant="primary", size="sm")
                    model_status = gr.Textbox(
                        label="Loaded",
                        value=sdxl.optimization_report.status() if sdxl else "Not loaded",
                        interactive=False,
                        lines=1,
                        max_lines=2,
                    )

                with gr.Accordion("Canvas / expand", open=False):
                    with gr.Row(elem_classes=["xwave-lc-compact-row"]):
                        canvas_preset = gr.Dropdown(
                            choices=list(CANVAS_PRESETS.keys()),
                            value="2048×2048",
                            label="Preset",
                            elem_classes=["xwave-lc-mini-dd"],
                            scale=2,
                            min_width=96,
                        )
                        create_btn = gr.Button(
                            "Create", variant="primary", size="sm", scale=0, min_width=64
                        )
                        reset_btn = gr.Button("Reset", size="sm", scale=0, min_width=56)
                    with gr.Row(elem_classes=["xwave-lc-compact-row"]):
                        canvas_w = gr.Number(
                            label="W",
                            value=DEFAULT_CANVAS,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=40,
                        )
                        canvas_h = gr.Number(
                            label="H",
                            value=DEFAULT_CANVAS,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=40,
                        )
                        exp_l = gr.Number(
                            label="L",
                            value=0,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=36,
                        )
                        exp_r = gr.Number(
                            label="R",
                            value=0,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=36,
                        )
                        exp_t = gr.Number(
                            label="T",
                            value=0,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=36,
                        )
                        exp_b = gr.Number(
                            label="B",
                            value=0,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=36,
                        )
                        expand_btn = gr.Button("Expand", size="sm", scale=0, min_width=60)

                with gr.Accordion("Region numbers", open=False):
                    with gr.Row(elem_classes=["xwave-lc-compact-row"]):
                        aspect = gr.Dropdown(
                            choices=list(STAMP_ASPECTS.keys()),
                            value="1:1",
                            label="Aspect",
                            elem_classes=["xwave-lc-mini-dd"],
                            scale=2,
                            min_width=72,
                        )
                        stamp_x = gr.Number(
                            label="X",
                            value=lc.stamp.x,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=48,
                        )
                        stamp_y = gr.Number(
                            label="Y",
                            value=lc.stamp.y,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=48,
                        )
                        stamp_w = gr.Number(
                            label="W",
                            value=lc.stamp.w,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=48,
                        )
                        stamp_h = gr.Number(
                            label="H",
                            value=lc.stamp.h,
                            precision=0,
                            elem_classes=["xwave-lc-mini"],
                            scale=1,
                            min_width=48,
                        )
                    with gr.Row(elem_classes=["xwave-lc-compact-row"]):
                        nudge_down = gr.Button("Size −", size="sm")
                        nudge_up = gr.Button("Size +", size="sm")
                        apply_stamp_btn = gr.Button("Apply", size="sm")

                prompt = gr.Textbox(
                    label="Prompt",
                    lines=2,
                    max_lines=6,
                    placeholder="Describe this region…",
                )
                with gr.Row():
                    style_family = gr.Dropdown(
                        choices=style_family_choices,
                        value=FAMILY_ALL,
                        label="Family",
                    )
                    style_dd = gr.Dropdown(
                        choices=style_names,
                        value=NO_STYLE,
                        label="Style",
                    )
                with gr.Accordion("Prompt options", open=False):
                    concat_dd = gr.Dropdown(
                        label="Order",
                        choices=list(CONCAT_ORDERS.keys()),
                        value="Prefix · Prompts · Suffix",
                    )
                    manual_chk = gr.Checkbox(
                        label="Manual prefix/suffix", value=False
                    )
                    manual_prefix = gr.Textbox(
                        label="Prefix",
                        lines=1,
                        visible=False,
                        placeholder="e.g. watercolor illustration of",
                    )
                    manual_suffix = gr.Textbox(
                        label="Suffix",
                        lines=1,
                        visible=False,
                        placeholder="e.g. soft pastel palette",
                    )
                    built_prompt = gr.Textbox(
                        label="Built prompt",
                        lines=2,
                        max_lines=4,
                        interactive=False,
                        placeholder="Style + order preview…",
                    )

                with gr.Row(elem_classes=["xwave-lc-compact-row", "xwave-lc-knobs"]):
                    cfg = gr.Number(
                        value=lc.cfg,
                        label="CFG",
                        minimum=0.0,
                        maximum=15.0,
                        step=0.1,
                        elem_classes=["xwave-lc-mini"],
                        scale=1,
                        min_width=52,
                        info="Values above 1 enable CFG and roughly double UNet work.",
                    )
                    steps = gr.Slider(
                        1,
                        20,
                        value=lc.steps,
                        step=1,
                        label="Steps",
                        elem_classes=["xwave-lc-mini-slider"],
                        scale=2,
                        min_width=90,
                    )
                    eta = gr.Number(
                        value=lc.eta,
                        label="Eta",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        elem_classes=["xwave-lc-mini"],
                        scale=1,
                        min_width=52,
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        elem_classes=["xwave-lc-mini"],
                        scale=1,
                        min_width=56,
                    )

                with gr.Accordion("Blending", open=False):
                    gr.Markdown(
                        '<p class="xwave-text-dim">SDF strength map + Differential '
                        "Diffusion. Denoise window = region + feather + overlap. "
                        "Blank = full rewrite; fade into old paint; overpaint soft-edged.</p>"
                    )
                    context_pad = gr.Slider(
                        0, 512, value=lc.context_pad, step=64, label="Context pad"
                    )
                    overlap = gr.Slider(
                        64, 512, value=lc.overlap, step=64, label="Read overlap"
                    )
                    feather = gr.Slider(
                        32, 384, value=lc.feather, step=1, label="Feather"
                    )
                    falloff = gr.Slider(
                        0, 1, value=lc.falloff, step=0.05, label="Falloff"
                    )
                    overpaint_denoise = gr.Slider(
                        0.35,
                        1.0,
                        value=lc.overpaint_denoise,
                        step=0.05,
                        label="Overpaint effect",
                        info=(
                            "Strength inside a fully painted selection. "
                            "Feather/Falloff still blend its edges."
                        ),
                    )

                with gr.Accordion("Final refine", open=True):
                    refine_mode = gr.Dropdown(
                        choices=list(REFINE_MODES),
                        value=lc.refine_mode,
                        label="Prompt source",
                    )
                    refine_prompt = gr.Textbox(
                        label="Custom refine prompt",
                        value=lc.refine_prompt,
                        lines=1,
                        placeholder="Prompt used by Custom modes",
                        visible=False,
                    )
                    refine_denoise = gr.Slider(
                        0.05, 0.5, value=lc.refine_denoise, step=0.01, label="Denoise"
                    )
                    with gr.Row():
                        refine_tile = gr.Slider(
                            512, 1536, value=lc.refine_tile, step=64, label="Tile"
                        )
                        refine_overlap = gr.Slider(
                            64, 512, value=lc.refine_overlap, step=32, label="Overlap"
                        )
                    refine_btn = gr.Button("Run fused refine", size="sm")

                with gr.Accordion("Import", open=False):
                    import_place = gr.Radio(
                        choices=list(IC_IMPORT_PLACEMENTS),
                        value="Fit canvas",
                        label="Placement",
                    )
                    import_file = gr.Image(
                        type="pil",
                        label="Drop or upload",
                        height=100,
                        buttons=[],
                    )
                    import_file_btn = gr.Button("Import file", size="sm")
                    ic_lib_seed = (
                        pack_library_views(library)[4:6]
                        if library is not None
                        else ("", "No images yet")
                    )
                    ic_lib_html = gr.HTML(
                        value=ic_lib_seed[0],
                        elem_classes=["xwave-lib-picker"],
                    )
                    with gr.Row(elem_classes=["xwave-lib-pager"]):
                        ic_lib_prev = gr.Button("Prev", size="sm", scale=0, min_width=64)
                        ic_lib_pager = gr.Textbox(
                            value=ic_lib_seed[1],
                            show_label=False,
                            interactive=False,
                            container=False,
                            scale=1,
                        )
                        ic_lib_next = gr.Button("Next", size="sm", scale=0, min_width=64)
                    gr.Markdown(
                        '<p class="xwave-text-dim">Click a library thumb to import with the placement above.</p>'
                    )

                with gr.Accordion("Export", open=False):
                    export_fmt = gr.Radio(
                        choices=["PNG", "JPEG", "WebP"],
                        value="PNG",
                        label="Format",
                    )
                    export_upscale = gr.Checkbox(
                        label="Upscale 2× (SeedVR2)", value=False
                    )
                    export_btn = gr.Button("Export canvas", variant="primary", size="sm")
                    export_file = gr.File(label="Download", interactive=False)
                    lib_save_name = gr.Textbox(
                        label="Library name",
                        placeholder="Optional name",
                        lines=1,
                    )
                    lib_save_btn = gr.Button(
                        "Save to library",
                        size="sm",
                        elem_classes=["xwave-lib-save-btn"],
                    )

            with gr.Group(elem_classes=["xwave-lc-actions"]):
                with gr.Row():
                    gen_btn = gr.Button("Generate region", variant="primary", size="sm")
                    undo_btn = gr.Button("Undo", size="sm")
                    free_vram_btn = gr.Button("Free VRAM", size="sm")

    # ── handlers ──────────────────────────────────────────────────
    def _style_choices(fam: str | None) -> list[str]:
        if not styles:
            return [NO_STYLE]
        if not fam or fam == FAMILY_ALL:
            names = styles.names()
        else:
            names = styles.names([fam])
        return [NO_STYLE] + names

    def on_load_sdxl(preset_name, custom, profile_name):
        ref = str(custom or "").strip() or BASE_MODEL_PRESETS.get(
            str(preset_name), BASE_MODEL_PRESETS[base_names[0]]
        )
        selected = normalize_profile(profile_name)
        detail = "Not loaded"
        with infer_lock:
            try:
                # Exclusive mode: drop Flux / Compose stack before (re)loading SDXL.
                if free_vram_fn is not None:
                    try:
                        free_vram_fn()
                    except Exception:  # noqa: BLE001
                        logger.debug("enter Infinite Canvas before SDXL load failed", exc_info=True)
                config.raw.setdefault("optimization", {})["profile"] = selected
                sdxl.set_profile(selected, reload=False)
                msg = sdxl.load(force=True, base_ref=ref)
                detail = sdxl.optimization_report.status()
                lc.status = f"{msg} | {profile_label(selected)}"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Infinite Canvas SDXL load")
                lc.status = f"SDXL load failed: {exc}"
                detail = str(exc)
        return (
            _scene_html(config, lc),
            lc.status,
            detail,
            lc.stamp.x,
            lc.stamp.y,
            lc.stamp.w,
            lc.stamp.h,
            gr.update(),
        )

    def on_create(_preset, w, h):
        # Honor Width/Height fields. Preset dropdown only fills those fields.
        lc.create(int(w or DEFAULT_CANVAS), int(h or DEFAULT_CANVAS))
        return pack()[:6] + (gr.update(),)

    def on_reset():
        lc.reset()
        return pack()[:6] + (gr.update(),)

    def on_expand(left, right, top, bottom):
        lc.expand(
            left=int(left or 0),
            right=int(right or 0),
            top=int(top or 0),
            bottom=int(bottom or 0),
        )
        return pack()[:6] + (gr.update(),)

    def on_aspect(a):
        lc.set_stamp_aspect(a)
        return pack()[:6] + (gr.update(),)

    def on_nudge(delta):
        lc.nudge_stamp_size(int(delta))
        return pack()[:6] + (gr.update(),)

    def on_apply_stamp(x, y, w, h):
        lc.set_stamp(x=int(x), y=int(y), w=int(w), h=int(h))
        return pack()[:6] + (gr.update(),)

    def on_action(payload):
        try:
            data = json.loads(payload or "")
        except json.JSONDecodeError:
            return lc.stamp.x, lc.stamp.y, lc.stamp.w, lc.stamp.h
        if data.get("type") in ("stamp", "region"):
            lc.set_stamp(
                x=int(data.get("x", lc.stamp.x)),
                y=int(data.get("y", lc.stamp.y)),
                w=int(data.get("w", lc.stamp.w)),
                h=int(data.get("h", lc.stamp.h)),
            )
        # The canvas already drew this drag locally. Returning the HTML scene
        # here replaced the canvas DOM on every drop, causing a visible
        # reload/flicker and resetting pointer state. Sync only numeric fields.
        return lc.stamp.x, lc.stamp.y, lc.stamp.w, lc.stamp.h

    def on_style_family(fam, current):
        choices = _style_choices(fam)
        value = current if current in choices else NO_STYLE
        return gr.update(choices=choices, value=value)

    def _apply_prompt_opts(
        prompt_v: str,
        style_name: str | None,
        order_label: str,
        manual: bool,
        m_prefix: str,
        m_suffix: str,
    ) -> tuple[str, str]:
        """Store local prompt opts; return exact style-built positive/negative."""
        lc.prompt = str(prompt_v or "")
        lc.style_name = None if not style_name or style_name == NO_STYLE else style_name
        lc.concat_order = CONCAT_ORDERS.get(str(order_label), "pcs")
        lc.manual_style = bool(manual)
        lc.manual_prefix = str(m_prefix or "")
        lc.manual_suffix = str(m_suffix or "")
        preset = styles.get(lc.style_name) if styles and lc.style_name else None
        # When manual is on, pass strings (including "") so preset prefix/suffix are skipped.
        if lc.manual_style:
            man_p: str | None = lc.manual_prefix
            man_s: str | None = lc.manual_suffix
        else:
            man_p = None
            man_s = None
        positive, neg = build_region_prompt(
            lc.prompt,
            preset,
            order=lc.concat_order,
            manual_prefix=man_p,
            manual_suffix=man_s,
        )
        # Always replace, including with "", so clearing a style cannot leak its
        # negative prompt into later generations.
        lc.negative_prompt = neg
        return positive, neg

    def _style_only_prompt() -> str:
        """Non-subject prompt for context tiles (research §5.5): the style's
        prefix/suffix without the user's subject text, so context views don't
        each try to render the full subject."""
        preset = styles.get(lc.style_name) if styles and lc.style_name else None
        man_p = lc.manual_prefix if lc.manual_style else None
        man_s = lc.manual_suffix if lc.manual_style else None
        pos, _neg = build_region_prompt(
            "",
            preset,
            order=lc.concat_order,
            manual_prefix=man_p,
            manual_suffix=man_s,
        )
        return pos.strip()

    def on_manual_toggle(enabled):
        return gr.update(visible=bool(enabled)), gr.update(visible=bool(enabled))

    def on_prompt_preview(prompt_v, style_name, order_label, manual, m_prefix, m_suffix):
        positive, _neg = _apply_prompt_opts(
            prompt_v, style_name, order_label, manual, m_prefix, m_suffix
        )
        return positive

    def on_style(name, prompt_v, order_label, manual, m_prefix, m_suffix):
        # Local only — do not call styles.set_active (Compose keeps its own).
        lc.style_name = None if not name or name == NO_STYLE else name
        preset = styles.get(lc.style_name) if styles and lc.style_name else None
        if preset:
            lc.negative_prompt = preset.negative
            lc.cfg = float(preset.cfg)
            lc.eta = float(preset.eta)
            lc.status = f"Style: {preset.name}"
        else:
            lc.negative_prompt = ""
            lc.status = "Style cleared."
        built, _ = _apply_prompt_opts(
            prompt_v, name, order_label, manual, m_prefix, m_suffix
        )
        return (
            lc.status,
            lc.steps,
            lc.cfg,
            lc.eta,
            built,
        )

    def on_generate(
        prompt_v,
        style_name,
        order_label,
        manual,
        m_prefix,
        m_suffix,
        st,
        pad,
        ov,
        fe,
        fo,
        opden,
        cfg_v,
        eta_v,
        seed_v,
    ):
        started = time.perf_counter()
        lc.steps = int(st)
        lc.context_pad = int(pad)
        lc.overlap = int(ov)
        lc.feather = float(fe)
        lc.falloff = float(fo)
        lc.overpaint_denoise = float(opden)
        lc.cfg = float(cfg_v)
        lc.eta = float(eta_v)
        lc.seed = int(seed_v)

        if not sdxl.ready:
            lc.status = "Load SDXL first (choose base + quant, then Load SDXL)."
            return pack()[:6] + (gr.update(),)

        positive, neg = _apply_prompt_opts(
            prompt_v, style_name, order_label, manual, m_prefix, m_suffix
        )
        if not positive.strip():
            lc.status = "Enter a region prompt before generating."
            return pack()[:6] + (gr.update(),)

        acquired = infer_lock.acquire(blocking=False)
        if not acquired:
            lc.status = "SDXL busy (Compose or another job). Try again."
            return pack()[:6] + (gr.update(),)
        try:
            # Drop Flux / optional models so the dilated DiffDiff window has room.
            if free_vram_fn is not None:
                try:
                    free_vram_fn()
                except Exception:  # noqa: BLE001
                    logger.debug("pre-generate free_vram failed", exc_info=True)
            else:
                empty_cache()
            with lc._lock:
                canvas = lc.image.copy()
                occupied = lc.occupied.copy()
                stamp = lc.stamp.normalized()
                existing_bp = lc.blueprint
            # Lazy blueprint when canvas is largely blank (global plan).
            bp = ensure_blueprint(
                canvas,
                occupied,
                sdxl,
                positive,
                neg,
                steps=lc.steps,
                cfg=lc.cfg,
                eta=lc.eta,
                seed=None if lc.seed < 0 else lc.seed,
                existing=existing_bp,
            )
            if bp is not None:
                lc.blueprint = bp
            # Snapshot BEFORE the fill: it writes into lc.latent in place.
            lc.push_undo()
            out, out_occ, status = fill_region(
                canvas=canvas,
                occupied=occupied,
                stamp=stamp,
                sdxl=sdxl,
                prompt=positive,
                negative_prompt=neg,
                steps=lc.steps,
                cfg=lc.cfg,
                eta=lc.eta,
                seed=None if lc.seed < 0 else lc.seed,
                context_pad=lc.context_pad,
                feather=lc.feather,
                falloff=lc.falloff,
                overlap=lc.overlap,
                blueprint=bp,
                world_seed=lc.world_seed,
                world_origin=(lc.world_ox, lc.world_oy),
                context_prompt=_style_only_prompt(),
                latent_canvas=lc.latent,
                overpaint_denoise=lc.overpaint_denoise,
            )
            lc.apply_result(out, out_occ)
            lc.status = f"{status} total={time.perf_counter() - started:.1f}s"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Infinite Canvas generate")
            lc.status = f"Generate failed: {exc}"
        finally:
            infer_lock.release()
        return pack()[:6] + (gr.update(),)

    def on_undo():
        lc.undo()
        return pack()[:6] + (gr.update(),)

    def on_free_vram():
        with infer_lock:
            if free_vram_fn is not None:
                try:
                    msg = free_vram_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Infinite Canvas Free VRAM")
                    msg = f"Free VRAM failed: {exc}"
            else:
                empty_cache()
                msg = f"CUDA cache cleared. {gpu_summary()}"
        lc.status = str(msg)
        return pack()[:6] + (gr.update(),)

    def on_refine(
        prompt_v,
        style_name,
        order_label,
        manual,
        m_prefix,
        m_suffix,
        rmode,
        rprompt,
        rden,
        rtile,
        rov,
    ):
        if not sdxl.ready:
            lc.status = "Load SDXL first."
            return pack()[:6] + (gr.update(),)
        _positive, _region_neg = _apply_prompt_opts(
            prompt_v, style_name, order_label, manual, m_prefix, m_suffix
        )
        lc.refine_mode = (
            str(rmode) if str(rmode) in REFINE_MODES else REFINE_STYLE
        )
        lc.refine_prompt = str(rprompt or "").strip()
        preset = styles.get(lc.style_name) if styles and lc.style_name else None
        man_p = lc.manual_prefix if lc.manual_style else None
        man_s = lc.manual_suffix if lc.manual_style else None
        try:
            tile_prompt, neg = resolve_refine_prompt(
                lc.refine_mode,
                lc.refine_prompt,
                preset,
                order=lc.concat_order,
                manual_prefix=man_p,
                manual_suffix=man_s,
            )
        except ValueError as exc:
            lc.status = str(exc)
            return pack()[:6] + (gr.update(),)
        acquired = infer_lock.acquire(blocking=False)
        if not acquired:
            lc.status = "SDXL busy. Try again."
            return pack()[:6] + (gr.update(),)
        try:
            with lc._lock:
                canvas = lc.image.copy()
            # Snapshot BEFORE the refine: it writes into lc.latent in place.
            lc.push_undo()
            out, status = tiled_refine(
                canvas=canvas,
                sdxl=sdxl,
                prompt=tile_prompt,
                negative_prompt=neg,
                denoise=float(rden),
                steps=max(12, lc.steps),
                cfg=lc.cfg,
                eta=lc.eta,
                seed=None if lc.seed < 0 else lc.seed,
                tile=int(rtile),
                overlap=int(rov),
                latent_canvas=lc.latent,
            )
            with lc._lock:
                # Refine does not clear occupancy — whole canvas was painted.
                occ = Image.new("L", out.size, 255)
            lc.apply_result(out, occ)
            lc.status = status
        except Exception as exc:  # noqa: BLE001
            logger.exception("Infinite Canvas refine")
            lc.status = f"Refine failed: {exc}"
        finally:
            infer_lock.release()
        return pack()[:6] + (gr.update(),)

    def on_export(fmt, do_upscale):
        with lc._lock:
            image = lc.image.copy()
        if do_upscale and upscaler is not None:
            acquired = infer_lock.acquire(blocking=False)
            if not acquired:
                lc.status = "Cannot upscale while SDXL is busy."
                return pack()[:6] + (gr.update(),)
            try:
                # Free SDXL VRAM for SeedVR2 — same idea as Compose export.
                was_ready = sdxl.ready
                if was_ready:
                    sdxl.unload()
                lc.status = "Upscaling Infinite Canvas…"
                image = upscaler.upscale_2x(image)
                if was_ready:
                    sdxl.load()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Infinite Canvas upscale")
                lc.status = f"Upscale failed: {exc}"
                try:
                    if not sdxl.ready:
                        sdxl.load()
                except Exception:  # noqa: BLE001
                    pass
                return pack()[:6] + (gr.update(),)
            finally:
                infer_lock.release()
        elif do_upscale and upscaler is None:
            lc.status = "Upscaler not available; exporting at native size."

        fmt_key = str(fmt or "PNG").upper()
        ext = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}.get(fmt_key, "png")
        try:
            path = _export_path(config, image, ext)
            lc.status = f"Exported {path}"
            return (*pack()[:6], str(path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Infinite Canvas export")
            lc.status = f"Export failed: {exc}"
            return pack()[:6] + (gr.update(),)

    def on_import_file(image, place):
        if image is None:
            lc.status = "Choose an image to import."
            return pack()[:6] + (gr.update(),)
        try:
            encode = sdxl if sdxl is not None and sdxl.ready else None
            lc.import_image(image, mode=placement_key(place), sdxl=encode)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Infinite Canvas import")
            lc.status = f"Import failed: {exc}"
        return pack()[:6] + (gr.update(),)

    out6 = [lc_html, lc_status, stamp_x, stamp_y, stamp_w, stamp_h, export_file]
    load_out = [
        lc_html,
        lc_status,
        model_status,
        stamp_x,
        stamp_y,
        stamp_w,
        stamp_h,
        export_file,
    ]

    load_btn.click(
        on_load_sdxl,
        inputs=[base_dd, base_custom, performance_dd],
        outputs=load_out,
    )
    create_btn.click(on_create, inputs=[canvas_preset, canvas_w, canvas_h], outputs=out6)
    reset_btn.click(on_reset, outputs=out6)
    canvas_preset.change(
        lambda p: (*CANVAS_PRESETS.get(p, (DEFAULT_CANVAS, DEFAULT_CANVAS)),),
        inputs=[canvas_preset],
        outputs=[canvas_w, canvas_h],
    )
    expand_btn.click(on_expand, inputs=[exp_l, exp_r, exp_t, exp_b], outputs=out6)
    aspect.change(on_aspect, inputs=[aspect], outputs=out6)
    nudge_down.click(lambda: on_nudge(-1), outputs=out6)
    nudge_up.click(lambda: on_nudge(1), outputs=out6)
    apply_stamp_btn.click(
        on_apply_stamp, inputs=[stamp_x, stamp_y, stamp_w, stamp_h], outputs=out6
    )
    lc_action.change(
        on_action,
        inputs=[lc_action],
        outputs=[stamp_x, stamp_y, stamp_w, stamp_h],
        show_progress="hidden",
    )
    style_family.change(
        on_style_family, inputs=[style_family, style_dd], outputs=[style_dd]
    )
    prompt_opt_inputs = [
        prompt,
        style_dd,
        concat_dd,
        manual_chk,
        manual_prefix,
        manual_suffix,
    ]
    style_dd.change(
        on_style,
        inputs=[style_dd, prompt, concat_dd, manual_chk, manual_prefix, manual_suffix],
        outputs=[lc_status, steps, cfg, eta, built_prompt],
    )
    manual_chk.change(
        on_manual_toggle,
        inputs=[manual_chk],
        outputs=[manual_prefix, manual_suffix],
    )
    for _comp in (prompt, concat_dd, manual_chk, manual_prefix, manual_suffix):
        _comp.change(
            on_prompt_preview,
            inputs=prompt_opt_inputs,
            outputs=[built_prompt],
            show_progress="hidden",
        )
    gen_btn.click(
        on_generate,
        inputs=[
            prompt,
            style_dd,
            concat_dd,
            manual_chk,
            manual_prefix,
            manual_suffix,
            steps,
            context_pad,
            overlap,
            feather,
            falloff,
            overpaint_denoise,
            cfg,
            eta,
            seed,
        ],
        outputs=out6,
    )
    undo_btn.click(on_undo, outputs=out6)
    free_vram_btn.click(on_free_vram, outputs=out6)
    refine_mode.change(
        lambda mode: gr.update(visible=str(mode) != REFINE_STYLE),
        inputs=[refine_mode],
        outputs=[refine_prompt],
        show_progress="hidden",
    )
    refine_btn.click(
        on_refine,
        inputs=[
            prompt,
            style_dd,
            concat_dd,
            manual_chk,
            manual_prefix,
            manual_suffix,
            refine_mode,
            refine_prompt,
            refine_denoise,
            refine_tile,
            refine_overlap,
        ],
        outputs=out6,
    )
    export_btn.click(on_export, inputs=[export_fmt, export_upscale], outputs=out6)
    import_file_btn.click(
        on_import_file,
        inputs=[import_file, import_place],
        outputs=out6,
    )

    return {
        "session": lc,
        "html": lc_html,
        "status": lc_status,
        "save_btn": lib_save_btn,
        "save_name": lib_save_name,
        "import_place": import_place,
        "picker_html": ic_lib_html,
        "picker_pager": ic_lib_pager,
        "picker_prev": ic_lib_prev,
        "picker_next": ic_lib_next,
        "stamp_x": stamp_x,
        "stamp_y": stamp_y,
        "stamp_w": stamp_w,
        "stamp_h": stamp_h,
        "export_file": export_file,
    }
