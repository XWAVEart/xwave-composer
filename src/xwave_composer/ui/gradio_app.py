"""Gradio Blocks UI — dual-canvas workspace.

Layout:
  [ load · status · VRAM meter ] [ CFG · Denoise · Steps · Eta · Refine · Export 2× ]
  [ WORK canvas                ] [ OUTPUT canvas                                    ]
  [ Layers (+ add) ] [ layer prompt+generate | style | model & export ]
"""

from __future__ import annotations

import base64
import hashlib
import html as html_lib
import io
import json
import logging
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import vram_stats
from xwave_composer.models.sdxl_hyper import BASE_MODEL_PRESETS
from xwave_composer.optimization import (
    PROFILE_CHOICES,
    normalize_profile,
    profile_label,
)
from xwave_composer.pipeline.session import ComposerSession

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).resolve().parent / "assets"

# Gradio defaults Image downloads to WebP; OUTPUT should be a high-quality JPEG.
_OUTPUT_JPEG_QUALITY = 95
_OUTPUT_JPEG_DIR = Path(tempfile.gettempdir()) / "xwave_output"
_OUTPUT_JPEG_CACHE: dict[int, str] = {}


def _output_jpeg_path(img: Image.Image) -> str:
    """Write OUTPUT as a cached high-quality JPEG for display + download."""
    cached = _OUTPUT_JPEG_CACHE.get(id(img))
    if cached and Path(cached).exists():
        return cached

    rgb = img.convert("RGB") if img.mode != "RGB" else img
    _OUTPUT_JPEG_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1()
    digest.update(str(rgb.size).encode())
    digest.update(rgb.mode.encode())
    digest.update(rgb.tobytes())
    path = _OUTPUT_JPEG_DIR / f"xwave_output_{digest.hexdigest()[:20]}.jpg"
    if not path.exists():
        rgb.save(
            path,
            format="JPEG",
            quality=_OUTPUT_JPEG_QUALITY,
            subsampling=0,
            optimize=True,
        )
    cached_path = str(path)
    _OUTPUT_JPEG_CACHE[id(img)] = cached_path
    return cached_path

ASPECT_PRESETS = {
    "1024×1024": (1024, 1024),
    "1152×896": (1152, 896),
    "896×1152": (896, 1152),
    "1216×832": (1216, 832),
}

NO_STYLE = "— none —"

CONCAT_ORDERS = {
    "Prefix · Prompts · Suffix": "pcs",
    "Prefix · Suffix · Prompts": "psc",
}


# ---------------------------------------------------------------------------
# Scene / render helpers
# ---------------------------------------------------------------------------

# Cache PNG-encoded data URLs; re-encoding every layer on every UI update is slow.
_URL_CACHE: dict[str, tuple[int, int, str]] = {}


def _encode_data_url(img: Image.Image, max_side: int) -> str:
    bands = img.getbands()
    im = img.convert("RGBA") if "A" in bands else img.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _data_url(key: str, img: Image.Image | None, max_side: int) -> str | None:
    if img is None:
        _URL_CACHE.pop(key, None)
        return None
    cached = _URL_CACHE.get(key)
    if cached and cached[0] == id(img) and cached[1] == max_side:
        return cached[2]
    url = _encode_data_url(img, max_side)
    _URL_CACHE[key] = (id(img), max_side, url)
    return url


def _work(session: ComposerSession) -> Image.Image:
    if session.last_work is not None:
        return session.last_work
    from xwave_composer.canvas.compositor import compose_work_image

    session.last_work = compose_work_image(session.doc)
    return session.last_work


def _blank(w: int = 1024, h: int = 1024) -> Image.Image:
    return Image.new("RGB", (w, h), (16, 17, 20))


def _prune_url_cache(session: ComposerSession) -> None:
    live = {f"layer:{o.id}" for o in session.doc.objects}
    live |= {f"thumb:{o.id}" for o in session.doc.objects}
    live |= {"bg", "bg-thumb"}
    for key in list(_URL_CACHE):
        if key not in live:
            _URL_CACHE.pop(key, None)


def _scene_dict(session: ComposerSession) -> dict[str, Any]:
    _prune_url_cache(session)
    layers = []
    for obj in session.doc.objects:
        layers.append(
            {
                "id": obj.id,
                "prompt": obj.prompt,
                "x": float(obj.transform.x),
                "y": float(obj.transform.y),
                "scale_x": float(obj.transform.scale_x),
                "scale_y": float(obj.transform.scale_y),
                "rotation": float(obj.transform.rotation),
                "opacity": float(obj.transform.opacity),
                "visible": bool(obj.transform.visible),
                "w": int(obj.image.width) if obj.image else 256,
                "h": int(obj.image.height) if obj.image else 256,
                "data_url": _data_url(f"layer:{obj.id}", obj.image, 640),
            }
        )
    return {
        "width": session.doc.width,
        "height": session.doc.height,
        "selected_id": session.doc.selected_id,
        "layers": layers,
        "bg_data_url": _data_url("bg", session.doc.background, 1024),
        "bg_prompt": session.doc.background_prompt,
        "rev": int(time.time() * 1000) % 10_000_000,
    }


def render_work_html(scene: dict[str, Any]) -> str:
    b64 = base64.b64encode(json.dumps(scene, separators=(",", ":")).encode()).decode("ascii")
    return f'<div id="xwave-work-root" data-scene="{b64}"><canvas id="xwave-work-canvas" width="512" height="512"></canvas></div>'


def render_layers_html(session: ComposerSession) -> str:
    """Layer stack cards. Cards show the layer prompt (no titles)."""
    cards: list[str] = []
    for obj in reversed(session.doc.objects):  # top-most first
        sel = " is-selected" if session.doc.selected_id == obj.id else ""
        muted = " is-muted" if not obj.prompt_enabled else ""
        thumb_url = _data_url(f"thumb:{obj.id}", obj.image, 96) or ""
        thumb = f' style="background-image:url({thumb_url})"' if thumb_url else ""
        text = (obj.prompt or "").strip() or "empty — type a prompt below"
        empty = "" if (obj.prompt or "").strip() else " is-empty"
        label_text = f"MUTED · {text}" if not obj.prompt_enabled else text
        label = html_lib.escape(label_text[:72])
        lid = html_lib.escape(obj.id)
        cards.append(
            f'<div class="xwave-card{sel}{empty}{muted}" data-layer-id="{lid}" draggable="true">'
            f'<div class="xwave-thumb"{thumb}></div>'
            f'<div class="xwave-card-text">{label}</div>'
            f'<button type="button" class="xwave-del" data-delete-id="{lid}" '
            f'aria-label="Delete layer">×</button></div>'
        )
    # Background card always last (bottom of stack)
    bg_url = _data_url("bg-thumb", session.doc.background, 96) or ""
    bg_sel = " is-selected" if session.doc.selected_id == "__bg__" else ""
    bg_text = (session.doc.background_prompt or "").strip() or "background — type a prompt below"
    bg_empty = "" if (session.doc.background_prompt or "").strip() else " is-empty"
    bg_thumb = f' style="background-image:url({bg_url})"' if bg_url else ""
    cards.append(
        f'<div class="xwave-card xwave-card-bg{bg_sel}{bg_empty}" data-layer-id="__bg__">'
        f'<div class="xwave-thumb"{bg_thumb}></div>'
        f'<div class="xwave-card-text">{html_lib.escape(bg_text[:72])}</div>'
        f'<span class="xwave-bg-tag">BG</span></div>'
    )
    return f'<div id="xwave-layers-root"><div id="xwave-layer-stack">{"".join(cards)}</div></div>'


def _inspector(session: ComposerSession) -> dict[str, Any]:
    sid = session.doc.selected_id
    if sid == "__bg__":
        return {
            "kind": "background",
            "prompt": session.doc.background_prompt or "",
            "opacity": 1.0,
            "iso": "",
            "raw": None,
            "prompt_enabled": True,
        }
    obj = session.doc.selected()
    if obj is None:
        return {
            "kind": "none",
            "prompt": "",
            "opacity": 1.0,
            "iso": "",
            "raw": None,
            "prompt_enabled": True,
        }
    return {
        "kind": "object",
        "prompt": obj.prompt or "",
        "opacity": float(obj.transform.opacity),
        "iso": obj.isolation_prompt or "",
        "raw": obj.raw_image,
        "prompt_enabled": obj.prompt_enabled,
    }


def render_vram_html() -> str:
    stats = vram_stats()
    if stats is None:
        return '<div class="xwave-vram"><span class="xwave-vram-label">CPU</span></div>'
    used, total = stats
    pct = max(0.0, min(100.0, used / total * 100.0))
    tone = "ok" if pct < 70 else ("warn" if pct < 90 else "hot")
    return (
        f'<div class="xwave-vram" title="VRAM {used:.1f} / {total:.1f} GB">'
        f'<div class="xwave-vram-bar"><div class="xwave-vram-fill is-{tone}" '
        f'style="width:{pct:.0f}%"></div></div>'
        f'<span class="xwave-vram-label">{used:.1f}/{total:.0f}G</span></div>'
    )


def _build_theme() -> gr.themes.Base:
    try:
        return gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="slate",
            neutral_hue="slate",
            radius_size=gr.themes.sizes.radius_sm,
            spacing_size=gr.themes.sizes.spacing_sm,
            text_size=gr.themes.sizes.text_sm,
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        )
    except Exception:  # noqa: BLE001
        return gr.themes.Base()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app(config: AppConfig | None = None) -> gr.Blocks:
    config = config or AppConfig.load()
    config.ensure_dirs()
    session = ComposerSession(config=config)
    session.doc.selected_id = "__bg__"
    cw, ch = config.canvas_size

    css = (ASSETS / "app.css").read_text(encoding="utf-8") if (ASSETS / "app.css").exists() else ""
    js = (ASSETS / "work_canvas.js").read_text(encoding="utf-8") if (ASSETS / "work_canvas.js").exists() else ""
    head_js = f"<script>\n{js}\n</script>"
    theme = _build_theme()

    style_names = [NO_STYLE] + (session.styles.names() if session.styles else [])
    base_names = list(BASE_MODEL_PRESETS.keys())

    # Debounced OUTPUT refresh driven by canvas transforms
    out_lock = threading.Lock()
    out_state: dict[str, Any] = {"token": 0, "running": False, "dirty": False}

    def run_output_now(preview: bool = False) -> Image.Image:
        try:
            return session.run_output(_work(session), preview=preview)
        except Exception as exc:  # noqa: BLE001
            logger.exception("OUTPUT")
            session.status = f"OUTPUT error: {exc}"
            return session.last_output or _work(session)

    def mark_output_dirty(delay_override: float | None = None) -> None:
        if not session.core_ready:
            return  # never trigger a surprise model download from a drag
        with out_lock:
            out_state["dirty"] = True
            out_state["token"] += 1
            token = out_state["token"]
        delay = (
            float(delay_override)
            if delay_override is not None
            else float(config.get("sdxl_hyper", "live_debounce_s", default=0.35))
        )

        def _worker() -> None:
            time.sleep(delay)
            with out_lock:
                if token != out_state["token"]:
                    return
                if out_state["running"]:
                    out_state["dirty"] = True
                    return
                out_state["running"] = True
                out_state["dirty"] = False
            try:
                # Re-check after the sleep — SDXL may have been unloaded for
                # a SeedVR2 export while this worker was waiting.
                if not session.core_ready:
                    return
                run_output_now()
            finally:
                with out_lock:
                    out_state["running"] = False
                    again = out_state["dirty"]
                if again and session.core_ready:
                    mark_output_dirty()

        threading.Thread(target=_worker, daemon=True).start()

    def pack(run_out: bool = True, sync_out: bool = False, preview: bool = False) -> tuple:
        """-> work_html, layers_html, output, status,
        insp_prompt, insp_iso, insp_opacity, raw_view, prompt_view,
        llm_prompt_view, mute_prompt_btn"""
        work = _work(session)
        if sync_out:
            # Cancel a sleeping debounce worker. Normal editing actions return
            # the completed image in this event instead of waiting for polling.
            with out_lock:
                out_state["dirty"] = False
                out_state["token"] += 1
            out = run_output_now(preview=preview)
            if preview:
                # Settle to full quality once the user stops moving things.
                mark_output_dirty(delay_override=1.2)
        else:
            if run_out:
                mark_output_dirty()
            out = session.last_output or work
        insp = _inspector(session)
        kind = insp["kind"]
        fields_on = kind != "none"
        obj_on = kind == "object"
        scene = _scene_dict(session)
        s = session.output_settings
        llm_on = bool(s.use_llm_rewrite)
        concat_val = session.last_concat_prompt or ""
        if not llm_on:
            if s.prompt_locked:
                concat_val = s.custom_prompt or session.last_output_prompt or concat_val
            else:
                concat_val = session.last_output_prompt or concat_val
        if llm_on and s.prompt_locked:
            llm_val = s.custom_prompt or session.last_output_prompt or ""
        else:
            llm_val = session.last_output_prompt if llm_on else ""
        return (
            render_work_html(scene),
            render_layers_html(session),
            _output_jpeg_path(out),
            session.status,
            gr.update(value=insp["prompt"], interactive=fields_on),
            gr.update(value=insp["iso"], interactive=obj_on, visible=obj_on),
            gr.update(value=insp["opacity"], interactive=obj_on),
            gr.update(value=insp["raw"], visible=insp["raw"] is not None),
            concat_val,
            gr.update(value=llm_val, visible=llm_on),
            gr.update(
                value=(
                    "Mute prompt"
                    if insp["prompt_enabled"]
                    else "Unmute prompt"
                ),
                interactive=obj_on,
            ),
        )

    with gr.Blocks(title="xwave-composer", elem_classes=["xwave-app"]) as demo:
        demo._xwave_css = css  # type: ignore[attr-defined]
        demo._xwave_head = head_js  # type: ignore[attr-defined]
        demo._xwave_theme = theme  # type: ignore[attr-defined]
        demo._xwave_session = session  # type: ignore[attr-defined]

        # JS→Python action bridge. Gradio 6 removes visible=False components
        # from the DOM entirely, so this must stay visible and be hidden by CSS.
        action_out = gr.Textbox(
            value="",
            elem_id="xwave-action-out",
            elem_classes=["xwave-hidden"],
            container=False,
            show_label=False,
        )
        rev_state = gr.State(-1)

        with gr.Row(elem_classes=["xwave-brand-wrap"], equal_height=True):
            gr.Markdown('<p class="xwave-brand">XWAVE COMPOSER</p>')
            # Plain HTML rather than a gr.Button: the toggle is handled entirely
            # in the browser, so it must not round-trip to Python at all.
            gr.HTML(
                '<button type="button" id="xwave-mode-toggle" '
                'class="xwave-mode-toggle" aria-pressed="false">Advanced</button>',
                padding=False,
            )

        # ══ ROW 1 — top bars ═══════════════════════════════════════
        with gr.Row(elem_classes=["xwave-row", "xwave-topbar"], equal_height=True):
            with gr.Column(scale=1, min_width=380, elem_classes=["xwave-col"]):
                with gr.Row(elem_classes=["xwave-bar-row"]):
                    load_btn = gr.Button("Load models", variant="primary", size="sm", scale=0, min_width=104)
                    free_btn = gr.Button("Free VRAM", size="sm", scale=0, min_width=88)
                    status = gr.Textbox(
                        show_label=False,
                        value="Ready — press Load models to begin.",
                        interactive=False,
                        lines=1,
                        max_lines=1,
                        scale=4,
                        min_width=140,
                        elem_classes=["xwave-status"],
                        container=False,
                    )
                    vram_html = gr.HTML(
                        value=render_vram_html(), padding=False,
                        elem_classes=["xwave-vram-block"],
                    )
                active_profile = session.compute_profile
                with gr.Row(elem_classes=["xwave-performance-buttons", "xwave-advanced"]):
                    bf16_btn = gr.Button(
                        "BF16",
                        size="sm",
                        variant="primary" if active_profile == "bf16" else "secondary",
                    )
                    mxfp8_btn = gr.Button(
                        "MXFP8",
                        size="sm",
                        variant="primary" if active_profile == "mxfp8" else "secondary",
                    )
                    nvfp4_btn = gr.Button(
                        "NVFP4",
                        size="sm",
                        variant="primary" if active_profile == "nvfp4" else "secondary",
                    )
            with gr.Column(scale=1, min_width=380, elem_classes=["xwave-col", "xwave-advanced"]):
                with gr.Row(elem_classes=["xwave-bar-row", "xwave-knob-row"]):
                    cfg = gr.Number(
                        value=session.output_settings.cfg,
                        label="CFG",
                        minimum=0.0,
                        maximum=15.0,
                        step=0.1,
                        scale=0,
                        min_width=64,
                        elem_classes=["xwave-knob"],
                    )
                    denoise = gr.Slider(
                        0.05, 0.95,
                        value=session.output_settings.denoise,
                        step=0.01,
                        label="Denoise",
                        scale=3,
                        min_width=100,
                        elem_classes=["xwave-knob"],
                    )
                    out_steps = gr.Slider(
                        1, 20,
                        value=session.output_settings.steps,
                        step=1,
                        label="Steps",
                        scale=2,
                        min_width=84,
                        elem_classes=["xwave-knob"],
                    )
                    eta = gr.Number(
                        value=session.output_settings.eta,
                        label="Eta",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        scale=0,
                        min_width=76,
                        elem_classes=["xwave-knob"],
                    )
        # ══ ROW 2 — canvases, perfectly side by side ═══════════════
        with gr.Row(elem_classes=["xwave-row", "xwave-canvases"], equal_height=False):
            with gr.Column(scale=1, min_width=380, elem_classes=["xwave-col"]):
                with gr.Group(elem_classes=["xwave-canvas-frame"]):
                    work_html = gr.HTML(value=render_work_html(_scene_dict(session)), padding=False)

            with gr.Column(scale=1, min_width=380, elem_classes=["xwave-col"]):
                with gr.Group(elem_classes=["xwave-canvas-frame"]):
                    output_image = gr.Image(
                        value=_output_jpeg_path(_blank(cw, ch)),
                        type="filepath",
                        format="jpeg",
                        show_label=False,
                        interactive=False,
                        buttons=["download", "fullscreen"],
                        elem_classes=["xwave-out-img"],
                    )

        # ══ ROW 3 — layers | properties ═══════════════════════════
        with gr.Row(elem_classes=["xwave-row", "xwave-bottom"], equal_height=False):
            with gr.Column(scale=1, min_width=230, elem_classes=["xwave-col", "xwave-layers-col"]):
                with gr.Row(elem_classes=["xwave-layers-head"]):
                    gr.Markdown('<p class="xwave-section-head">Layers</p>')
                    reset_workspace_btn = gr.Button(
                        "Reset", size="sm", scale=0, min_width=64,
                    )
                    add_obj_btn = gr.Button(
                        "+ Layer", size="sm", scale=0, min_width=72,
                        elem_classes=["xwave-add-btn"],
                    )
                layers_html = gr.HTML(value=render_layers_html(session), padding=False)

            with gr.Column(scale=3, min_width=420, elem_classes=["xwave-col", "xwave-props-col"]):
                with gr.Row(elem_classes=["xwave-props-grid"]):
                    # —— Selected layer: prompt + generate ——
                    with gr.Column(scale=1, min_width=250, elem_classes=["xwave-panel"]):
                        gr.Markdown('<p class="xwave-section-head">Layer</p>')
                        insp_prompt = gr.Textbox(
                            show_label=False,
                            container=False,
                            placeholder="Layer prompt — select a layer, type, Generate…",
                            lines=2,
                            max_lines=3,
                        )
                        insp_iso = gr.Textbox(
                            label="Isolation prompt",
                            value="isolated on plain white background, centered",
                            lines=1,
                            visible=False,
                        )
                        cutout_chk = gr.Checkbox(
                            label="Cut out object (uncheck to place the full image)",
                            value=True,
                            elem_classes=["xwave-advanced"],
                        )
                        with gr.Row():
                            gen_seed = gr.Number(
                                value=-1, precision=0, show_label=False, container=False,
                                scale=0, min_width=76,
                                elem_classes=["xwave-seed", "xwave-advanced"],
                            )
                            gen_btn = gr.Button(
                                "⟡ Generate", variant="primary", size="sm", scale=1, min_width=110
                            )
                        with gr.Row():
                            mute_prompt_btn = gr.Button(
                                "Mute prompt", size="sm", scale=1, min_width=92,
                                elem_classes=["xwave-advanced"],
                            )
                            duplicate_btn = gr.Button(
                                "Duplicate", size="sm", scale=1, min_width=82
                            )
                        with gr.Row():
                            reset_xform_btn = gr.Button(
                                "Reset pose", size="sm", scale=1, min_width=88,
                                elem_classes=["xwave-advanced"],
                            )
                            reisolate_btn = gr.Button(
                                "Re-cut", size="sm", scale=1, min_width=72,
                                elem_classes=["xwave-advanced"],
                            )
                            delete_btn = gr.Button("Delete", size="sm", variant="stop", scale=1, min_width=72)
                        insp_opacity = gr.Slider(
                            0.0, 1.0, value=1.0, step=0.01, label="Opacity",
                            interactive=False,
                            elem_classes=["xwave-advanced"],
                        )
                        raw_view = gr.Image(
                            label="Raw — click the subject to re-cut with SAM2",
                            type="pil",
                            interactive=False,
                            visible=False,
                            height=170,
                            buttons=[],
                            elem_classes=["xwave-raw-view"],
                        )
                        gr.Markdown(
                            '<p class="xwave-section-head">Import image</p>',
                            elem_classes=["xwave-advanced"],
                        )
                        import_img = gr.Image(
                            type="pil",
                            label="Drop or upload",
                            height=120,
                            buttons=[],
                            elem_classes=["xwave-import", "xwave-advanced"],
                        )
                        import_cutout = gr.Radio(
                            choices=["rembg", "SAM2", "none"],
                            value="rembg",
                            label="Cutout",
                            elem_classes=["xwave-import-cutout", "xwave-advanced"],
                        )
                        import_btn = gr.Button(
                            "Import into layer", size="sm", elem_classes=["xwave-advanced"]
                        )

                    # —— Improve: critique the picture, propose a better prompt ——
                    with gr.Column(scale=1, min_width=250, elem_classes=["xwave-panel"]):
                        gr.Markdown('<p class="xwave-section-head">Improve</p>')
                        improve_notes = gr.Textbox(
                            show_label=False,
                            container=False,
                            placeholder="What's wrong with it? (optional — it can also just look)",
                            lines=2,
                            max_lines=3,
                        )
                        improve_edit_chk = gr.Checkbox(
                            label="Edit mode — refine this image instead of rewriting the prompt",
                            value=False,
                        )
                        with gr.Row(elem_classes=["xwave-improve-row"]):
                            improve_btn = gr.Button(
                                "✧ Improve", variant="primary", size="sm", scale=1, min_width=110
                            )
                            improve_apply_btn = gr.Button(
                                "Use this", size="sm", scale=1, min_width=88
                            )
                        improve_critique = gr.Textbox(
                            label="Critique",
                            value="",
                            interactive=False,
                            lines=5,
                            max_lines=10,
                            elem_classes=["xwave-critique"],
                        )
                        improve_prompt_view = gr.Textbox(
                            label="Improved prompt (edit before using if you like)",
                            value="",
                            lines=3,
                            max_lines=8,
                        )

                    # —— Output style ——
                    with gr.Column(
                        scale=1, min_width=250, elem_classes=["xwave-panel", "xwave-advanced"]
                    ):
                        gr.Markdown('<p class="xwave-section-head">Style</p>')
                        style_dd = gr.Dropdown(
                            choices=style_names,
                            value=NO_STYLE,
                            show_label=False,
                            container=False,
                            filterable=True,
                        )
                        concat_dd = gr.Dropdown(
                            label="Prompt order",
                            choices=list(CONCAT_ORDERS.keys()),
                            value="Prefix · Prompts · Suffix",
                        )
                        manual_chk = gr.Checkbox(
                            label="Manual style (own prefix/suffix)", value=False
                        )
                        manual_prefix = gr.Textbox(
                            label="Manual prefix", lines=1, visible=False,
                            placeholder="e.g. watercolor illustration of",
                        )
                        manual_suffix = gr.Textbox(
                            label="Manual suffix", lines=1, visible=False,
                            placeholder="e.g. soft pastel palette, paper texture",
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative prompt",
                            value=session.output_settings.negative_prompt,
                            lines=2,
                            max_lines=3,
                        )
                        with gr.Row():
                            use_llm = gr.Checkbox(label="LLM rewrite", value=False, scale=1)
                            out_seed = gr.Number(
                                label="Seed",
                                value=session.output_settings.seed,
                                precision=0,
                                scale=1,
                                min_width=100,
                            )
                            roll_seed_btn = gr.Button(
                                "Roll seed", size="sm", scale=0, min_width=88
                            )
                        with gr.Row():
                            aspect = gr.Dropdown(
                                label="Canvas size",
                                choices=list(ASPECT_PRESETS.keys()),
                                value="1024×1024",
                                scale=2,
                            )
                            apply_size_btn = gr.Button("Apply", size="sm", scale=0, min_width=70)
                        # Concatenation result (always available)
                        prompt_lock = gr.Checkbox(
                            label="Edit built prompt (lock auto-build)",
                            value=False,
                        )
                        prompt_view = gr.Textbox(
                            label="Built prompt",
                            lines=2,
                            max_lines=4,
                            interactive=False,
                        )
                        # Shown when LLM rewrite is on — the rewritten prompt, optionally editable
                        llm_prompt_lock = gr.Checkbox(
                            label="Edit LLM prompt (lock rewrite)",
                            value=False,
                            visible=False,
                        )
                        llm_prompt_view = gr.Textbox(
                            label="LLM rewritten prompt",
                            lines=3,
                            max_lines=6,
                            interactive=False,
                            visible=False,
                            placeholder="Enable LLM rewrite to generate…",
                        )

                    # —— Model + export ——
                    with gr.Column(scale=1, min_width=250, elem_classes=["xwave-panel"]):
                        gr.Markdown(
                            '<p class="xwave-section-head">Output model</p>',
                            elem_classes=["xwave-advanced"],
                        )
                        with gr.Row(elem_classes=["xwave-advanced"]):
                            base_dd = gr.Dropdown(
                                choices=base_names,
                                value=base_names[0],
                                show_label=False,
                                container=False,
                                scale=2,
                            )
                            base_btn = gr.Button("Load", size="sm", scale=0, min_width=64)
                        base_custom = gr.Textbox(
                            show_label=False,
                            container=False,
                            placeholder="…or HF repo id / CivitAI .safetensors link",
                            lines=1,
                            elem_classes=["xwave-advanced"],
                        )
                        performance_dd = gr.Dropdown(
                            label="Performance",
                            choices=PROFILE_CHOICES,
                            value=profile_label(session.compute_profile),
                            elem_classes=["xwave-advanced"],
                        )
                        performance_status = gr.Textbox(
                            label="Applied compute",
                            value=session.optimization_status(),
                            interactive=False,
                            lines=2,
                            max_lines=3,
                            elem_classes=["xwave-advanced"],
                        )
                        with gr.Accordion(
                            "Style adapters (LoRA / TI)",
                            open=False,
                            elem_classes=["xwave-advanced"],
                        ):
                            lora_path = gr.Textbox(label="LoRA path or HF id", lines=1)
                            lora_scale = gr.Slider(0, 1.5, value=0.8, step=0.05, label="LoRA scale")
                            load_lora_btn = gr.Button("Load LoRA", size="sm")
                            emb_path = gr.Textbox(label="Textual inversion path or id", lines=1)
                            load_emb_btn = gr.Button("Load TI", size="sm")
                        gr.Markdown('<p class="xwave-section-head">Final output</p>')
                        with gr.Row(elem_classes=["xwave-advanced"]):
                            final_refine_strength = gr.Slider(
                                0.05,
                                0.95,
                                value=float(
                                    config.get(
                                        "export",
                                        "default_refine_strength",
                                        default=0.3,
                                    )
                                ),
                                step=0.01,
                                label="Refine strength",
                                scale=2,
                            )
                            final_refine_steps = gr.Slider(
                                1,
                                40,
                                value=int(
                                    config.get(
                                        "export",
                                        "default_refine_steps",
                                        default=8,
                                    )
                                ),
                                step=1,
                                label="Refine steps",
                                scale=2,
                            )
                        with gr.Row():
                            update_output_btn = gr.Button(
                                "Update OUTPUT now", size="sm",
                            )
                            refine_output_btn = gr.Button(
                                "Refine OUTPUT", variant="secondary", size="sm",
                            )
                        with gr.Accordion(
                            "SeedVR2 export settings", open=False, elem_classes=["xwave-advanced"]
                        ):
                            seedvr_model = gr.Dropdown(
                                label="Model",
                                choices=[
                                    (
                                        "7B FP16 — highest fidelity",
                                        "seedvr2_ema_7b_fp16.safetensors",
                                    ),
                                    (
                                        "7B Sharp FP16 — enhanced detail",
                                        "seedvr2_ema_7b_sharp_fp16.safetensors",
                                    ),
                                ],
                                value=config.get(
                                    "export",
                                    "seedvr2_model",
                                    default="seedvr2_ema_7b_fp16.safetensors",
                                ),
                            )
                            seedvr_color = gr.Dropdown(
                                label="Color fidelity",
                                choices=["lab", "wavelet", "wavelet_adaptive", "none"],
                                value=config.get(
                                    "export",
                                    "seedvr2_color_correction",
                                    default="lab",
                                ),
                            )
                            with gr.Row():
                                seedvr_input_noise = gr.Slider(
                                    0.0,
                                    0.3,
                                    value=float(
                                        config.get(
                                            "export",
                                            "seedvr2_input_noise_scale",
                                            default=0.0,
                                        )
                                    ),
                                    step=0.01,
                                    label="Artifact reduction",
                                )
                                seedvr_latent_noise = gr.Slider(
                                    0.0,
                                    0.2,
                                    value=float(
                                        config.get(
                                            "export",
                                            "seedvr2_latent_noise_scale",
                                            default=0.0,
                                        )
                                    ),
                                    step=0.01,
                                    label="Detail softness",
                                )
                            seedvr_seed = gr.Number(
                                label="Seed",
                                value=int(
                                    config.get(
                                        "export",
                                        "seedvr2_seed",
                                        default=42,
                                    )
                                ),
                                precision=0,
                            )
                        export_btn = gr.Button(
                            "Export accepted OUTPUT 2× with SeedVR2",
                            variant="primary",
                            size="sm",
                            elem_classes=["xwave-export-btn"],
                        )
                        export_path = gr.Textbox(
                            label="Export path", interactive=False, lines=1
                        )
                        export_image = gr.Image(
                            label="Export preview",
                            type="filepath",
                            format="jpeg",
                            interactive=False,
                            height=140,
                            buttons=["download"],
                        )

        # ── Callbacks ───────────────────────────────────────────
        pack_out = [
            work_html,
            layers_html,
            output_image,
            status,
            insp_prompt,
            insp_iso,
            insp_opacity,
            raw_view,
            prompt_view,
            llm_prompt_view,
            mute_prompt_btn,
        ]
        settings_in = [denoise, out_steps, cfg, eta, use_llm, neg_prompt, out_seed]

        def apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed):
            s = session.output_settings
            s.denoise = float(den)
            s.steps = int(steps)
            s.cfg = float(cfg_v)
            s.eta = float(eta_v)
            s.use_llm_rewrite = bool(llm)
            s.negative_prompt = str(neg or "")
            s.seed = int(oseed)

        def on_load():
            return session.preload_core()

        def on_free():
            return session.free_optional()

        def on_action(payload, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if not payload or not str(payload).strip():
                return tuple(gr.update() for _ in pack_out)
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                session.status = "Bad action payload"
                return pack(run_out=False)

            atype = data.get("type")

            if atype == "transform":
                session.update_transform_by_id(
                    str(data.get("id", "")),
                    x=data.get("x"),
                    y=data.get("y"),
                    scale_x=data.get("scale_x"),
                    scale_y=data.get("scale_y"),
                    rotation=data.get("rotation"),
                )
                session.status = "Transforming…"
                # Older tabs (opened before the canvas JS update) omit
                # ``final``. Keep those functional until the user refreshes;
                # current tabs send false while dragging and true on release.
                final = data.get("final")
                if final is True:
                    # Pointer-up returns a fast low-resolution pass immediately,
                    # so the OUTPUT tracks the drag; pack() then schedules the
                    # full-quality render for when movement stops.
                    return pack(run_out=False, sync_out=True, preview=True)
                elif final is None:
                    # Legacy browser assets emit every 120 ms and do not label
                    # pointer-up. A longer debounce collapses the stream into
                    # one render after movement stops.
                    mark_output_dirty(delay_override=0.8)
                # Live events synchronize the transform without starting SDXL.
                return tuple(gr.update() for _ in pack_out)

            if atype == "history_push":
                # Sent on pointer-down, before a drag mutates the pose. The
                # gesture's live transforms have not been applied yet, so this
                # is the only moment the pre-drag state still exists.
                session.push_history()
                # A click that also changes selection carries it here rather
                # than as a second action, because the bridge holds one action
                # at a time and the later write would win.
                sid = data.get("select")
                if sid:
                    session.select_layer_id(str(sid))
                    return pack(run_out=False)
                return tuple(gr.update() for _ in pack_out)

            if atype == "undo":
                session.undo()
                return pack(run_out=True)

            if atype == "redo":
                session.redo()
                return pack(run_out=True)

            if atype == "select":
                sid = data.get("id")
                if sid == "__bg__":
                    session.doc.selected_id = "__bg__"
                elif sid:
                    session.select_layer_id(str(sid))
                else:
                    session.doc.selected_id = None
                return pack(run_out=False)

            if atype == "delete":
                lid = data.get("id")
                if lid and lid != "__bg__":
                    session.delete_layer(str(lid))
                return pack(run_out=True)

            if atype == "reorder":
                session.reorder_layers([str(i) for i in (data.get("ids") or [])])
                return pack(run_out=True)

            return pack(run_out=False)

        def on_add_layer(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            session.add_empty_object()
            return pack(run_out=False)

        def on_improve(notes, edit_mode):
            """Look at the current image and propose a better prompt."""
            result = session.improve(user_notes=str(notes or ""), edit_mode=bool(edit_mode))
            if result.ok:
                return result.critique, result.improved_prompt, session.status
            # A failed parse still usually carries a readable critique; show it
            # rather than discarding work the model already did.
            note = result.error or "Improve failed."
            return (result.critique or note), gr.update(), note

        def on_improve_apply(edited_prompt):
            message = session.apply_improved(str(edited_prompt or ""))
            selected = session.doc.selected()
            layer_prompt = (
                selected.prompt if selected is not None else gr.update()
            )
            built = (
                session.output_settings.custom_prompt
                if session.output_settings.prompt_locked
                else gr.update()
            )
            return layer_prompt, built, message

        def on_generate(prompt, iso, seed, cutout, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if not prompt or not str(prompt).strip():
                session.status = "Type a prompt for the selected layer first."
                return pack(run_out=False)
            try:
                session.generate_selected(
                    str(prompt), str(iso or ""), seed=int(seed), isolate=bool(cutout)
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Generation failed")
                session.status = f"Generation failed: {exc}"
                return pack(run_out=False)
            return pack(run_out=False, sync_out=True)

        def on_import(image, cutout, prompt, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if image is None:
                session.status = "Choose an image to import."
                return pack(run_out=False)
            mode = {"rembg": "rembg", "SAM2": "sam2", "none": "none"}.get(
                str(cutout or "rembg"), "rembg"
            )
            try:
                session.import_into_selected(image, cutout=mode, prompt=str(prompt or ""))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Import failed")
                session.status = f"Import failed: {exc}"
                return pack(run_out=False)
            return pack(run_out=False, sync_out=True)

        def on_prompt_lock(locked, current_text):
            s = session.output_settings
            # Built-prompt lock only applies when LLM rewrite is off.
            if s.use_llm_rewrite:
                return gr.update(value=False), gr.update(), session.status
            s.prompt_locked = bool(locked)
            if locked:
                seed = str(current_text or session.last_output_prompt or "").strip()
                if not seed:
                    seed = session.build_prompt()
                s.custom_prompt = seed
                session.last_output_prompt = seed
                session.last_concat_prompt = seed
                mark_output_dirty()
                return gr.update(value=True), gr.update(value=seed, interactive=True), session.status
            s.custom_prompt = ""
            rebuilt = session.build_prompt()
            mark_output_dirty()
            return gr.update(value=False), gr.update(value=rebuilt, interactive=False), session.status

        def on_prompt_edit_built(text):
            s = session.output_settings
            if not s.prompt_locked or s.use_llm_rewrite:
                return session.status
            s.custom_prompt = str(text or "")
            session.last_output_prompt = s.custom_prompt
            session.last_concat_prompt = s.custom_prompt
            mark_output_dirty()
            return session.status

        def on_llm_toggle(enabled, den, steps, cfg_v, eta_v, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, enabled, neg, oseed)
            s = session.output_settings
            # Clear any prior lock so a fresh rewrite (or concat) can run.
            s.prompt_locked = False
            s.custom_prompt = ""
            mark_output_dirty()
            llm_on = bool(enabled)
            return (
                gr.update(visible=not llm_on, value=False),  # built edit lock
                gr.update(interactive=False),  # built prompt
                gr.update(visible=llm_on, value=False),  # llm edit lock
                gr.update(visible=llm_on, interactive=False, value=""),  # llm prompt
                session.status,
            )

        def on_llm_prompt_lock(locked, current_text):
            s = session.output_settings
            if not s.use_llm_rewrite:
                return gr.update(value=False), gr.update(visible=False), session.status
            s.prompt_locked = bool(locked)
            if locked:
                seed = str(current_text or session.last_output_prompt or "").strip()
                if not seed:
                    # Force a rewrite once to populate the box.
                    seed = session.build_prompt()
                s.custom_prompt = seed
                session.last_output_prompt = seed
                mark_output_dirty()
                return (
                    gr.update(value=True),
                    gr.update(value=seed, interactive=True, visible=True),
                    session.status,
                )
            s.custom_prompt = ""
            mark_output_dirty()
            return (
                gr.update(value=False),
                gr.update(interactive=False, visible=True),
                session.status,
            )

        def on_llm_prompt_edit(text):
            s = session.output_settings
            if not s.prompt_locked or not s.use_llm_rewrite:
                return session.status
            s.custom_prompt = str(text or "")
            session.last_output_prompt = s.custom_prompt
            mark_output_dirty()
            return session.status

        def on_reisolate(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            obj = session.doc.selected()
            if obj is None or obj.raw_image is None:
                session.status = "Select an object with a raw image."
                return pack(run_out=False)
            session.reisolate_selected(prefer="rembg")
            return pack(run_out=False, sync_out=True)

        def on_raw_click(evt: gr.SelectData, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            obj = session.doc.selected()
            if obj is None or obj.raw_image is None:
                session.status = "Select an object with a raw image."
                return pack(run_out=False)
            try:
                x, y = float(evt.index[0]), float(evt.index[1])
            except Exception:  # noqa: BLE001
                session.status = "Click inside the raw image."
                return pack(run_out=False)
            session.reisolate_selected(click_xy=(x, y), prefer="sam2")
            return pack(run_out=False, sync_out=True)

        def on_delete(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id in (None, "__bg__"):
                session.status = "Select an object layer to delete."
                return pack(run_out=False)
            session.delete_selected()
            return pack(run_out=True)

        def on_duplicate(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id in (None, "__bg__"):
                session.status = "Select an object layer to duplicate."
                return pack(run_out=False)
            session.duplicate_selected()
            return pack(run_out=True)

        def on_mute_prompt(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id in (None, "__bg__"):
                session.status = "Select an object layer to mute its prompt."
                return pack(run_out=False)
            session.toggle_selected_prompt()
            # Prompt composition changed — rebuild OUTPUT without changing WORK.
            return pack(run_out=False, sync_out=True)

        def on_reset_workspace():
            with out_lock:
                out_state["dirty"] = False
                out_state["token"] += 1
            session.reset_workspace()
            s = session.output_settings
            packed = pack(run_out=False)
            return (
                *packed,
                s.denoise,
                s.steps,
                s.cfg,
                s.eta,
                bool(s.use_llm_rewrite),
                s.negative_prompt,
                s.seed,
                NO_STYLE,
                False,
                gr.update(value="", interactive=False),
                False,
                gr.update(value="", visible=False, interactive=False),
            )

        def on_reset_xform(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id in (None, "__bg__"):
                session.status = "Select an object layer to reset."
                return pack(run_out=False)
            session.reset_selected_transform()
            return pack(run_out=True)

        def on_roll_seed():
            seed = session.roll_output_seed()
            mark_output_dirty()
            return seed, session.status

        def on_opacity_live(opacity, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            obj = session.doc.selected()
            if obj is None:
                return pack(run_out=False)
            session.update_transform_by_id(obj.id, opacity=float(opacity))
            return pack(run_out=False, sync_out=True)

        def on_prompt_edit(prompt):
            """Persist prompt edits to the selected layer without regenerating."""
            sid = session.doc.selected_id
            if sid == "__bg__":
                session.doc.background_prompt = str(prompt or "")
            else:
                obj = session.doc.selected()
                if obj is not None:
                    obj.prompt = str(prompt or "")
            return render_layers_html(session)

        def on_style(name):
            s = session.apply_style_preset(None if name == NO_STYLE else name)
            return s.cfg, s.denoise, s.eta, s.negative_prompt, session.status

        def on_params_now(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.background is not None or session.doc.objects:
                return pack(run_out=False, sync_out=True)
            return pack(run_out=False)

        def on_size(preset, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            w, h = ASPECT_PRESETS.get(preset, (1024, 1024))
            session.set_canvas_size(w, h)
            session.status = f"Canvas set to {w}×{h}."
            return pack(run_out=True)

        def on_base(preset_name, custom):
            ref = str(custom or "").strip() or BASE_MODEL_PRESETS.get(
                preset_name, BASE_MODEL_PRESETS[base_names[0]]
            )
            # Load in a background thread so the UI (and the poll timer) stays
            # responsive while big checkpoints download. Status/output update
            # through the timer when the switch completes.
            threading.Thread(
                target=session.switch_base_model, args=(ref,), daemon=True
            ).start()
            session.status = f"Loading SDXL base in background: {ref}…"
            return session.status

        def on_performance(profile_name):
            selected = normalize_profile(profile_name)
            threading.Thread(
                target=session.switch_compute_profile,
                args=(selected,),
                daemon=True,
            ).start()
            session.status = (
                f"Switching performance profile to {profile_label(selected)} "
                "(models will reload)…"
            )
            return session.status

        def on_performance_button(profile_name):
            message = on_performance(profile_name)
            selected = normalize_profile(profile_name)
            variants = tuple(
                gr.update(variant="primary" if mode == selected else "secondary")
                for mode in ("bf16", "mxfp8", "nvfp4")
            )
            return message, *variants

        def on_concat(order_label):
            session.output_settings.concat_order = CONCAT_ORDERS.get(order_label, "pcs")
            mark_output_dirty()
            return session.status

        def on_manual_toggle(enabled):
            session.output_settings.manual_style = bool(enabled)
            mark_output_dirty()
            return (
                gr.update(visible=bool(enabled)),
                gr.update(visible=bool(enabled)),
                session.status,
            )

        def on_manual_text(prefix, suffix):
            session.output_settings.manual_prefix = str(prefix or "")
            session.output_settings.manual_suffix = str(suffix or "")
            if session.output_settings.manual_style:
                mark_output_dirty()
            return session.status

        def on_lora(path, scale):
            if not path:
                return "Need a LoRA path."
            try:
                return session.sdxl.load_style_lora(str(path).strip(), scale=float(scale))
            except Exception as exc:  # noqa: BLE001
                return f"LoRA failed: {exc}"

        def on_emb(path):
            if not path:
                return "Need a TI path."
            try:
                return session.sdxl.load_textual_inversion(str(path).strip())
            except Exception as exc:  # noqa: BLE001
                return f"TI failed: {exc}"

        def on_final_refine(strength, steps):
            try:
                session.refine_final(
                    steps=int(steps),
                    denoise=float(strength),
                )
                return pack(run_out=False)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Final refine failed")
                session.status = f"Final refine failed: {exc}"
                return pack(run_out=False)

        def on_export(model, color, input_noise, latent_noise, seed):
            # Cancel any sleeping OUTPUT debounce while SeedVR2 owns the GPU.
            with out_lock:
                out_state["dirty"] = False
                out_state["token"] += 1
            try:
                _img, path = session.export_final(
                    seedvr2_options={
                        "model": str(model),
                        "color_correction": str(color),
                        "input_noise_scale": float(input_noise),
                        "latent_noise_scale": float(latent_noise),
                        "seed": int(seed),
                    }
                )
                return str(path), str(path), session.status
            except Exception as exc:  # noqa: BLE001
                logger.exception("Export failed")
                return None, "", f"Export failed: {exc}"

        def poll_output(last_rev):
            """Refresh OUTPUT + scene when the composition changed (any tab)."""
            if session.last_output is None or session.output_rev == last_rev:
                return (
                    gr.skip(), session.status, gr.skip(), last_rev,
                    gr.skip(), gr.skip(), gr.skip(),
                )
            s = session.output_settings
            llm_on = bool(s.use_llm_rewrite)
            concat_val = session.last_concat_prompt or ""
            if not llm_on:
                if s.prompt_locked:
                    concat_val = s.custom_prompt or session.last_output_prompt or concat_val
                else:
                    concat_val = session.last_output_prompt or concat_val
            if llm_on and s.prompt_locked:
                llm_val = s.custom_prompt or session.last_output_prompt or ""
            elif llm_on:
                llm_val = session.last_output_prompt or ""
            else:
                llm_val = ""
            return (
                _output_jpeg_path(session.last_output),
                session.status,
                concat_val,
                session.output_rev,
                render_work_html(_scene_dict(session)),
                render_layers_html(session),
                gr.update(value=llm_val, visible=llm_on),
            )

        load_btn.click(on_load, outputs=[status]).then(
            render_vram_html, outputs=[vram_html], show_progress="hidden"
        )
        free_btn.click(on_free, outputs=[status]).then(
            render_vram_html, outputs=[vram_html], show_progress="hidden"
        )
        action_out.change(
            on_action, inputs=[action_out, *settings_in], outputs=pack_out, show_progress="hidden"
        )
        add_obj_btn.click(on_add_layer, inputs=settings_in, outputs=pack_out, show_progress="hidden")
        reset_workspace_btn.click(
            on_reset_workspace,
            outputs=[
                *pack_out,
                denoise,
                out_steps,
                cfg,
                eta,
                use_llm,
                neg_prompt,
                out_seed,
                style_dd,
                prompt_lock,
                prompt_view,
                llm_prompt_lock,
                llm_prompt_view,
            ],
            show_progress="hidden",
        )
        mute_prompt_btn.click(
            on_mute_prompt, inputs=settings_in, outputs=pack_out, show_progress="hidden"
        )
        duplicate_btn.click(
            on_duplicate, inputs=settings_in, outputs=pack_out, show_progress="hidden"
        )
        gen_btn.click(
            on_generate,
            inputs=[insp_prompt, insp_iso, gen_seed, cutout_chk, *settings_in],
            outputs=pack_out,
        )
        insp_prompt.submit(
            on_generate,
            inputs=[insp_prompt, insp_iso, gen_seed, cutout_chk, *settings_in],
            outputs=pack_out,
        )
        insp_prompt.blur(
            on_prompt_edit, inputs=[insp_prompt], outputs=[layers_html], show_progress="hidden"
        )
        import_btn.click(
            on_import,
            inputs=[import_img, import_cutout, insp_prompt, *settings_in],
            outputs=pack_out,
        )
        improve_btn.click(
            on_improve,
            inputs=[improve_notes, improve_edit_chk],
            outputs=[improve_critique, improve_prompt_view, status],
        )
        improve_apply_btn.click(
            on_improve_apply,
            inputs=[improve_prompt_view],
            outputs=[insp_prompt, prompt_view, status],
        )
        prompt_lock.input(
            on_prompt_lock,
            inputs=[prompt_lock, prompt_view],
            outputs=[prompt_lock, prompt_view, status],
            show_progress="hidden",
        )
        prompt_view.blur(
            on_prompt_edit_built, inputs=[prompt_view], outputs=[status], show_progress="hidden"
        )
        use_llm.input(
            on_llm_toggle,
            inputs=[use_llm, denoise, out_steps, cfg, eta, neg_prompt, out_seed],
            outputs=[prompt_lock, prompt_view, llm_prompt_lock, llm_prompt_view, status],
            show_progress="hidden",
        )
        llm_prompt_lock.input(
            on_llm_prompt_lock,
            inputs=[llm_prompt_lock, llm_prompt_view],
            outputs=[llm_prompt_lock, llm_prompt_view, status],
            show_progress="hidden",
        )
        llm_prompt_view.blur(
            on_llm_prompt_edit, inputs=[llm_prompt_view], outputs=[status], show_progress="hidden"
        )
        reisolate_btn.click(on_reisolate, inputs=settings_in, outputs=pack_out)
        reset_xform_btn.click(
            on_reset_xform, inputs=settings_in, outputs=pack_out, show_progress="hidden"
        )
        roll_seed_btn.click(
            on_roll_seed, outputs=[out_seed, status], show_progress="hidden"
        )
        raw_view.select(on_raw_click, inputs=settings_in, outputs=pack_out)
        delete_btn.click(on_delete, inputs=settings_in, outputs=pack_out, show_progress="hidden")
        insp_opacity.release(
            on_opacity_live, inputs=[insp_opacity, *settings_in], outputs=pack_out,
            show_progress="hidden",
        )
        style_dd.change(
            on_style, inputs=[style_dd], outputs=[cfg, denoise, eta, neg_prompt, status],
            show_progress="hidden",
        ).then(
            on_params_now, inputs=settings_in, outputs=pack_out, show_progress="hidden",
        )
        for comp in (denoise, out_steps):
            comp.release(
                on_params_now, inputs=settings_in, outputs=pack_out, show_progress="hidden"
            )
        for comp in (cfg, eta, out_seed):
            comp.blur(
                on_params_now, inputs=settings_in, outputs=pack_out, show_progress="hidden"
            )
        neg_prompt.blur(
            on_params_now, inputs=settings_in, outputs=pack_out, show_progress="hidden"
        )
        # use_llm has its own on_llm_toggle handler above
        apply_size_btn.click(on_size, inputs=[aspect, *settings_in], outputs=pack_out)
        concat_dd.change(on_concat, inputs=[concat_dd], outputs=[status], show_progress="hidden")
        manual_chk.input(
            on_manual_toggle, inputs=[manual_chk],
            outputs=[manual_prefix, manual_suffix, status], show_progress="hidden",
        )
        for comp in (manual_prefix, manual_suffix):
            comp.blur(
                on_manual_text, inputs=[manual_prefix, manual_suffix],
                outputs=[status], show_progress="hidden",
            )
        base_btn.click(on_base, inputs=[base_dd, base_custom], outputs=[status])
        performance_dd.change(
            on_performance,
            inputs=[performance_dd],
            outputs=[status],
            show_progress="hidden",
        )
        for button, mode in (
            (bf16_btn, "bf16"),
            (mxfp8_btn, "mxfp8"),
            (nvfp4_btn, "nvfp4"),
        ):
            button.click(
                partial(on_performance_button, mode),
                outputs=[status, bf16_btn, mxfp8_btn, nvfp4_btn],
                show_progress="hidden",
            )
        load_lora_btn.click(on_lora, inputs=[lora_path, lora_scale], outputs=[status])
        load_emb_btn.click(on_emb, inputs=[emb_path], outputs=[status])
        update_output_btn.click(
            on_params_now, inputs=settings_in, outputs=pack_out, show_progress="full"
        )
        refine_output_btn.click(
            on_final_refine,
            inputs=[final_refine_strength, final_refine_steps],
            outputs=pack_out,
            show_progress="full",
        )
        export_btn.click(
            on_export,
            inputs=[
                seedvr_model,
                seedvr_color,
                seedvr_input_noise,
                seedvr_latent_noise,
                seedvr_seed,
            ],
            outputs=[export_image, export_path, status],
        )

        timer = gr.Timer(0.2)
        timer.tick(
            poll_output,
            inputs=[rev_state],
            outputs=[
                output_image, status, prompt_view, rev_state,
                work_html, layers_html, llm_prompt_view,
            ],
            show_progress="hidden",
        )
        vram_timer = gr.Timer(3.0)

        def runtime_status():
            return render_vram_html(), session.optimization_status()

        vram_timer.tick(
            runtime_status,
            outputs=[vram_html, performance_status],
            show_progress="hidden",
        )

        def _init():
            return render_work_html(_scene_dict(session)), render_layers_html(session)

        demo.load(_init, outputs=[work_html, layers_html])

    return demo


def _launch_kwargs(demo: gr.Blocks, config: AppConfig) -> dict:
    kwargs: dict[str, Any] = {
        "server_name": str(config.get("server", "host", default="0.0.0.0")),
        "server_port": int(config.get("server", "port", default=7860)),
        "share": bool(config.get("server", "share", default=False)),
        "show_error": bool(config.get("server", "show_error", default=True)),
    }
    theme = getattr(demo, "_xwave_theme", None)
    css = getattr(demo, "_xwave_css", None)
    head = getattr(demo, "_xwave_head", None)
    if theme is not None:
        kwargs["theme"] = theme
    if css:
        kwargs["css"] = css
    if head:
        kwargs["head"] = head
    return kwargs


def launch_app(config: AppConfig | None = None) -> None:
    config = config or AppConfig.load()
    demo = build_app(config)
    demo.queue(default_concurrency_limit=1).launch(**_launch_kwargs(demo, config))
