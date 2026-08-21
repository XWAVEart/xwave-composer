"""Gradio UI for the Edit tab — SAM2 segmenting and registered layers."""

from __future__ import annotations

import base64
import hashlib
import html as html_lib
import io
import json
import logging
import threading
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, gpu_summary
from xwave_composer.library.render import pack_library_views
from xwave_composer.library.store import ImageLibrary
from xwave_composer.models.upscaler import ImageUpscaler
from xwave_composer.glitch.registry import effects_schema
from xwave_composer.pipeline.edit import EditSession

logger = logging.getLogger(__name__)

MODE_CHOICES = ("Off", "Include (+)", "Exclude (−)")
_MODE_KEYS = {
    "Off": "off",
    "Include (+)": "include",
    "Exclude (−)": "exclude",
    "Exclude (-)": "exclude",
}


def mode_key(label: str | None) -> str:
    raw = str(label or "").strip()
    return _MODE_KEYS.get(raw, raw.lower() or "off")


def _thumb_data_url(img: Image.Image | None, max_side: int = 96) -> str:
    if img is None:
        return ""
    im = img.convert("RGBA")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h, 1))
    if scale < 1.0:
        im = im.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_edit_layers_html(session: EditSession) -> str:
    cards: list[str] = []
    layers = list(reversed(session.doc.layers))  # top-most first
    for layer in layers:
        sel = " is-selected" if session.doc.selected_id == layer.id else ""
        hidden = " is-hidden" if not layer.visible else ""
        thumb_url = _thumb_data_url(layer.image)
        thumb = (
            f' style="background-image:url(&quot;{html_lib.escape(thumb_url)}&quot;)"'
            if thumb_url
            else ""
        )
        tags: list[str] = []
        if not layer.visible:
            tags.append("HIDDEN")
        if layer.is_base:
            tags.append("BASE")
        label_text = layer.name
        if tags:
            label_text = f"{' · '.join(tags)} · {layer.name}"
        label = html_lib.escape(label_text[:72])
        lid = html_lib.escape(layer.id)
        extra = " xwave-card-bg" if layer.is_base else ""
        drag = "" if layer.is_base else ' draggable="true"'
        delete = ""
        if not layer.is_base:
            delete = (
                f'<button type="button" class="xwave-del" data-delete-id="{lid}" '
                f'title="Delete this layer" aria-label="Delete this layer">×</button>'
            )
        tag = '<span class="xwave-bg-tag">BASE</span>' if layer.is_base else ""
        cards.append(
            f'<div class="xwave-card{extra}{sel}{hidden}" data-layer-id="{lid}"{drag}>'
            f'<div class="xwave-thumb"{thumb}></div>'
            f'<div class="xwave-card-text">{label}</div>'
            f"{tag}{delete}</div>"
        )
    if not cards:
        cards.append(
            '<p class="xwave-text-dim">Load an image to create a Base layer.</p>'
        )
    meta = [
        {"id": layer.id, "name": layer.name, "base": layer.is_base}
        for layer in session.doc.layers
    ]
    meta_b64 = base64.b64encode(json.dumps(meta, separators=(",", ":")).encode()).decode()
    return (
        f'<div id="xwave-edit-layers-root" data-layers="{meta_b64}">'
        f'<div id="xwave-edit-layer-stack">{"".join(cards)}</div></div>'
    )


def _export_path(config: AppConfig, image: Image.Image) -> Path:
    out_dir = config.path("export", "output_dir", default="exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(image.tobytes()).hexdigest()[:10]
    path = out_dir / f"edit_{image.width}x{image.height}_{digest}.png"
    image.save(path, format="PNG", optimize=True)
    return path


def render_effects_html() -> str:
    schema = effects_schema()
    b64 = base64.b64encode(json.dumps(schema, separators=(",", ":")).encode()).decode()
    return (
        f'<div id="xwave-fx-root" class="xwave-fx-root" data-schema="{b64}">'
        f'<div class="xwave-fx-bar">'
        f'<select id="xwave-fx-group" title="Group" aria-label="Effect group"></select>'
        f'<select id="xwave-fx-effect" title="Effect" aria-label="Effect"></select>'
        f'<select id="xwave-fx-secondary" class="is-hidden" title="Secondary source" '
        f'aria-label="Secondary source"></select>'
        f'<label id="xwave-fx-warp-wrap" class="xwave-fx-warp is-hidden">'
        f'<input type="checkbox" id="xwave-fx-warp"> Warp alpha</label>'
        f'<button type="button" id="xwave-fx-apply">Apply</button>'
        f"</div>"
        f'<div id="xwave-fx-params" class="xwave-fx-params"></div>'
        f"</div>"
    )


def inspector_values(session: EditSession) -> tuple[str, bool, float]:
    layer = session.selected()
    if layer is None:
        return "", True, 1.0
    return layer.name, bool(layer.visible), float(layer.opacity)


def _edit_cache_dir(config: AppConfig) -> Path:
    d = config.path("paths", "workspace_dir", default="workspace") / "edit_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def _file_url(path: Path) -> str:
    return f"/gradio_api/file={path.resolve()}"


def _flatten_url(config: AppConfig, image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha1()
    digest.update(str(rgb.size).encode())
    digest.update(rgb.tobytes())
    path = _edit_cache_dir(config) / f"edit_flat_{digest.hexdigest()[:16]}.jpg"
    if not path.exists():
        rgb.save(path, format="JPEG", quality=90, optimize=True)
    return _file_url(path)


def _overlay_url(config: AppConfig, overlay: Image.Image | None) -> str:
    if overlay is None:
        return ""
    rgba = overlay.convert("RGBA")
    digest = hashlib.sha1()
    digest.update(str(rgba.size).encode())
    digest.update(rgba.tobytes())
    path = _edit_cache_dir(config) / f"edit_glow_{digest.hexdigest()[:16]}.png"
    if not path.exists():
        rgba.save(path, format="PNG", optimize=True)
    return _file_url(path)


def render_edit_scene_html(session: EditSession, config: AppConfig) -> str:
    if not session.has_image or session.doc.source is None:
        scene = {
            "width": 1024,
            "height": 1024,
            "preview_url": "",
            "overlay_url": "",
            "mode": session.mode,
            "empty": True,
        }
    else:
        src = session.doc.source
        scene = {
            "width": src.width,
            "height": src.height,
            "preview_url": _flatten_url(config, session.flatten()),
            "overlay_url": _overlay_url(config, session.glow_overlay()),
            "mode": session.mode,
            "empty": False,
        }
    b64 = base64.b64encode(json.dumps(scene, separators=(",", ":")).encode()).decode()
    return (
        f'<div id="xwave-edit-root" class="xwave-edit-root" data-scene="{b64}">'
        f'<div id="xwave-edit-wrap" class="xwave-edit-wrap">'
        f'<canvas id="xwave-edit-canvas" class="xwave-edit-canvas"></canvas>'
        f'<p class="xwave-edit-hint">Hover an object — it glows. Click to lift it onto a layer.</p>'
        f"</div></div>"
    )


def pack_edit(session: EditSession, config: AppConfig) -> tuple:
    name, visible, opacity = inspector_values(session)
    return (
        render_edit_scene_html(session, config),
        render_edit_layers_html(session),
        session.status,
        name,
        visible,
        opacity,
        gr.update(),
    )


def build_edit_tab(
    *,
    session: EditSession,
    config: AppConfig,
    infer_lock: threading.Lock,
    upscaler: ImageUpscaler | None,
    composer: Any,
    library: ImageLibrary | None,
) -> dict[str, Any]:
    """Build Edit tab widgets inside the current Tab context."""
    empty = pack_edit(session, config)
    picker_seed = (
        pack_library_views(library)[6:8]
        if library is not None
        else ("", "No images yet")
    )
    edit_offset = gr.State(0)

    with gr.Row(elem_classes=["xwave-row", "xwave-edit-top"], equal_height=False):
        with gr.Column(scale=1, min_width=260, elem_classes=["xwave-col", "xwave-edit-side"]):
            edit_status = gr.Textbox(
                value=empty[2],
                label="Status",
                interactive=False,
                lines=2,
                elem_classes=["xwave-status"],
            )
            with gr.Accordion("Import", open=True):
                import_file = gr.Image(
                    type="pil",
                    label="Drop or upload",
                    height=120,
                    buttons=[],
                    elem_classes=["xwave-import"],
                )
                import_file_btn = gr.Button("Load file", size="sm")
                edit_lib_html = gr.HTML(
                    value=picker_seed[0],
                    elem_classes=["xwave-lib-picker"],
                )
                with gr.Row(elem_classes=["xwave-lib-pager"]):
                    edit_lib_prev = gr.Button("Prev", size="sm", scale=0, min_width=64)
                    edit_lib_pager = gr.Textbox(
                        value=picker_seed[1],
                        show_label=False,
                        interactive=False,
                        container=False,
                        scale=1,
                    )
                    edit_lib_next = gr.Button("Next", size="sm", scale=0, min_width=64)
                gr.Markdown(
                    '<p class="xwave-text-dim">Click a library thumb to open it in Edit.</p>'
                )

            gr.Markdown('<p class="xwave-section-head">Segment</p>')
            seg_mode = gr.Radio(
                choices=list(MODE_CHOICES),
                value="Include (+)",
                label="Mode",
                elem_classes=["xwave-compact-radio"],
            )
            gr.Markdown(
                '<p class="xwave-text-dim">Hover to glow, click to lift. Exclude trims.</p>'
            )
            undo_pt_btn = gr.Button("Undo", size="sm")

            gr.Markdown('<p class="xwave-section-head">Save</p>')
            lib_save_name = gr.Textbox(
                label="Library name",
                placeholder="Optional name",
                lines=1,
            )
            save_btn = gr.Button(
                "Save to library",
                size="sm",
                elem_classes=["xwave-lib-save-btn"],
            )
            export_btn = gr.Button("Export PNG", size="sm")
            upscale_btn = gr.Button("Upscale 2×", size="sm")
            export_file = gr.File(label="Download", interactive=False)

        with gr.Column(scale=3, min_width=420, elem_classes=["xwave-col", "xwave-edit-viewport"]):
            preview = gr.HTML(
                value=empty[0],
                padding=False,
                elem_classes=["xwave-edit-html"],
            )
            fx_html = gr.HTML(
                value=render_effects_html(),
                padding=False,
                elem_classes=["xwave-fx-html"],
            )
            edit_action = gr.Textbox(
                value="",
                elem_id="xwave-edit-action",
                elem_classes=["xwave-hidden"],
                container=False,
                show_label=False,
            )

        with gr.Column(scale=1, min_width=230, elem_classes=["xwave-col", "xwave-layers-col"]):
            gr.Markdown('<p class="xwave-section-head">Layers</p>')
            layers_html = gr.HTML(value=empty[1], padding=False)
            name_box = gr.Textbox(
                value=empty[3],
                label="Name",
                lines=1,
            )
            visible_chk = gr.Checkbox(value=empty[4], label="Visible")
            opacity = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=empty[5],
                label="Opacity",
            )

    pack_out = [
        preview,
        layers_html,
        edit_status,
        name_box,
        visible_chk,
        opacity,
        export_file,
    ]

    def pack(export_path: str | None = None) -> tuple:
        preview_html, html, status, name, vis, opac, _ = pack_edit(session, config)
        file_out = export_path if export_path else gr.update()
        return preview_html, html, status, name, vis, opac, file_out

    def hover_pack() -> tuple:
        return (
            render_edit_scene_html(session, config),
            gr.skip(),
            session.status,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    def on_load_file(image):
        if image is None:
            session.status = "Choose an image to load."
            return pack()
        try:
            session.load(image, name="file")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Edit load failed")
            session.status = f"Load failed: {exc}"
        return pack()

    def on_mode(label):
        session.set_mode(mode_key(label))
        session.clear_hover()
        return pack()

    def on_undo():
        session.undo_point()
        return pack()

    def on_name(name):
        layer = session.selected()
        if layer is not None:
            session.set_name(layer.id, name)
        return pack()

    def on_visible(flag):
        layer = session.selected()
        if layer is not None:
            session.set_visible(layer.id, bool(flag))
        return pack()

    def on_opacity(value):
        layer = session.selected()
        if layer is not None:
            session.set_opacity(layer.id, float(value))
        return pack()

    def on_action(payload):
        if not payload or not str(payload).strip():
            return tuple(gr.update() for _ in pack_out)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            session.status = "Bad edit action."
            return pack()
        atype = data.get("type")
        if atype == "hover":
            try:
                x, y = float(data.get("x")), float(data.get("y"))
            except Exception:  # noqa: BLE001
                return tuple(gr.skip() for _ in pack_out)
            acquired = infer_lock.acquire(blocking=False)
            if not acquired:
                return tuple(gr.skip() for _ in pack_out)
            try:
                session.hover_at(x, y)
            finally:
                infer_lock.release()
            return hover_pack()
        if atype == "hover_end":
            session.clear_hover()
            return hover_pack()
        if atype == "cut":
            try:
                x, y = float(data.get("x")), float(data.get("y"))
            except Exception:  # noqa: BLE001
                session.status = "Click inside the image."
                return pack()
            acquired = infer_lock.acquire(blocking=False)
            if not acquired:
                session.status = "GPU is busy — wait, then click again."
                return pack()
            try:
                session.cut_at(x, y)
            finally:
                infer_lock.release()
            return pack()
        if atype == "effect":
            session.apply_effect(
                str(data.get("id") or ""),
                data.get("params") or {},
                secondary_key=str(data.get("secondary") or "rest"),
                warp_alpha=bool(data.get("warp")),
            )
            return pack()
        if atype == "select":
            session.select(str(data.get("id") or ""))
        elif atype == "delete":
            session.delete_layer(str(data.get("id") or ""))
        elif atype == "reorder":
            ids = data.get("ids") or []
            session.reorder_cuts([str(i) for i in ids])
        return pack()

    def on_export():
        if not session.has_image:
            session.status = "Load an image first."
            return pack()
        try:
            path = _export_path(config, session.flatten())
            session.status = f"Exported {path}"
            return pack(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Edit export failed")
            session.status = f"Export failed: {exc}"
            return pack()

    def on_upscale():
        if not session.has_image:
            session.status = "Load an image first."
            return pack()
        if upscaler is None:
            session.status = "Upscaler not available."
            return pack()
        acquired = infer_lock.acquire(blocking=False)
        if not acquired:
            session.status = "Cannot upscale while another model is busy."
            return pack()
        isolator = getattr(composer, "isolator", None)
        sdxl = getattr(composer, "sdxl", None)
        try:
            if isolator is not None:
                isolator.unload()
            if sdxl is not None and getattr(sdxl, "ready", False):
                sdxl.unload()
            empty_cache()
            session.status = "Upscaling with SeedVR2…"
            image = upscaler.upscale_2x(session.flatten())
            path = _export_path(config, image)
            session.status = f"Upscaled {image.width}×{image.height}: {path} · {gpu_summary()}"
            return pack(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Edit upscale failed")
            session.status = f"Upscale failed: {exc}"
            return pack()
        finally:
            try:
                if upscaler is not None:
                    upscaler.unload()
                empty_cache()
            except Exception:  # noqa: BLE001
                logger.debug("upscaler unload after edit failed", exc_info=True)
            infer_lock.release()

    import_file_btn.click(on_load_file, inputs=[import_file], outputs=pack_out)
    seg_mode.change(on_mode, inputs=[seg_mode], outputs=pack_out, show_progress="hidden")
    undo_pt_btn.click(on_undo, outputs=pack_out, show_progress="hidden")
    name_box.blur(on_name, inputs=[name_box], outputs=pack_out, show_progress="hidden")
    visible_chk.change(on_visible, inputs=[visible_chk], outputs=pack_out, show_progress="hidden")
    opacity.release(on_opacity, inputs=[opacity], outputs=pack_out, show_progress="hidden")
    edit_action.change(on_action, inputs=[edit_action], outputs=pack_out, show_progress="hidden")
    export_btn.click(on_export, outputs=pack_out)
    upscale_btn.click(on_upscale, outputs=pack_out)

    return {
        "session": session,
        "preview": preview,
        "layers": layers_html,
        "status": edit_status,
        "name": name_box,
        "visible": visible_chk,
        "opacity": opacity,
        "export_file": export_file,
        "save_btn": save_btn,
        "save_name": lib_save_name,
        "picker_html": edit_lib_html,
        "picker_pager": edit_lib_pager,
        "picker_prev": edit_lib_prev,
        "picker_next": edit_lib_next,
        "offset": edit_offset,
        "pack_out": pack_out,
        "pack": pack,
        "load_image": session.load,
    }
