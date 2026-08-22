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
from xwave_composer.pipeline.quality_modes import normalize_quality_mode
from xwave_composer.pipeline.session import ComposerSession
from xwave_composer.pipeline.edit import EditSession
from xwave_composer.style.style_manager import CONCAT_ORDERS, STYLE_FAMILIES
from xwave_composer.ui.large_canvas_tab import _scene_html, build_large_canvas_tab
from xwave_composer.ui.edit_tab import build_edit_tab
from xwave_composer.ui.settings_tab import build_settings_tab
from xwave_composer.ui.library_tab import (
    build_library_tab,
    placement_key,
    shift_page,
)
from xwave_composer.library.render import pack_library_views, selected_item_outputs
from xwave_composer.library.store import ImageLibrary
from xwave_composer.models.upscaler import (
    DEFAULT_SEEDVR2_MODEL,
    SEEDVR2_MODEL_CHOICES,
    SEEDVR2_PRESET_CHOICES,
    SEEDVR2_PRESETS,
)
from xwave_composer.canvas.compositor import (
    BLEND_MODE_LABELS,
    BLEND_TO_CANVAS,
    blend_mode_label,
    feather_alpha_inward,
    normalize_blend_mode,
)

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

# Label format: aspect · pixels. Portrait → square → landscape so 1:1 sits mid-list.
ASPECT_PRESETS = {
    "9:21 · 640×1536": (640, 1536),
    "9:16 · 768×1344": (768, 1344),
    "2:3 · 832×1216": (832, 1216),
    "3:4 · 896×1152": (896, 1152),
    "1:1 · 1024×1024": (1024, 1024),
    "4:3 · 1152×896": (1152, 896),
    "3:2 · 1216×832": (1216, 832),
    "16:9 · 1344×768": (1344, 768),
    "21:9 · 1536×640": (1536, 640),
}
DEFAULT_ASPECT = "1:1 · 1024×1024"

NO_STYLE = "— none —"
FAMILY_ALL = "All families"

# WORK-only alignment overlays (canvas JS reads #xwave-grid-state).
WORK_GRID_TYPES = [
    "Off",
    "Center",
    "Thirds",
    "Golden",
    "Quadrants",
    "Diagonals",
    "Safe margins",
]
WORK_GRID_COLORS = ["Black", "White", "Cyan", "Magenta"]
_WORK_GRID_TYPE_KEYS = {
    "off": "off",
    "center": "center",
    "thirds": "thirds",
    "golden": "golden",
    "quadrants": "quadrants",
    "diagonals": "diagonals",
    "safe margins": "safe_margins",
    "safe_margins": "safe_margins",
}


def _work_grid_state_html(
    grid_type: str | None,
    grid_color: str | None,
    *,
    user_set: bool = False,
) -> str:
    t_raw = str(grid_type or "Off").strip().lower()
    t = _WORK_GRID_TYPE_KEYS.get(t_raw, "off")
    c = str(grid_color or "Cyan").strip().lower()
    if c not in ("black", "white", "cyan", "magenta"):
        c = "cyan"
    user_attr = ' data-user="1"' if user_set else ""
    return (
        f'<div id="xwave-grid-state" data-type="{html_lib.escape(t)}" '
        f'data-color="{html_lib.escape(c)}"{user_attr} '
        f'hidden aria-hidden="true"></div>'
    )

ISO_WHITE = "isolated on plain white background, centered"
ISO_BLACK = "isolated on plain black background, centered"
# Radio display labels are square emojis; values stay White/Black for session logic.
# Isolation backdrop: stored as "White"/"Black"; UI uses emoji swatch buttons.


def _iso_backdrop_label(prompt: str | None) -> str:
    text = (prompt or "").lower()
    return "Black" if "black" in text else "White"


def _iso_prompt_for_backdrop(label: str | None) -> str:
    return ISO_BLACK if str(label or "").strip().title() == "Black" else ISO_WHITE


# ---------------------------------------------------------------------------
# Scene / render helpers
# ---------------------------------------------------------------------------

# Cache preview media as on-disk PNG/JPEG files served via Gradio's /file=
# endpoint. Scene HTML stays small (JSON + short URLs) instead of megabyte
# base64 data-URLs embedded in the DOM.
# key -> (python id(img), max_side, url)
_URL_CACHE: dict[str, tuple[int, int, str]] = {}


def _scene_cache_dir(config: AppConfig) -> Path:
    d = config.path("paths", "workspace_dir", default="workspace") / "scene_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def _prepare_preview(img: Image.Image, max_side: int) -> Image.Image:
    bands = img.getbands()
    im = img.convert("RGBA") if "A" in bands else img.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h, 1))
    if scale < 1.0:
        im = im.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.BILINEAR,
        )
    return im


def _write_scene_file(img: Image.Image, max_side: int, cache_dir: Path) -> Path:
    """Persist a resized preview; filename is content-hash for reuse."""
    im = _prepare_preview(img, max_side)
    digest = hashlib.sha1()
    digest.update(str(im.size).encode())
    digest.update(im.mode.encode())
    digest.update(im.tobytes())
    # JPEG for opaque RGB previews (bg); PNG when alpha matters (objects).
    if im.mode == "RGBA":
        path = cache_dir / f"xwave_scene_{digest.hexdigest()[:20]}.png"
        if not path.exists():
            im.save(path, format="PNG", optimize=True)
    else:
        path = cache_dir / f"xwave_scene_{digest.hexdigest()[:20]}.jpg"
        if not path.exists():
            im.save(path, format="JPEG", quality=88, optimize=True)
    return path


def _gradio_file_url(path: Path) -> str:
    """Browser URL for a local file Gradio is allowed to serve."""
    return f"/gradio_api/file={path.resolve()}"


def _media_url(
    key: str,
    img: Image.Image | None,
    max_side: int,
    cache_dir: Path,
) -> str | None:
    if img is None:
        _URL_CACHE.pop(key, None)
        return None
    cached = _URL_CACHE.get(key)
    if cached and cached[0] == id(img) and cached[1] == max_side:
        return cached[2]
    path = _write_scene_file(img, max_side, cache_dir)
    url = _gradio_file_url(path)
    _URL_CACHE[key] = (id(img), max_side, url)
    return url


def _work(session: ComposerSession) -> Image.Image:
    return session.ensure_work_composed()


def _blank(w: int = 1024, h: int = 1024) -> Image.Image:
    return Image.new("RGB", (w, h), (16, 17, 20))


def _prune_url_cache(session: ComposerSession) -> None:
    live: set[str] = set()
    for obj in session.doc.objects:
        feather = round(float(getattr(obj, "feather", 0.0)), 1)
        live.add(f"layer:{obj.id}:f{feather}")
        live.add(f"thumb:{obj.id}")
    live |= {"bg", "bg-thumb"}
    for key in list(_URL_CACHE):
        if key not in live:
            _URL_CACHE.pop(key, None)


def _scene_dict(session: ComposerSession) -> dict[str, Any]:
    _prune_url_cache(session)
    cache_dir = _scene_cache_dir(session.config)
    layers = []
    for obj in session.doc.objects:
        feather = float(getattr(obj, "feather", 0.0))
        display = obj.image
        if display is not None and feather > 0.05:
            display = feather_alpha_inward(display, feather)
        blend = normalize_blend_mode(getattr(obj, "blend_mode", "normal"))
        # Keep key name ``data_url`` for the canvas JS contract; value is now
        # a Gradio /file= URL (or null), not a base64 data URI.
        media = _media_url(
            f"layer:{obj.id}:f{round(feather, 1)}", display, 640, cache_dir
        )
        layers.append(
            {
                "id": obj.id,
                "prompt": obj.prompt,
                "x": float(obj.transform.x),
                "y": float(obj.transform.y),
                "scale_x": float(obj.transform.scale_x),
                "scale_y": float(obj.transform.scale_y),
                "rotation": float(obj.transform.rotation),
                "flip_x": bool(getattr(obj.transform, "flip_x", False)),
                "flip_y": bool(getattr(obj.transform, "flip_y", False)),
                "opacity": float(obj.transform.opacity),
                "visible": bool(obj.transform.visible),
                "blend_mode": blend,
                "blend_canvas": BLEND_TO_CANVAS.get(blend, "source-over"),
                "w": int(obj.image.width) if obj.image else 256,
                "h": int(obj.image.height) if obj.image else 256,
                "data_url": media,
            }
        )
    return {
        "width": session.doc.width,
        "height": session.doc.height,
        "selected_id": session.doc.selected_id,
        "layers": layers,
        "bg_data_url": _media_url("bg", session.doc.background, 1024, cache_dir),
        "bg_prompt": session.doc.background_prompt,
        "bg_scale": float(session.doc.bg_scale),
        "bg_rotation": float(session.doc.bg_rotation),
        "bg_offset_x": float(session.doc.bg_offset_x),
        "bg_offset_y": float(session.doc.bg_offset_y),
        "bg_flip_x": bool(session.doc.bg_flip_x),
        "bg_flip_y": bool(session.doc.bg_flip_y),
        "rev": int(time.time() * 1000) % 10_000_000,
    }


def render_work_html(scene: dict[str, Any]) -> str:
    b64 = base64.b64encode(json.dumps(scene, separators=(",", ":")).encode()).decode("ascii")
    return f'<div id="xwave-work-root" data-scene="{b64}"><canvas id="xwave-work-canvas" width="512" height="512"></canvas></div>'


def render_layers_html(session: ComposerSession) -> str:
    """Layer stack cards. Cards show the layer prompt (no titles)."""
    cache_dir = _scene_cache_dir(session.config)
    cards: list[str] = []
    for obj in reversed(session.doc.objects):  # top-most first
        sel = " is-selected" if session.doc.selected_id == obj.id else ""
        muted = " is-muted" if not obj.prompt_enabled else ""
        hidden = " is-hidden" if not obj.transform.visible else ""
        thumb_url = _media_url(f"thumb:{obj.id}", obj.image, 96, cache_dir) or ""
        thumb = (
            f' style="background-image:url(&quot;{html_lib.escape(thumb_url)}&quot;)"'
            if thumb_url
            else ""
        )
        text = (obj.prompt or "").strip() or "empty — type a prompt below"
        empty = "" if (obj.prompt or "").strip() else " is-empty"
        tags: list[str] = []
        if not obj.transform.visible:
            tags.append("HIDDEN")
        if not obj.prompt_enabled:
            tags.append("MUTED")
        label_text = f"{' · '.join(tags)} · {text}" if tags else text
        label = html_lib.escape(label_text[:72])
        lid = html_lib.escape(obj.id)
        cards.append(
            f'<div class="xwave-card{sel}{empty}{muted}{hidden}" data-layer-id="{lid}" draggable="true">'
            f'<div class="xwave-thumb"{thumb}></div>'
            f'<div class="xwave-card-text">{label}</div>'
            f'<button type="button" class="xwave-del" data-delete-id="{lid}" '
            f'title="Delete this layer" aria-label="Delete this layer">×</button></div>'
        )
    # Background card always last (bottom of stack)
    bg_url = _media_url("bg-thumb", session.doc.background, 96, cache_dir) or ""
    bg_sel = " is-selected" if session.doc.selected_id == "__bg__" else ""
    bg_text = (session.doc.background_prompt or "").strip() or "background — type a prompt below"
    bg_empty = "" if (session.doc.background_prompt or "").strip() else " is-empty"
    bg_thumb = (
        f' style="background-image:url(&quot;{html_lib.escape(bg_url)}&quot;)"'
        if bg_url
        else ""
    )
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
            "iso_backdrop": "White",
            "scale": 1.0,
            "rotation": float(session.doc.bg_rotation),
            "flip_x": False,
            "flip_y": False,
            "feather": 0.0,
            "blend_mode": "Normal",
            "raw": None,
            "prompt_enabled": True,
            "visible": True,
            "bg_scale": float(session.doc.bg_scale),
            "bg_rotation": float(session.doc.bg_rotation),
            "bg_offset_x": float(session.doc.bg_offset_x),
            "bg_offset_y": float(session.doc.bg_offset_y),
            "bg_flip_x": bool(session.doc.bg_flip_x),
            "bg_flip_y": bool(session.doc.bg_flip_y),
        }
    obj = session.doc.selected()
    if obj is None:
        return {
            "kind": "none",
            "prompt": "",
            "opacity": 1.0,
            "iso_backdrop": "White",
            "scale": 1.0,
            "rotation": 0.0,
            "flip_x": False,
            "flip_y": False,
            "feather": 0.0,
            "blend_mode": "Normal",
            "raw": None,
            "prompt_enabled": True,
            "visible": True,
            "bg_scale": 1.0,
            "bg_rotation": 0.0,
            "bg_offset_x": 0.0,
            "bg_offset_y": 0.0,
            "bg_flip_x": False,
            "bg_flip_y": False,
        }
    sx = abs(float(obj.transform.scale_x))
    sy = abs(float(obj.transform.scale_y))
    return {
        "kind": "object",
        "prompt": obj.prompt or "",
        "opacity": float(obj.transform.opacity),
        "iso_backdrop": _iso_backdrop_label(obj.isolation_prompt),
        "scale": (sx + sy) * 0.5,
        "rotation": float(obj.transform.rotation),
        "flip_x": bool(getattr(obj.transform, "flip_x", False)),
        "flip_y": bool(getattr(obj.transform, "flip_y", False)),
        "feather": float(getattr(obj, "feather", 0.0)),
        "blend_mode": blend_mode_label(getattr(obj, "blend_mode", "normal")),
        "raw": obj.raw_image,
        "prompt_enabled": obj.prompt_enabled,
        "visible": bool(obj.transform.visible),
        "bg_scale": 1.0,
        "bg_rotation": 0.0,
        "bg_offset_x": 0.0,
        "bg_offset_y": 0.0,
        "bg_flip_x": False,
        "bg_flip_y": False,
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
            font=[
                gr.themes.GoogleFont("IBM Plex Sans"),
                gr.themes.GoogleFont("Space Grotesk"),
            ],
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
    library = ImageLibrary(config.path("paths", "library_dir", default="library"))

    css = (ASSETS / "app.css").read_text(encoding="utf-8") if (ASSETS / "app.css").exists() else ""
    js_parts: list[str] = []
    for name in ("settings.js", "work_canvas.js", "tooltips.js", "large_canvas.js", "library.js", "edit.js"):
        path = ASSETS / name
        if path.exists():
            js_parts.append(path.read_text(encoding="utf-8"))
    head_js = "<script>\n" + "\n".join(js_parts) + "\n</script>" if js_parts else ""
    theme = _build_theme()

    style_family_choices = [FAMILY_ALL] + (
        session.styles.families_present() if session.styles else list(STYLE_FAMILIES)
    )
    style_names = [NO_STYLE] + (session.styles.names() if session.styles else [])
    flipbook_family_choices = (
        session.styles.families_present() if session.styles else list(STYLE_FAMILIES)
    )
    base_names = list(BASE_MODEL_PRESETS.keys())
    _cfg_base = str(
        config.get(
            "sdxl_hyper",
            "base_model_id",
            default=BASE_MODEL_PRESETS.get("Juggernaut XL v9", BASE_MODEL_PRESETS[base_names[0]]),
        )
    )
    default_base_name = next(
        (name for name, ref in BASE_MODEL_PRESETS.items() if ref == _cfg_base),
        "Juggernaut XL v9" if "Juggernaut XL v9" in BASE_MODEL_PRESETS else base_names[0],
    )

    # Debounced OUTPUT refresh driven by canvas transforms.
    # One full-quality refine after movement stops — no preview/settle swap.
    out_lock = threading.Lock()
    # Shared SDXL mutex between Compose OUTPUT and Infinite Canvas generation.
    infer_lock = threading.Lock()
    out_state: dict[str, Any] = {
        "token": 0,
        "running": False,
        "dirty": False,
        # Sticky: composition changed while SDXL was unloaded (export / free).
        "pending": False,
    }

    def run_output_now() -> Image.Image:
        try:
            # Always re-read last_work under the session lock inside
            # run_output(). Snapshotting WORK here races with later
            # transforms and can refine a stale pose.
            with infer_lock:
                return session.run_output(preview=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("OUTPUT")
            session.status = f"OUTPUT error: {exc}"
            return session.last_output or _work(session)

    def flush_pending_output() -> None:
        """Run a deferred OUTPUT refine after SDXL becomes ready again."""
        with out_lock:
            should = out_state["pending"] or out_state["dirty"]
            out_state["pending"] = False
        if should and session.core_ready:
            mark_output_dirty(delay_override=0.05)

    def mark_output_dirty(delay_override: float | None = None) -> None:
        with out_lock:
            out_state["dirty"] = True
            if not session.compose_active:
                # Infinite Canvas owns the GPU — defer OUTPUT until Compose mode.
                out_state["pending"] = True
                return
            if not session.core_ready:
                # Never trigger a surprise model download from a drag — remember
                # the request and flush after Load / post-export SDXL restore.
                out_state["pending"] = True
                return
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
                # a SeedVR2 export while this worker was waiting, or the user
                # switched to Infinite Canvas.
                if not session.compose_active or not session.core_ready:
                    with out_lock:
                        out_state["dirty"] = True
                        out_state["pending"] = True
                    return
                run_output_now()
            finally:
                with out_lock:
                    out_state["running"] = False
                    again = out_state["dirty"]
                if again and session.core_ready:
                    mark_output_dirty(delay_override=0.05)

        threading.Thread(target=_worker, daemon=True).start()

    def pack_work_then_output() -> tuple:
        """Push WORK/layers immediately; refine OUTPUT asynchronously via poll.

        Never block the WORK HTML return on SDXL. Kick dirty OUTPUT without
        writing the image slot here so a late JPEG cannot clobber a fresher
        poll, and so WORK always paints before OUTPUT.
        """
        mark_output_dirty(delay_override=0.05)
        return pack(run_out=False, sync_out=False, include_output=False)

    def pack_output_only(*, include_layers: bool = False) -> tuple:
        """Refine OUTPUT without rewriting WORK (prompt/params-only edits)."""
        mark_output_dirty(delay_override=0.05)
        return pack(
            run_out=False,
            sync_out=False,
            include_work=False,
            include_layers=include_layers,
            include_output=False,
        )

    def pack(
        run_out: bool = True,
        sync_out: bool = False,
        *,
        include_work: bool = True,
        include_layers: bool = True,
        include_output: bool | None = None,
    ) -> tuple:
        """-> work_html, layers_html, output, status,
        insp_prompt, iso_backdrop, iso_white_btn, iso_black_btn,
        insp_opacity, insp_feather, insp_blend,
        raw_view, prompt_view,
        llm_prompt_view, mute_prompt_btn, hide_layer_btn, cutout_chk, cutout_mode,
        obj_scale, obj_rotation, obj_flip_x, obj_flip_y,
        bg_scale, bg_rotation, bg_offset_x, bg_offset_y, bg_flip_x, bg_flip_y

        Set include_work/include_layers False for lightweight responses that
        must not clobber the optimistic WORK canvas / layer stack.

        OUTPUT ordering rule: when OUTPUT is refined asynchronously
        (``run_out`` and not ``sync_out``), the image slot is skipped so
        poll_output delivers it after WORK has already updated. Pass
        ``include_output=True/False`` to override.
        """
        work = _work(session)
        if sync_out:
            # Cancel a sleeping debounce worker. Used for chained follow-ups
            # after WORK was already delivered (e.g. generate phase 2).
            with out_lock:
                out_state["dirty"] = False
                out_state["token"] += 1
            out = run_output_now()
        else:
            if run_out:
                mark_output_dirty(delay_override=0.05)
            out = session.last_output or work
        if include_output is None:
            # Async refine (run_out): poll owns the image so WORK can paint
            # first. Sync refine: return the fresh JPEG. Otherwise keep the
            # current OUTPUT visible (e.g. final refine already in last_output).
            if sync_out:
                include_output = True
            elif run_out:
                include_output = False
            else:
                include_output = True
        insp = _inspector(session)
        kind = insp["kind"]
        fields_on = kind != "none"
        obj_on = kind == "object"
        bg_on = kind == "background"
        cutout_on = obj_on and bool(session.layer_cutout)
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
        work_update: Any = (
            render_work_html(_scene_dict(session)) if include_work else gr.skip()
        )
        layers_update: Any = (
            render_layers_html(session) if include_layers else gr.skip()
        )
        output_update: Any = (
            _output_jpeg_path(out) if include_output else gr.skip()
        )
        return (
            work_update,
            layers_update,
            output_update,
            session.status,
            gr.update(value=insp["prompt"], interactive=fields_on),
            insp["iso_backdrop"],
            gr.update(
                visible=cutout_on,
                variant=(
                    "primary" if insp["iso_backdrop"] == "White" else "secondary"
                ),
            ),
            gr.update(
                visible=cutout_on,
                variant=(
                    "primary" if insp["iso_backdrop"] == "Black" else "secondary"
                ),
            ),
            gr.update(value=insp["opacity"], interactive=obj_on, visible=obj_on),
            gr.update(value=insp["feather"], interactive=obj_on, visible=obj_on),
            gr.update(value=insp["blend_mode"], interactive=obj_on, visible=obj_on),
            gr.update(value=insp["raw"], visible=insp["raw"] is not None),
            concat_val,
            gr.update(value=llm_val, visible=llm_on),
            gr.update(
                value=("Mute" if insp["prompt_enabled"] else "Unmute"),
                interactive=obj_on,
            ),
            gr.update(
                value=("Hide" if insp.get("visible", True) else "Show"),
                interactive=obj_on,
            ),
            gr.update(
                value=bool(session.layer_cutout),
                visible=obj_on,
            ),
            gr.update(visible=obj_on),
            gr.update(value=insp["scale"], visible=obj_on),
            gr.update(value=insp["rotation"], visible=obj_on),
            gr.update(value=insp["flip_x"], visible=obj_on),
            gr.update(value=insp["flip_y"], visible=obj_on),
            gr.update(value=insp["bg_scale"], visible=bg_on),
            gr.update(value=insp["bg_rotation"], visible=bg_on),
            gr.update(value=insp["bg_offset_x"], visible=bg_on),
            gr.update(value=insp["bg_offset_y"], visible=bg_on),
            gr.update(value=insp["bg_flip_x"], visible=bg_on),
            gr.update(value=insp["bg_flip_y"], visible=bg_on),
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
        lib_action = gr.Textbox(
            value="",
            elem_id="xwave-lib-action",
            elem_classes=["xwave-hidden"],
            container=False,
            show_label=False,
        )

        gr.Markdown(
            '<p class="xwave-brand"><span>XWAVE</span> COMPOSER</p>',
            elem_classes=["xwave-brand-wrap"],
        )

        with gr.Tabs(elem_classes=["xwave-main-tabs"]) as main_tabs:
            with gr.Tab("Compose", elem_id="xwave-compose-tab"):
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
                        active_profile = session.compute_profile
                        with gr.Row(elem_classes=["xwave-performance-buttons"]):
                            vram_html = gr.HTML(
                                value=render_vram_html(),
                                padding=False,
                                elem_classes=["xwave-vram-block"],
                            )
                            bf16_btn = gr.Button(
                                "BF16",
                                size="sm",
                                scale=1,
                                min_width=64,
                                variant="primary" if active_profile == "bf16" else "secondary",
                            )
                            mxfp8_btn = gr.Button(
                                "MXFP8",
                                size="sm",
                                scale=1,
                                min_width=72,
                                variant="primary" if active_profile == "mxfp8" else "secondary",
                            )
                            nvfp4_btn = gr.Button(
                                "NVFP4",
                                size="sm",
                                scale=1,
                                min_width=72,
                                variant="primary" if active_profile == "nvfp4" else "secondary",
                            )
                    with gr.Column(scale=1, min_width=380, elem_classes=["xwave-col", "xwave-knob-col"]):
                        active_quality = normalize_quality_mode(session.quality_mode)
                        with gr.Row(elem_classes=["xwave-bar-row", "xwave-quality-row"]):
                            fast_btn = gr.Button(
                                "Fast",
                                size="sm",
                                scale=0,
                                min_width=64,
                                variant="primary" if active_quality == "fast" else "secondary",
                                elem_classes=["xwave-quality-fast"],
                            )
                            quality_btn = gr.Button(
                                "Quality",
                                size="sm",
                                scale=0,
                                min_width=80,
                                variant="primary" if active_quality == "quality" else "secondary",
                                elem_classes=["xwave-quality-hq"],
                            )
                        with gr.Row(elem_classes=["xwave-bar-row", "xwave-knob-row"]):
                            cfg = gr.Number(
                                value=session.output_settings.cfg,
                                label="CFG",
                                minimum=0.0,
                                maximum=15.0,
                                step=0.1,
                                scale=0,
                                min_width=60,
                                elem_classes=["xwave-knob", "xwave-knob-num"],
                            )
                            denoise = gr.Slider(
                                0.05, 0.95,
                                value=session.output_settings.denoise,
                                step=0.01,
                                label="Denoise",
                                scale=1,
                                min_width=140,
                                elem_classes=["xwave-knob", "xwave-knob-slider"],
                            )
                            out_steps = gr.Slider(
                                1, 20,
                                value=session.output_settings.steps,
                                step=1,
                                label="Steps",
                                scale=1,
                                min_width=120,
                                elem_classes=["xwave-knob", "xwave-knob-slider"],
                            )
                            eta = gr.Number(
                                value=session.output_settings.eta,
                                label="Eta",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                scale=0,
                                min_width=60,
                                elem_classes=["xwave-knob", "xwave-knob-num"],
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
                            roll_all_btn = gr.Button(
                                "Roll all", size="sm", scale=0, min_width=72,
                                elem_classes=["xwave-roll-all-btn"],
                            )
                            add_obj_btn = gr.Button(
                                "+ Layer", size="sm", scale=0, min_width=72,
                                elem_classes=["xwave-add-btn"],
                            )
                        layers_html = gr.HTML(value=render_layers_html(session), padding=False)
                        with gr.Row(elem_classes=["xwave-canvas-size"]):
                            aspect = gr.Dropdown(
                                label="Canvas size",
                                choices=list(ASPECT_PRESETS.keys()),
                                value=DEFAULT_ASPECT,
                                scale=1,
                                elem_classes=["xwave-canvas-aspect"],
                            )
                            apply_size_btn = gr.Button(
                                "Apply",
                                size="sm",
                                scale=0,
                                min_width=64,
                                elem_classes=["xwave-canvas-apply"],
                            )
                        with gr.Row(elem_classes=["xwave-work-grid-bar"]):
                            work_grid_type = gr.Dropdown(
                                choices=WORK_GRID_TYPES,
                                value="Off",
                                label="Grid",
                                scale=2,
                                elem_id="xwave-work-grid-type",
                                elem_classes=["xwave-work-grid-type"],
                            )
                            work_grid_color = gr.Dropdown(
                                choices=WORK_GRID_COLORS,
                                value="Cyan",
                                label="Color",
                                scale=1,
                                elem_id="xwave-work-grid-color",
                                elem_classes=["xwave-work-grid-color"],
                            )
                        work_grid_state = gr.HTML(
                            value=_work_grid_state_html("Off", "Cyan"),
                            padding=False,
                            elem_classes=["xwave-work-grid-state"],
                        )

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
                                with gr.Row(elem_classes=["xwave-iso-row"]):
                                    cutout_chk = gr.Checkbox(
                                        label="✂️",
                                        value=True,
                                        visible=False,
                                        scale=0,
                                        min_width=40,
                                        elem_classes=["xwave-mini-check", "xwave-cutout-chk"],
                                    )
                                    iso_backdrop = gr.State(value="White")
                                    iso_white_btn = gr.Button(
                                        "⬜",
                                        size="sm",
                                        scale=0,
                                        min_width=36,
                                        visible=False,
                                        elem_classes=["xwave-iso-swatch", "xwave-iso-white"],
                                    )
                                    iso_black_btn = gr.Button(
                                        "⬛",
                                        size="sm",
                                        scale=0,
                                        min_width=36,
                                        visible=False,
                                        elem_classes=["xwave-iso-swatch", "xwave-iso-black"],
                                    )
                                    gen_btn = gr.Button(
                                        "▶️",
                                        variant="primary",
                                        size="sm",
                                        scale=0,
                                        min_width=40,
                                        elem_classes=["xwave-gen-btn"],
                                    )
                                    gen_seed = gr.Number(
                                        value=-1,
                                        precision=0,
                                        show_label=False,
                                        container=False,
                                        scale=1,
                                        min_width=88,
                                        elem_classes=["xwave-seed", "xwave-iso-seed"],
                                    )
                                # Background pose (numbers) + flips on their own row
                                # so FlipV never clips off the panel edge.
                                with gr.Row(elem_classes=["xwave-xform-strip", "xwave-bg-xform"]):
                                    bg_scale = gr.Number(
                                        label="Scale",
                                        value=1.0,
                                        precision=3,
                                        minimum=0.05,
                                        maximum=8.0,
                                        step=0.05,
                                        visible=True,
                                        scale=1,
                                        min_width=56,
                                        elem_classes=["xwave-knob", "xwave-mini"],
                                    )
                                    bg_rotation = gr.Number(
                                        label="Rot°",
                                        value=0.0,
                                        precision=1,
                                        step=1.0,
                                        visible=True,
                                        scale=1,
                                        min_width=52,
                                        elem_classes=["xwave-knob", "xwave-mini"],
                                    )
                                    bg_offset_x = gr.Number(
                                        label="X",
                                        value=0.0,
                                        precision=1,
                                        step=1.0,
                                        visible=True,
                                        scale=1,
                                        min_width=52,
                                        elem_classes=["xwave-knob", "xwave-mini"],
                                    )
                                    bg_offset_y = gr.Number(
                                        label="Y",
                                        value=0.0,
                                        precision=1,
                                        step=1.0,
                                        visible=True,
                                        scale=1,
                                        min_width=52,
                                        elem_classes=["xwave-knob", "xwave-mini"],
                                    )
                                with gr.Row(
                                    elem_classes=["xwave-xform-strip", "xwave-flip-row", "xwave-bg-xform"]
                                ):
                                    bg_flip_x = gr.Checkbox(
                                        label="Flip H",
                                        value=False,
                                        visible=True,
                                        scale=1,
                                        min_width=72,
                                        elem_classes=["xwave-mini-check"],
                                    )
                                    bg_flip_y = gr.Checkbox(
                                        label="Flip V",
                                        value=False,
                                        visible=True,
                                        scale=1,
                                        min_width=72,
                                        elem_classes=["xwave-mini-check"],
                                    )
                                # Object pose: scale / rotate / flip
                                with gr.Row(elem_classes=["xwave-xform-strip", "xwave-obj-xform"]):
                                    obj_scale = gr.Number(
                                        label="Scale",
                                        value=1.0,
                                        precision=3,
                                        minimum=0.05,
                                        maximum=8.0,
                                        step=0.05,
                                        visible=False,
                                        scale=1,
                                        min_width=56,
                                        elem_classes=["xwave-knob", "xwave-mini"],
                                    )
                                    obj_rotation = gr.Number(
                                        label="Rot°",
                                        value=0.0,
                                        precision=1,
                                        step=1.0,
                                        visible=False,
                                        scale=1,
                                        min_width=52,
                                        elem_classes=["xwave-knob", "xwave-mini"],
                                    )
                                with gr.Row(
                                    elem_classes=["xwave-xform-strip", "xwave-flip-row", "xwave-obj-xform"]
                                ):
                                    obj_flip_x = gr.Checkbox(
                                        label="Flip H",
                                        value=False,
                                        visible=False,
                                        scale=1,
                                        min_width=72,
                                        elem_classes=["xwave-mini-check"],
                                    )
                                    obj_flip_y = gr.Checkbox(
                                        label="Flip V",
                                        value=False,
                                        visible=False,
                                        scale=1,
                                        min_width=72,
                                        elem_classes=["xwave-mini-check"],
                                    )
                                # Object look: opacity / feather / blend
                                with gr.Row(elem_classes=["xwave-xform-strip", "xwave-obj-xform"]):
                                    insp_opacity = gr.Slider(
                                        0.0,
                                        1.0,
                                        value=1.0,
                                        step=0.01,
                                        label="Opacity",
                                        visible=False,
                                        scale=1,
                                        min_width=100,
                                        elem_classes=["xwave-mini-slider"],
                                    )
                                    insp_feather = gr.Slider(
                                        0.0,
                                        128.0,
                                        value=0.0,
                                        step=1.0,
                                        label="Feather",
                                        visible=False,
                                        scale=1,
                                        min_width=100,
                                        elem_classes=["xwave-mini-slider"],
                                    )
                                with gr.Row(elem_classes=["xwave-xform-strip", "xwave-obj-xform"]):
                                    insp_blend = gr.Dropdown(
                                        choices=BLEND_MODE_LABELS,
                                        value="Normal",
                                        label="Blend",
                                        visible=False,
                                        scale=1,
                                        min_width=120,
                                        elem_classes=["xwave-mini-dd"],
                                    )
                                with gr.Row(elem_classes=["xwave-action-row"]):
                                    mute_prompt_btn = gr.Button(
                                        "Mute", size="sm", scale=1, min_width=56
                                    )
                                    hide_layer_btn = gr.Button(
                                        "Hide", size="sm", scale=1, min_width=56
                                    )
                                    duplicate_btn = gr.Button(
                                        "Dup", size="sm", scale=1, min_width=48
                                    )
                                with gr.Row(elem_classes=["xwave-action-row"]):
                                    reset_xform_btn = gr.Button(
                                        "Reset", size="sm", scale=1, min_width=56
                                    )
                                    reisolate_btn = gr.Button(
                                        "Re-cut", size="sm", scale=1, min_width=56
                                    )
                                    delete_btn = gr.Button(
                                        "Delete",
                                        size="sm",
                                        variant="stop",
                                        scale=1,
                                        min_width=56,
                                    )
                                cutout_mode = gr.Radio(
                                    choices=["rembg", "SAM2", "none"],
                                    value="rembg",
                                    show_label=False,
                                    container=False,
                                    visible=False,
                                    elem_classes=["xwave-cutout-mode", "xwave-compact-radio"],
                                )
                                raw_view = gr.Image(
                                    label="Raw — click the subject to re-cut with SAM2",
                                    type="pil",
                                    interactive=False,
                                    visible=False,
                                    height=140,
                                    buttons=[],
                                    elem_classes=["xwave-raw-view"],
                                )
                                with gr.Accordion("Import image", open=False):
                                    import_img = gr.Image(
                                        type="pil",
                                        label="Drop or upload",
                                        height=100,
                                        buttons=[],
                                        elem_classes=["xwave-import"],
                                    )
                                    import_btn = gr.Button("Import into layer", size="sm")
                                    compose_lib_seed = pack_library_views(library)
                                    compose_lib_html = gr.HTML(
                                        value=compose_lib_seed[2],
                                        elem_classes=["xwave-lib-picker"],
                                    )
                                    with gr.Row(elem_classes=["xwave-lib-pager"]):
                                        compose_lib_prev = gr.Button(
                                            "Prev", size="sm", scale=0, min_width=64
                                        )
                                        compose_lib_pager = gr.Textbox(
                                            value=compose_lib_seed[3],
                                            show_label=False,
                                            interactive=False,
                                            container=False,
                                            scale=1,
                                        )
                                        compose_lib_next = gr.Button(
                                            "Next", size="sm", scale=0, min_width=64
                                        )
                                    gr.Markdown(
                                        '<p class="xwave-text-dim">Click a library thumb to import into the selected layer.</p>'
                                    )

                            # —— Output style ——
                            with gr.Column(scale=1, min_width=250, elem_classes=["xwave-panel"]):
                                gr.Markdown('<p class="xwave-section-head">Style</p>')
                                style_family_dd = gr.Dropdown(
                                    choices=style_family_choices,
                                    value=FAMILY_ALL,
                                    label="Family",
                                    show_label=False,
                                    container=False,
                                    filterable=False,
                                    elem_classes=["xwave-style-family"],
                                )
                                style_dd = gr.Dropdown(
                                    choices=style_names,
                                    value=NO_STYLE,
                                    label="Style",
                                    show_label=False,
                                    container=False,
                                    filterable=True,
                                    elem_classes=["xwave-style-name"],
                                )
                                with gr.Row(elem_classes=["xwave-seed-row", "xwave-style-seed"]):
                                    out_seed = gr.Number(
                                        value=session.output_settings.seed,
                                        precision=0,
                                        show_label=False,
                                        container=False,
                                        scale=1,
                                        min_width=88,
                                        elem_classes=["xwave-seed", "xwave-style-seed-num"],
                                    )
                                    roll_seed_btn = gr.Button(
                                        "🎲",
                                        size="sm",
                                        scale=0,
                                        min_width=36,
                                        elem_classes=["xwave-roll-seed"],
                                    )
                                with gr.Accordion(
                                    "Prompt options",
                                    open=False,
                                    elem_classes=["xwave-style-opts"],
                                ):
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
                                # Concatenation result (always available)
                                with gr.Group(elem_classes=["xwave-built-prompt"]):
                                    with gr.Row():
                                        use_llm = gr.Checkbox(
                                            label="LLM rewrite", value=False, scale=1,
                                        )
                                        prompt_lock = gr.Checkbox(
                                            label="Edit built prompt (lock auto-build)",
                                            value=False,
                                            scale=1,
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
                                gr.Markdown('<p class="xwave-section-head">Output model</p>')
                                with gr.Row():
                                    base_dd = gr.Dropdown(
                                        choices=base_names,
                                        value=default_base_name,
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
                                )
                                performance_dd = gr.Dropdown(
                                    label="Performance",
                                    choices=PROFILE_CHOICES,
                                    value=profile_label(session.compute_profile),
                                )
                                performance_status = gr.Textbox(
                                    label="Applied compute",
                                    value=session.optimization_status(),
                                    interactive=False,
                                    lines=2,
                                    max_lines=3,
                                )
                                with gr.Accordion("Style adapters (LoRA / TI)", open=False):
                                    lora_path = gr.Textbox(label="LoRA path or HF id", lines=1)
                                    lora_scale = gr.Slider(0, 1.5, value=0.8, step=0.05, label="LoRA scale")
                                    load_lora_btn = gr.Button("Load LoRA", size="sm")
                                    emb_path = gr.Textbox(label="Textual inversion path or id", lines=1)
                                    load_emb_btn = gr.Button("Load TI", size="sm")
                                gr.Markdown('<p class="xwave-section-head">Final output</p>')
                                with gr.Row():
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
                                with gr.Accordion("SeedVR2 export settings", open=False):
                                    seedvr_preset = gr.Dropdown(
                                        label="Preset",
                                        choices=SEEDVR2_PRESET_CHOICES,
                                        value="quality",
                                    )
                                    seedvr_model = gr.Dropdown(
                                        label="Model",
                                        choices=SEEDVR2_MODEL_CHOICES,
                                        value=config.get(
                                            "export",
                                            "seedvr2_model",
                                            default=DEFAULT_SEEDVR2_MODEL,
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
                                    with gr.Accordion("Advanced — VRAM / speed", open=False):
                                        seedvr_blocks = gr.Slider(
                                            0,
                                            36,
                                            value=int(
                                                config.get(
                                                    "export",
                                                    "seedvr2_blocks_to_swap",
                                                    default=0,
                                                )
                                            ),
                                            step=1,
                                            label="BlockSwap (0 = off)",
                                            info="Offloads DiT blocks to CPU. Auto-sets offload=cpu when > 0.",
                                        )
                                        seedvr_swap_io = gr.Checkbox(
                                            label="Swap I/O components",
                                            value=bool(
                                                config.get(
                                                    "export",
                                                    "seedvr2_swap_io_components",
                                                    default=False,
                                                )
                                            ),
                                        )
                                        with gr.Row():
                                            seedvr_dit_offload = gr.Dropdown(
                                                label="DiT offload",
                                                choices=["none", "cpu"],
                                                value=str(
                                                    config.get(
                                                        "export",
                                                        "seedvr2_dit_offload_device",
                                                        default="none",
                                                    )
                                                ),
                                            )
                                            seedvr_vae_offload = gr.Dropdown(
                                                label="VAE offload",
                                                choices=["none", "cpu"],
                                                value=str(
                                                    config.get(
                                                        "export",
                                                        "seedvr2_vae_offload_device",
                                                        default="none",
                                                    )
                                                ),
                                            )
                                        seedvr_compile = gr.Checkbox(
                                            label="Compile DiT (torch.compile)",
                                            value=bool(
                                                config.get(
                                                    "export",
                                                    "seedvr2_compile_dit",
                                                    default=False,
                                                )
                                            ),
                                            info="Faster later exports; first run pays compile cost.",
                                        )
                                export_btn = gr.Button(
                                    "Export accepted OUTPUT 2× with SeedVR2",
                                    variant="primary",
                                    size="sm",
                                    elem_classes=["xwave-export-btn"],
                                )
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
                                with gr.Accordion(
                                    "Export Style Flipbook",
                                    open=False,
                                    elem_classes=["xwave-flipbook"],
                                ):
                                    style_preset_count = (
                                        len(session.styles.names()) if session.styles else 0
                                    )
                                    flipbook_mode = gr.Radio(
                                        choices=["Styles", "Seeds"],
                                        value="Styles",
                                        label="Mode",
                                        elem_classes=["xwave-flipbook-mode"],
                                    )
                                    flipbook_families = gr.CheckboxGroup(
                                        choices=flipbook_family_choices,
                                        value=list(flipbook_family_choices),
                                        label="Families",
                                        elem_classes=["xwave-flipbook-families"],
                                    )
                                    flipbook_count = gr.Number(
                                        label="Styles (0 = all in checked families)",
                                        value=min(32, style_preset_count) if style_preset_count else 32,
                                        precision=0,
                                        minimum=0,
                                        maximum=max(style_preset_count, 1),
                                    )
                                    flipbook_fps = gr.Radio(
                                        choices=[24, 30, 48, 60],
                                        value=30,
                                        label="FPS",
                                    )
                                    flipbook_hold = gr.Number(
                                        label="Frames per image",
                                        value=8,
                                        precision=0,
                                        minimum=1,
                                        maximum=120,
                                    )
                                    flipbook_seed_mode = gr.State(value="lock")
                                    with gr.Row(elem_classes=["xwave-flipbook-seed"]):
                                        flipbook_lock_btn = gr.Button(
                                            "🔒",
                                            size="sm",
                                            scale=0,
                                            min_width=36,
                                            variant="primary",
                                            elem_classes=["xwave-flipbook-lock"],
                                        )
                                        flipbook_dice_btn = gr.Button(
                                            "🎲",
                                            size="sm",
                                            scale=0,
                                            min_width=36,
                                            variant="secondary",
                                            elem_classes=["xwave-flipbook-dice"],
                                        )
                                        flipbook_seed_hint = gr.Markdown(
                                            '<p class="xwave-flipbook-seed-hint">'
                                            "🔒 same seed · 🎲 new seed per style</p>"
                                        )
                                    flipbook_btn = gr.Button(
                                        "Run style flipbook",
                                        variant="primary",
                                        size="sm",
                                        elem_classes=["xwave-flipbook-run"],
                                    )
                                    flipbook_path = gr.Textbox(
                                        label="Flipbook path",
                                        interactive=False,
                                        lines=1,
                                    )
                                    flipbook_video = gr.Video(
                                        label="Flipbook preview",
                                        interactive=False,
                                        height=160,
                                    )


            with gr.Tab("Infinite Canvas", elem_id="xwave-infinite-canvas-tab"):
                _lc_ui = build_large_canvas_tab(
                    sdxl=session.sdxl,
                    styles=session.styles,
                    config=config,
                    infer_lock=infer_lock,
                    upscaler=session.upscaler,
                    free_vram_fn=session.enter_infinite_canvas_mode,
                    library=library,
                )
                demo._xwave_large_canvas = _lc_ui  # type: ignore[attr-defined]
                lc_status = _lc_ui["status"]

            with gr.Tab("Edit", elem_id="xwave-edit-tab"):
                edit_session = EditSession(isolator=session.isolator)
                _edit_ui = build_edit_tab(
                    session=edit_session,
                    config=config,
                    infer_lock=infer_lock,
                    upscaler=session.upscaler,
                    composer=session,
                    library=library,
                )
                demo._xwave_edit = _edit_ui  # type: ignore[attr-defined]
                edit_status = _edit_ui["status"]

            with gr.Tab("Library", elem_id="xwave-library-tab"):
                _lib_ui = build_library_tab(library)
                demo._xwave_library = _lib_ui  # type: ignore[attr-defined]

            with gr.Tab("Settings", elem_id="xwave-settings-tab"):
                build_settings_tab()
        # ── Callbacks ───────────────────────────────────────────
        pack_out = [
            work_html,
            layers_html,
            output_image,
            status,
            insp_prompt,
            iso_backdrop,
            iso_white_btn,
            iso_black_btn,
            insp_opacity,
            insp_feather,
            insp_blend,
            raw_view,
            prompt_view,
            llm_prompt_view,
            mute_prompt_btn,
            hide_layer_btn,
            cutout_chk,
            cutout_mode,
            obj_scale,
            obj_rotation,
            obj_flip_x,
            obj_flip_y,
            bg_scale,
            bg_rotation,
            bg_offset_x,
            bg_offset_y,
            bg_flip_x,
            bg_flip_y,
        ]
        settings_in = [denoise, out_steps, cfg, eta, use_llm, neg_prompt, out_seed]
        lc = _lc_ui["session"]
        lib_browse_off = _lib_ui["browse_offset"]
        lib_compose_off = _lib_ui["compose_offset"]
        lib_ic_off = _lib_ui["ic_offset"]
        lib_edit_off = _edit_ui["offset"]
        lib_selected = _lib_ui["selected_id"]
        lib_view_out = [
            lib_browse_off,
            lib_compose_off,
            lib_ic_off,
            lib_edit_off,
            lib_selected,
            _lib_ui["html"],
            _lib_ui["pager"],
            compose_lib_html,
            compose_lib_pager,
            _lc_ui["picker_html"],
            _lc_ui["picker_pager"],
            _edit_ui["picker_html"],
            _edit_ui["picker_pager"],
            _lib_ui["preview"],
            _lib_ui["name"],
            _lib_ui["meta"],
            _lib_ui["download"],
        ]
        lc_pack_out = [
            _lc_ui["html"],
            lc_status,
            _lc_ui["stamp_x"],
            _lc_ui["stamp_y"],
            _lc_ui["stamp_w"],
            _lc_ui["stamp_h"],
            _lc_ui["export_file"],
        ]
        edit_pack_out = _edit_ui["pack_out"]

        def _lib_pack(
            browse_off=0,
            compose_off=0,
            ic_off=0,
            edit_off=0,
            selected_id="",
        ):
            b = library.page(int(browse_off or 0)).offset
            c = library.page(int(compose_off or 0)).offset
            i = library.page(int(ic_off or 0)).offset
            e = library.page(int(edit_off or 0)).offset
            sid = str(selected_id or "")
            if sid and library.get(sid) is None:
                sid = ""
            views = pack_library_views(
                library,
                browse_offset=b,
                compose_offset=c,
                ic_offset=i,
                edit_offset=e,
                selected_id=sid or None,
            )
            preview = selected_item_outputs(library, sid)
            return (b, c, i, e, sid, *views, *preview)

        def _skip_pack():
            return tuple(gr.skip() for _ in pack_out)

        def _skip_lc():
            return tuple(gr.skip() for _ in lc_pack_out)

        def _skip_edit():
            return tuple(gr.skip() for _ in edit_pack_out)

        def _edit_pack():
            return _edit_ui["pack"]()

        def _lc_pack():
            return (
                _scene_html_lc(),
                lc.status,
                lc.stamp.x,
                lc.stamp.y,
                lc.stamp.w,
                lc.stamp.h,
                gr.update(),
            )

        def _scene_html_lc():
            return _scene_html(config, lc)

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
            with infer_lock:
                session.enter_compose_mode()
                msg = session.preload_core()
            flush_pending_output()
            return msg

        def on_free():
            with infer_lock:
                return session.free_optional()

        def on_main_tab(evt: gr.SelectData):
            """Exclusive GPU mode when switching Compose ↔ Infinite Canvas ↔ Edit."""
            label = str(getattr(evt, "value", "") or "").strip().lower()
            # Gradio may pass the tab label as a dict-like value.
            if isinstance(getattr(evt, "value", None), dict):
                label = str(evt.value.get("label") or evt.value.get("value") or "").strip().lower()
            if "library" in label:
                return gr.skip(), gr.skip(), gr.skip()
            want_ic = "infinite" in label
            want_compose = "compose" in label
            want_edit = label == "edit" or label.startswith("edit")
            if not want_ic and not want_compose and not want_edit:
                return gr.skip(), gr.skip(), gr.skip()
            with infer_lock:
                if want_ic:
                    with out_lock:
                        remember = bool(out_state["dirty"] or out_state["pending"])
                        out_state["token"] += 1
                        out_state["dirty"] = False
                        out_state["running"] = False
                        out_state["pending"] = remember
                    msg = session.enter_infinite_canvas_mode()
                    return msg, msg, msg
                if want_edit:
                    with out_lock:
                        remember = bool(out_state["dirty"] or out_state["pending"])
                        out_state["token"] += 1
                        out_state["dirty"] = False
                        out_state["running"] = False
                        out_state["pending"] = remember
                    msg = session.enter_edit_mode()
                    return msg, msg, msg
                msg = session.enter_compose_mode()
            flush_pending_output()
            return msg, msg, msg

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
                lid = str(data.get("id", ""))
                ts = int(data.get("_ts") or 0)
                last_ts = session._last_transform_ts.get(lid, 0)
                final = data.get("final")
                if ts and ts < last_ts:
                    # Stale Gradio-queued drag event — ignore so we don't
                    # snap the layer back to an older position. Still kick
                    # OUTPUT on a rejected final: a newer live pose may
                    # already be applied without having scheduled refine.
                    if final is True:
                        session.ensure_work_composed()
                        mark_output_dirty(delay_override=0.05)
                    return tuple(gr.update() for _ in pack_out)
                if ts:
                    session._last_transform_ts[lid] = ts
                # Live drags: pose fields only. Pointer-up / legacy: compose WORK.
                is_final = final is True or final is None
                session.update_transform_by_id(
                    lid,
                    x=data.get("x"),
                    y=data.get("y"),
                    scale_x=data.get("scale_x"),
                    scale_y=data.get("scale_y"),
                    rotation=data.get("rotation"),
                    compose=is_final,
                )
                if final is not True:
                    session.status = "Transforming…"
                # Older tabs (opened before the canvas JS update) omit
                # ``final``. Keep those functional until the user refreshes;
                # current tabs send false while dragging and true on release.
                if final is True:
                    # Keep the WORK canvas on the optimistic local pose.
                    # Kick one full-quality OUTPUT refine after release.
                    # Do not write output_image here — a late pack with the
                    # pre-refine JPEG can clobber a fresher poll_output and
                    # leave rev_state stuck until the next move.
                    mark_output_dirty(delay_override=0.05)
                    return pack(
                        run_out=False,
                        sync_out=False,
                        include_work=False,
                        include_layers=False,
                        include_output=False,
                    )
                if final is None:
                    # Legacy browser assets emit every 120 ms and do not label
                    # pointer-up. A longer debounce collapses the stream into
                    # one full refine after movement stops.
                    mark_output_dirty(delay_override=0.8)
                # Live events update server state only — never rewrite the scene.
                return tuple(gr.update() for _ in pack_out)

            if atype == "select":
                sid = data.get("id")
                if sid == "__bg__":
                    session.doc.selected_id = "__bg__"
                elif sid:
                    session.select_layer_id(str(sid))
                else:
                    session.doc.selected_id = None
                # JS already highlights the card + canvas selection. Skip the
                # heavy work_html rebuild so the click feels instant. Skip
                # OUTPUT too — select does not change composition, and a
                # stale JPEG write can clobber a just-polled refine.
                return pack(
                    run_out=False,
                    include_work=False,
                    include_output=False,
                )

            if atype == "delete":
                lid = data.get("id")
                if lid and lid != "__bg__":
                    session.delete_layer(str(lid))
                return pack_work_then_output()

            if atype == "reorder":
                session.reorder_layers([str(i) for i in (data.get("ids") or [])])
                return pack_work_then_output()

            return pack(run_out=False)

        def on_add_layer(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            session.add_empty_object()
            return pack(run_out=False)

        def on_roll_all(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            try:
                session.roll_all()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Roll all failed")
                session.status = f"Roll all failed: {exc}"
                packed = pack(run_out=False)
                return (*packed, int(session.output_settings.seed))
            # WORK already updated; OUTPUT was refined inside roll_all.
            with out_lock:
                out_state["dirty"] = False
                out_state["token"] += 1
            packed = pack(run_out=False, sync_out=False, include_output=True)
            return (*packed, int(session.output_settings.seed))

        # When True, the chained generate follow-up should refine OUTPUT.
        # Cleared on failure / empty prompt so we don't re-refine a stale pose.
        generate_needs_output = {"ok": False}

        def on_generate(prompt, backdrop, seed, cutout, den, steps, cfg_v, eta_v, llm, neg, oseed):
            generate_needs_output["ok"] = False
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if not prompt or not str(prompt).strip():
                session.status = "Type a prompt for the selected layer first."
                return pack(run_out=False)
            isolate = bool(cutout) and session.doc.selected_id not in (None, "__bg__")
            session.layer_cutout = bool(cutout) if session.doc.selected_id not in (None, "__bg__") else session.layer_cutout
            iso_text = _iso_prompt_for_backdrop(backdrop) if isolate else ""
            try:
                session.generate_selected(
                    str(prompt), iso_text, seed=int(seed), isolate=isolate
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Generation failed")
                session.status = f"Generation failed: {exc}"
                return pack(run_out=False)
            # Push WORK/layers first. Refining OUTPUT in this same response
            # (sync_out) meant the JPEG could paint before the canvas scene
            # HTML arrived and before layer data-URLs finished decoding.
            generate_needs_output["ok"] = True
            return pack(run_out=False, sync_out=False, include_output=False)

        def on_generate_output():
            if not generate_needs_output["ok"]:
                return tuple(gr.update() for _ in pack_out)
            generate_needs_output["ok"] = False
            return pack(
                run_out=False,
                sync_out=True,
                include_work=False,
                include_layers=False,
            )

        def _iso_swatch_pack(label: str, *, visible: bool):
            sel = str(label or "White")
            return (
                sel,
                gr.update(
                    visible=visible,
                    variant="primary" if sel == "White" else "secondary",
                ),
                gr.update(
                    visible=visible,
                    variant="primary" if sel == "Black" else "secondary",
                ),
            )

        def on_cutout(checked):
            session.set_layer_cutout(bool(checked))
            if not session.layer_cutout:
                return (*_iso_swatch_pack("White", visible=False), session.status)
            obj = session.doc.selected()
            label = _iso_backdrop_label(obj.isolation_prompt if obj else ISO_WHITE)
            if obj is not None and not (obj.isolation_prompt or "").strip():
                session.set_isolation_backdrop(label)
            return (*_iso_swatch_pack(label, visible=True), session.status)

        def on_iso_swatch(label):
            if session.doc.selected_id in (None, "__bg__") or not session.layer_cutout:
                return (*_iso_swatch_pack(str(label or "White"), visible=False), session.status)
            session.set_isolation_backdrop(str(label or "White"))
            return (*_iso_swatch_pack(str(label or "White"), visible=True), session.status)

        def on_obj_transform(
            scale, rotation, flip_x, flip_y,
            den, steps, cfg_v, eta_v, llm, neg, oseed,
        ):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            obj = session.doc.selected()
            if obj is None:
                return pack(run_out=False)
            if scale is None or rotation is None:
                return tuple(gr.update() for _ in pack_out)
            new_scale = max(0.05, float(scale))
            new_rot = float(rotation)
            new_fx = bool(flip_x)
            new_fy = bool(flip_y)
            t = obj.transform
            if (
                abs(float(t.scale_x) - new_scale) < 1e-6
                and abs(float(t.scale_y) - new_scale) < 1e-6
                and abs(float(t.rotation) - new_rot) < 1e-6
                and bool(getattr(t, "flip_x", False)) == new_fx
                and bool(getattr(t, "flip_y", False)) == new_fy
            ):
                return tuple(gr.update() for _ in pack_out)
            session.update_transform_by_id(
                obj.id,
                scale_x=new_scale,
                scale_y=new_scale,
                rotation=new_rot,
                flip_x=new_fx,
                flip_y=new_fy,
            )
            return pack_work_then_output()

        def on_feather(feather, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            obj = session.doc.selected()
            if obj is None:
                return pack(run_out=False)
            new_f = max(0.0, min(128.0, float(feather)))
            if abs(float(getattr(obj, "feather", 0.0)) - new_f) < 1e-6:
                return tuple(gr.update() for _ in pack_out)
            obj.feather = new_f
            session.last_work = session.refresh_work()
            session.status = f"Edge feather → {new_f:.0f}px"
            return pack_work_then_output()

        def on_blend_mode(mode_label, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            obj = session.doc.selected()
            if obj is None:
                return pack(run_out=False)
            new_mode = normalize_blend_mode(mode_label)
            if normalize_blend_mode(obj.blend_mode) == new_mode:
                return tuple(gr.update() for _ in pack_out)
            obj.blend_mode = new_mode
            session.last_work = session.refresh_work()
            session.status = f"Blend mode → {blend_mode_label(new_mode)}"
            return pack_work_then_output()

        def on_bg_transform(
            scale, rotation, offset_x, offset_y, flip_x, flip_y,
            den, steps, cfg_v, eta_v, llm, neg, oseed,
        ):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id != "__bg__":
                return pack(run_out=False)
            if None in (scale, rotation, offset_x, offset_y):
                return tuple(gr.update() for _ in pack_out)
            new_scale = max(0.05, float(scale))
            new_rot = float(rotation)
            new_ox = float(offset_x)
            new_oy = float(offset_y)
            new_fx = bool(flip_x)
            new_fy = bool(flip_y)
            if (
                abs(session.doc.bg_scale - new_scale) < 1e-6
                and abs(session.doc.bg_rotation - new_rot) < 1e-6
                and abs(session.doc.bg_offset_x - new_ox) < 1e-6
                and abs(session.doc.bg_offset_y - new_oy) < 1e-6
                and session.doc.bg_flip_x == new_fx
                and session.doc.bg_flip_y == new_fy
            ):
                return tuple(gr.update() for _ in pack_out)
            session.update_background_transform(
                scale=new_scale,
                rotation=new_rot,
                offset_x=new_ox,
                offset_y=new_oy,
                flip_x=new_fx,
                flip_y=new_fy,
            )
            return pack_work_then_output()

        def _cutout_prefer(label) -> str:
            return {"rembg": "rembg", "SAM2": "sam2", "none": "none"}.get(
                str(label or "rembg"), "rembg"
            )

        def on_import(image, cutout, prompt, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if image is None:
                session.status = "Choose an image to import."
                return pack(run_out=False)
            mode = _cutout_prefer(cutout)
            try:
                session.import_into_selected(image, cutout=mode, prompt=str(prompt or ""))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Import failed")
                session.status = f"Import failed: {exc}"
                return pack(run_out=False)
            return pack_work_then_output()

        def on_compose_save_lib(name, browse_off, compose_off, ic_off, edit_off, selected):
            img = session.last_output or session.last_work
            if img is None:
                session.status = "Nothing to save — generate an OUTPUT first."
                return (session.status, *_lib_pack(browse_off, compose_off, ic_off, edit_off, selected))
            try:
                item = library.add(img, "compose", name)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Library save failed")
                session.status = f"Library save failed: {exc}"
                return (session.status, *_lib_pack(browse_off, compose_off, ic_off, edit_off, selected))
            session.status = f"Saved to library ({item.width}×{item.height})."
            return (session.status, *_lib_pack(0, 0, 0, 0, item.id))

        def on_ic_save_lib(name, browse_off, compose_off, ic_off, edit_off, selected):
            with lc._lock:
                img = lc.image.copy()
            try:
                item = library.add(img, "infinite_canvas", name)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Library save failed")
                lc.status = f"Library save failed: {exc}"
                return (lc.status, *_lib_pack(browse_off, compose_off, ic_off, edit_off, selected))
            lc.status = f"Saved to library ({item.width}×{item.height})."
            return (lc.status, *_lib_pack(0, 0, 0, 0, item.id))

        def on_edit_save_lib(name, browse_off, compose_off, ic_off, edit_off, selected):
            if not edit_session.has_image:
                edit_session.status = "Load an image first."
                return (edit_session.status, *_lib_pack(browse_off, compose_off, ic_off, edit_off, selected))
            try:
                item = library.add(edit_session.flatten(), "edit", name)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Library save failed")
                edit_session.status = f"Library save failed: {exc}"
                return (edit_session.status, *_lib_pack(browse_off, compose_off, ic_off, edit_off, selected))
            edit_session.status = f"Saved to library ({item.width}×{item.height})."
            return (edit_session.status, *_lib_pack(0, 0, 0, 0, item.id))

        def on_lib_page(which, delta, browse_off, compose_off, ic_off, edit_off, selected):
            if which == "browse":
                browse_off = shift_page(library, browse_off, delta)
            elif which == "compose":
                compose_off = shift_page(library, compose_off, delta)
            elif which == "edit":
                edit_off = shift_page(library, edit_off, delta)
            else:
                ic_off = shift_page(library, ic_off, delta)
            return _lib_pack(browse_off, compose_off, ic_off, edit_off, selected)

        def on_lib_rename(name, selected, browse_off, compose_off, ic_off, edit_off):
            if selected:
                library.set_name(str(selected), name)
            return _lib_pack(browse_off, compose_off, ic_off, edit_off, selected)

        def on_lib_delete(selected, browse_off, compose_off, ic_off, edit_off):
            if selected:
                library.delete(str(selected))
            return _lib_pack(browse_off, compose_off, ic_off, edit_off, "")

        def _import_lib_compose(
            item_id,
            cutout,
            prompt,
            den,
            steps,
            cfg_v,
            eta_v,
            llm,
            neg,
            oseed,
            browse_off,
            compose_off,
            ic_off,
            edit_off,
        ):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            img = library.open_full(str(item_id or ""))
            if img is None:
                session.status = "Select a library image first."
                return (
                    *pack(run_out=False),
                    *_skip_lc(),
                    *_skip_edit(),
                    *_lib_pack(browse_off, compose_off, ic_off, edit_off, item_id),
                )
            try:
                session.import_into_selected(
                    img, cutout=_cutout_prefer(cutout), prompt=str(prompt or "")
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Library import failed")
                session.status = f"Import failed: {exc}"
                return (
                    *pack(run_out=False),
                    *_skip_lc(),
                    *_skip_edit(),
                    *_lib_pack(browse_off, compose_off, ic_off, edit_off, item_id),
                )
            return (
                *pack_work_then_output(),
                *_skip_lc(),
                *_skip_edit(),
                *_lib_pack(browse_off, compose_off, ic_off, edit_off, item_id),
            )

        def _import_lib_ic(item_id, place, browse_off, compose_off, ic_off, edit_off):
            img = library.open_full(str(item_id or ""))
            if img is None:
                lc.status = "Select a library image first."
                return (
                    *_skip_pack(),
                    *_lc_pack(),
                    *_skip_edit(),
                    *_lib_pack(browse_off, compose_off, ic_off, edit_off, item_id),
                )
            try:
                encode = session.sdxl if session.sdxl is not None and session.sdxl.ready else None
                lc.import_image(img, mode=placement_key(place), sdxl=encode)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Library import to Infinite Canvas failed")
                lc.status = f"Import failed: {exc}"
            return (
                *_skip_pack(),
                *_lc_pack(),
                *_skip_edit(),
                *_lib_pack(browse_off, compose_off, ic_off, edit_off, item_id),
            )

        def _import_lib_edit(item_id, browse_off, compose_off, ic_off, edit_off):
            img = library.open_full(str(item_id or ""))
            item = library.get(str(item_id or ""))
            if img is None:
                edit_session.status = "Select a library image first."
                return (
                    *_skip_pack(),
                    *_skip_lc(),
                    *_edit_pack(),
                    *_lib_pack(browse_off, compose_off, ic_off, edit_off, item_id),
                )
            try:
                edit_session.load(img, name=(item.name if item else ""))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Library import to Edit failed")
                edit_session.status = f"Import failed: {exc}"
            return (
                *_skip_pack(),
                *_skip_lc(),
                *_edit_pack(),
                *_lib_pack(browse_off, compose_off, ic_off, edit_off, item_id),
            )

        def on_lib_action(
            payload,
            selected,
            browse_off,
            compose_off,
            ic_off,
            edit_off,
            cutout,
            prompt,
            den,
            steps,
            cfg_v,
            eta_v,
            llm,
            neg,
            oseed,
            lib_place,
            lc_place,
        ):
            if not payload or not str(payload).strip():
                return (
                    *_skip_pack(),
                    *_skip_lc(),
                    *_skip_edit(),
                    *_lib_pack(browse_off, compose_off, ic_off, edit_off, selected),
                )
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return (
                    *_skip_pack(),
                    *_skip_lc(),
                    *_skip_edit(),
                    *_lib_pack(browse_off, compose_off, ic_off, edit_off, selected),
                )
            item_id = str(data.get("id") or "")
            ctx = str(data.get("context") or "browse")
            if ctx == "compose":
                return _import_lib_compose(
                    item_id,
                    cutout,
                    prompt,
                    den,
                    steps,
                    cfg_v,
                    eta_v,
                    llm,
                    neg,
                    oseed,
                    browse_off,
                    compose_off,
                    ic_off,
                    edit_off,
                )
            if ctx == "infinite":
                return _import_lib_ic(
                    item_id, lc_place, browse_off, compose_off, ic_off, edit_off
                )
            if ctx == "edit":
                return _import_lib_edit(
                    item_id, browse_off, compose_off, ic_off, edit_off
                )
            return (
                *_skip_pack(),
                *_skip_lc(),
                *_skip_edit(),
                *_lib_pack(browse_off, compose_off, ic_off, edit_off, item_id),
            )

        def on_lib_to_compose(
            selected,
            cutout,
            prompt,
            den,
            steps,
            cfg_v,
            eta_v,
            llm,
            neg,
            oseed,
            browse_off,
            compose_off,
            ic_off,
            edit_off,
        ):
            return _import_lib_compose(
                selected,
                cutout,
                prompt,
                den,
                steps,
                cfg_v,
                eta_v,
                llm,
                neg,
                oseed,
                browse_off,
                compose_off,
                ic_off,
                edit_off,
            )

        def on_lib_to_ic(selected, place, browse_off, compose_off, ic_off, edit_off):
            return _import_lib_ic(selected, place, browse_off, compose_off, ic_off, edit_off)

        def on_lib_to_edit(selected, browse_off, compose_off, ic_off, edit_off):
            return _import_lib_edit(selected, browse_off, compose_off, ic_off, edit_off)

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

        def on_reisolate(cutout, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            obj = session.doc.selected()
            if obj is None or obj.raw_image is None:
                session.status = "Select an object with a raw image."
                return pack(run_out=False)
            mode = _cutout_prefer(cutout)
            if mode == "none":
                with session._lock:
                    obj.image = obj.raw_image.convert("RGBA")
                    session.status = "Cutout cleared — full raw image."
                session.refresh_work()
                return pack_work_then_output()
            session.reisolate_selected(prefer=mode)
            return pack_work_then_output()

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
            return pack_work_then_output()

        def on_delete(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id in (None, "__bg__"):
                session.status = "Select an object layer to delete."
                return pack(run_out=False)
            session.delete_selected()
            return pack_work_then_output()

        def on_duplicate(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id in (None, "__bg__"):
                session.status = "Select an object layer to duplicate."
                return pack(run_out=False)
            session.duplicate_selected()
            return pack_work_then_output()

        def on_mute_prompt(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id in (None, "__bg__"):
                session.status = "Select an object layer to mute its prompt."
                return pack(run_out=False)
            session.toggle_selected_prompt()
            # Prompt composition changed — refresh layer cards + OUTPUT;
            # WORK pixels are unchanged.
            return pack_output_only(include_layers=True)

        def on_hide_layer(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id in (None, "__bg__"):
                session.status = "Select an object layer to hide or show."
                return pack(run_out=False)
            session.toggle_selected_visibility()
            # Pixels leave/return on WORK and OUTPUT; prompt concat unchanged.
            return pack_work_then_output()

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
                FAMILY_ALL,
                NO_STYLE,
                False,
                gr.update(value="", interactive=False),
                False,
                gr.update(value="", visible=False, interactive=False),
            )

        def on_reset_xform(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.selected_id is None:
                session.status = "Select a layer to reset."
                return pack(run_out=False)
            session.reset_selected_transform()
            return pack_work_then_output()

        def on_roll_seed():
            seed = session.roll_output_seed()
            mark_output_dirty(delay_override=0.05)
            return seed, session.status

        def on_opacity_live(opacity, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            obj = session.doc.selected()
            if obj is None:
                return pack(run_out=False)
            session.update_transform_by_id(obj.id, opacity=float(opacity))
            return pack_work_then_output()

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

        def _style_choices_for_family(fam: str | None) -> list[str]:
            if not session.styles:
                return [NO_STYLE]
            if not fam or fam == FAMILY_ALL:
                names = session.styles.names()
            else:
                names = session.styles.names([fam])
            return [NO_STYLE] + names

        def on_style_family(fam, current_style):
            choices = _style_choices_for_family(fam)
            value = current_style if current_style in choices else NO_STYLE
            s = session.apply_style_preset(None if value == NO_STYLE else value)
            return (
                gr.update(choices=choices, value=value),
                s.cfg,
                s.denoise,
                s.eta,
                s.negative_prompt,
                session.status,
            )

        def on_style(name):
            s = session.apply_style_preset(None if name == NO_STYLE else name)
            return s.cfg, s.denoise, s.eta, s.negative_prompt, session.status

        def on_params_now(den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            if session.doc.background is not None or session.doc.objects:
                # Params only affect OUTPUT refine — leave WORK alone.
                return pack_output_only()
            return pack(run_out=False)

        def on_size(preset, den, steps, cfg_v, eta_v, llm, neg, oseed):
            apply_settings(den, steps, cfg_v, eta_v, llm, neg, oseed)
            w, h = ASPECT_PRESETS.get(preset, ASPECT_PRESETS[DEFAULT_ASPECT])
            session.set_canvas_size(w, h)
            session.status = f"Canvas set to {w}×{h}."
            return pack_work_then_output()

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
                with infer_lock:
                    session.refine_final(
                        steps=int(steps),
                        denoise=float(strength),
                    )
                return pack(run_out=False)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Final refine failed")
                session.status = f"Final refine failed: {exc}"
                return pack(run_out=False)

        def on_seedvr_preset(preset_key):
            meta = SEEDVR2_PRESETS.get(str(preset_key)) or SEEDVR2_PRESETS["quality"]
            return (
                meta["model"],
                int(meta["blocks_to_swap"]),
                bool(meta["swap_io_components"]),
                str(meta["dit_offload_device"]),
                str(meta["vae_offload_device"]),
                bool(meta["compile_dit"]),
            )

        def on_export(
            model,
            color,
            input_noise,
            latent_noise,
            seed,
            blocks,
            swap_io,
            dit_offload,
            vae_offload,
            compile_dit,
        ):
            # Cancel any sleeping OUTPUT debounce while SeedVR2 owns the GPU.
            # Keep a sticky pending bit so mid-export edits refine after restore.
            with out_lock:
                if out_state["dirty"]:
                    out_state["pending"] = True
                out_state["dirty"] = False
                out_state["token"] += 1
            try:
                with infer_lock:
                    _img, path = session.export_final(
                        seedvr2_options={
                            "model": str(model),
                            "color_correction": str(color),
                            "input_noise_scale": float(input_noise),
                            "latent_noise_scale": float(latent_noise),
                            "seed": int(seed),
                            "blocks_to_swap": int(blocks),
                            "swap_io_components": bool(swap_io),
                            "dit_offload_device": str(dit_offload),
                            "vae_offload_device": str(vae_offload),
                            "compile_dit": bool(compile_dit),
                        }
                    )
                flush_pending_output()
                return str(path), str(path), session.status
            except Exception as exc:  # noqa: BLE001
                logger.exception("Export failed")
                flush_pending_output()
                return None, "", f"Export failed: {exc}"

        def poll_output(last_rev):
            """Refresh OUTPUT when output_rev advances (cheap no-op otherwise)."""
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
            # Never rewrite WORK/layers from the poll timer — that clobbers
            # in-progress (or just-finished) canvas drags with a stale scene.
            return (
                _output_jpeg_path(session.last_output),
                session.status,
                concat_val,
                session.output_rev,
                gr.skip(),
                gr.skip(),
                gr.update(value=llm_val, visible=llm_on),
            )

        load_btn.click(on_load, outputs=[status]).then(
            render_vram_html, outputs=[vram_html], show_progress="hidden"
        )
        free_btn.click(on_free, outputs=[status]).then(
            render_vram_html, outputs=[vram_html], show_progress="hidden"
        )
        main_tabs.select(
            on_main_tab,
            outputs=[status, lc_status, edit_status],
            show_progress="hidden",
        ).then(
            render_vram_html, outputs=[vram_html], show_progress="hidden"
        )
        action_out.change(
            on_action, inputs=[action_out, *settings_in], outputs=pack_out, show_progress="hidden"
        )
        add_obj_btn.click(on_add_layer, inputs=settings_in, outputs=pack_out, show_progress="hidden")
        roll_all_btn.click(
            on_roll_all,
            inputs=settings_in,
            outputs=[*pack_out, out_seed],
            show_progress="full",
        )
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
                style_family_dd,
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
        hide_layer_btn.click(
            on_hide_layer, inputs=settings_in, outputs=pack_out, show_progress="hidden"
        )
        duplicate_btn.click(
            on_duplicate, inputs=settings_in, outputs=pack_out, show_progress="hidden"
        )
        gen_btn.click(
            on_generate,
            inputs=[insp_prompt, iso_backdrop, gen_seed, cutout_chk, *settings_in],
            outputs=pack_out,
        ).then(
            on_generate_output,
            outputs=pack_out,
            show_progress="hidden",
        )
        insp_prompt.submit(
            on_generate,
            inputs=[insp_prompt, iso_backdrop, gen_seed, cutout_chk, *settings_in],
            outputs=pack_out,
        ).then(
            on_generate_output,
            outputs=pack_out,
            show_progress="hidden",
        )
        cutout_chk.change(
            on_cutout,
            inputs=[cutout_chk],
            outputs=[iso_backdrop, iso_white_btn, iso_black_btn, status],
            show_progress="hidden",
        )
        iso_white_btn.click(
            lambda: on_iso_swatch("White"),
            outputs=[iso_backdrop, iso_white_btn, iso_black_btn, status],
            show_progress="hidden",
        )
        iso_black_btn.click(
            lambda: on_iso_swatch("Black"),
            outputs=[iso_backdrop, iso_white_btn, iso_black_btn, status],
            show_progress="hidden",
        )
        for obj_comp in (obj_scale, obj_rotation):
            obj_comp.submit(
                on_obj_transform,
                inputs=[
                    obj_scale, obj_rotation, obj_flip_x, obj_flip_y, *settings_in,
                ],
                outputs=pack_out,
                show_progress="hidden",
            )
            obj_comp.blur(
                on_obj_transform,
                inputs=[
                    obj_scale, obj_rotation, obj_flip_x, obj_flip_y, *settings_in,
                ],
                outputs=pack_out,
                show_progress="hidden",
            )
        for obj_flip in (obj_flip_x, obj_flip_y):
            obj_flip.change(
                on_obj_transform,
                inputs=[
                    obj_scale, obj_rotation, obj_flip_x, obj_flip_y, *settings_in,
                ],
                outputs=pack_out,
                show_progress="hidden",
            )
        insp_feather.release(
            on_feather,
            inputs=[insp_feather, *settings_in],
            outputs=pack_out,
            show_progress="hidden",
        )
        insp_blend.change(
            on_blend_mode,
            inputs=[insp_blend, *settings_in],
            outputs=pack_out,
            show_progress="hidden",
        )
        for bg_comp in (bg_scale, bg_rotation, bg_offset_x, bg_offset_y):
            bg_comp.submit(
                on_bg_transform,
                inputs=[
                    bg_scale, bg_rotation, bg_offset_x, bg_offset_y,
                    bg_flip_x, bg_flip_y, *settings_in,
                ],
                outputs=pack_out,
                show_progress="hidden",
            )
            bg_comp.blur(
                on_bg_transform,
                inputs=[
                    bg_scale, bg_rotation, bg_offset_x, bg_offset_y,
                    bg_flip_x, bg_flip_y, *settings_in,
                ],
                outputs=pack_out,
                show_progress="hidden",
            )
        for bg_flip in (bg_flip_x, bg_flip_y):
            bg_flip.change(
                on_bg_transform,
                inputs=[
                    bg_scale, bg_rotation, bg_offset_x, bg_offset_y,
                    bg_flip_x, bg_flip_y, *settings_in,
                ],
                outputs=pack_out,
                show_progress="hidden",
            )
        insp_prompt.blur(
            on_prompt_edit, inputs=[insp_prompt], outputs=[layers_html], show_progress="hidden"
        )
        import_btn.click(
            on_import,
            inputs=[import_img, cutout_mode, insp_prompt, *settings_in],
            outputs=pack_out,
        )
        lib_event_out = [*pack_out, *lc_pack_out, *edit_pack_out, *lib_view_out]
        lib_off_in = [lib_browse_off, lib_compose_off, lib_ic_off, lib_edit_off]
        lib_action.change(
            on_lib_action,
            inputs=[
                lib_action,
                lib_selected,
                *lib_off_in,
                cutout_mode,
                insp_prompt,
                *settings_in,
                _lib_ui["ic_place"],
                _lc_ui["import_place"],
            ],
            outputs=lib_event_out,
            show_progress="hidden",
        )
        lib_save_btn.click(
            on_compose_save_lib,
            inputs=[lib_save_name, *lib_off_in, lib_selected],
            outputs=[status, *lib_view_out],
            show_progress="hidden",
        )
        _lc_ui["save_btn"].click(
            on_ic_save_lib,
            inputs=[_lc_ui["save_name"], *lib_off_in, lib_selected],
            outputs=[lc_status, *lib_view_out],
            show_progress="hidden",
        )
        _edit_ui["save_btn"].click(
            on_edit_save_lib,
            inputs=[_edit_ui["save_name"], *lib_off_in, lib_selected],
            outputs=[edit_status, *lib_view_out],
            show_progress="hidden",
        )
        _lib_ui["prev"].click(
            lambda *a: on_lib_page("browse", -1, *a),
            inputs=[*lib_off_in, lib_selected],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        _lib_ui["next"].click(
            lambda *a: on_lib_page("browse", 1, *a),
            inputs=[*lib_off_in, lib_selected],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        compose_lib_prev.click(
            lambda *a: on_lib_page("compose", -1, *a),
            inputs=[*lib_off_in, lib_selected],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        compose_lib_next.click(
            lambda *a: on_lib_page("compose", 1, *a),
            inputs=[*lib_off_in, lib_selected],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        _lc_ui["picker_prev"].click(
            lambda *a: on_lib_page("infinite", -1, *a),
            inputs=[*lib_off_in, lib_selected],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        _lc_ui["picker_next"].click(
            lambda *a: on_lib_page("infinite", 1, *a),
            inputs=[*lib_off_in, lib_selected],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        _edit_ui["picker_prev"].click(
            lambda *a: on_lib_page("edit", -1, *a),
            inputs=[*lib_off_in, lib_selected],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        _edit_ui["picker_next"].click(
            lambda *a: on_lib_page("edit", 1, *a),
            inputs=[*lib_off_in, lib_selected],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        _lib_ui["rename"].click(
            on_lib_rename,
            inputs=[_lib_ui["name"], lib_selected, *lib_off_in],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        _lib_ui["delete"].click(
            on_lib_delete,
            inputs=[lib_selected, *lib_off_in],
            outputs=lib_view_out,
            show_progress="hidden",
        )
        _lib_ui["to_compose"].click(
            on_lib_to_compose,
            inputs=[
                lib_selected,
                cutout_mode,
                insp_prompt,
                *settings_in,
                *lib_off_in,
            ],
            outputs=lib_event_out,
        )
        _lib_ui["to_ic"].click(
            on_lib_to_ic,
            inputs=[lib_selected, _lib_ui["ic_place"], *lib_off_in],
            outputs=lib_event_out,
        )
        _lib_ui["to_edit"].click(
            on_lib_to_edit,
            inputs=[lib_selected, *lib_off_in],
            outputs=lib_event_out,
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
        reisolate_btn.click(
            on_reisolate,
            inputs=[cutout_mode, *settings_in],
            outputs=pack_out,
        )
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
        style_family_dd.change(
            on_style_family,
            inputs=[style_family_dd, style_dd],
            outputs=[style_dd, cfg, denoise, eta, neg_prompt, status],
            show_progress="hidden",
        ).then(
            on_params_now, inputs=settings_in, outputs=pack_out, show_progress="hidden",
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

        def on_quality_mode(mode: str):
            s = session.apply_quality_mode(mode)
            key = normalize_quality_mode(mode)
            packed = pack_output_only()
            return (
                gr.update(variant="primary" if key == "fast" else "secondary"),
                gr.update(variant="primary" if key == "quality" else "secondary"),
                s.denoise,
                s.steps,
                *packed,
            )

        for button, mode in ((fast_btn, "fast"), (quality_btn, "quality")):
            button.click(
                partial(on_quality_mode, mode),
                outputs=[fast_btn, quality_btn, denoise, out_steps, *pack_out],
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
        seedvr_preset.change(
            on_seedvr_preset,
            inputs=[seedvr_preset],
            outputs=[
                seedvr_model,
                seedvr_blocks,
                seedvr_swap_io,
                seedvr_dit_offload,
                seedvr_vae_offload,
                seedvr_compile,
            ],
            show_progress="hidden",
        )
        export_btn.click(
            on_export,
            inputs=[
                seedvr_model,
                seedvr_color,
                seedvr_input_noise,
                seedvr_latent_noise,
                seedvr_seed,
                seedvr_blocks,
                seedvr_swap_io,
                seedvr_dit_offload,
                seedvr_vae_offload,
                seedvr_compile,
            ],
            outputs=[export_image, export_path, status],
        )

        def _flipbook_seed_pack(mode: str):
            sel = "lock" if mode != "random" else "random"
            return (
                sel,
                gr.update(variant="primary" if sel == "lock" else "secondary"),
                gr.update(variant="primary" if sel == "random" else "secondary"),
            )

        def on_flipbook_lock():
            return (*_flipbook_seed_pack("lock"), session.status)

        def on_flipbook_dice():
            return (*_flipbook_seed_pack("random"), session.status)

        def _flipbook_style_count_update(fams):
            selected = [f for f in (fams or []) if f]
            n = len(session.styles.names(selected)) if session.styles and selected else 0
            return gr.update(
                label="Styles (0 = all in checked families)",
                maximum=max(n, 1),
                value=min(32, n) if n else 0,
                minimum=0,
            )

        def on_flipbook_families(fams):
            return _flipbook_style_count_update(fams)

        def on_flipbook_mode(mode, fams):
            if str(mode or "").strip().lower().startswith("seed"):
                return (
                    gr.update(visible=False),
                    gr.update(
                        label="Frames (incl. current OUTPUT)",
                        minimum=2,
                        maximum=128,
                        value=32,
                    ),
                    (
                        '<p class="xwave-flipbook-seed-hint">'
                        "🔒 seed+1,+2… · 🎲 random seed per frame</p>"
                    ),
                    gr.update(value="Run seed flipbook"),
                )
            return (
                gr.update(visible=True),
                _flipbook_style_count_update(fams),
                (
                    '<p class="xwave-flipbook-seed-hint">'
                    "🔒 same seed · 🎲 new seed per style</p>"
                ),
                gr.update(value="Run style flipbook"),
            )

        def on_flipbook(mode, count, fps, hold, seed_mode, fams):
            with infer_lock:
                path, msg = session.export_style_flipbook(
                    style_count=int(count or 0),
                    fps=int(fps or 30),
                    frames_per_image=int(hold or 8),
                    lock_seed=(str(seed_mode or "lock") != "random"),
                    families=list(fams or []),
                    mode=str(mode or "styles"),
                )
            video = str(path) if path is not None else None
            out_path = str(path) if path is not None else ""
            return video, out_path, msg

        flipbook_mode.change(
            on_flipbook_mode,
            inputs=[flipbook_mode, flipbook_families],
            outputs=[
                flipbook_families,
                flipbook_count,
                flipbook_seed_hint,
                flipbook_btn,
            ],
            show_progress="hidden",
        )
        flipbook_families.change(
            on_flipbook_families,
            inputs=[flipbook_families],
            outputs=[flipbook_count],
            show_progress="hidden",
        )
        flipbook_lock_btn.click(
            on_flipbook_lock,
            outputs=[flipbook_seed_mode, flipbook_lock_btn, flipbook_dice_btn, status],
            show_progress="hidden",
        )
        flipbook_dice_btn.click(
            on_flipbook_dice,
            outputs=[flipbook_seed_mode, flipbook_lock_btn, flipbook_dice_btn, status],
            show_progress="hidden",
        )
        flipbook_btn.click(
            on_flipbook,
            inputs=[
                flipbook_mode,
                flipbook_count,
                flipbook_fps,
                flipbook_hold,
                flipbook_seed_mode,
                flipbook_families,
            ],
            outputs=[flipbook_video, flipbook_path, status],
        )

        # 0.75s: less contention with concurrency=1 than 200ms; OUTPUT still
        # feels timely after a refine (poll already no-ops when rev unchanged).
        timer = gr.Timer(0.75)
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

        def on_work_grid(grid_type, grid_color):
            return _work_grid_state_html(grid_type, grid_color, user_set=True)

        work_grid_type.change(
            on_work_grid,
            inputs=[work_grid_type, work_grid_color],
            outputs=[work_grid_state],
            show_progress="hidden",
        )
        work_grid_color.change(
            on_work_grid,
            inputs=[work_grid_type, work_grid_color],
            outputs=[work_grid_state],
            show_progress="hidden",
        )

        demo.load(_init, outputs=[work_html, layers_html])

    return demo


def _launch_kwargs(demo: gr.Blocks, config: AppConfig) -> dict:
    kwargs: dict[str, Any] = {
        "server_name": str(config.get("server", "host", default="0.0.0.0")),
        "server_port": int(config.get("server", "port", default=7860)),
        "share": bool(config.get("server", "share", default=False)),
        "show_error": bool(config.get("server", "show_error", default=True)),
    }
    # WORK canvas + layer thumbs load previews from these dirs via /gradio_api/file=.
    workspace = config.path("paths", "workspace_dir", default="workspace").resolve()
    layers = config.path("paths", "layers_dir", default="workspace/layers").resolve()
    scene_cache = _scene_cache_dir(config)
    lc_cache = (workspace / "large_canvas_cache").resolve()
    lc_cache.mkdir(parents=True, exist_ok=True)
    exports = config.path("export", "output_dir", default="exports").resolve()
    exports.mkdir(parents=True, exist_ok=True)
    _OUTPUT_JPEG_DIR.mkdir(parents=True, exist_ok=True)
    library = config.path("paths", "library_dir", default="library").resolve()
    library.mkdir(parents=True, exist_ok=True)
    kwargs["allowed_paths"] = [
        str(workspace),
        str(layers),
        str(scene_cache),
        str(lc_cache),
        str(exports),
        str(library),
        str(_OUTPUT_JPEG_DIR.resolve()),
    ]
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
