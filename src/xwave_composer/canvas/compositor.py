"""Compose background + object layers into a single WORK image."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

if TYPE_CHECKING:
    from .layers import ObjectLayer, WorkDocument

# UI labels → internal keys (and canvas globalCompositeOperation where available)
BLEND_MODES: dict[str, str] = {
    "Normal": "normal",
    "Multiply": "multiply",
    "Screen": "screen",
    "Overlay": "overlay",
    "Soft Light": "soft_light",
    "Hard Light": "hard_light",
    "Add": "add",
    "Subtract": "subtract",
    "Difference": "difference",
    "Darken": "darken",
    "Lighten": "lighten",
}
BLEND_MODE_LABELS = list(BLEND_MODES.keys())
_BLEND_KEY_TO_LABEL = {v: k for k, v in BLEND_MODES.items()}

# Canvas 2D composite ops for live WORK preview
BLEND_TO_CANVAS: dict[str, str] = {
    "normal": "source-over",
    "multiply": "multiply",
    "screen": "screen",
    "overlay": "overlay",
    "soft_light": "soft-light",
    "hard_light": "hard-light",
    "add": "lighter",
    "subtract": "difference",  # closest canvas stand-in; OUTPUT uses true subtract
    "difference": "difference",
    "darken": "darken",
    "lighten": "lighten",
}


def _blank_rgba(width: int, height: int, color=(32, 32, 36, 255)) -> Image.Image:
    return Image.new("RGBA", (width, height), color)


def normalize_blend_mode(value: object, default: str = "normal") -> str:
    raw = str(value or "").strip()
    if raw in BLEND_MODES:
        return BLEND_MODES[raw]
    key = raw.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "source_over": "normal",
        "plus": "add",
        "lighter": "lighten",
        "darker": "darken",
        "softlight": "soft_light",
        "hardlight": "hard_light",
    }
    key = aliases.get(key, key)
    return key if key in BLEND_TO_CANVAS else default


def blend_mode_label(value: object) -> str:
    key = normalize_blend_mode(value)
    return _BLEND_KEY_TO_LABEL.get(key, "Normal")


def feather_alpha_inward(image: Image.Image, radius: float) -> Image.Image:
    """Shrink the alpha matte inward and soften the new edge.

    ``radius`` is in source pixels. 0 leaves the image unchanged.
    Cutouts erode the existing alpha edge; fully opaque images feather
    inward from the rectangular frame.
    """
    r = float(radius)
    if r <= 0.05 or image is None:
        return image
    src = image.convert("RGBA")
    alpha = src.getchannel("A")
    w, h = src.size
    extrema = alpha.getextrema()
    fully_opaque = extrema is not None and extrema[0] >= 250

    if fully_opaque:
        from PIL import ImageDraw

        mask = Image.new("L", (w, h), 0)
        inset = max(1, int(round(r)))
        if w <= inset * 2 or h <= inset * 2:
            soft = Image.new("L", (w, h), 0)
        else:
            draw = ImageDraw.Draw(mask)
            draw.rectangle(
                (inset, inset, w - 1 - inset, h - 1 - inset),
                fill=255,
            )
            soft = mask.filter(
                ImageFilter.GaussianBlur(radius=max(0.5, min(r, 64.0) * 0.45))
            )
    else:
        # Iterative MinFilter so large radii (up to 128) stay tractable.
        eroded = alpha
        remaining = max(1, int(round(r)))
        while remaining > 0:
            step = min(16, remaining)
            k = step * 2 + 1
            eroded = eroded.filter(ImageFilter.MinFilter(size=k))
            remaining -= step
        soft = eroded.filter(
            ImageFilter.GaussianBlur(radius=max(0.5, min(r, 64.0) * 0.45))
        )

    out = src.copy()
    out.putalpha(soft)
    return out


def _blend_rgb(base_rgb: Image.Image, over_rgb: Image.Image, mode: str) -> Image.Image:
    """Blend two RGB images with the named mode."""
    ops: dict[str, Callable[[Image.Image, Image.Image], Image.Image]] = {
        "multiply": ImageChops.multiply,
        "screen": ImageChops.screen,
        "overlay": ImageChops.overlay,
        "soft_light": ImageChops.soft_light,
        "hard_light": ImageChops.hard_light,
        "add": ImageChops.add,
        "subtract": ImageChops.subtract,
        "difference": ImageChops.difference,
        "darken": ImageChops.darker,
        "lighten": ImageChops.lighter,
    }
    op = ops.get(mode)
    if op is None:
        return over_rgb
    try:
        return op(base_rgb, over_rgb)
    except Exception:
        # Some Pillow builds omit soft/hard light; fall back sensibly.
        if mode in ("soft_light", "hard_light"):
            return ImageChops.overlay(base_rgb, over_rgb)
        return over_rgb


def composite_layer(
    base: Image.Image,
    overlay: Image.Image,
    blend_mode: str = "normal",
) -> Image.Image:
    """Composite ``overlay`` onto ``base`` with optional Photoshop-style blend."""
    mode = normalize_blend_mode(blend_mode)
    base_rgba = base.convert("RGBA")
    over_rgba = overlay.convert("RGBA")
    if mode == "normal":
        return Image.alpha_composite(base_rgba, over_rgba)

    base_rgb = base_rgba.convert("RGB")
    over_rgb = over_rgba.convert("RGB")
    blended_rgb = _blend_rgb(base_rgb, over_rgb, mode)
    blended = blended_rgb.convert("RGBA")
    blended.putalpha(over_rgba.getchannel("A"))
    return Image.alpha_composite(base_rgba, blended)


def transform_object_layer(
    layer: "ObjectLayer",
    canvas_w: int,
    canvas_h: int,
) -> Image.Image | None:
    """Return an RGBA image the size of the canvas with the object placed."""
    if layer.image is None or not layer.transform.visible:
        return None

    src = feather_alpha_inward(layer.image.convert("RGBA"), getattr(layer, "feather", 0.0))
    t = layer.transform

    # Scale (stretch supported via independent scale_x / scale_y)
    new_w = max(1, int(round(src.width * t.scale_x)))
    new_h = max(1, int(round(src.height * t.scale_y)))
    if (new_w, new_h) != src.size:
        resample = Image.Resampling.LANCZOS
        src = src.resize((new_w, new_h), resample)

    # Opacity
    if t.opacity < 0.999:
        alpha = src.split()[-1]
        alpha = ImageEnhance.Brightness(alpha).enhance(max(0.0, min(1.0, t.opacity)))
        src.putalpha(alpha)

    # Rotate around center; expand so corners are not clipped
    if abs(t.rotation) > 1e-3:
        src = src.rotate(
            -t.rotation,  # PIL rotates counter-clockwise for positive angles
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    # Place so layer center sits at (x, y). paste() clips off-canvas safely.
    left = int(round(t.x - src.width / 2))
    top = int(round(t.y - src.height / 2))
    canvas.paste(src, (left, top), src)
    return canvas


def transform_background(
    background: Image.Image,
    canvas_w: int,
    canvas_h: int,
    *,
    scale: float = 1.0,
    rotation: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> Image.Image:
    """Place a background image on a canvas-sized RGBA layer.

    Scale 1.0 with no flip/rotation/offset matches the previous fill-to-canvas
    stretch. Transform order matches the WORK canvas JS: scale, flip,
    then rotate around center, then apply X/Y offset from center.
    """
    src = background.convert("RGBA")
    if src.size != (canvas_w, canvas_h):
        src = src.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

    s = max(0.05, float(scale))
    new_w = max(1, int(round(canvas_w * s)))
    new_h = max(1, int(round(canvas_h * s)))
    if (new_w, new_h) != src.size:
        src = src.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if flip_x:
        src = src.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_y:
        src = src.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if abs(float(rotation)) > 1e-3:
        src = src.rotate(
            -float(rotation),
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

    canvas = _blank_rgba(canvas_w, canvas_h)
    left = int(round((canvas_w - src.width) / 2 + float(offset_x)))
    top = int(round((canvas_h - src.height) / 2 + float(offset_y)))
    canvas.paste(src, (left, top), src)
    return canvas


def compose_work_image(doc: "WorkDocument") -> Image.Image:
    """Flatten the document to an RGB image for display and OUTPUT init."""
    w, h = doc.width, doc.height

    if doc.background is not None:
        bg = transform_background(
            doc.background,
            w,
            h,
            scale=doc.bg_scale,
            rotation=doc.bg_rotation,
            offset_x=doc.bg_offset_x,
            offset_y=doc.bg_offset_y,
            flip_x=doc.bg_flip_x,
            flip_y=doc.bg_flip_y,
        )
    else:
        bg = _blank_rgba(w, h)

    composed = bg.copy()
    for obj in doc.objects:
        placed = transform_object_layer(obj, w, h)
        if placed is not None:
            composed = composite_layer(
                composed, placed, getattr(obj, "blend_mode", "normal")
            )

    return composed.convert("RGB")


def compose_work_rgba(doc: "WorkDocument") -> Image.Image:
    """Flatten with alpha preserved (background still opaque)."""
    return compose_work_image(doc).convert("RGBA")


def estimated_bbox(layer: "ObjectLayer") -> tuple[int, int, int, int] | None:
    """Axis-aligned bounding box of a transformed layer on the canvas."""
    if layer.image is None:
        return None
    t = layer.transform
    w = layer.image.width * t.scale_x
    h = layer.image.height * t.scale_y
    # Rotate bbox extremes
    rad = math.radians(t.rotation)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
    bw = w * cos_a + h * sin_a
    bh = w * sin_a + h * cos_a
    left = int(t.x - bw / 2)
    top = int(t.y - bh / 2)
    return left, top, left + int(bw), top + int(bh)
