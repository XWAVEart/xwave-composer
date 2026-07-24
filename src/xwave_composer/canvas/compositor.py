"""Compose background + object layers into a single WORK image."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PIL import Image, ImageEnhance

if TYPE_CHECKING:
    from .layers import ObjectLayer, WorkDocument


def _blank_rgba(width: int, height: int, color=(32, 32, 36, 255)) -> Image.Image:
    return Image.new("RGBA", (width, height), color)


def transform_object_layer(
    layer: "ObjectLayer",
    canvas_w: int,
    canvas_h: int,
) -> Image.Image | None:
    """Return an RGBA image the size of the canvas with the object placed."""
    if layer.image is None or not layer.transform.visible:
        return None

    src = layer.image.convert("RGBA")
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
            composed = Image.alpha_composite(composed, placed)

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
