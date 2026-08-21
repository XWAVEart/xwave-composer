"""Edit-tab document: pixel-aligned SAM2 cuts stacked over a punched Base."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

BASE_ID = "base"
MODE_OFF = "off"
MODE_INCLUDE = "include"
MODE_EXCLUDE = "exclude"
POINT_INCLUDE = 1
POINT_EXCLUDE = 0
OVERLAY_FILL = (94, 234, 212, 92)
INCLUDE_COLOR = (52, 211, 153)
EXCLUDE_COLOR = (248, 113, 113)
PREVIEW_BG = (16, 17, 20)


class IsolatorLike(Protocol):
    """SAM2-capable isolator (ObjectIsolator or a test double)."""

    sam2_ready: bool

    def load_sam2(self) -> str: ...

    def set_image(self, image: Image.Image) -> None: ...

    def reset_image(self) -> None: ...

    def predict_mask(
        self,
        image: Image.Image,
        points: list[tuple[float, float]],
        labels: list[int],
    ) -> np.ndarray: ...


@dataclass
class EditLayer:
    """One registered layer. Base is the leftover original; cuts are SAM2 commits."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = "Layer"
    kind: str = "cut"  # "base" | "cut"
    image: Image.Image | None = None
    mask: Image.Image | None = None
    visible: bool = True
    opacity: float = 1.0

    @property
    def is_base(self) -> bool:
        return self.kind == "base" or self.id == BASE_ID


@dataclass
class EditDocument:
    source: Image.Image | None = None
    source_name: str = ""
    layers: list[EditLayer] = field(default_factory=list)
    selected_id: str | None = None
    draft_points: list[tuple[float, float, int]] = field(default_factory=list)
    draft_mask: Image.Image | None = None

    def base(self) -> EditLayer | None:
        return self.layers[0] if self.layers else None

    def find(self, layer_id: str | None) -> EditLayer | None:
        key = str(layer_id or "")
        for layer in self.layers:
            if layer.id == key:
                return layer
        return None

    def selected(self) -> EditLayer | None:
        return self.find(self.selected_id)

    def cuts(self) -> list[EditLayer]:
        return [layer for layer in self.layers if not layer.is_base]


class EditSession:
    """In-memory edit document + SAM2 draft mask."""

    def __init__(self, isolator: Any | None = None):
        self.isolator = isolator
        self.doc = EditDocument()
        self.mode: str = MODE_INCLUDE
        self.status: str = "Load an image from the library or drop a file."
        self.hover_xy: tuple[float, float] | None = None
        self._cut_serial: int = 0
        self._lock = threading.RLock()
        self._history: list[tuple[str, Image.Image, Image.Image | None]] = []
        self._effect_busy = False

    @property
    def has_image(self) -> bool:
        return self.doc.source is not None

    def selected(self) -> EditLayer | None:
        return self.doc.selected()

    def load(self, image: Image.Image, name: str = "") -> None:
        if image is None:
            raise ValueError("No image to load.")
        rgb = image.convert("RGB")
        rgba = rgb.convert("RGBA")
        mask = Image.new("L", rgb.size, 255)
        base = EditLayer(
            id=BASE_ID,
            name="Base",
            kind="base",
            image=rgba,
            mask=mask,
        )
        with self._lock:
            self.doc = EditDocument(
                source=rgb,
                source_name=str(name or "").strip(),
                layers=[base],
                selected_id=BASE_ID,
            )
            self._cut_serial = 0
            self.mode = MODE_INCLUDE
            self.hover_xy = None
            self._history = []
            self.status = (
                f"Loaded {rgb.width}×{rgb.height}. Hover an object, then click to lift it."
            )
        isolator = self.isolator
        if isolator is not None:
            reset = getattr(isolator, "reset_image", None)
            if callable(reset):
                try:
                    reset()
                except Exception:  # noqa: BLE001
                    logger.debug("isolator reset_image failed", exc_info=True)

    def set_mode(self, mode: str) -> None:
        key = str(mode or MODE_OFF).strip().lower()
        if key in ("+", "include", "include (+)"):
            key = MODE_INCLUDE
        elif key in ("-", "exclude", "exclude (−)", "exclude (-)"):
            key = MODE_EXCLUDE
        elif key not in (MODE_OFF, MODE_INCLUDE, MODE_EXCLUDE):
            key = MODE_OFF
        with self._lock:
            self.mode = key
            if key == MODE_OFF:
                self.status = "Segment mode off — clicks ignored."
            elif key == MODE_INCLUDE:
                self.status = "Hover an object to preview the cut, then click to lift it onto a layer."
            else:
                self.status = "Exclude (−): hover a region, then click to trim the selected layer."

    def add_point(self, x: float, y: float) -> EditLayer | None:
        """Click the source image. Include cuts a new layer; exclude trims the selection."""
        with self._lock:
            if self.doc.source is None:
                self.status = "Load an image first."
                return None
            if self.mode == MODE_OFF:
                self.status = "Choose Include (+) or Exclude (−) to segment."
                return None
            w, h = self.doc.source.size
            px = float(min(max(x, 0.0), w - 1))
            py = float(min(max(y, 0.0), h - 1))
            exclude = self.mode == MODE_EXCLUDE
            if exclude:
                target = self.doc.selected()
                if target is None or target.is_base:
                    cuts = self.doc.cuts()
                    if not cuts:
                        self.status = "Cut an object first, then use Exclude to trim it."
                        return None
                    target = cuts[-1]
                    self.doc.selected_id = target.id
            # SAM2 still needs a positive click on the region to isolate.
            self.doc.draft_points = [(px, py, POINT_INCLUDE)]
        self._run_sam()
        if self.doc.draft_mask is None:
            return None
        if exclude:
            return self._subtract_draft_from_selected()
        return self.commit()

    def hover_at(self, x: float, y: float) -> None:
        """SAM2 preview under the cursor. Does not create a layer."""
        with self._lock:
            if self.doc.source is None or self.mode == MODE_OFF or self._effect_busy:
                self.doc.draft_mask = None
                self.hover_xy = None
                return
            w, h = self.doc.source.size
            px = float(min(max(x, 0.0), w - 1))
            py = float(min(max(y, 0.0), h - 1))
            if self.hover_xy is not None and self.doc.draft_mask is not None:
                dx = px - self.hover_xy[0]
                dy = py - self.hover_xy[1]
                if dx * dx + dy * dy < 9.0:
                    return
            self.doc.draft_points = [(px, py, POINT_INCLUDE)]
            self.hover_xy = (px, py)
        self._run_sam()
        if self.doc.draft_mask is not None:
            self.status = "Click to lift this object onto its own layer."

    def cut_at(self, x: float, y: float) -> EditLayer | None:
        """Click: reuse the hover mask when the cursor is still on it."""
        with self._lock:
            if self.doc.source is None:
                self.status = "Load an image first."
                return None
            w, h = self.doc.source.size
            px = float(min(max(x, 0.0), w - 1))
            py = float(min(max(y, 0.0), h - 1))
            reuse = False
            if self.doc.draft_mask is not None and self.hover_xy is not None:
                dx = px - self.hover_xy[0]
                dy = py - self.hover_xy[1]
                reuse = (dx * dx + dy * dy) < (32.0 * 32.0)
        if reuse:
            if self.mode == MODE_EXCLUDE:
                return self._subtract_draft_from_selected()
            return self.commit()
        return self.add_point(x, y)

    def clear_hover(self) -> None:
        with self._lock:
            self.doc.draft_points = []
            self.doc.draft_mask = None
            self.hover_xy = None

    def glow_overlay(self) -> Image.Image | None:
        """Teal fill + edge ring for the live hover mask."""
        with self._lock:
            mask = self.doc.draft_mask
        if mask is None:
            return None
        matte = mask.convert("L")
        fill_a = matte.point(lambda p: 72 if p > 127 else 0)
        ring = ImageChops.subtract(
            matte.filter(ImageFilter.MaxFilter(5)),
            matte.filter(ImageFilter.MinFilter(3)),
        )
        ring_a = ring.point(lambda p: 230 if p > 12 else 0)
        fill = Image.new("RGBA", matte.size, (94, 234, 212, 0))
        fill.putalpha(fill_a)
        edge = Image.new("RGBA", matte.size, (190, 255, 245, 0))
        edge.putalpha(ring_a)
        return Image.alpha_composite(fill, edge)

    def undo_point(self) -> None:
        """Undo the last effect apply, or the last cut / leftover draft."""
        with self._lock:
            if self._history:
                layer_id, image, mask = self._history.pop()
                layer = self.doc.find(layer_id)
                if layer is not None:
                    layer.image = image
                    layer.mask = mask
                    self.status = f"Undid last effect on {layer.name}."
                    return
            if self.doc.draft_points or self.doc.draft_mask is not None:
                self.doc.draft_points = []
                self.doc.draft_mask = None
                self.status = "Draft cleared."
                return
            cuts = self.doc.cuts()
            if not cuts:
                self.status = "Nothing to undo."
                return
            layer_id = cuts[-1].id
        self.delete_layer(layer_id)

    def rest_composite(self, skip_id: str | None = None) -> Image.Image:
        """RGB flatten of every visible layer except ``skip_id``."""
        with self._lock:
            source = self.doc.source
            if source is None:
                return Image.new("RGB", (1024, 1024), PREVIEW_BG)
            out = Image.new("RGBA", source.size, (0, 0, 0, 0))
            skip = str(skip_id or "")
            for layer in self.doc.layers:
                if layer.id == skip or not layer.visible or layer.image is None:
                    continue
                im = layer.image.convert("RGBA")
                if im.size != source.size:
                    im = im.resize(source.size, Image.Resampling.NEAREST)
                opacity = float(layer.opacity)
                if opacity < 0.999:
                    alpha = im.getchannel("A")
                    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
                    im.putalpha(alpha)
                out = Image.alpha_composite(out, im)
            bg = Image.new("RGB", out.size, PREVIEW_BG)
            bg.paste(out, mask=out.getchannel("A"))
            return bg

    def resolve_secondary(
        self,
        secondary_key: str | None,
        extra: Image.Image | None = None,
        skip_id: str | None = None,
    ) -> Image.Image | None:
        key = str(secondary_key or "rest").strip()
        if key in ("", "rest"):
            return self.rest_composite(skip_id)
        if key in ("file", "upload") and extra is not None:
            return extra.convert("RGB")
        layer = self.doc.find(key)
        if layer is None or layer.image is None:
            return self.rest_composite(skip_id)
        return layer.image.convert("RGB")

    def apply_effect(
        self,
        effect_id: str,
        params: dict[str, Any] | None = None,
        *,
        secondary_key: str | None = "rest",
        secondary_image: Image.Image | None = None,
        warp_alpha: bool = False,
    ) -> EditLayer | None:
        """Run a CPU glitch effect on the selected layer."""
        from xwave_composer.glitch.registry import get_effect, run_effect

        self._effect_busy = True
        try:
            with self._lock:
                layer = self.doc.selected()
                if layer is None or layer.image is None:
                    self.status = "Select a layer first."
                    return None
                spec = get_effect(effect_id)
                snapshot = (
                    layer.id,
                    layer.image.copy(),
                    layer.mask.copy() if layer.mask is not None else None,
                )
            secondary = None
            if spec.two_image:
                secondary = self.resolve_secondary(
                    secondary_key, extra=secondary_image, skip_id=layer.id
                )
                if secondary is not None and secondary.size != layer.image.size:
                    secondary = secondary.resize(layer.image.size, Image.Resampling.BICUBIC)
            rgb = layer.image.convert("RGB")
            result = run_effect(spec.id, rgb, params or {}, secondary)
            result = result.convert("RGB")
            if result.size != layer.image.size:
                result = result.resize(layer.image.size, Image.Resampling.NEAREST)
            alpha = layer.image.convert("RGBA").getchannel("A")
            if warp_alpha and spec.warp:
                gray = Image.merge("RGB", (alpha, alpha, alpha))
                warped = run_effect(spec.id, gray, params or {}, secondary).convert("L")
                if warped.size != alpha.size:
                    warped = warped.resize(alpha.size, Image.Resampling.NEAREST)
                alpha = warped
            out = result.convert("RGBA")
            out.putalpha(alpha)
            with self._lock:
                self._history.append(snapshot)
                if len(self._history) > 12:
                    self._history = self._history[-12:]
                layer.image = out
                if warp_alpha and spec.warp:
                    layer.mask = alpha
                self.status = f"Applied {spec.label} to {layer.name}."
            return layer
        except KeyError:
            self.status = f"Unknown effect: {effect_id}"
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Edit effect failed")
            self.status = f"Effect failed: {exc}"
            return None
        finally:
            self._effect_busy = False

    def clear_points(self) -> None:
        with self._lock:
            self.doc.draft_points = []
            self.doc.draft_mask = None
            self.status = "Draft points cleared."

    def commit(self) -> EditLayer | None:
        with self._lock:
            if self.doc.source is None:
                self.status = "Load an image first."
                return None
            mask = self.doc.draft_mask
            if mask is None:
                self.status = "Click include/exclude points first, then Commit layer."
                return None
            mask_l = mask.convert("L")
            if not np.any(np.array(mask_l) > 127):
                self.status = "Draft mask is empty — add include clicks."
                return None
            self._cut_serial += 1
            rgba = self.doc.source.convert("RGBA")
            rgba.putalpha(mask_l)
            layer = EditLayer(
                name=f"Cut {self._cut_serial}",
                kind="cut",
                image=rgba,
                mask=mask_l,
            )
            self.doc.layers.append(layer)
            self.doc.selected_id = layer.id
            self.doc.draft_points = []
            self.doc.draft_mask = None
            self.hover_xy = None
            self._rebuild_base_alpha()
            self.status = f"{layer.name} cut from the image. Click another object to add a layer."
            return layer

    def _subtract_draft_from_selected(self) -> EditLayer | None:
        with self._lock:
            layer = self.doc.selected()
            mask = self.doc.draft_mask
            source = self.doc.source
            if layer is None or layer.is_base or layer.mask is None or mask is None or source is None:
                self.doc.draft_points = []
                self.doc.draft_mask = None
                if "SAM2" not in (self.status or ""):
                    self.status = "Exclude needs a selected cut layer and a SAM2 mask."
                return None
            cut_np = np.array(layer.mask.convert("L"))
            sub_np = np.array(mask.convert("L"))
            if sub_np.shape != cut_np.shape:
                mask = mask.convert("L").resize(layer.mask.size, Image.Resampling.NEAREST)
                sub_np = np.array(mask)
            cut_np[sub_np > 127] = 0
            layer.mask = Image.fromarray(cut_np, mode="L")
            rgba = source.convert("RGBA")
            rgba.putalpha(layer.mask)
            layer.image = rgba
            self.doc.draft_points = []
            self.doc.draft_mask = None
            self.hover_xy = None
            self._rebuild_base_alpha()
            self.status = f"Trimmed {layer.name}."
            return layer

    def delete_layer(self, layer_id: str) -> None:
        with self._lock:
            layer = self.doc.find(layer_id)
            if layer is None:
                self.status = "Select a cut layer to delete."
                return
            if layer.is_base:
                self.status = "Cannot delete Base."
                return
            self.doc.layers = [item for item in self.doc.layers if item.id != layer.id]
            self._rebuild_base_alpha()
            self.doc.selected_id = (
                self.doc.layers[-1].id if self.doc.layers else BASE_ID
            )
            self.status = f"Deleted {layer.name}."

    def select(self, layer_id: str) -> None:
        with self._lock:
            layer = self.doc.find(layer_id)
            if layer is None:
                return
            self.doc.selected_id = layer.id
            self.status = f"Selected {layer.name}."

    def set_visible(self, layer_id: str, visible: bool) -> None:
        with self._lock:
            layer = self.doc.find(layer_id)
            if layer is None:
                return
            layer.visible = bool(visible)
            state = "visible" if layer.visible else "hidden"
            self.status = f"{layer.name} {state}."

    def set_opacity(self, layer_id: str, opacity: float) -> None:
        with self._lock:
            layer = self.doc.find(layer_id)
            if layer is None:
                return
            layer.opacity = float(min(max(opacity, 0.0), 1.0))

    def set_name(self, layer_id: str, name: str) -> None:
        with self._lock:
            layer = self.doc.find(layer_id)
            if layer is None:
                return
            text = " ".join(str(name or "").strip().split())[:40]
            if not text:
                return
            layer.name = text

    def reorder_cuts(self, bottom_to_top_ids: list[str]) -> None:
        """Re-order cut layers. Base stays at index 0. ``ids`` are bottom→top."""
        with self._lock:
            base = self.doc.base()
            if base is None:
                return
            by_id = {layer.id: layer for layer in self.doc.cuts()}
            ordered: list[EditLayer] = []
            seen: set[str] = set()
            for lid in bottom_to_top_ids:
                layer = by_id.get(str(lid))
                if layer is None or layer.id in seen:
                    continue
                ordered.append(layer)
                seen.add(layer.id)
            for layer in self.doc.cuts():
                if layer.id not in seen:
                    ordered.append(layer)
            self.doc.layers = [base] + ordered

    def composite(self) -> Image.Image:
        with self._lock:
            source = self.doc.source
            if source is None:
                return Image.new("RGBA", (1024, 1024), (*PREVIEW_BG, 255))
            out = Image.new("RGBA", source.size, (0, 0, 0, 0))
            for layer in self.doc.layers:
                if not layer.visible or layer.image is None:
                    continue
                im = layer.image.convert("RGBA")
                if im.size != source.size:
                    im = im.resize(source.size, Image.Resampling.NEAREST)
                opacity = float(layer.opacity)
                if opacity < 0.999:
                    alpha = im.getchannel("A")
                    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
                    im.putalpha(alpha)
                out = Image.alpha_composite(out, im)
            return out

    def flatten(self) -> Image.Image:
        """RGB composite on a dark backdrop (holes show as near-black)."""
        comp = self.composite()
        bg = Image.new("RGB", comp.size, PREVIEW_BG)
        bg.paste(comp, mask=comp.getchannel("A"))
        return bg

    def preview(self) -> Image.Image:
        """Flattened composite plus draft mask overlay and point markers."""
        rgb = self.flatten()
        with self._lock:
            mask = self.doc.draft_mask
            points = list(self.doc.draft_points)
        canvas = rgb.convert("RGBA")
        if mask is not None:
            tint = Image.new("RGBA", canvas.size, OVERLAY_FILL)
            alpha = mask.convert("L").point(lambda p: 92 if p > 127 else 0)
            tint.putalpha(alpha)
            canvas = Image.alpha_composite(canvas, tint)
        if points:
            draw = ImageDraw.Draw(canvas)
            radius = max(5, min(canvas.size) // 120)
            for x, y, label in points:
                color = INCLUDE_COLOR if int(label) == POINT_INCLUDE else EXCLUDE_COLOR
                box = [x - radius, y - radius, x + radius, y + radius]
                draw.ellipse(box, outline=color + (255,), width=max(2, radius // 3))
                if int(label) == POINT_INCLUDE:
                    draw.line(
                        [(x - radius, y), (x + radius, y)],
                        fill=color + (255,),
                        width=2,
                    )
                    draw.line(
                        [(x, y - radius), (x, y + radius)],
                        fill=color + (255,),
                        width=2,
                    )
                else:
                    draw.line(
                        [(x - radius, y), (x + radius, y)],
                        fill=color + (255,),
                        width=2,
                    )
        return canvas.convert("RGB")

    def _rebuild_base_alpha(self) -> None:
        """Base alpha = original opaque minus the union of remaining cuts."""
        source = self.doc.source
        base = self.doc.base()
        if source is None or base is None:
            return
        alpha = np.full((source.height, source.width), 255, dtype=np.uint8)
        for layer in self.doc.cuts():
            if layer.mask is None:
                continue
            mask_np = np.array(layer.mask.convert("L"))
            if mask_np.shape != alpha.shape:
                mask_img = layer.mask.convert("L").resize(
                    source.size, Image.Resampling.NEAREST
                )
                mask_np = np.array(mask_img)
            alpha[mask_np > 127] = 0
        base.mask = Image.fromarray(alpha, mode="L")
        rgba = source.convert("RGBA")
        rgba.putalpha(base.mask)
        base.image = rgba

    def _run_sam(self) -> None:
        isolator = self.isolator
        with self._lock:
            source = self.doc.source
            points = list(self.doc.draft_points)
        if source is None:
            return
        if isolator is None:
            self.status = "SAM2 isolator is not available."
            return
        coords = [(p[0], p[1]) for p in points]
        labels = [int(p[2]) for p in points]
        if not any(lab == POINT_INCLUDE for lab in labels):
            with self._lock:
                self.doc.draft_mask = None
                self.status = "Add an include (+) click."
            return
        try:
            if not getattr(isolator, "sam2_ready", True):
                self.status = "Loading SAM2…"
                load = getattr(isolator, "load_sam2", None)
                if callable(load):
                    load()
            mask_np = isolator.predict_mask(source, coords, labels)
        except Exception as exc:  # noqa: BLE001
            logger.exception("SAM2 predict failed")
            self.status = f"SAM2 failed: {exc}"
            return
        mask_img = _mask_image(mask_np, source.size)
        with self._lock:
            self.doc.draft_mask = mask_img
            self.status = "Hover preview ready."


def _mask_image(mask_np: np.ndarray, size: tuple[int, int]) -> Image.Image:
    arr = np.asarray(mask_np)
    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.dtype != np.uint8:
        arr = ((arr > 0.5) * 255).astype(np.uint8)
    elif arr.max() <= 1:
        arr = arr * 255
    img = Image.fromarray(arr, mode="L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.NEAREST)
    return img
