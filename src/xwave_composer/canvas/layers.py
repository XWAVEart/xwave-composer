"""Layer data model for the WORK canvas."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class LayerTransform:
    """Affine-ish transform for an object on the WORK canvas.

    Position is the center of the layer in canvas pixels.
    scale_x / scale_y stretch relative to the source image size.
    rotation is degrees, clockwise-positive in PIL.
    """

    x: float = 512.0
    y: float = 512.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    visible: bool = True
    opacity: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "rotation": self.rotation,
            "visible": self.visible,
            "opacity": self.opacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerTransform":
        return cls(
            x=float(data.get("x", 512.0)),
            y=float(data.get("y", 512.0)),
            scale_x=float(data.get("scale_x", 1.0)),
            scale_y=float(data.get("scale_y", 1.0)),
            rotation=float(data.get("rotation", 0.0)),
            visible=bool(data.get("visible", True)),
            opacity=float(data.get("opacity", 1.0)),
        )


@dataclass
class ObjectLayer:
    """One isolated object layer with prompt and transform."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = "Object"
    prompt: str = ""
    # Muting excludes this text from OUTPUT prompt construction while leaving
    # the visual layer untouched on the WORK canvas.
    prompt_enabled: bool = True
    isolation_prompt: str = "isolated object on plain white background"
    # Soften cutout edges by shrinking/blurring alpha inward (pixels).
    feather: float = 0.0
    # Photoshop-style blend when compositing onto the WORK stack.
    blend_mode: str = "normal"
    # RGBA image with transparent background
    image: Image.Image | None = None
    # Raw generation before isolation (for SAM2 click re-run)
    raw_image: Image.Image | None = None
    transform: LayerTransform = field(default_factory=LayerTransform)
    path: Path | None = None

    def label(self) -> str:
        short = (self.prompt[:40] + "…") if len(self.prompt) > 40 else self.prompt
        return f"{self.name} [{self.id}] {short}".strip()


@dataclass
class WorkDocument:
    """Full composition state: background + ordered object layers."""

    width: int = 1024
    height: int = 1024
    background: Image.Image | None = None
    background_prompt: str = ""
    # Background placement on the canvas (center-anchored). Scale 1.0 matches
    # the previous fill-to-canvas size; flip/rotation apply around center.
    bg_scale: float = 1.0
    bg_rotation: float = 0.0
    bg_offset_x: float = 0.0
    bg_offset_y: float = 0.0
    bg_flip_x: bool = False
    bg_flip_y: bool = False
    objects: list[ObjectLayer] = field(default_factory=list)
    selected_id: str | None = None

    def selected(self) -> ObjectLayer | None:
        if not self.selected_id:
            return None
        for obj in self.objects:
            if obj.id == self.selected_id:
                return obj
        return None

    def select(self, layer_id: str | None) -> None:
        self.selected_id = layer_id

    def add_object(self, layer: ObjectLayer) -> ObjectLayer:
        # Place new objects near center by default
        layer.transform.x = self.width / 2
        layer.transform.y = self.height / 2
        self.objects.append(layer)
        self.selected_id = layer.id
        return layer

    def remove_object(self, layer_id: str) -> bool:
        before = len(self.objects)
        self.objects = [o for o in self.objects if o.id != layer_id]
        if self.selected_id == layer_id:
            self.selected_id = self.objects[-1].id if self.objects else None
        return len(self.objects) < before

    def reorder(self, layer_id: str, direction: str) -> None:
        """Move layer up (later draw = on top) or down (earlier draw)."""
        ids = [o.id for o in self.objects]
        if layer_id not in ids:
            return
        i = ids.index(layer_id)
        if direction == "up" and i < len(self.objects) - 1:
            self.objects[i], self.objects[i + 1] = self.objects[i + 1], self.objects[i]
        elif direction == "down" and i > 0:
            self.objects[i], self.objects[i - 1] = self.objects[i - 1], self.objects[i]
        elif direction == "top":
            layer = self.objects.pop(i)
            self.objects.append(layer)
        elif direction == "bottom":
            layer = self.objects.pop(i)
            self.objects.insert(0, layer)

    def reorder_by_ids(self, ordered_ids: list[str]) -> None:
        """Set draw order from bottom→top by id list (unknown ids ignored)."""
        by_id = {o.id: o for o in self.objects}
        new_list: list[ObjectLayer] = []
        for lid in ordered_ids:
            if lid in by_id:
                new_list.append(by_id.pop(lid))
        # Append any missing at end
        new_list.extend(by_id.values())
        self.objects = new_list

    def find_by_id(self, layer_id: str) -> ObjectLayer | None:
        for o in self.objects:
            if o.id == layer_id:
                return o
        return None

    def rename(self, layer_id: str, name: str) -> None:
        obj = self.find_by_id(layer_id)
        if obj is not None and name.strip():
            obj.name = name.strip()

    def layer_choices(self) -> list[str]:
        return [o.label() for o in self.objects]

    def find_by_label(self, label: str) -> ObjectLayer | None:
        for o in self.objects:
            if o.label() == label or o.id in label:
                return o
        return None

    def set_size(self, width: int, height: int) -> None:
        self.width = max(256, int(width))
        self.height = max(256, int(height))
