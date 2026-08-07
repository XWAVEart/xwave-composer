"""Thin facade over Infinite Canvas session + Hyper pipe (keeps Gradio thin)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from xwave_composer.pipeline.large_canvas import LargeCanvasSession, StampRect
from xwave_composer.pipeline.region_fill import (
    ensure_blueprint,
    fill_region,
    tiled_refine,
)

if TYPE_CHECKING:
    from xwave_composer.models.sdxl_hyper import SDXLHyperPipeline


class CanvasEngine:
    """Session-scoped helpers for generate / refine / blueprint."""

    def __init__(self, session: LargeCanvasSession, sdxl: "SDXLHyperPipeline"):
        self.session = session
        self.sdxl = sdxl

    def generate_region(
        self,
        stamp: StampRect,
        prompt: str,
        negative_prompt: str = "",
        **kwargs,
    ) -> tuple[Image.Image, Image.Image, str]:
        with self.session._lock:
            canvas = self.session.image.copy()
            occupied = self.session.occupied.copy()
            bp = self.session.blueprint
        bp = ensure_blueprint(
            canvas,
            occupied,
            self.sdxl,
            prompt,
            negative_prompt,
            steps=kwargs.get("steps", self.session.steps),
            cfg=kwargs.get("cfg", self.session.cfg),
            eta=kwargs.get("eta", self.session.eta),
            seed=kwargs.get("seed"),
            existing=bp,
        )
        if bp is not None:
            self.session.blueprint = bp
        return fill_region(
            canvas,
            occupied,
            stamp,
            self.sdxl,
            prompt,
            negative_prompt,
            blueprint=bp,
            **kwargs,
        )

    def refine(self, prompt: str, negative_prompt: str = "", **kwargs) -> tuple[Image.Image, str]:
        with self.session._lock:
            canvas = self.session.image.copy()
        return tiled_refine(canvas, self.sdxl, prompt, negative_prompt, **kwargs)
