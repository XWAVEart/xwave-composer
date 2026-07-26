"""Object isolation: SAM2 (preferred) with rembg fallback."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, hard_release

logger = logging.getLogger(__name__)


class ObjectIsolator:
    """Remove background from a generated object image.

    Preferred path: SAM2 with optional click points.
    Fallback: rembg (u2net / similar).
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.device
        self._sam2_predictor: Any = None
        self._rembg_session: Any = None
        self.backend_in_use: str | None = None
        self._sam2_error: str | None = None

    @property
    def sam2_ready(self) -> bool:
        return self._sam2_predictor is not None

    def load_sam2(self) -> str:
        """Try to load SAM2. Returns status message."""
        if self._sam2_predictor is not None:
            return "SAM2 already loaded."

        model_id = str(
            self.config.get("isolation", "sam2", "model_id", default="facebook/sam2.1-hiera-large")
        )
        tf_error: str | None = None
        try:
            # Path A: transformers SAM2
            from transformers import Sam2Model, Sam2Processor

            processor = Sam2Processor.from_pretrained(model_id)
            dtype = (
                torch.bfloat16
                if str(self.device).startswith("cuda") and torch.cuda.is_available()
                else torch.float32
            )
            model = Sam2Model.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
            model.eval()
            self._sam2_predictor = {"kind": "transformers", "model": model, "processor": processor}
            self.backend_in_use = "sam2"
            return f"SAM2 loaded via transformers: {model_id} ({dtype})"
        except Exception as exc_tf:  # noqa: BLE001
            tf_error = str(exc_tf)
            logger.info("transformers SAM2 unavailable: %s", exc_tf)

        try:
            # Path B: official sam2 package if installed
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            ckpt = self.config.get("isolation", "sam2", "checkpoint", default=None)
            # User must provide config + checkpoint for the official package.
            if ckpt:
                sam = build_sam2("sam2_hiera_l.yaml", ckpt, device=self.device)
                self._sam2_predictor = {
                    "kind": "sam2_pkg",
                    "predictor": SAM2ImagePredictor(sam),
                }
                self.backend_in_use = "sam2"
                return f"SAM2 loaded via sam2 package: {ckpt}"
            raise RuntimeError("Official sam2 package needs a local checkpoint path in config.")
        except Exception as exc_pkg:  # noqa: BLE001
            self._sam2_error = f"tf={tf_error}; pkg={exc_pkg}"
            logger.warning("SAM2 not available: %s", self._sam2_error)
            raise RuntimeError(f"SAM2 load failed: {self._sam2_error}") from exc_pkg

    def load_rembg(self) -> str:
        if self._rembg_session is not None:
            return "rembg already loaded."
        from rembg import new_session

        model_name = str(self.config.get("isolation", "rembg", "model", default="u2net"))
        self._rembg_session = new_session(model_name)
        if self.backend_in_use is None:
            self.backend_in_use = "rembg"
        return f"rembg session ready: {model_name}"

    def ensure_fallback(self) -> None:
        if self._rembg_session is None:
            self.load_rembg()

    def isolate_rembg(self, image: Image.Image) -> Image.Image:
        """Background removal with rembg. Returns RGBA."""
        from rembg import remove

        self.ensure_fallback()
        rgb = image.convert("RGB")
        out = remove(rgb, session=self._rembg_session)
        if isinstance(out, bytes):
            from io import BytesIO

            out = Image.open(BytesIO(out))
        return out.convert("RGBA")

    def isolate_sam2_click(
        self,
        image: Image.Image,
        point_xy: tuple[float, float] | None = None,
        point_labels: list[int] | None = None,
    ) -> Image.Image:
        """Isolate using a positive click (x, y) in image pixel coords.

        If no point is given, use the image center as a positive point.
        """
        if self._sam2_predictor is None:
            self.load_sam2()

        assert self._sam2_predictor is not None
        rgb = image.convert("RGB")
        w, h = rgb.size
        if point_xy is None:
            point_xy = (w / 2.0, h / 2.0)
        labels = point_labels or [1]

        kind = self._sam2_predictor["kind"]
        if kind == "transformers":
            return self._isolate_transformers(rgb, point_xy, labels)
        if kind == "sam2_pkg":
            return self._isolate_sam2_pkg(rgb, point_xy, labels)
        raise RuntimeError(f"Unknown SAM2 backend kind: {kind}")

    def _isolate_transformers(
        self,
        rgb: Image.Image,
        point_xy: tuple[float, float],
        labels: list[int],
    ) -> Image.Image:
        import torch

        model = self._sam2_predictor["model"]
        processor = self._sam2_predictor["processor"]
        input_points = [[[list(point_xy)]]]
        input_labels = [[labels]]

        inputs = processor(
            images=rgb,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )
        # masks[0] shape depends on version; pick highest-score mask when possible
        mask_t = masks[0]
        if hasattr(outputs, "iou_scores"):
            scores = outputs.iou_scores[0].detach().cpu().numpy().reshape(-1)
            best = int(np.argmax(scores))
        else:
            best = 0
        mask_np = mask_t[0, best].numpy() if mask_t.ndim == 4 else mask_t[best].numpy()
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255
        return self._apply_mask(rgb, mask_np)

    def _isolate_sam2_pkg(
        self,
        rgb: Image.Image,
        point_xy: tuple[float, float],
        labels: list[int],
    ) -> Image.Image:
        predictor = self._sam2_predictor["predictor"]
        arr = np.array(rgb)
        predictor.set_image(arr)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([point_xy], dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        mask_np = (masks[best].astype(np.uint8)) * 255
        return self._apply_mask(rgb, mask_np)

    @staticmethod
    def _apply_mask(rgb: Image.Image, mask_np: np.ndarray) -> Image.Image:
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze()
        mask_img = Image.fromarray(mask_np.astype(np.uint8), mode="L")
        if mask_img.size != rgb.size:
            mask_img = mask_img.resize(rgb.size, Image.Resampling.NEAREST)
        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask_img)
        return rgba

    def isolate(
        self,
        image: Image.Image,
        click_xy: tuple[float, float] | None = None,
        prefer: str | None = None,
    ) -> tuple[Image.Image, str]:
        """Run isolation. Returns (RGBA image, backend name used)."""
        prefer = prefer or str(self.config.get("isolation", "preferred", default="sam2"))

        # SAM2 without a user click falls back to a center-point guess, which
        # often lands on empty background (e.g. between bicycle frame tubes)
        # and masks the wrong region. Whole-object matting (rembg) is the
        # reliable default when no point is given.
        if prefer == "sam2" and click_xy is None:
            prefer = "rembg"

        if prefer == "sam2":
            try:
                out = self.isolate_sam2_click(image, point_xy=click_xy)
                self.backend_in_use = "sam2"
                return out, "sam2"
            except Exception as exc:  # noqa: BLE001
                logger.warning("SAM2 isolation failed, falling back to rembg: %s", exc)

        out = self.isolate_rembg(image)
        self.backend_in_use = "rembg"
        return out, "rembg"

    def unload(self) -> None:
        pred = self._sam2_predictor
        rembg = self._rembg_session
        self._sam2_predictor = None
        self._rembg_session = None
        self.backend_in_use = None
        release: list[Any] = [rembg]
        if isinstance(pred, dict):
            release.append(pred.get("model"))
            release.append(pred.get("predictor"))
            release.append(pred.get("processor"))
        else:
            release.append(pred)
        # Isolator is not torch.compiled; keep Flux/SDXL compile caches.
        hard_release(*release, reset_compiler=False)
