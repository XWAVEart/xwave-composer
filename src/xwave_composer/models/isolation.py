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


def tf_sam2_prompts(
    points: list[tuple[float, float]],
    labels: list[int],
) -> tuple[list, list]:
    """Nest clicks the way transformers Sam2Processor expects.

    Points:  [image, object, point, xy]
    Labels:  [image, object, point]
    """
    coords = [list(p) for p in points]
    return [[coords]], [[list(labels)]]


def refine_sam_mask(
    mask_np: np.ndarray,
    points: list[tuple[float, float]] | None = None,
    labels: list[int] | None = None,
    image_size: tuple[int, int] | None = None,
    min_area_px: int = 48,
    min_area_frac: float = 0.0004,
    search_radius: int = 8,
) -> np.ndarray:
    """Drop disconnected specks so a click selects one object.

    SAM2 often lights up tiny islands far from the prompt. Keep the connected
    component under each positive click (or the largest blob if the click
    missed), and discard everything else.
    """
    from scipy.ndimage import generate_binary_structure, label

    arr = np.asarray(mask_np)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray(mask_np)
    binary = arr > 127 if arr.dtype == np.uint8 and int(arr.max()) > 1 else arr > 0.5
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.uint8)

    labeled, count = label(binary, structure=generate_binary_structure(2, 2))
    if count == 0:
        return np.zeros(binary.shape, dtype=np.uint8)

    mh, mw = binary.shape
    sx = sy = 1.0
    if image_size is not None:
        iw, ih = int(image_size[0]), int(image_size[1])
        if iw > 0 and ih > 0 and (mw, mh) != (iw, ih):
            sx = mw / float(iw)
            sy = mh / float(ih)

    keep: set[int] = set()
    labs = list(labels) if labels is not None else [1] * len(points or ())
    for idx, pt in enumerate(points or ()):
        lab = labs[idx] if idx < len(labs) else 1
        if int(lab) != 1:
            continue
        cid = _component_near(labeled, pt[0] * sx, pt[1] * sy, search_radius)
        if cid:
            keep.add(cid)

    if not keep:
        areas = np.bincount(labeled.ravel())
        min_area = max(int(min_area_px), int(round(mh * mw * float(min_area_frac))))
        large = [i for i in range(1, count + 1) if int(areas[i]) >= min_area]
        if large:
            keep.add(max(large, key=lambda i: int(areas[i])))
        elif count:
            keep.add(int(np.argmax(areas[1:]) + 1))

    cleaned = np.isin(labeled, list(keep))
    return cleaned.astype(np.uint8) * 255


def _component_near(labeled: np.ndarray, x: float, y: float, radius: int) -> int:
    h, w = labeled.shape
    xi = int(round(x))
    yi = int(round(y))
    if 0 <= yi < h and 0 <= xi < w:
        cid = int(labeled[yi, xi])
        if cid:
            return cid
    rad = max(0, int(radius))
    y0, y1 = max(0, yi - rad), min(h, yi + rad + 1)
    x0, x1 = max(0, xi - rad), min(w, xi + rad + 1)
    patch = labeled[y0:y1, x0:x1]
    ys, xs = np.nonzero(patch)
    if ys.size == 0:
        return 0
    dy = (ys.astype(np.int32) + y0) - yi
    dx = (xs.astype(np.int32) + x0) - xi
    k = int(np.argmin(dx * dx + dy * dy))
    return int(patch[ys[k], xs[k]])


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
        self._cached_rgb: Image.Image | None = None
        self._tf_cache: dict[str, Any] | None = None

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

    def reset_image(self) -> None:
        """Drop cached SAM2 image embeddings (call when the source image changes)."""
        self._cached_rgb = None
        self._tf_cache = None
        pred = self._sam2_predictor
        if isinstance(pred, dict) and pred.get("kind") == "sam2_pkg":
            predictor = pred.get("predictor")
            reset = getattr(predictor, "reset_predictor", None)
            if callable(reset):
                try:
                    reset()
                except Exception:  # noqa: BLE001
                    logger.debug("SAM2 reset_predictor failed", exc_info=True)

    def set_image(self, image: Image.Image) -> None:
        """Encode ``image`` once so later point prompts reuse the embedding."""
        if self._sam2_predictor is None:
            self.load_sam2()
        assert self._sam2_predictor is not None
        rgb = image.convert("RGB")
        if (
            self._cached_rgb is not None
            and self._cached_rgb.size == rgb.size
            and id(self._cached_rgb) == id(rgb)
        ):
            return
        self._cached_rgb = rgb
        kind = self._sam2_predictor["kind"]
        if kind == "sam2_pkg":
            predictor = self._sam2_predictor["predictor"]
            predictor.set_image(np.array(rgb))
            self._tf_cache = None
            return
        if kind != "transformers":
            return
        import torch

        model = self._sam2_predictor["model"]
        processor = self._sam2_predictor["processor"]
        encoded = processor(images=rgb, return_tensors="pt")
        pixel_values = encoded["pixel_values"]
        pixel_values = self._to_model(pixel_values)
        self._tf_cache = {
            "pixel_values": pixel_values,
            "size": rgb.size,
            "image_embeddings": None,
        }
        get_emb = getattr(model, "get_image_embeddings", None) or getattr(
            model, "get_image_features", None
        )
        if not callable(get_emb):
            return
        try:
            with torch.inference_mode():
                self._tf_cache["image_embeddings"] = get_emb(pixel_values)
        except Exception:  # noqa: BLE001
            logger.debug("SAM2 embedding cache unavailable", exc_info=True)
            self._tf_cache["image_embeddings"] = None

    def isolate_sam2_click(
        self,
        image: Image.Image,
        point_xy: tuple[float, float] | None = None,
        point_labels: list[int] | None = None,
        points: list[tuple[float, float]] | None = None,
    ) -> Image.Image:
        """Isolate using one or more clicks in image pixel coords.

        If no point is given, use the image center as a positive point.
        """
        rgb = image.convert("RGB")
        w, h = rgb.size
        if points:
            coords = [(float(x), float(y)) for x, y in points]
        else:
            if point_xy is None:
                point_xy = (w / 2.0, h / 2.0)
            coords = [(float(point_xy[0]), float(point_xy[1]))]
        labels = list(point_labels) if point_labels is not None else [1] * len(coords)
        if len(labels) != len(coords):
            labels = (labels + [1] * len(coords))[: len(coords)]
        mask_np = self.predict_mask(rgb, coords, labels)
        return self._apply_mask(rgb, mask_np)

    def predict_mask(
        self,
        image: Image.Image,
        points: list[tuple[float, float]],
        labels: list[int],
    ) -> np.ndarray:
        """Return a uint8 0/255 mask the same size as ``image``."""
        if self._sam2_predictor is None:
            self.load_sam2()
        assert self._sam2_predictor is not None
        rgb = image.convert("RGB")
        if not points:
            return np.zeros((rgb.height, rgb.width), dtype=np.uint8)
        coords = [(float(x), float(y)) for x, y in points]
        labs = list(labels) if labels is not None else [1] * len(coords)
        if len(labs) != len(coords):
            labs = (labs + [1] * len(coords))[: len(coords)]
        kind = self._sam2_predictor["kind"]
        if kind == "transformers":
            mask_np = self._predict_transformers(rgb, coords, labs)
        elif kind == "sam2_pkg":
            mask_np = self._predict_sam2_pkg(rgb, coords, labs)
        else:
            raise RuntimeError(f"Unknown SAM2 backend kind: {kind}")
        min_px = int(self.config.get("isolation", "sam2", "min_speckle_px", default=48) or 48)
        min_frac = float(self.config.get("isolation", "sam2", "min_speckle_frac", default=0.0004) or 0.0)
        return refine_sam_mask(
            mask_np,
            coords,
            labs,
            image_size=rgb.size,
            min_area_px=min_px,
            min_area_frac=min_frac,
        )

    def _to_model(self, value: Any) -> Any:
        """Move tensors onto the SAM2 device; floats follow the model dtype."""
        if not hasattr(value, "to"):
            return value
        model = (self._sam2_predictor or {}).get("model")
        dtype = getattr(model, "dtype", None)
        if dtype is not None and torch.is_floating_point(value):
            return value.to(device=self.device, dtype=dtype)
        return value.to(self.device)

    def _predict_transformers(
        self,
        rgb: Image.Image,
        points: list[tuple[float, float]],
        labels: list[int],
    ) -> np.ndarray:
        import torch

        model = self._sam2_predictor["model"]
        processor = self._sam2_predictor["processor"]
        # transformers Sam2Processor wants 4-level points / 3-level labels.
        input_points, input_labels = tf_sam2_prompts(points, labels)

        if self._tf_cache is None or self._tf_cache.get("size") != rgb.size:
            self.set_image(rgb)

        def _best_mask(outputs: Any, original_sizes: Any) -> np.ndarray:
            # CPU interpolate / numpy cannot read bfloat16. Decode in float32.
            pred = outputs.pred_masks.detach().float()
            orig = original_sizes.cpu() if hasattr(original_sizes, "cpu") else original_sizes
            masks = processor.post_process_masks(pred, orig)
            mask_t = masks[0]
            if hasattr(mask_t, "detach"):
                mask_t = mask_t.detach().float().cpu()
            if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
                scores = outputs.iou_scores[0].detach().float().cpu().numpy().reshape(-1)
                best = int(np.argmax(scores))
            else:
                best = 0
            if mask_t.ndim == 4:
                mask_np = mask_t[0, min(best, mask_t.shape[1] - 1)].numpy()
            elif mask_t.ndim == 3:
                mask_np = mask_t[min(best, mask_t.shape[0] - 1)].numpy()
            else:
                mask_np = mask_t.numpy()
            return (mask_np > 0.5).astype(np.uint8) * 255

        cache = self._tf_cache or {}
        try:
            point_inputs = processor(
                images=rgb,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )
            orig = point_inputs["original_sizes"]
            embeddings = cache.get("image_embeddings")
            if embeddings is not None:
                kwargs = {
                    k: self._to_model(v)
                    for k, v in point_inputs.items()
                    if k != "pixel_values"
                }
                kwargs["image_embeddings"] = embeddings
                with torch.inference_mode():
                    outputs = model(**kwargs)
                return _best_mask(outputs, orig)
            if cache.get("pixel_values") is not None:
                point_inputs["pixel_values"] = cache["pixel_values"]
            inputs = {k: self._to_model(v) for k, v in point_inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            return _best_mask(outputs, inputs["original_sizes"])
        except Exception:  # noqa: BLE001
            logger.debug("SAM2 cached predict failed; running full forward", exc_info=True)

        inputs = processor(
            images=rgb,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        )
        inputs = {k: self._to_model(v) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        return _best_mask(outputs, inputs["original_sizes"])

    def _predict_sam2_pkg(
        self,
        rgb: Image.Image,
        points: list[tuple[float, float]],
        labels: list[int],
    ) -> np.ndarray:
        predictor = self._sam2_predictor["predictor"]
        if self._cached_rgb is None or self._cached_rgb.size != rgb.size:
            self.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        mask_np = masks[best]
        if np.issubdtype(mask_np.dtype, np.floating) or mask_np.dtype == np.bool_:
            mask_np = (mask_np > 0.5).astype(np.uint8) * 255
        elif mask_np.max() <= 1:
            mask_np = mask_np.astype(np.uint8) * 255
        else:
            mask_np = mask_np.astype(np.uint8)
        return mask_np

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
        self.reset_image()
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
