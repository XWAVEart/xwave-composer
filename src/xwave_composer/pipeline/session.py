"""Application session: wires models, canvas document, and OUTPUT updates."""

from __future__ import annotations

import hashlib
import logging
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from xwave_composer.canvas.compositor import compose_work_image
from xwave_composer.canvas.layers import LayerTransform, ObjectLayer, WorkDocument
from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, gpu_summary
from xwave_composer.models.flux_generator import FluxGenerator
from xwave_composer.models.isolation import ObjectIsolator
from xwave_composer.models.llm_rewriter import PromptRewriter
from xwave_composer.models.sdxl_hyper import SDXLHyperPipeline
from xwave_composer.models.upscaler import ImageUpscaler
from xwave_composer.optimization import (
    configured_profile,
    normalize_profile,
    profile_label,
)
from xwave_composer.style.style_manager import StyleManager, build_output_prompt

logger = logging.getLogger(__name__)


@dataclass
class OutputSettings:
    denoise: float = 0.3
    steps: int = 6
    cfg: float = 1.0
    eta: float = 0.0
    use_llm_rewrite: bool = False
    negative_prompt: str = "blurry, low quality, deformed, watermark, text"
    seed: int = -1
    # Prompt concatenation order:
    #   pcs = prefix, content, suffix (prefix fused with space)
    #   psc = prefix, suffix, content
    #   cps = content, prefix, suffix
    #   spc = suffix, prefix, content
    concat_order: str = "pcs"
    # Manual style: overrides the CSV preset prefix/suffix when enabled.
    manual_style: bool = False
    manual_prefix: str = ""
    manual_suffix: str = ""
    # When True, OUTPUT uses custom_prompt instead of auto-built concatenation.
    prompt_locked: bool = False
    custom_prompt: str = ""


@dataclass
class ComposerSession:
    """Holds runtime state for one Gradio user session (single-user local app)."""

    config: AppConfig
    doc: WorkDocument = field(default_factory=WorkDocument)
    styles: StyleManager | None = None
    flux: FluxGenerator | None = None
    isolator: ObjectIsolator | None = None
    sdxl: SDXLHyperPipeline | None = None
    llm: PromptRewriter | None = None
    upscaler: ImageUpscaler | None = None
    output_settings: OutputSettings = field(default_factory=OutputSettings)
    last_work: Image.Image | None = None
    last_output: Image.Image | None = None
    last_output_prompt: str = ""
    last_concat_prompt: str = ""
    _rewrite_cache_key: str = ""
    _rewrite_cache_result: str = ""
    _flux_reload_pending: bool = False
    output_rev: int = 0
    status: str = "Ready"
    # Pending object awaiting SAM2 click isolation
    pending_raw: Image.Image | None = None
    pending_prompt: str = ""
    pending_isolation_prompt: str = ""
    # Object-layer UI preference: when False, generate/import places the full
    # image and the isolation prompt is cleared / unused.
    layer_cutout: bool = True
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        self.config.ensure_dirs()
        w, h = self.config.canvas_size
        self.doc.width = w
        self.doc.height = h
        self.styles = StyleManager(self.config)
        self.flux = FluxGenerator(self.config)
        self.isolator = ObjectIsolator(self.config)
        self.sdxl = SDXLHyperPipeline(self.config)
        self.llm = PromptRewriter(self.config)
        self.upscaler = ImageUpscaler(self.config)
        self.output_settings.denoise = float(
            self.config.get("sdxl_hyper", "default_denoise", default=0.3)
        )
        self.output_settings.steps = int(
            self.config.get("sdxl_hyper", "default_steps", default=6)
        )
        self.output_settings.cfg = float(
            self.config.get("sdxl_hyper", "guidance_scale", default=1.0)
        )
        # Sticky OUTPUT seed — only changes when the user rolls a new one.
        self.output_settings.seed = random.randint(0, 2_147_483_647)
        self.output_settings.use_llm_rewrite = bool(
            self.config.get("llm", "enabled_by_default", default=False)
        )

    # ------------------------------------------------------------------ models
    def preload_core(self) -> str:
        """Load Flux + SDXL Hyper + isolation for interactive use."""
        parts: list[str] = []
        try:
            self.flux.load()
            parts.append("Flux ✓")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"Flux ✗ ({exc})")
        try:
            self.sdxl.load()
            parts.append("SDXL Hyper ✓")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"SDXL ✗ ({exc})")
        if str(self.config.get("isolation", "preferred", default="sam2")) == "sam2":
            try:
                self.isolator.load_sam2()
                parts.append("SAM2 ✓")
            except Exception as exc:  # noqa: BLE001
                logger.warning("SAM2 unavailable: %s", exc)
                parts.append("SAM2 ✗ (rembg fallback)")
        try:
            self.isolator.load_rembg()
            parts.append("rembg ✓")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"rembg ✗ ({exc})")
        self.status = f"{' · '.join(parts)} | {gpu_summary()}"
        return self.status

    @property
    def compute_profile(self) -> str:
        if self.flux:
            return self.flux.profile
        return configured_profile(self.config)

    def optimization_status(self) -> str:
        reports = []
        if self.flux:
            reports.append(self.flux.optimization_report.status())
        if self.sdxl:
            reports.append(self.sdxl.optimization_report.status())
        return " · ".join(reports)

    def switch_compute_profile(self, profile: str) -> str:
        """Rebuild core diffusion pipelines under a new compute profile."""
        selected = normalize_profile(profile)
        with self._lock:
            self.status = f"Switching performance profile to {profile_label(selected)}…"
            self.config.raw.setdefault("optimization", {})["profile"] = selected
            errors: list[str] = []
            try:
                self.flux.set_profile(selected, reload=False)
                self.sdxl.set_profile(selected, reload=False)
                self.flux.unload()
                self.sdxl.unload()
                self.flux.load()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Flux profile switch failed")
                errors.append(f"Flux: {exc}")
            try:
                self.sdxl.load()
            except Exception as exc:  # noqa: BLE001
                logger.exception("SDXL profile switch failed")
                errors.append(f"SDXL: {exc}")
            detail = self.optimization_status()
            if errors:
                self.status = f"Profile switch partial: {' | '.join(errors)} | {detail}"
            else:
                self.status = f"Performance profile ready: {profile_label(selected)} | {detail}"
            if self.last_work is not None and self.sdxl.ready:
                try:
                    self.run_output(self.last_work)
                except Exception:  # noqa: BLE001
                    logger.exception("OUTPUT refresh after profile switch failed")
            return self.status

    @property
    def core_ready(self) -> bool:
        """True when the OUTPUT pipeline can run without a surprise load."""
        return bool(self.sdxl and self.sdxl.ready)

    def switch_base_model(self, base_ref: str) -> str:
        """Swap the SDXL base model under the Hyper LoRA (HF id / CivitAI link)."""
        with self._lock:
            self.status = f"Loading SDXL base: {base_ref}…"
            try:
                msg = self.sdxl.load(force=True, base_ref=base_ref)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Base model switch failed")
                self.status = f"Base model failed: {exc}"
                return self.status
            self.status = msg
            if self.last_work is not None:
                try:
                    self.run_output(self.last_work)
                except Exception:  # noqa: BLE001
                    logger.exception("OUTPUT refresh after base switch failed")
            return self.status

    # --------------------------------------------------------------- generation
    def generate_background(self, prompt: str, seed: int = -1) -> Image.Image:
        with self._lock:
            self.status = "Generating background with Flux…"
            img = self.flux.generate_background(
                prompt=prompt,
                width=self.doc.width,
                height=self.doc.height,
                seed=seed if seed >= 0 else None,
            )
            self.doc.background = img
            self.doc.background_prompt = prompt
            self._save_layer_image(img, "background")
            self.last_work = compose_work_image(self.doc)
            self.status = "Background ready."
            return self.last_work

    def generate_object_raw(
        self,
        prompt: str,
        isolation_prompt: str,
        seed: int = -1,
        auto_isolate: bool = True,
        prefer: str | None = None,
    ) -> Image.Image:
        """Generate object, optionally auto-isolate, and add it to the WORK canvas.

        When auto_isolate is True (default), rembg/SAM2 runs immediately so the
        object appears on WORK without a second step. Raw image is kept for
        optional SAM2 re-isolate with a click.
        """
        with self._lock:
            self.status = "Generating object with Flux…"
            img = self.flux.generate_object(
                object_prompt=prompt,
                isolation_prompt=isolation_prompt,
                width=min(1024, self.doc.width),
                height=min(1024, self.doc.height),
                seed=seed if seed >= 0 else None,
            )
            self.pending_raw = img
            self.pending_prompt = prompt
            self.pending_isolation_prompt = isolation_prompt

            if not auto_isolate:
                self.status = "Object generated. Run isolate to add it to WORK."
                return img

            # prefer=None honors config isolation.preferred (SAM2 center-point,
            # rembg fallback happens inside the isolator).
            return self._isolate_pending_unlocked(click_xy=None, prefer=prefer)

    def _isolate_pending_unlocked(
        self,
        click_xy: tuple[float, float] | None = None,
        prefer: str | None = None,
    ) -> Image.Image:
        """Isolate pending raw and add object layer. Caller must hold self._lock."""
        if self.pending_raw is None:
            self.status = "No pending object to isolate."
            if self.last_work is None:
                self.last_work = compose_work_image(self.doc)
            return self.last_work

        self.status = "Isolating object…"
        try:
            rgba, backend = self.isolator.isolate(
                self.pending_raw, click_xy=click_xy, prefer=prefer
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Isolation failed")
            # Still add the raw RGB as an opaque layer so WORK updates.
            rgba = self.pending_raw.convert("RGBA")
            backend = f"none (isolation error: {exc})"

        # Sanity: ensure some visible alpha exists
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        alpha = rgba.getchannel("A")
        extrema = alpha.getextrema()
        if extrema is not None and extrema[1] == 0:
            # Fully transparent mask — fall back to full opacity so layer is visible
            logger.warning("Isolation produced empty alpha; using opaque object image.")
            rgba = self.pending_raw.convert("RGBA")
            backend = f"{backend}+opaque_fallback"

        layer = ObjectLayer(
            name=f"Object {len(self.doc.objects) + 1}",
            prompt=self.pending_prompt,
            isolation_prompt=self.pending_isolation_prompt,
            image=rgba,
            raw_image=self.pending_raw.copy(),
        )
        # Fit large objects so they do not cover the whole canvas by default
        max_dim = max(rgba.width, rgba.height, 1)
        target = min(self.doc.width, self.doc.height) * 0.45
        if max_dim > target:
            s = target / max_dim
            layer.transform.scale_x = s
            layer.transform.scale_y = s

        self.doc.add_object(layer)
        self._save_layer_image(rgba, f"object_{layer.id}")
        # Clear pending so re-isolate cannot double-add the same raw image.
        # Raw stays on layer.raw_image for SAM2 re-cut.
        self.pending_raw = None
        self.last_work = compose_work_image(self.doc)
        self.status = (
            f"Object layer '{layer.name}' added via {backend} "
            f"({len(self.doc.objects)} object(s) on WORK)."
        )
        return self.last_work

    def isolate_pending(
        self,
        click_xy: tuple[float, float] | None = None,
        prefer: str | None = None,
    ) -> Image.Image | None:
        """Isolate pending raw image and add as object layer."""
        with self._lock:
            return self._isolate_pending_unlocked(click_xy=click_xy, prefer=prefer)

    def add_object_from_image(
        self,
        image: Image.Image,
        prompt: str = "",
        isolation_prompt: str = "",
        isolate: bool = True,
        prefer: str | None = "rembg",
    ) -> Image.Image:
        """Add an existing image as an object layer (for tests / re-import)."""
        with self._lock:
            self.pending_raw = image.convert("RGB")
            self.pending_prompt = prompt
            self.pending_isolation_prompt = isolation_prompt
            if isolate:
                return self._isolate_pending_unlocked(prefer=prefer)
            layer = ObjectLayer(
                name=f"Object {len(self.doc.objects) + 1}",
                prompt=prompt,
                isolation_prompt=isolation_prompt,
                image=image.convert("RGBA"),
                raw_image=image.convert("RGB"),
            )
            self.doc.add_object(layer)
            self.last_work = compose_work_image(self.doc)
            return self.last_work

    def import_into_selected(
        self,
        image: Image.Image,
        cutout: str = "rembg",
        prompt: str = "",
    ) -> Image.Image:
        """Import a user image into the selected layer.

        cutout: "rembg" | "sam2" | "none"
        - Background selected → image becomes the canvas background (full, no cutout).
        - Object selected (or none) → image becomes / creates an object layer.
        SAM2 without a click uses the image center as the prompt point.
        """
        with self._lock:
            raw = image.convert("RGB")
            mode = (cutout or "none").strip().lower()
            if mode not in ("rembg", "sam2", "none"):
                mode = "none"

            # Background: always full-bleed to canvas size
            if self.doc.selected_id == "__bg__":
                bg = raw.resize((self.doc.width, self.doc.height), Image.Resampling.LANCZOS)
                self.doc.background = bg
                if prompt.strip():
                    self.doc.background_prompt = prompt.strip()
                elif not self.doc.background_prompt:
                    self.doc.background_prompt = "imported background"
                self._save_layer_image(bg, "background")
                self.last_work = compose_work_image(self.doc)
                self.status = "Background imported."
                return self.last_work

            # Ensure an object layer exists
            obj = self.doc.selected()
            if obj is None:
                obj = ObjectLayer(name=f"Object {len(self.doc.objects) + 1}")
                self.doc.add_object(obj)

            if prompt.strip():
                obj.prompt = prompt.strip()
            elif not obj.prompt:
                obj.prompt = "imported image"

            if mode == "none":
                rgba, backend = raw.convert("RGBA"), "imported (full image)"
            else:
                click_xy = None
                prefer = mode
                if mode == "sam2":
                    # Center-point prompt so SAM2 can run without a UI click.
                    click_xy = (raw.width / 2.0, raw.height / 2.0)
                self.status = f"Isolating import via {mode}…"
                try:
                    rgba, backend = self.isolator.isolate(raw, click_xy=click_xy, prefer=prefer)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Import isolation failed")
                    rgba, backend = raw.convert("RGBA"), f"imported opaque ({exc})"
                if rgba.mode != "RGBA":
                    rgba = rgba.convert("RGBA")

            first_image = obj.image is None
            obj.image = rgba
            obj.raw_image = raw.copy()
            if first_image:
                max_dim = max(rgba.width, rgba.height, 1)
                target = min(self.doc.width, self.doc.height) * 0.45
                if max_dim > target:
                    s = target / max_dim
                    obj.transform.scale_x = s
                    obj.transform.scale_y = s
            self._save_layer_image(rgba, f"object_{obj.id}")
            self.last_work = compose_work_image(self.doc)
            self.status = f"Image imported via {backend}."
            return self.last_work

    def add_empty_object(self) -> "ObjectLayer":
        """Add an empty object layer (no image yet) and select it."""
        with self._lock:
            layer = ObjectLayer(name=f"Object {len(self.doc.objects) + 1}")
            self.doc.add_object(layer)
            self.status = "Layer added — type its prompt and Generate."
            return layer

    def generate_selected(
        self,
        prompt: str,
        isolation_prompt: str = "",
        seed: int = -1,
        isolate: bool = True,
    ) -> Image.Image:
        """Generate into the selected layer: background layer -> Flux background,
        object layer -> Flux object (+ optional isolation) replacing its image."""
        with self._lock:
            sid = self.doc.selected_id
            if sid in (None, "__bg__"):
                return self.generate_background(prompt, seed=seed)
            obj = self.doc.selected()
            if obj is None:
                return self.generate_background(prompt, seed=seed)

            self.status = "Generating object with Flux…"
            if isolate:
                iso = (isolation_prompt or obj.isolation_prompt or "").strip()
                raw = self.flux.generate_object(
                    object_prompt=prompt,
                    isolation_prompt=iso,
                    width=min(1024, self.doc.width),
                    height=min(1024, self.doc.height),
                    seed=seed if seed >= 0 else None,
                )
            else:
                # Full-image placement: no plain-backdrop coaxing, no cutout.
                iso = ""
                raw = self.flux.generate(
                    prompt=prompt,
                    width=min(1024, self.doc.width),
                    height=min(1024, self.doc.height),
                    seed=seed if seed >= 0 else None,
                )
            if not isolate:
                rgba, backend = raw.convert("RGBA"), "full image (no cutout)"
            else:
                self.status = "Isolating object…"
                try:
                    rgba, backend = self.isolator.isolate(raw, prefer=None)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Isolation failed")
                    rgba, backend = raw.convert("RGBA"), f"none ({exc})"
                if rgba.mode != "RGBA":
                    rgba = rgba.convert("RGBA")
                extrema = rgba.getchannel("A").getextrema()
                if extrema is not None and extrema[1] == 0:
                    logger.warning("Isolation produced empty alpha; using opaque image.")
                    rgba = raw.convert("RGBA")
                    backend = f"{backend}+opaque_fallback"

            first_image = obj.image is None
            obj.prompt = prompt
            obj.isolation_prompt = iso
            obj.image = rgba
            obj.raw_image = raw.copy()
            if first_image:
                # Fit so a new object doesn't cover the whole canvas.
                max_dim = max(rgba.width, rgba.height, 1)
                target = min(self.doc.width, self.doc.height) * 0.45
                if max_dim > target:
                    s = target / max_dim
                    obj.transform.scale_x = s
                    obj.transform.scale_y = s
            self._save_layer_image(rgba, f"object_{obj.id}")
            self.last_work = compose_work_image(self.doc)
            self.status = f"Layer generated via {backend}."
            return self.last_work

    def reisolate_selected(
        self,
        click_xy: tuple[float, float] | None = None,
        prefer: str | None = None,
    ) -> Image.Image | None:
        with self._lock:
            obj = self.doc.selected()
            if obj is None or obj.raw_image is None:
                self.status = "Select an object that has a raw image."
                return self.last_work
            rgba, backend = self.isolator.isolate(
                obj.raw_image, click_xy=click_xy, prefer=prefer
            )
            obj.image = rgba
            self.last_work = compose_work_image(self.doc)
            self.status = f"Re-isolated via {backend}."
            return self.last_work

    # ------------------------------------------------------------------ canvas
    def refresh_work(self) -> Image.Image:
        with self._lock:
            self.last_work = compose_work_image(self.doc)
            return self.last_work

    def set_canvas_size(self, width: int, height: int) -> Image.Image:
        with self._lock:
            self.doc.set_size(width, height)
            self.last_work = compose_work_image(self.doc)
            return self.last_work

    def update_selected_transform(
        self,
        x: float | None = None,
        y: float | None = None,
        scale_x: float | None = None,
        scale_y: float | None = None,
        rotation: float | None = None,
        opacity: float | None = None,
    ) -> Image.Image:
        with self._lock:
            obj = self.doc.selected()
            if obj is None:
                return self.refresh_work()
            return self._apply_transform(obj, x, y, scale_x, scale_y, rotation, opacity)

    def reset_selected_transform(self) -> Image.Image:
        """Reset selected layer pose. BG → scale/rotation/flip; object → center."""
        with self._lock:
            if self.doc.selected_id == "__bg__":
                self.doc.bg_scale = 1.0
                self.doc.bg_rotation = 0.0
                self.doc.bg_offset_x = 0.0
                self.doc.bg_offset_y = 0.0
                self.doc.bg_flip_x = False
                self.doc.bg_flip_y = False
                self.last_work = compose_work_image(self.doc)
                self.status = "Background pose reset."
                return self.last_work
            obj = self.doc.selected()
            if obj is None:
                self.status = "Select a layer to reset."
                return self.last_work or compose_work_image(self.doc)
            t = obj.transform
            t.x = self.doc.width / 2
            t.y = self.doc.height / 2
            t.scale_x = 1.0
            t.scale_y = 1.0
            t.rotation = 0.0
            self.last_work = compose_work_image(self.doc)
            self.status = f"Reset transform for {obj.name or obj.id}."
            return self.last_work

    def update_background_transform(
        self,
        *,
        scale: float | None = None,
        rotation: float | None = None,
        offset_x: float | None = None,
        offset_y: float | None = None,
        flip_x: bool | None = None,
        flip_y: bool | None = None,
    ) -> Image.Image:
        """Update background placement controls and recompose WORK."""
        with self._lock:
            if scale is not None:
                self.doc.bg_scale = max(0.05, float(scale))
            if rotation is not None:
                self.doc.bg_rotation = float(rotation)
            if offset_x is not None:
                self.doc.bg_offset_x = float(offset_x)
            if offset_y is not None:
                self.doc.bg_offset_y = float(offset_y)
            if flip_x is not None:
                self.doc.bg_flip_x = bool(flip_x)
            if flip_y is not None:
                self.doc.bg_flip_y = bool(flip_y)
            self.last_work = compose_work_image(self.doc)
            self.status = "Background transform updated."
            return self.last_work

    def set_layer_cutout(self, enabled: bool) -> None:
        """Toggle object cut-out; clear isolation prompt when disabled."""
        self.layer_cutout = bool(enabled)
        if not self.layer_cutout:
            obj = self.doc.selected()
            if obj is not None:
                obj.isolation_prompt = ""
            self.pending_isolation_prompt = ""
        elif self.doc.selected() is not None:
            obj = self.doc.selected()
            assert obj is not None
            if not (obj.isolation_prompt or "").strip():
                obj.isolation_prompt = (
                    "isolated on plain white background, centered"
                )

    def set_isolation_backdrop(self, backdrop: str) -> str:
        """Set white/black isolation backdrop for the selected object layer."""
        label = str(backdrop or "White").strip().title()
        prompt = (
            "isolated on plain black background, centered"
            if label == "Black"
            else "isolated on plain white background, centered"
        )
        obj = self.doc.selected()
        if obj is not None:
            obj.isolation_prompt = prompt
        self.pending_isolation_prompt = prompt
        return prompt

    def duplicate_selected(self) -> Image.Image:
        """Duplicate the selected object, including its images and settings."""
        with self._lock:
            source = self.doc.selected()
            if source is None:
                self.status = "Select an object layer to duplicate."
                return self.last_work or compose_work_image(self.doc)
            duplicate = ObjectLayer(
                name=f"{source.name} copy",
                prompt=source.prompt,
                prompt_enabled=source.prompt_enabled,
                isolation_prompt=source.isolation_prompt,
                feather=float(source.feather),
                blend_mode=str(source.blend_mode or "normal"),
                image=source.image.copy() if source.image is not None else None,
                raw_image=(
                    source.raw_image.copy() if source.raw_image is not None else None
                ),
                transform=LayerTransform.from_dict(source.transform.to_dict()),
            )
            self.doc.add_object(duplicate)
            duplicate.transform.x = min(
                self.doc.width, source.transform.x + 24
            )
            duplicate.transform.y = min(
                self.doc.height, source.transform.y + 24
            )
            if duplicate.image is not None:
                duplicate.path = self._save_layer_image(
                    duplicate.image, f"object_{duplicate.id}"
                )
            self.last_work = compose_work_image(self.doc)
            self.status = f"Duplicated {source.name or source.id}."
            return self.last_work

    def toggle_selected_prompt(self) -> bool | None:
        """Toggle whether the selected object's prompt contributes to OUTPUT."""
        with self._lock:
            obj = self.doc.selected()
            if obj is None:
                self.status = "Select an object layer to mute its prompt."
                return None
            obj.prompt_enabled = not obj.prompt_enabled
            state = "included" if obj.prompt_enabled else "muted"
            self.status = f"Layer prompt {state}: {obj.name or obj.id}."
            return obj.prompt_enabled

    def reset_workspace(self) -> None:
        """Clear composition state while keeping loaded models ready for use."""
        with self._lock:
            width, height = self.config.canvas_size
            self.doc = WorkDocument(
                width=width,
                height=height,
                selected_id="__bg__",
            )
            self.output_settings = OutputSettings(
                denoise=float(
                    self.config.get(
                        "sdxl_hyper", "default_denoise", default=0.3
                    )
                ),
                steps=int(
                    self.config.get("sdxl_hyper", "default_steps", default=6)
                ),
                cfg=float(
                    self.config.get(
                        "sdxl_hyper", "guidance_scale", default=1.0
                    )
                ),
                seed=random.randint(0, 2_147_483_647),
                use_llm_rewrite=bool(
                    self.config.get(
                        "llm", "enabled_by_default", default=False
                    )
                ),
            )
            if self.styles:
                self.styles.set_active(None)
            self.last_work = compose_work_image(self.doc)
            self.last_output = None
            self.last_output_prompt = ""
            self.last_concat_prompt = ""
            self._rewrite_cache_key = ""
            self._rewrite_cache_result = ""
            self.pending_raw = None
            self.pending_prompt = ""
            self.pending_isolation_prompt = ""
            self.output_rev += 1
            self.status = "Workspace reset — ready for a fresh composition."

    def roll_output_seed(self) -> int:
        """Pick a new sticky OUTPUT seed and return it."""
        seed = random.randint(0, 2_147_483_647)
        self.output_settings.seed = seed
        self.status = f"OUTPUT seed → {seed}"
        return seed

    def update_transform_by_id(
        self,
        layer_id: str,
        x: float | None = None,
        y: float | None = None,
        scale_x: float | None = None,
        scale_y: float | None = None,
        rotation: float | None = None,
        opacity: float | None = None,
    ) -> Image.Image:
        with self._lock:
            obj = self.doc.find_by_id(layer_id)
            if obj is None:
                return self.refresh_work()
            self.doc.selected_id = layer_id
            return self._apply_transform(obj, x, y, scale_x, scale_y, rotation, opacity)

    def _apply_transform(
        self,
        obj: ObjectLayer,
        x: float | None,
        y: float | None,
        scale_x: float | None,
        scale_y: float | None,
        rotation: float | None,
        opacity: float | None,
    ) -> Image.Image:
        t = obj.transform
        if x is not None:
            t.x = float(x)
        if y is not None:
            t.y = float(y)
        if scale_x is not None:
            t.scale_x = max(0.05, float(scale_x))
        if scale_y is not None:
            t.scale_y = max(0.05, float(scale_y))
        if rotation is not None:
            t.rotation = float(rotation)
        if opacity is not None:
            t.opacity = max(0.0, min(1.0, float(opacity)))
        self.last_work = compose_work_image(self.doc)
        return self.last_work

    def select_layer(self, label: str | None) -> ObjectLayer | None:
        with self._lock:
            if not label:
                self.doc.selected_id = None
                return None
            obj = self.doc.find_by_label(label) or self.doc.find_by_id(label)
            if obj:
                self.doc.selected_id = obj.id
            return obj

    def select_layer_id(self, layer_id: str | None) -> ObjectLayer | None:
        with self._lock:
            if not layer_id:
                self.doc.selected_id = None
                return None
            obj = self.doc.find_by_id(layer_id)
            self.doc.selected_id = obj.id if obj else None
            return obj

    def rename_layer(self, layer_id: str, name: str) -> Image.Image:
        with self._lock:
            self.doc.rename(layer_id, name)
            self.status = f"Renamed layer to '{name.strip()}'."
            return self.refresh_work()

    def delete_layer(self, layer_id: str | None = None) -> Image.Image:
        with self._lock:
            lid = layer_id or self.doc.selected_id
            if lid:
                self.doc.remove_object(lid)
                self.status = f"Deleted layer {lid}."
            self.last_work = compose_work_image(self.doc)
            return self.last_work

    def delete_selected(self) -> Image.Image:
        return self.delete_layer(None)

    def reorder_selected(self, direction: str) -> Image.Image:
        with self._lock:
            if self.doc.selected_id:
                self.doc.reorder(self.doc.selected_id, direction)
            self.last_work = compose_work_image(self.doc)
            return self.last_work

    def reorder_layers(self, ordered_ids: list[str]) -> Image.Image:
        with self._lock:
            self.doc.reorder_by_ids(ordered_ids)
            self.last_work = compose_work_image(self.doc)
            self.status = "Layer order updated."
            return self.last_work

    # ------------------------------------------------------------------- output
    def apply_style_preset(self, name: str | None) -> "OutputSettings":
        """Activate a CSV style preset and copy its sampler values into settings."""
        preset = self.styles.set_active(name) if self.styles else None
        if preset is not None:
            self.output_settings.cfg = preset.cfg
            self.output_settings.denoise = preset.denoise
            self.output_settings.eta = preset.eta
            self.output_settings.negative_prompt = preset.negative
            self.status = f"Style loaded: {preset.name}"
        else:
            self.output_settings.cfg = float(
                self.config.get("sdxl_hyper", "guidance_scale", default=1.0)
            )
            self.output_settings.eta = 0.0
            self.status = "Style cleared."
        return self.output_settings

    def build_prompt(self) -> str:
        """Default: simple concatenation (prefix fused into content + suffix).

        When prompt_locked is set, the user's custom_prompt is used as-is.
        When LLM rewrite is enabled (and not locked), Qwen2.5-VL rewrites
        the concatenation using the WORK canvas for composition only.
        """
        s = self.output_settings
        if s.prompt_locked and str(s.custom_prompt or "").strip():
            self.last_output_prompt = str(s.custom_prompt).strip()
            return self.last_output_prompt

        preset = self.styles.active if self.styles else None
        concat = build_output_prompt(
            self.doc,
            preset,
            order=s.concat_order,
            manual_prefix=s.manual_prefix if s.manual_style else None,
            manual_suffix=s.manual_suffix if s.manual_style else None,
        )
        self.last_concat_prompt = concat

        if s.use_llm_rewrite and concat.strip():
            cache_key = self._prompt_rewrite_cache_key(concat)
            if cache_key == self._rewrite_cache_key and self._rewrite_cache_result:
                self.last_output_prompt = self._rewrite_cache_result
                self.status = "LLM rewrite ready (cached)."
                return self.last_output_prompt
            try:
                rewritten = self._rewrite_with_vram_headroom(concat)
                if rewritten.strip() and rewritten.strip() != concat.strip():
                    self.last_output_prompt = rewritten.strip()
                    self._rewrite_cache_key = cache_key
                    self._rewrite_cache_result = self.last_output_prompt
                    self.status = "LLM rewrite ready."
                    return self.last_output_prompt
                self.status = "LLM rewrite returned unchanged text — using concatenation."
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM rewrite failed: %s", exc)
                self.status = f"LLM rewrite failed, using concatenation: {exc}"
        self.last_output_prompt = concat
        return concat

    def _prompt_rewrite_cache_key(self, concat: str) -> str:
        """Key rewrites by style/text and a cheap WORK-canvas fingerprint."""
        digest = hashlib.sha1(concat.encode("utf-8"))
        if self.last_work is not None:
            sample = self.last_work.convert("RGB")
            sample.thumbnail((64, 64), Image.Resampling.BILINEAR)
            digest.update(str(sample.size).encode("ascii"))
            digest.update(sample.tobytes())
        return digest.hexdigest()

    def _rewrite_with_vram_headroom(self, concat: str) -> str:
        """Drop Flux, run Qwen-VL beside SDXL, then release Qwen."""
        keep_llm = bool(self.config.get("llm", "keep_loaded", default=False))
        flux_was = bool(self.flux and self.flux.ready)

        try:
            # Flux is the biggest resident and is no longer needed once WORK
            # has been generated. Unloading it leaves ample room for Qwen-VL
            # while keeping SDXL resident for the immediate OUTPUT pass.
            if flux_was:
                self.flux.unload()
                self._flux_reload_pending = True
            empty_cache()
            content_parts = [self.doc.background_prompt.strip()]
            content_parts.extend(
                obj.prompt.strip()
                for obj in self.doc.objects
                if obj.prompt_enabled
            )
            content_prompt = ", ".join(part for part in content_parts if part)
            preset = self.styles.active if self.styles else None
            settings = self.output_settings
            if settings.manual_style:
                style_name = "Manual style"
                style_prefix = settings.manual_prefix
                style_suffix = settings.manual_suffix
            else:
                style_name = preset.name if preset else ""
                style_prefix = preset.prefix if preset else ""
                style_suffix = preset.suffix if preset else ""
            return self.llm.rewrite(
                concat,
                reference_image=self.last_work,
                content_prompt=content_prompt,
                style_name=style_name,
                style_prefix=style_prefix,
                style_suffix=style_suffix,
            )
        finally:
            if not keep_llm and self.llm:
                self.llm.unload()
            empty_cache()

    def _schedule_flux_reload(self) -> None:
        """Warm Flux in the background after Qwen releases its VRAM."""
        if not self._flux_reload_pending or not self.flux:
            return
        if not bool(self.config.get("flux", "keep_loaded", default=True)):
            self._flux_reload_pending = False
            return
        if self.llm and self.llm.ready:
            return
        self._flux_reload_pending = False

        def _reload() -> None:
            with self._lock:
                if self.flux and not self.flux.ready:
                    try:
                        logger.info("Background-reloading Flux after LLM rewrite")
                        self.flux.load()
                        logger.info("Background Flux reload complete")
                    except Exception:  # noqa: BLE001
                        logger.exception("Background Flux reload failed")

        threading.Thread(target=_reload, daemon=True, name="flux-warm-reload").start()

    def run_output(self, work_image: Image.Image | None = None) -> Image.Image:
        """Run SDXL Hyper img2img on the WORK composition."""
        with self._lock:
            work = work_image or self.last_work or compose_work_image(self.doc)
            self.last_work = work
            if not self.core_ready:
                # Never replace an accepted OUTPUT with WORK while SDXL is
                # temporarily unloaded (e.g. during a SeedVR2 export).
                self.status = "SDXL not ready — OUTPUT left unchanged."
                if self.last_output is None:
                    self.last_output = work
                    self.output_rev += 1
                return self.last_output
            prompt = self.build_prompt()
            if not prompt.strip():
                prompt = "high quality image, detailed"
            # Sticky seed: never leave OUTPUT to chance mid-session.
            if self.output_settings.seed < 0:
                self.output_settings.seed = random.randint(0, 2_147_483_647)
            self.status = "Refining OUTPUT with SDXL Hyper…"
            try:
                out = self.sdxl.refine(
                    init_image=work,
                    prompt=prompt,
                    negative_prompt=self.output_settings.negative_prompt,
                    denoise=self.output_settings.denoise,
                    steps=self.output_settings.steps,
                    seed=int(self.output_settings.seed),
                    guidance_scale=self.output_settings.cfg,
                    eta=self.output_settings.eta,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("OUTPUT refine failed")
                self.status = f"OUTPUT failed: {exc}"
                # Keep the last good OUTPUT when possible; only fall back to
                # WORK if nothing has been refined yet.
                if self.last_output is None:
                    self.last_output = work
                    self.output_rev += 1
                self._schedule_flux_reload()
                return self.last_output
            self.last_output = out
            self.output_rev += 1
            self.status = "OUTPUT updated."
            self._schedule_flux_reload()
            return out

    # ------------------------------------------------------------------- export
    def refine_final(self, steps: int, denoise: float) -> Image.Image:
        """Refine the current OUTPUT again without replacing the WORK source."""
        with self._lock:
            if self.last_output is None:
                self.run_output()
            source = self.last_output
            if source is None:
                raise RuntimeError("No OUTPUT image is available to refine.")
            prompt = self.last_output_prompt or self.build_prompt()
            if not prompt.strip():
                prompt = "high quality image, detailed"
            self.status = "Refining the current OUTPUT with SDXL Hyper…"
            try:
                image = self.sdxl.refine(
                    init_image=source,
                    prompt=prompt,
                    negative_prompt=self.output_settings.negative_prompt,
                    denoise=float(denoise),
                    steps=int(steps),
                    seed=int(self.output_settings.seed),
                    guidance_scale=self.output_settings.cfg,
                    eta=self.output_settings.eta,
                )
            except Exception as exc:  # noqa: BLE001
                self.status = f"Final OUTPUT refinement failed: {exc}"
                raise
            self.last_output = image
            self.output_rev += 1
            self.status = (
                f"Refined current OUTPUT ({int(steps)} steps, "
                f"strength {float(denoise):.2f}). Review it, then export."
            )
            return image

    def export_final(
        self,
        refine_steps: int | None = None,
        seedvr2_options: dict[str, object] | None = None,
    ) -> tuple[Image.Image, Path]:
        """Upscale the currently accepted OUTPUT and save it."""
        with self._lock:
            # Ensure we have a current OUTPUT
            if self.last_output is None:
                self.run_output()
            source = self.last_output or self.last_work or compose_work_image(self.doc)

            # Optional final refine with more steps before upscale
            if refine_steps and refine_steps > 0:
                old_steps = self.output_settings.steps
                self.output_settings.steps = int(refine_steps)
                try:
                    source = self.run_output(self.last_work)
                finally:
                    self.output_settings.steps = old_steps

            seedvr2_export = (
                str(self.config.get("export", "upscaler", default="seedvr2")).lower()
                == "seedvr2"
            )
            if seedvr2_export:
                self.status = "Freeing VRAM for export-only SeedVR2…"
                if self.flux:
                    self.flux.unload()
                if self.sdxl:
                    self.sdxl.unload()
                if self.llm:
                    self.llm.unload()
                empty_cache()

            self.status = "Upscaling final export with SeedVR2…" if seedvr2_export else "Upscaling export (2x)…"
            try:
                path = self.upscaler.save_export(
                    source,
                    stem="xwave_export",
                    seedvr2_options=seedvr2_options,
                )
                backend = self.upscaler.backend or "upscaler"
            finally:
                # SeedVR2 runs in a child process, so all of its allocations
                # are gone here. Restore SDXL before returning so the very next
                # edit is processed instead of falling back to the WORK image.
                if seedvr2_export:
                    self.upscaler.unload()
                    empty_cache()
                    if self.sdxl and not self.sdxl.ready:
                        self.status = "Restoring SDXL after SeedVR2 export…"
                        try:
                            self.sdxl.load()
                        except Exception:  # noqa: BLE001
                            logger.exception("SDXL restore after export failed")
                            self.status = (
                                "Export saved, but SDXL restore failed — "
                                "press Load models before editing OUTPUT."
                            )
                    self._flux_reload_pending = bool(
                        self.flux and not self.flux.ready
                    )
                    self._schedule_flux_reload()
            exported = Image.open(path).convert("RGB")
            restored = bool(self.sdxl and self.sdxl.ready)
            self.status = (
                f"Exported {exported.width}×{exported.height} via {backend}: {path}"
                + (" · SDXL ready" if restored else " · SDXL not ready")
            )
            return exported, path

    # ------------------------------------------------------------------- helpers
    def _save_layer_image(self, image: Image.Image, name: str) -> Path:
        d = self.config.path("paths", "layers_dir", default="workspace/layers")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.png"
        image.save(path)
        return path

    def free_optional(self) -> str:
        """Unload LLM and upscaler to free VRAM."""
        if self.llm:
            self.llm.unload()
        if self.upscaler:
            self.upscaler.unload()
        empty_cache()
        return f"Optional models unloaded. {gpu_summary()}"
