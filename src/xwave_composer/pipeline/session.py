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
    # True when last_output came from the fast low-res preview pass.
    output_is_preview: bool = False
    quality_mode: str = "quality"
    last_concat_prompt: str = ""
    _rewrite_cache_key: str = ""
    _rewrite_cache_result: str = ""
    _flux_reload_pending: bool = False
    # Exclusive UI mode: "compose" (Flux+WORK/OUTPUT) or "infinite_canvas" (SDXL stamps).
    active_system: str = "compose"
    output_rev: int = 0
    status: str = "Ready"
    # Pending object awaiting SAM2 click isolation
    pending_raw: Image.Image | None = None
    pending_prompt: str = ""
    pending_isolation_prompt: str = ""
    # Object-layer UI preference: when False, generate/import places the full
    # image and the isolation prompt is cleared / unused.
    layer_cutout: bool = True
    # Reject out-of-order canvas transform events (Gradio can deliver late).
    _last_transform_ts: dict[str, int] = field(default_factory=dict, repr=False)
    # True when pose fields changed without recomposing last_work (live drag).
    _work_stale: bool = field(default=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _flipbook_running: bool = field(default=False, repr=False)

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
        from xwave_composer.pipeline.quality_modes import (
            normalize_quality_mode,
            quality_mode_pack,
        )

        self.quality_mode = normalize_quality_mode(
            self.config.get("sdxl_hyper", "default_quality_mode", default="quality")
        )
        pack = quality_mode_pack(self.config, self.quality_mode)
        self.output_settings.denoise = float(
            self.config.get("sdxl_hyper", "default_denoise", default=pack["denoise"])
        )
        self.output_settings.steps = int(
            self.config.get("sdxl_hyper", "default_steps", default=pack["steps"])
        )
        self.output_settings.cfg = float(
            self.config.get("sdxl_hyper", "guidance_scale", default=1.0)
        )
        # Sticky OUTPUT seed — only changes when the user rolls a new one.
        self.output_settings.seed = random.randint(0, 2_147_483_647)
        self.output_settings.use_llm_rewrite = bool(
            self.config.get("llm", "enabled_by_default", default=False)
        )

    def apply_quality_mode(self, mode: str) -> OutputSettings:
        """Apply a Fast/Quality pack to OUTPUT denoise + steps."""
        from xwave_composer.pipeline.quality_modes import (
            QUALITY_MODE_LABELS,
            normalize_quality_mode,
            quality_mode_pack,
        )

        key = normalize_quality_mode(mode)
        pack = quality_mode_pack(self.config, key)
        with self._lock:
            self.quality_mode = key
            self.output_settings.denoise = float(pack["denoise"])
            self.output_settings.steps = int(pack["steps"])
            label = QUALITY_MODE_LABELS.get(key, key)
            self.status = (
                f"OUTPUT mode: {label} "
                f"(denoise {self.output_settings.denoise:.2f}, "
                f"steps {self.output_settings.steps})."
            )
            return self.output_settings

    def preview_steps_for_mode(self) -> int:
        from xwave_composer.pipeline.quality_modes import quality_mode_pack

        pack = quality_mode_pack(self.config, self.quality_mode)
        return max(1, int(pack["preview_steps"]))

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

    def ensure_work_composed(self) -> Image.Image:
        """Return an up-to-date WORK image, recomposing if pose-only edits pending."""
        with self._lock:
            if self._work_stale or self.last_work is None:
                self.last_work = compose_work_image(self.doc)
                self._work_stale = False
            return self.last_work

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
                if self.active_system == "compose":
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
            if (
                self.active_system == "compose"
                and self.last_work is not None
                and self.sdxl.ready
            ):
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
            width = self.doc.width
            height = self.doc.height
            flux = self.flux
        assert flux is not None
        # Flux runs outside the session lock so canvas transforms stay responsive.
        img = flux.generate_background(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed if seed >= 0 else None,
        )
        with self._lock:
            self.doc.background = img
            self.doc.background_prompt = prompt
            self.doc.background_source = "generated"
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
            width = min(1024, self.doc.width)
            height = min(1024, self.doc.height)
            flux = self.flux
        assert flux is not None
        img = flux.generate_object(
            object_prompt=prompt,
            isolation_prompt=isolation_prompt,
            width=width,
            height=height,
            seed=seed if seed >= 0 else None,
        )
        with self._lock:
            self.pending_raw = img
            self.pending_prompt = prompt
            self.pending_isolation_prompt = isolation_prompt

            if not auto_isolate:
                self.status = "Object generated. Run isolate to add it to WORK."
                return img

        # prefer=None honors config isolation.preferred (SAM2 center-point,
        # rembg fallback happens inside the isolator).
        return self.isolate_pending(click_xy=None, prefer=prefer)

    def _commit_isolated_object(
        self,
        raw: Image.Image,
        rgba: Image.Image,
        backend: str,
        prompt: str,
        isolation_prompt: str,
    ) -> Image.Image:
        """Add an isolated RGBA as a new object layer. Caller must hold self._lock."""
        if rgba.mode != "RGBA":
            rgba = rgba.convert("RGBA")
        extrema = rgba.getchannel("A").getextrema()
        if extrema is not None and extrema[1] == 0:
            logger.warning("Isolation produced empty alpha; using opaque object image.")
            rgba = raw.convert("RGBA")
            backend = f"{backend}+opaque_fallback"

        layer = ObjectLayer(
            name=f"Object {len(self.doc.objects) + 1}",
            prompt=prompt,
            isolation_prompt=isolation_prompt,
            image=rgba,
            raw_image=raw.copy(),
            source="generated",
            cutout=True,
        )
        max_dim = max(rgba.width, rgba.height, 1)
        target = min(self.doc.width, self.doc.height) * 0.45
        if max_dim > target:
            s = target / max_dim
            layer.transform.scale_x = s
            layer.transform.scale_y = s

        self.doc.add_object(layer)
        self._save_layer_image(rgba, f"object_{layer.id}")
        self.pending_raw = None
        self.last_work = compose_work_image(self.doc)
        self.status = (
            f"Object layer '{layer.name}' added via {backend} "
            f"({len(self.doc.objects)} object(s) on WORK)."
        )
        return self.last_work

    def _isolate_pending_unlocked(
        self,
        click_xy: tuple[float, float] | None = None,
        prefer: str | None = None,
    ) -> Image.Image:
        """Isolate pending raw and add object layer. Caller must hold self._lock.

        Prefer :meth:`isolate_pending` for UI paths — it releases the lock
        during the isolator GPU/CPU work.
        """
        if self.pending_raw is None:
            self.status = "No pending object to isolate."
            if self.last_work is None:
                self.last_work = compose_work_image(self.doc)
            return self.last_work

        self.status = "Isolating object…"
        raw = self.pending_raw
        try:
            rgba, backend = self.isolator.isolate(
                raw, click_xy=click_xy, prefer=prefer
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Isolation failed")
            rgba = raw.convert("RGBA")
            backend = f"none (isolation error: {exc})"

        return self._commit_isolated_object(
            raw,
            rgba,
            backend,
            self.pending_prompt,
            self.pending_isolation_prompt,
        )

    def isolate_pending(
        self,
        click_xy: tuple[float, float] | None = None,
        prefer: str | None = None,
    ) -> Image.Image | None:
        """Isolate pending raw image and add as object layer."""
        with self._lock:
            if self.pending_raw is None:
                self.status = "No pending object to isolate."
                if self.last_work is None:
                    self.last_work = compose_work_image(self.doc)
                return self.last_work
            self.status = "Isolating object…"
            raw = self.pending_raw.copy()
            prompt = self.pending_prompt
            isolation_prompt = self.pending_isolation_prompt
            isolator = self.isolator
        assert isolator is not None
        try:
            rgba, backend = isolator.isolate(
                raw, click_xy=click_xy, prefer=prefer
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Isolation failed")
            rgba = raw.convert("RGBA")
            backend = f"none (isolation error: {exc})"
        with self._lock:
            # Another generate may have replaced pending; only commit if raw still matches.
            if self.pending_raw is None:
                return self.last_work or compose_work_image(self.doc)
            return self._commit_isolated_object(
                self.pending_raw,
                rgba,
                backend,
                prompt,
                isolation_prompt,
            )

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
                self.doc.background_source = "imported"
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
            obj.source = "imported"
            obj.cutout = mode != "none"
            obj.origin_id = None
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
            obj = self.doc.selected() if sid not in (None, "__bg__") else None
            if sid in (None, "__bg__") or obj is None:
                go_background = True
            else:
                go_background = False
                obj_id = obj.id
                iso = (
                    (isolation_prompt or obj.isolation_prompt or "").strip()
                    if isolate
                    else ""
                )
                width = min(1024, self.doc.width)
                height = min(1024, self.doc.height)
                flux = self.flux
                isolator = self.isolator
                self.status = "Generating object with Flux…"

        if go_background:
            return self.generate_background(prompt, seed=seed)

        assert flux is not None
        if isolate:
            raw = flux.generate_object(
                object_prompt=prompt,
                isolation_prompt=iso,
                width=width,
                height=height,
                seed=seed if seed >= 0 else None,
            )
        else:
            # Full-image placement: no plain-backdrop coaxing, no cutout.
            raw = flux.generate(
                prompt=prompt,
                width=width,
                height=height,
                seed=seed if seed >= 0 else None,
            )

        if not isolate:
            rgba, backend = raw.convert("RGBA"), "full image (no cutout)"
        else:
            assert isolator is not None
            with self._lock:
                self.status = "Isolating object…"
            try:
                rgba, backend = isolator.isolate(raw, prefer=None)
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

        with self._lock:
            obj = self.doc.find_by_id(obj_id)
            if obj is None:
                self.status = "Selected layer disappeared during generate."
                return self.last_work or compose_work_image(self.doc)
            first_image = obj.image is None
            obj.prompt = prompt
            obj.isolation_prompt = iso
            obj.image = rgba
            obj.raw_image = raw.copy()
            obj.source = "generated"
            obj.cutout = bool(isolate)
            if first_image:
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
            obj_id = obj.id
            raw = obj.raw_image.copy()
            isolator = self.isolator
            self.status = "Isolating object…"
        assert isolator is not None
        rgba, backend = isolator.isolate(raw, click_xy=click_xy, prefer=prefer)
        with self._lock:
            obj = self.doc.find_by_id(obj_id)
            if obj is None:
                self.status = "Selected layer disappeared during re-isolate."
                return self.last_work
            obj.image = rgba
            self.last_work = compose_work_image(self.doc)
            self.status = f"Re-isolated via {backend}."
            return self.last_work

    # ------------------------------------------------------------------ canvas
    def refresh_work(self) -> Image.Image:
        with self._lock:
            self.last_work = compose_work_image(self.doc)
            self._work_stale = False
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
        flip_x: bool | None = None,
        flip_y: bool | None = None,
    ) -> Image.Image:
        with self._lock:
            obj = self.doc.selected()
            if obj is None:
                return self.refresh_work()
            return self._apply_transform(
                obj,
                x,
                y,
                scale_x,
                scale_y,
                rotation,
                opacity,
                flip_x=flip_x,
                flip_y=flip_y,
            )

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
            t.flip_x = False
            t.flip_y = False
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
                source=str(source.source or "generated"),
                cutout=bool(source.cutout),
                # Point at the ultimate origin so Roll all regenerates once.
                origin_id=source.origin_id or source.id,
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

    def toggle_selected_visibility(self) -> bool | None:
        """Hide/show the selected object on WORK and OUTPUT (prompt unchanged)."""
        with self._lock:
            obj = self.doc.selected()
            if obj is None:
                self.status = "Select an object layer to hide or show."
                return None
            obj.transform.visible = not bool(obj.transform.visible)
            self.last_work = compose_work_image(self.doc)
            state = "shown" if obj.transform.visible else "hidden"
            self.status = f"Layer {state}: {obj.name or obj.id}."
            return obj.transform.visible

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

    def roll_all(self) -> Image.Image:
        """Regenerate every generated layer (muted included); leave imports alone.

        Clears pending SAM state first. For duplicates, re-rolls the origin
        once, then copies the new pixels onto each duplicate while keeping
        that duplicate's position, rotation, and x/y flips (and the rest of
        its transform). Re-rolls the OUTPUT seed and runs a full refine.
        """
        with self._lock:
            self.pending_raw = None
            self.pending_prompt = ""
            self.pending_isolation_prompt = ""
            bg_prompt = (self.doc.background_prompt or "").strip()
            bg_imported = (
                str(self.doc.background_source or "").strip().lower() == "imported"
            )
            objects = list(self.doc.objects)
            by_id = {obj.id: obj for obj in objects}

        rolled = 0
        skipped_import = 0

        if bg_imported:
            skipped_import += 1
        elif bg_prompt:
            self.status = "Roll all — regenerating background…"
            try:
                self.generate_background(bg_prompt, seed=-1)
                rolled += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("roll_all: background failed")
                self.status = f"Roll all: background failed: {exc}"

        # Group by ultimate origin so each unique image is generated once.
        groups: dict[str, list[str]] = {}
        for obj in objects:
            if obj.is_imported:
                skipped_import += 1
                continue
            if obj.origin_id and obj.origin_id in by_id:
                origin = by_id[obj.origin_id]
                if origin.is_imported:
                    skipped_import += 1
                    continue
                root = obj.origin_id
            elif obj.origin_id:
                root = f"orphan:{obj.origin_id}"
            else:
                root = obj.id
            if not (obj.prompt or "").strip():
                # Still allow origin-less empty shells to be skipped; members
                # with prompts will pull an origin that may lack a prompt.
                if root == obj.id:
                    continue
            groups.setdefault(root, []).append(obj.id)

        # Include origins that only appear via duplicates (no prompt on origin).
        for root, member_ids in list(groups.items()):
            if root.startswith("orphan:"):
                continue
            if root not in member_ids and root in by_id:
                member_ids.insert(0, root)

        for root, member_ids in groups.items():
            with self._lock:
                members = [
                    self.doc.find_by_id(mid) for mid in member_ids
                ]
                members = [m for m in members if m is not None]
                if not members:
                    continue
                if root.startswith("orphan:"):
                    primary = members[0]
                else:
                    primary = self.doc.find_by_id(root) or members[0]
                prompt_src = next(
                    (
                        m
                        for m in ([primary] + members)
                        if (m.prompt or "").strip()
                    ),
                    None,
                )
                if prompt_src is None:
                    continue
                primary_id = primary.id
                prompt = prompt_src.prompt.strip()
                iso = (prompt_src.isolation_prompt or primary.isolation_prompt or "")
                isolate = bool(primary.cutout if primary.image is not None else prompt_src.cutout)
                self.doc.selected_id = primary_id
                self.status = (
                    f"Roll all — regenerating {primary.name or primary_id}…"
                )

            try:
                self.generate_selected(
                    prompt, iso, seed=-1, isolate=isolate
                )
                rolled += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("roll_all: layer %s failed", primary_id)
                self.status = f"Roll all: layer failed: {exc}"
                continue

            with self._lock:
                origin = self.doc.find_by_id(primary_id)
                if origin is None or origin.image is None:
                    continue
                for mid in member_ids:
                    if mid == primary_id:
                        continue
                    clone = self.doc.find_by_id(mid)
                    if clone is None or clone.is_imported:
                        continue
                    # Replace pixels only — keep pose (x/y, rotation, flips,
                    # scale, opacity, visibility) and blend/feather.
                    clone.image = origin.image.copy()
                    clone.raw_image = (
                        origin.raw_image.copy()
                        if origin.raw_image is not None
                        else None
                    )
                    clone.source = "generated"
                    clone.cutout = bool(origin.cutout)
                    clone.path = self._save_layer_image(
                        clone.image, f"object_{clone.id}"
                    )
                self.last_work = compose_work_image(self.doc)

        self.roll_output_seed()
        out = self.run_output()
        with self._lock:
            parts = [f"Roll all done — regenerated {rolled} layer(s)"]
            if skipped_import:
                parts.append(f"skipped {skipped_import} import(s)")
            self.status = ", ".join(parts) + "."
        return out if out is not None else (
            self.last_work or compose_work_image(self.doc)
        )

    def update_transform_by_id(
        self,
        layer_id: str,
        x: float | None = None,
        y: float | None = None,
        scale_x: float | None = None,
        scale_y: float | None = None,
        rotation: float | None = None,
        opacity: float | None = None,
        flip_x: bool | None = None,
        flip_y: bool | None = None,
        *,
        compose: bool = True,
    ) -> Image.Image:
        """Update a layer pose. Live drags pass ``compose=False`` to skip Pillow."""
        with self._lock:
            obj = self.doc.find_by_id(layer_id)
            if obj is None:
                return self.refresh_work()
            self.doc.selected_id = layer_id
            return self._apply_transform(
                obj,
                x,
                y,
                scale_x,
                scale_y,
                rotation,
                opacity,
                flip_x=flip_x,
                flip_y=flip_y,
                compose=compose,
            )

    def _apply_transform(
        self,
        obj: ObjectLayer,
        x: float | None,
        y: float | None,
        scale_x: float | None,
        scale_y: float | None,
        rotation: float | None,
        opacity: float | None,
        flip_x: bool | None = None,
        flip_y: bool | None = None,
        *,
        compose: bool = True,
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
        if flip_x is not None:
            t.flip_x = bool(flip_x)
        if flip_y is not None:
            t.flip_y = bool(flip_y)
        if compose:
            self.last_work = compose_work_image(self.doc)
            self._work_stale = False
            return self.last_work
        # Live drag: keep doc poses current; defer expensive compose until
        # pointer-up / OUTPUT refine.
        self._work_stale = True
        return self.last_work or compose_work_image(self.doc)

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

        Always runs under ``self._lock`` so Flux unload / Qwen load cannot
        race unlocked UI callers or debounce workers.
        """
        with self._lock:
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
                    self.status = (
                        "LLM rewrite returned unchanged text — using concatenation."
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("LLM rewrite failed: %s", exc)
                    self.status = f"LLM rewrite failed, using concatenation: {exc}"
            self.last_output_prompt = concat
            return concat

    def _prompt_rewrite_cache_key(self, concat: str) -> str:
        """Key rewrites by style/text and a cheap WORK-canvas fingerprint.

        Caller must hold ``self._lock``.
        """
        digest = hashlib.sha1(concat.encode("utf-8"))
        if self.last_work is not None:
            sample = self.last_work.convert("RGB")
            sample.thumbnail((64, 64), Image.Resampling.BILINEAR)
            digest.update(str(sample.size).encode("ascii"))
            digest.update(sample.tobytes())
        return digest.hexdigest()

    def _rewrite_with_vram_headroom(self, concat: str) -> str:
        """Drop Flux, run Qwen-VL beside SDXL, then release Qwen.

        Caller must hold ``self._lock`` for the entire unload → rewrite →
        unload choreography so generation cannot touch Flux mid-flight.
        """
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
            assert self.llm is not None
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
        if self.active_system != "compose":
            # Never pull Flux back while Infinite Canvas owns the GPU.
            self._flux_reload_pending = False
            return
        if not bool(self.config.get("flux", "keep_loaded", default=True)):
            self._flux_reload_pending = False
            return
        if self.llm and self.llm.ready:
            return
        self._flux_reload_pending = False

        def _reload() -> None:
            with self._lock:
                if self.active_system != "compose":
                    return
                if self.flux and not self.flux.ready:
                    try:
                        logger.info("Background-reloading Flux after LLM rewrite")
                        self.flux.load()
                        logger.info("Background Flux reload complete")
                    except Exception:  # noqa: BLE001
                        logger.exception("Background Flux reload failed")

        threading.Thread(target=_reload, daemon=True, name="flux-warm-reload").start()

    def run_output(
        self,
        work_image: Image.Image | None = None,
        *,
        preview: bool = False,
    ) -> Image.Image:
        """Run SDXL Hyper img2img on the WORK composition.

        Snapshot under the session lock; prompt build (incl. LLM) takes the
        lock on its own; SDXL refine runs unlocked so canvas transforms are
        not blocked for the full inference.

        ``preview`` trades resolution and steps for latency so OUTPUT can
        keep up with dragging. A full-quality pass follows once movement
        stops. Export / refine / flipbook call ``ensure_full_output`` first.
        """
        with self._lock:
            if work_image is not None:
                work = work_image
                self.last_work = work
                self._work_stale = False
            elif self._work_stale or self.last_work is None:
                work = compose_work_image(self.doc)
                self.last_work = work
                self._work_stale = False
            else:
                work = self.last_work
            work_snap = work.copy()
            if not self.core_ready:
                # Never replace an accepted OUTPUT with WORK while SDXL is
                # temporarily unloaded (e.g. during a SeedVR2 export).
                self.status = "SDXL not ready — OUTPUT left unchanged."
                if self.last_output is None:
                    self.last_output = work
                    self.output_rev += 1
                return self.last_output

        # May unload Flux under lock for Qwen; do not hold our outer critical
        # section across this call (RLock would still block other waiters).
        prompt = self.build_prompt()
        if not prompt.strip():
            prompt = "high quality image, detailed"

        with self._lock:
            if not self.core_ready:
                self.status = "SDXL not ready — OUTPUT left unchanged."
                if self.last_output is None:
                    self.last_output = work_snap
                    self.output_rev += 1
                return self.last_output
            if self.output_settings.seed < 0:
                self.output_settings.seed = random.randint(0, 2_147_483_647)
            init = work_snap
            steps = int(self.output_settings.steps)
            if preview:
                scale = float(
                    self.config.get("sdxl_hyper", "preview_scale", default=0.625)
                )
                steps = self.preview_steps_for_mode()
                pw = max(256, (int(work_snap.width * scale) // 8) * 8)
                ph = max(256, (int(work_snap.height * scale) // 8) * 8)
                if (pw, ph) != work_snap.size:
                    init = work_snap.resize((pw, ph), Image.Resampling.BILINEAR)
                status_msg = "Previewing…"
            else:
                status_msg = "Refining OUTPUT with SDXL Hyper…"
            refine_kwargs = {
                "init_image": init,
                "prompt": prompt,
                "negative_prompt": self.output_settings.negative_prompt,
                "denoise": self.output_settings.denoise,
                "steps": steps,
                "seed": int(self.output_settings.seed),
                "guidance_scale": self.output_settings.cfg,
                "eta": self.output_settings.eta,
            }
            target_size = work_snap.size
            sdxl = self.sdxl
            self.status = status_msg

        assert sdxl is not None
        try:
            out = sdxl.refine(**refine_kwargs)
            if preview and out.size != target_size:
                # Back to canvas size so the OUTPUT pane does not jump.
                out = out.resize(target_size, Image.Resampling.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            logger.exception("OUTPUT refine failed")
            with self._lock:
                self.status = f"OUTPUT failed: {exc}"
                if self.last_output is None:
                    self.last_output = work_snap
                    self.output_rev += 1
                result = self.last_output
                self._schedule_flux_reload()
            return result

        with self._lock:
            self.last_output = out
            self.output_is_preview = bool(preview)
            self.output_rev += 1
            self.status = "Preview — settling…" if preview else "OUTPUT updated."
            self._schedule_flux_reload()
            return out

    def ensure_full_output(self) -> Image.Image | None:
        """Re-render at full quality when the current OUTPUT is only a preview."""
        with self._lock:
            needs = self.output_is_preview or self.last_output is None
        if needs:
            return self.run_output(preview=False)
        return self.last_output

    # ------------------------------------------------------------------- export
    def refine_final(self, steps: int, denoise: float) -> Image.Image:
        """Refine the current OUTPUT again without replacing the WORK source."""
        self.ensure_full_output()
        if self.last_output is None:
            self.run_output(preview=False)

        with self._lock:
            source = self.last_output
            if source is None:
                raise RuntimeError("No OUTPUT image is available to refine.")
            source_snap = source.copy()
            prompt = self.last_output_prompt or ""
            neg = self.output_settings.negative_prompt
            seed = int(self.output_settings.seed)
            cfg = self.output_settings.cfg
            eta = self.output_settings.eta
            sdxl = self.sdxl
            self.status = "Refining the current OUTPUT with SDXL Hyper…"

        if not prompt.strip():
            prompt = self.build_prompt()
        if not prompt.strip():
            prompt = "high quality image, detailed"

        assert sdxl is not None
        try:
            image = sdxl.refine(
                init_image=source_snap,
                prompt=prompt,
                negative_prompt=neg,
                denoise=float(denoise),
                steps=int(steps),
                seed=seed,
                guidance_scale=cfg,
                eta=eta,
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.status = f"Final OUTPUT refinement failed: {exc}"
            raise

        with self._lock:
            self.last_output = image
            self.output_is_preview = False
            self.output_rev += 1
            self.status = (
                f"Refined current OUTPUT ({int(steps)} steps, "
                f"strength {float(denoise):.2f}). Review it, then export."
            )
            return image

    def export_style_flipbook(
        self,
        style_count: int = 32,
        fps: int = 30,
        frames_per_image: int = 8,
        lock_seed: bool = True,
        families: list[str] | None = None,
        mode: str = "styles",
    ) -> tuple[Path | None, str]:
        """Refine WORK into a flipbook MP4.

        Modes:
        - ``styles``: vary style prompts (CFG/Denoise/Eta fixed). Frame 0 is
          the current OUTPUT; remaining stills are shuffled after it.
          ``lock_seed`` keeps one seed; otherwise each style gets a new seed.
          ``families`` limits the pool (``None`` = all; empty = error).
        - ``seeds``: keep the current OUTPUT style/prompt settings; vary only
          the seed. Frame 0 is the current OUTPUT; remaining stills stay in
          generation order. ``lock_seed`` increments ``seed+1…``; otherwise
          each frame gets a random seed. ``style_count`` is total stills
          (including frame 0); must be ≥ 2.
        """
        from datetime import datetime

        from xwave_composer.pipeline.style_flipbook import (
            assemble_flipbook_stills,
            flipbook_seeds_for_frames,
            sample_style_names,
            styles_to_generate,
            write_flipbook_mp4,
        )

        mode_key = str(mode or "styles").strip().lower()
        if mode_key in ("seed", "seeds"):
            mode_key = "seeds"
        else:
            mode_key = "styles"

        with self._lock:
            if self._flipbook_running:
                return None, "Style flipbook already running."
            if self.last_output is None:
                return None, "Refine OUTPUT first — flipbook needs a starting frame."
            if self.last_work is None and self.doc.background is None:
                return None, "Generate a WORK composition before running a flipbook."
            if not self.core_ready or self.sdxl is None or not self.sdxl.ready:
                return None, "SDXL not ready — load models before flipbook export."

            all_names: list[str] = []
            if mode_key == "styles":
                if not self.styles or not self.styles.names():
                    return None, "No style presets loaded."
                if families is None:
                    all_names = list(self.styles.names())
                else:
                    fam_filter = [f for f in families if f]
                    if not fam_filter:
                        return None, "Select at least one style family for the flipbook."
                    all_names = list(self.styles.names(fam_filter))
                    if not all_names:
                        return None, "No styles in the selected families."
            else:
                n_stills = int(style_count or 0)
                if n_stills < 2:
                    return None, "Seed flipbook needs at least 2 frames (incl. current OUTPUT)."

        # Never stitch a low-res interactive preview into a flipbook.
        self.ensure_full_output()

        with self._lock:
            if self._flipbook_running:
                return None, "Style flipbook already running."
            if self.last_output is None:
                return None, "Refine OUTPUT first — flipbook needs a starting frame."
            self._flipbook_running = True
            frame0 = self.last_output.copy()
            work_snap = (
                self.last_work.copy()
                if self.last_work is not None
                else compose_work_image(self.doc)
            )
            s = self.output_settings
            saved = {
                "active_name": self.styles.active_name if self.styles else None,
                "seed": int(s.seed),
                "negative": str(s.negative_prompt or ""),
                "use_llm": bool(s.use_llm_rewrite),
                "prompt_locked": bool(s.prompt_locked),
                "custom_prompt": str(s.custom_prompt or ""),
                "manual_style": bool(s.manual_style),
                "manual_prefix": str(s.manual_prefix or ""),
                "manual_suffix": str(s.manual_suffix or ""),
                "cfg": float(s.cfg),
                "denoise": float(s.denoise),
                "eta": float(s.eta),
                "steps": int(s.steps),
            }
            current_name = self.styles.active_name if self.styles else None
            if s.seed < 0:
                s.seed = random.randint(0, 2_147_483_647)
            locked_seed = int(s.seed)
            if mode_key == "styles":
                # Force concat path for style-swap frames.
                s.use_llm_rewrite = False
                s.prompt_locked = False
                s.manual_style = False
            else:
                # Keep current style/prompt path; only seeds change.
                s.use_llm_rewrite = False

        path: Path | None = None
        msg = "Flipbook cancelled."
        try:
            generated: list[Image.Image] = []
            if mode_key == "styles":
                selected = sample_style_names(all_names, current_name, int(style_count))
                to_gen = styles_to_generate(selected, current_name)
                total = len(to_gen)
                for i, name in enumerate(to_gen, start=1):
                    with self._lock:
                        preset = self.styles.set_active(name) if self.styles else None
                        if preset is None:
                            continue
                        self.output_settings.negative_prompt = preset.negative
                        self.output_settings.cfg = saved["cfg"]
                        self.output_settings.denoise = saved["denoise"]
                        self.output_settings.eta = saved["eta"]
                        self.output_settings.steps = saved["steps"]
                        if lock_seed:
                            self.output_settings.seed = locked_seed
                        else:
                            self.output_settings.seed = random.randint(0, 2_147_483_647)
                        self.status = f"Style flipbook {i}/{total}: {name}…"

                    out = self.run_output(work_snap)
                    generated.append(out.copy())
                stills = assemble_flipbook_stills(frame0, generated, shuffle_rest=True)
                stamp_prefix = "style_flipbook"
                kind = "Style"
            else:
                n_stills = int(style_count)
                seeds = flipbook_seeds_for_frames(
                    locked_seed,
                    n_stills - 1,
                    increment=bool(lock_seed),
                )
                total = len(seeds)
                style_label = current_name or ("manual" if saved["manual_style"] else "current")
                for i, seed in enumerate(seeds, start=1):
                    with self._lock:
                        self.output_settings.cfg = saved["cfg"]
                        self.output_settings.denoise = saved["denoise"]
                        self.output_settings.eta = saved["eta"]
                        self.output_settings.steps = saved["steps"]
                        self.output_settings.negative_prompt = saved["negative"]
                        self.output_settings.manual_style = saved["manual_style"]
                        self.output_settings.manual_prefix = saved["manual_prefix"]
                        self.output_settings.manual_suffix = saved["manual_suffix"]
                        self.output_settings.prompt_locked = saved["prompt_locked"]
                        self.output_settings.custom_prompt = saved["custom_prompt"]
                        self.output_settings.seed = int(seed)
                        self.status = (
                            f"Seed flipbook {i}/{total}: {style_label} · seed {seed}…"
                        )

                    out = self.run_output(work_snap)
                    generated.append(out.copy())
                stills = assemble_flipbook_stills(frame0, generated, shuffle_rest=False)
                stamp_prefix = "seed_flipbook"
                kind = "Seed"

            out_dir = self.config.path("export", "output_dir", default="exports")
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"{stamp_prefix}_{stamp}.mp4"
            with self._lock:
                self.status = f"Encoding flipbook ({len(stills)} stills)…"
            write_flipbook_mp4(
                stills,
                path,
                fps=int(fps),
                frames_per_image=int(frames_per_image),
            )
            msg = (
                f"{kind} flipbook saved ({len(stills)} stills, "
                f"{fps} fps, {frames_per_image} frames/image): {path}"
            )
            return path, msg
        except Exception as exc:  # noqa: BLE001
            logger.exception("Flipbook failed")
            msg = f"Flipbook failed: {exc}"
            return None, msg
        finally:
            with self._lock:
                if self.styles is not None:
                    self.styles.set_active(saved["active_name"])
                s = self.output_settings
                s.seed = saved["seed"]
                s.negative_prompt = saved["negative"]
                s.use_llm_rewrite = saved["use_llm"]
                s.prompt_locked = saved["prompt_locked"]
                s.custom_prompt = saved["custom_prompt"]
                s.manual_style = saved["manual_style"]
                s.manual_prefix = saved["manual_prefix"]
                s.manual_suffix = saved["manual_suffix"]
                s.cfg = saved["cfg"]
                s.denoise = saved["denoise"]
                s.eta = saved["eta"]
                s.steps = saved["steps"]
                self.last_output = frame0
                self.output_rev += 1
                self._flipbook_running = False
                self.status = msg

    def export_final(
        self,
        refine_steps: int | None = None,
        seedvr2_options: dict[str, object] | None = None,
    ) -> tuple[Image.Image, Path]:
        """Upscale the currently accepted OUTPUT and save it."""
        # Ensure OUTPUT without holding the lock across SDXL.
        # Never upscale a low-res interactive preview.
        self.ensure_full_output()
        if self.last_output is None:
            self.run_output(preview=False)

        with self._lock:
            source = self.last_output or self.last_work or compose_work_image(self.doc)
            source = source.copy()
            need_step_refine = bool(refine_steps and refine_steps > 0)
            denoise = float(self.output_settings.denoise)
            seedvr2_export = (
                str(self.config.get("export", "upscaler", default="seedvr2")).lower()
                == "seedvr2"
            )

        if need_step_refine:
            # Same semantics as refine_final: deeper img2img on accepted OUTPUT.
            source = self.refine_final(
                steps=int(refine_steps), denoise=denoise
            ).copy()

        with self._lock:
            if seedvr2_export:
                self.status = "Freeing VRAM for export-only SeedVR2…"
                if self.flux:
                    self.flux.unload()
                if self.sdxl:
                    self.sdxl.unload()
                if self.llm:
                    self.llm.unload()
                if self.isolator:
                    self.isolator.unload()
                empty_cache()
            self.status = (
                "Upscaling final export with SeedVR2…"
                if seedvr2_export
                else "Upscaling export (2x)…"
            )
            upscaler = self.upscaler

        assert upscaler is not None
        try:
            path = upscaler.save_export(
                source,
                stem="xwave_export",
                seedvr2_options=seedvr2_options,
            )
            backend = upscaler.backend or "upscaler"
        finally:
            # SeedVR2 runs in a child process, so all of its allocations
            # are gone here. Restore SDXL before returning so the very next
            # edit is processed instead of falling back to the WORK image.
            if seedvr2_export:
                with self._lock:
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
        with self._lock:
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

    def _unload_compose_stack(self) -> list[str]:
        """Unload Flux / isolator / LLM / upscaler. Keeps shared SDXL."""
        unloaded: list[str] = []
        self._flux_reload_pending = False
        if self.flux and self.flux.ready:
            self.flux.unload()
            unloaded.append("Flux")
        if self.llm:
            was = bool(getattr(self.llm, "ready", False))
            self.llm.unload()
            if was:
                unloaded.append("LLM")
        if self.upscaler:
            self.upscaler.unload()
            unloaded.append("upscaler")
        if self.isolator:
            try:
                self.isolator.unload()
                unloaded.append("isolator")
            except Exception:  # noqa: BLE001
                logger.debug("isolator unload failed", exc_info=True)
        empty_cache()
        return unloaded

    def enter_compose_mode(self) -> str:
        """Activate Compose (Flux + WORK/OUTPUT). Does not auto-load models."""
        with self._lock:
            self.active_system = "compose"
            empty_cache()
            self.status = (
                f"Compose mode active. Press Load models if Flux is unloaded. "
                f"{gpu_summary()}"
            )
            return self.status

    def enter_infinite_canvas_mode(self) -> str:
        """Activate Infinite Canvas: unload Compose stack, keep SDXL for stamps."""
        with self._lock:
            self.active_system = "infinite_canvas"
            flux_was_loaded = bool(self.flux and self.flux.ready)
            unloaded = self._unload_compose_stack()
            # Drop CUDA-graph pools only when Flux actually held them. This
            # method runs before every generate; an unconditional reset forced
            # a full UNet recompile per patch (~16s instead of ~2s).
            if flux_was_loaded and self.sdxl and getattr(
                self.sdxl, "optimization_report", None
            ):
                if self.sdxl.optimization_report.compiled:
                    try:
                        import torch

                        torch.compiler.reset()
                    except Exception:  # noqa: BLE001
                        pass
            empty_cache()
            what = ", ".join(unloaded) if unloaded else "Compose stack already clear"
            self.status = (
                f"Infinite Canvas mode — unloaded {what}. "
                f"SDXL kept for region gen. {gpu_summary()}"
            )
            return self.status

    def free_optional(self) -> str:
        """Unload Compose stack (Flux etc.); keep SDXL. Used by Free VRAM buttons."""
        with self._lock:
            unloaded = self._unload_compose_stack()
            what = ", ".join(unloaded) if unloaded else "caches"
            self.status = f"Freed {what}. SDXL kept loaded. {gpu_summary()}"
            return self.status

    @property
    def compose_active(self) -> bool:
        return self.active_system == "compose"

    @property
    def infinite_canvas_active(self) -> bool:
        return self.active_system == "infinite_canvas"
