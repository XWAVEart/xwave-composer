"""Background and object generation with Flux 2 klein (Apache 2.0).

Falls back to another Flux-family model when the primary id is missing.
All generation targets the configured CUDA device (RTX 5090).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, gpu_summary, hard_release
from xwave_composer.optimization import (
    OptimizationReport,
    compile_component,
    configured_profile,
    normalize_profile,
    profile_label,
    quantize_component,
)

logger = logging.getLogger(__name__)


class FluxGenerator:
    """Thin wrapper around a Flux diffusion pipeline for layer generation."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.pipe: Any = None
        self.model_id_loaded: str | None = None
        self._load_error: str | None = None
        self.profile = configured_profile(config)
        self.optimization_report = OptimizationReport(
            component="flux",
            requested=self.profile,
            applied=f"unloaded (next: {profile_label(self.profile)})",
        )

    @property
    def ready(self) -> bool:
        return self.pipe is not None

    def load(self, force: bool = False) -> str:
        """Load Flux 2 klein or the configured fallback. Returns status text."""
        if self.pipe is not None and not force:
            return f"Flux already loaded: {self.model_id_loaded}"

        primary = str(self.config.get("flux", "model_id", default="black-forest-labs/FLUX.2-klein-4B"))
        fallback = str(
            self.config.get(
                "flux", "fallback_model_id", default="black-forest-labs/FLUX.2-klein-base-4B"
            )
        )
        candidates = [primary]
        if fallback and fallback != primary:
            candidates.append(fallback)

        last_err: Exception | None = None
        for model_id in candidates:
            try:
                status = self._load_model(model_id)
                self._load_error = None
                return status
            except Exception as exc:  # noqa: BLE001 — try next candidate
                last_err = exc
                logger.warning("Failed to load Flux model %s: %s", model_id, exc)
                self.pipe = None
                empty_cache()

        self._load_error = str(last_err) if last_err else "unknown error"
        raise RuntimeError(
            f"Could not load any Flux model. Tried: {candidates}. Last error: {self._load_error}"
        )

    def _build_pipeline(self, model_id: str) -> Any:
        """Build an eager BF16 pipeline before optional optimization."""
        pipe = None
        # Prefer the dedicated FLUX.2 klein pipeline when available.
        try:
            from diffusers import Flux2KleinPipeline  # type: ignore

            pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=self.dtype)
            logger.info("Using Flux2KleinPipeline")
        except Exception as exc_klein:  # noqa: BLE001
            logger.info("Flux2KleinPipeline unavailable (%s); trying generic loaders.", exc_klein)

        if pipe is None:
            try:
                from diffusers import FluxPipeline  # type: ignore

                pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=self.dtype)
            except Exception:
                from diffusers import DiffusionPipeline

                try:
                    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=self.dtype)
                except TypeError:
                    pipe = DiffusionPipeline.from_pretrained(model_id)
        return pipe.to(self.device)

    def _load_model(self, model_id: str) -> str:
        logger.info("Loading Flux model: %s on %s (%s)", model_id, self.device, self.dtype)
        empty_cache()

        pipe = self._build_pipeline(model_id)
        report = quantize_component(
            pipe.transformer, self.profile, "flux", self.config
        )
        if self.profile != "bf16" and report.applied == "bf16":
            # quantize_ is in-place and can fail after converting some layers.
            # Never keep a potentially mixed/corrupt graph as the fallback.
            reason = report.fallback_reason
            logger.warning(
                "Flux %s unavailable; rebuilding clean BF16 pipeline: %s",
                profile_label(self.profile),
                reason,
            )
            del pipe
            empty_cache()
            pipe = self._build_pipeline(model_id)
            report = OptimizationReport(
                component="flux",
                requested=self.profile,
                fallback_reason=reason,
            )
        report = compile_component(pipe.transformer, report, self.config)
        self.optimization_report = report

        # Memory helpers when available
        if hasattr(pipe, "enable_vae_tiling"):
            try:
                pipe.enable_vae_tiling()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=False)

        self.pipe = pipe
        self.model_id_loaded = model_id
        msg = (
            f"Loaded Flux: {model_id} | {self.optimization_report.status()} | "
            f"{gpu_summary()}"
        )
        logger.info(msg)
        return msg

    def set_profile(self, profile: str, reload: bool = True) -> str:
        """Change the requested profile and optionally rebuild the pipeline."""
        selected = normalize_profile(profile)
        if selected == self.profile and (self.pipe is not None or not reload):
            return f"Flux profile already {profile_label(selected)}."
        self.profile = selected
        if not reload:
            return f"Flux profile set to {profile_label(selected)}."
        self.unload()
        return self.load()

    def unload(self) -> None:
        pipe = self.pipe
        self.pipe = None
        self.model_id_loaded = None
        self.optimization_report = OptimizationReport(
            component="flux",
            requested=self.profile,
            applied=f"unloaded (next: {profile_label(self.profile)})",
        )
        # Regional compile pins Inductor/CUDA caches — reset so SeedVR2 / LLM
        # headroom is real, not just a cleared Python reference.
        hard_release(pipe, reset_compiler=True)

    def ensure_loaded(self) -> None:
        if self.pipe is None:
            self.load()

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        seed: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        """Generate a single RGB image from a text prompt."""
        self.ensure_loaded()
        assert self.pipe is not None

        steps = steps if steps is not None else int(self.config.get("flux", "steps", default=4))
        guidance = (
            guidance_scale
            if guidance_scale is not None
            else float(self.config.get("flux", "guidance_scale", default=0.0))
        )
        max_seq = int(self.config.get("flux", "max_sequence_length", default=512))

        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "generator": generator,
        }
        if self.optimization_report.compiled:
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except Exception:  # noqa: BLE001
                pass
        with torch.inference_mode():
            try:
                call_kwargs["max_sequence_length"] = max_seq
                if negative_prompt:
                    call_kwargs["negative_prompt"] = negative_prompt
                result = self.pipe(**call_kwargs)
            except TypeError:
                # Retry with a minimal argument set
                minimal = {
                    "prompt": prompt,
                    "width": int(width),
                    "height": int(height),
                    "num_inference_steps": int(steps),
                    "generator": generator,
                }
                try:
                    minimal["guidance_scale"] = float(guidance)
                    result = self.pipe(**minimal)
                except TypeError:
                    result = self.pipe(
                        prompt=prompt,
                        num_inference_steps=int(steps),
                        generator=generator,
                    )

        image = result.images[0]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def generate_background(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate a full-frame background (no transparency)."""
        return self.generate(prompt=prompt, width=width, height=height, steps=steps, seed=seed)

    def generate_object(
        self,
        object_prompt: str,
        isolation_prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate an object on a plain background for later isolation.

        The isolation prompt is user-editable and describes the plain backdrop.
        Style is intentionally kept out of object layers.
        """
        isolation = isolation_prompt.strip() or "isolated object on plain white background"
        full_prompt = (
            f"{object_prompt.strip()}, {isolation}, "
            "single subject, centered, no artistic style filter, plain backdrop"
        )
        return self.generate(prompt=full_prompt, width=width, height=height, steps=steps, seed=seed)
