"""OUTPUT stage: SDXL Hyper / Hyper-SDXL img2img refinement.

Uses the composed WORK canvas as the init image.
User controls denoise strength and step count for live vs quality balance.
Supports style LoRAs and textual inversions on this stage only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, gpu_summary, hard_release
from xwave_composer.optimization import (
    OptimizationReport,
    compile_component,
    configured_profile,
    make_nvfp4_linear_inputs_contiguous,
    normalize_profile,
    profile_label,
    quantize_component,
)

logger = logging.getLogger(__name__)

# Diffusers-format SDXL fine-tunes that pair well with the Hyper LoRA.
BASE_MODEL_PRESETS: dict[str, str] = {
    "SDXL Base 1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "DreamShaper XL": "Lykon/dreamshaper-xl-1-0",
    "Juggernaut XL v9": "RunDiffusion/Juggernaut-XL-v9",
    "epiCRealism XL": "John6666/epicrealism-xl-vxvii-crystal-clear-realism-sdxl",
    "RealVisXL V5.0": "SG161222/RealVisXL_V5.0",
}


class SDXLHyperPipeline:
    """Fast SDXL img2img pipeline tuned for low-step Hyper-SD style adapters."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.pipe: Any = None
        self.loaded_loras: list[str] = []
        self.loaded_embeddings: list[str] = []
        self._base_id: str | None = None
        self._style_lora_specs: list[tuple[str, str | None, float]] = []
        self._embedding_specs: list[tuple[str, str | None]] = []
        self.profile = configured_profile(config)
        self.optimization_report = OptimizationReport(
            component="sdxl",
            requested=self.profile,
            applied=f"unloaded (next: {profile_label(self.profile)})",
        )

    @property
    def ready(self) -> bool:
        return self.pipe is not None

    @property
    def base_id(self) -> str | None:
        return self._base_id

    def _load_base_pipe(self, base_ref: str) -> Any:
        """Load an SDXL img2img pipeline (on CPU) from an HF repo id,
        single-file URL (CivitAI download link), or local .safetensors path."""
        from diffusers import AutoPipelineForImage2Image, StableDiffusionXLImg2ImgPipeline

        ref = base_ref.strip()
        # An HF web URL like https://huggingface.co/org/name -> repo id
        if ref.startswith("http") and "huggingface.co/" in ref and ".safetensors" not in ref:
            ref = ref.split("huggingface.co/", 1)[1].strip("/")
            ref = "/".join(ref.split("/")[:2])
        is_single_file = ref.endswith(".safetensors") or "civitai.com" in ref
        if is_single_file:
            return StableDiffusionXLImg2ImgPipeline.from_single_file(
                ref, torch_dtype=self.dtype, use_safetensors=True
            )
        # Some fine-tune repos (e.g. Juggernaut XL v9) only ship fp16-variant
        # weight files, so try both variants before giving up.
        errors: list[str] = []
        for loader in (AutoPipelineForImage2Image, StableDiffusionXLImg2ImgPipeline):
            for variant in (None, "fp16"):
                try:
                    return loader.from_pretrained(
                        ref,
                        torch_dtype=self.dtype,
                        variant=variant,
                        use_safetensors=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{loader.__name__}(variant={variant}): {exc}")
        raise RuntimeError(
            f"Could not load SDXL base '{ref}'. Tried: " + " | ".join(errors[-2:])
        )

    def _prepare_pipeline(
        self,
        base_id: str,
        hyper_lora_id: str | None,
        hyper_weight: str | None,
    ) -> tuple[Any, list[str]]:
        """Load, move, schedule, and attach the fixed Hyper adapter."""
        pipe = self._load_base_pipe(base_id).to(self.device)

        try:
            from diffusers import DDIMScheduler, TCDScheduler

            try:
                pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
            except Exception:
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        except Exception as exc:  # noqa: BLE001
            logger.info("Scheduler swap skipped: %s", exc)

        if hasattr(pipe, "enable_vae_tiling"):
            try:
                pipe.enable_vae_tiling()
            except Exception:  # noqa: BLE001
                pass

        loaded_loras: list[str] = []
        if hyper_lora_id:
            try:
                pipe.load_lora_weights(
                    hyper_lora_id,
                    weight_name=hyper_weight,
                    adapter_name="hyper",
                )
                if hasattr(pipe, "set_adapters"):
                    pipe.set_adapters(["hyper"], adapter_weights=[1.0])
                elif hasattr(pipe, "fuse_lora"):
                    pipe.fuse_lora(lora_scale=1.0)
                loaded_loras.append(f"{hyper_lora_id}/{hyper_weight}")
                logger.info("Hyper LoRA loaded: %s / %s", hyper_lora_id, hyper_weight)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Hyper LoRA not loaded (%s). OUTPUT still works with more steps.",
                    exc,
                )
        self._restore_optional_adapters(pipe, loaded_loras)
        return pipe, loaded_loras

    def _restore_optional_adapters(self, pipe: Any, loaded_loras: list[str]) -> None:
        """Restore user adapters before quantization/compilation."""
        adapter_names = ["hyper"] if any("Hyper-SDXL" in item for item in loaded_loras) else []
        adapter_weights = [1.0] if adapter_names else []
        for path_or_id, weight_name, scale in self._style_lora_specs:
            name = Path(path_or_id).stem
            kwargs: dict[str, Any] = {"adapter_name": name}
            if weight_name:
                kwargs["weight_name"] = weight_name
            try:
                pipe.load_lora_weights(path_or_id, **kwargs)
                adapter_names.append(name)
                adapter_weights.append(scale)
                loaded_loras.append(path_or_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not restore style LoRA %s: %s", path_or_id, exc)
        if adapter_names and hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not restore adapter weights: %s", exc)
        for path_or_id, token in self._embedding_specs:
            kwargs = {"token": token} if token else {}
            try:
                pipe.load_textual_inversion(path_or_id, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not restore textual inversion %s: %s", path_or_id, exc)

    def load(self, force: bool = False, base_ref: str | None = None) -> str:
        if self.pipe is not None and not force and base_ref is None:
            return f"SDXL Hyper already loaded: {self._base_id}"

        base_id = str(
            base_ref
            or self.config.get(
                "sdxl_hyper", "base_model_id", default="RunDiffusion/Juggernaut-XL-v9"
            )
        )
        hyper_lora_id = self.config.get("sdxl_hyper", "hyper_lora_id", default=None)
        hyper_weight = self.config.get(
            "sdxl_hyper", "hyper_lora_weight_name", default="Hyper-SDXL-8steps-lora.safetensors"
        )

        logger.info("Loading SDXL img2img base: %s", base_id)

        # Build a new eager pipeline before replacing the current one.
        pipe = self._load_base_pipe(base_id)
        old = self.pipe
        self.pipe = None
        self.loaded_loras.clear()
        del old
        empty_cache()
        # Reuse the successfully downloaded CPU pipeline instead of downloading
        # twice for the normal path.
        pipe = pipe.to(self.device)
        try:
            from diffusers import DDIMScheduler, TCDScheduler

            try:
                pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
            except Exception:
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        except Exception as exc:  # noqa: BLE001
            logger.info("Scheduler swap skipped: %s", exc)
        if hasattr(pipe, "enable_vae_tiling"):
            try:
                pipe.enable_vae_tiling()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(pipe, "enable_vae_slicing"):
            try:
                pipe.enable_vae_slicing()
            except Exception:  # noqa: BLE001
                pass
        loaded_loras: list[str] = []
        if hyper_lora_id:
            try:
                pipe.load_lora_weights(
                    hyper_lora_id,
                    weight_name=hyper_weight,
                    adapter_name="hyper",
                )
                if hasattr(pipe, "set_adapters"):
                    pipe.set_adapters(["hyper"], adapter_weights=[1.0])
                elif hasattr(pipe, "fuse_lora"):
                    pipe.fuse_lora(lora_scale=1.0)
                loaded_loras.append(f"{hyper_lora_id}/{hyper_weight}")
                logger.info("Hyper LoRA loaded: %s / %s", hyper_lora_id, hyper_weight)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Hyper LoRA not loaded (%s). OUTPUT still works with more steps.",
                    exc,
                )
        self._restore_optional_adapters(pipe, loaded_loras)

        report = quantize_component(pipe.unet, self.profile, "sdxl", self.config)
        if self.profile != "bf16" and report.applied == "bf16":
            reason = report.fallback_reason
            logger.warning(
                "SDXL optimized path unavailable; rebuilding clean BF16 pipeline: %s",
                reason,
            )
            del pipe
            empty_cache()
            pipe, loaded_loras = self._prepare_pipeline(
                base_id, hyper_lora_id, hyper_weight
            )
            report = OptimizationReport(
                component="sdxl",
                requested=self.profile,
                fallback_reason=reason,
            )
        if report.applied == "nvfp4":
            make_nvfp4_linear_inputs_contiguous(pipe.unet)
        report = compile_component(pipe.unet, report, self.config)
        self.optimization_report = report
        self.loaded_loras.extend(loaded_loras)

        self.pipe = pipe
        self._base_id = base_id
        return (
            f"SDXL Hyper OUTPUT ready: {base_id} | "
            f"{self.optimization_report.status()} | {gpu_summary()}"
        )

    def set_profile(self, profile: str, reload: bool = True) -> str:
        selected = normalize_profile(profile)
        if selected == self.profile and (self.pipe is not None or not reload):
            return f"SDXL profile already {profile_label(selected)}."
        self.profile = selected
        if not reload:
            return f"SDXL profile set to {profile_label(selected)}."
        base_id = self._base_id
        return self.load(force=True, base_ref=base_id)

    def ensure_loaded(self) -> None:
        if self.pipe is None:
            self.load()

    def load_style_lora(self, path_or_id: str, weight_name: str | None = None, scale: float = 0.8) -> str:
        """Load an additional style LoRA for the OUTPUT stage."""
        self.ensure_loaded()
        assert self.pipe is not None
        name = Path(path_or_id).stem
        kwargs: dict[str, Any] = {"adapter_name": name}
        if weight_name:
            kwargs["weight_name"] = weight_name
        spec = (path_or_id, weight_name, float(scale))
        try:
            self.pipe.load_lora_weights(path_or_id, **kwargs)
        except Exception:
            if self.optimization_report.applied == "bf16":
                raise
            # Some PEFT versions cannot inject a LoRA into an already
            # quantized Linear. Rebuild with the adapter attached first, then
            # quantize/compile the final graph.
            if spec not in self._style_lora_specs:
                self._style_lora_specs.append(spec)
            self.load(force=True, base_ref=self._base_id)
            if path_or_id not in self.loaded_loras:
                raise RuntimeError(
                    f"Style LoRA {path_or_id} is incompatible with the active profile"
                )
            return (
                f"Style LoRA loaded after pipeline rebuild: {path_or_id} | "
                f"{self.optimization_report.status()}"
            )

        # Combine hyper + style adapters when possible
        try:
            current = (
                ["hyper", name]
                if any("hyper" in x.lower() for x in self.loaded_loras)
                else [name]
            )
            # Filter to adapters that exist
            self.pipe.set_adapters(current, adapter_weights=[1.0 if a == "hyper" else scale for a in current])
        except Exception:
            try:
                self.pipe.fuse_lora(lora_scale=scale)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Style LoRA apply partial: %s", exc)

        self.loaded_loras.append(path_or_id)
        if spec not in self._style_lora_specs:
            self._style_lora_specs.append(spec)
        if self.optimization_report.compiled:
            # Adapter injection changes the graph. Clear Dynamo's cache so the
            # compiled repeated blocks are regenerated on the next OUTPUT pass.
            try:
                torch.compiler.reset()
            except Exception:  # noqa: BLE001
                pass
        return f"Style LoRA loaded: {path_or_id}"

    def load_textual_inversion(self, path_or_id: str, token: str | None = None) -> str:
        """Load a textual inversion embedding into the OUTPUT pipeline."""
        self.ensure_loaded()
        assert self.pipe is not None
        kwargs: dict[str, Any] = {}
        if token:
            kwargs["token"] = token
        self.pipe.load_textual_inversion(path_or_id, **kwargs)
        self.loaded_embeddings.append(path_or_id)
        spec = (path_or_id, token)
        if spec not in self._embedding_specs:
            self._embedding_specs.append(spec)
        return f"Textual inversion loaded: {path_or_id}"

    def _prepare_diffdiff_noise(
        self,
        init_image: Image.Image,
        steps: int,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Encode init + shared noise for Differential Diffusion (lazy re-noise).

        Returns ``(init_latents, noise, timesteps, num_steps)``. The callback
        re-noises ``z0`` per step instead of stacking all noise levels on GPU
        (which OOMs on large dilated Infinite Canvas windows).
        """
        from diffusers.utils.torch_utils import randn_tensor

        pipe = self.pipe
        assert pipe is not None
        device = pipe._execution_device
        try:
            dtype = next(pipe.unet.parameters()).dtype
        except StopIteration:
            dtype = self.dtype

        empty_cache()

        # Scheduler timesteps for full-strength (strength=1) img2img.
        pipe.scheduler.set_timesteps(steps, device=device)
        timesteps, num_steps = pipe.get_timesteps(
            num_inference_steps=steps, strength=1.0, device=device
        )
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise RuntimeError("Differential Diffusion: empty timestep schedule")

        image = pipe.image_processor.preprocess(init_image)
        image = image.to(device=device, dtype=dtype)
        # Encode once without noise; pin reference is re-derived per step.
        init_latents = pipe.prepare_latents(
            image,
            timesteps[:1],
            batch_size=1,
            num_images_per_prompt=1,
            dtype=dtype,
            device=device,
            generator=generator,
            add_noise=False,
        )
        del image
        empty_cache()

        noise = randn_tensor(
            init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype
        )
        return init_latents, noise, timesteps, num_steps

    def refine(
        self,
        init_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        denoise: float | None = None,
        steps: int | None = None,
        seed: int | None = None,
        guidance_scale: float | None = None,
        eta: float | None = None,
        change_map: Image.Image | None = None,
    ) -> Image.Image:
        """Run img2img refinement. init_image is the composed WORK canvas.

        Optional ``change_map`` (L mode, 0..255): Differential Diffusion soft-inpaint.
        White = rewrite strongly; black = keep. Soft gray edges = seamless blend.
        When set, global strength is forced to 1.0 and the map drives per-pixel strength.
        """
        self.ensure_loaded()
        assert self.pipe is not None

        denoise = (
            float(denoise)
            if denoise is not None
            else float(self.config.get("sdxl_hyper", "default_denoise", default=0.35))
        )
        steps = (
            int(steps)
            if steps is not None
            else int(self.config.get("sdxl_hyper", "default_steps", default=6))
        )
        guidance = (
            float(guidance_scale)
            if guidance_scale is not None
            else float(self.config.get("sdxl_hyper", "guidance_scale", default=1.0))
        )

        init = init_image.convert("RGB")
        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        use_diff = change_map is not None
        strength = 1.0 if use_diff else max(0.01, min(1.0, denoise))
        # Few-step Hyper still works better with a little headroom for soft maps.
        if use_diff:
            steps = max(steps, 8)
            # Note: do NOT torch.compiler.reset() here — that forces a full
            # UNet recompile on every call. Mode switches handle pool cleanup.
            empty_cache()

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": init,
            "strength": strength,
            "num_inference_steps": max(1, steps),
            "guidance_scale": guidance,
            "generator": generator,
        }
        if eta is not None:
            # DDIM/TCD stochasticity; ignored by schedulers that don't use it.
            call_kwargs["eta"] = max(0.0, min(1.0, float(eta)))
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        if use_diff:
            from xwave_composer.pipeline.infinite_canvas.strength import to_latent_strength

            cmap = change_map.convert("L")
            if cmap.size != init.size:
                cmap = cmap.resize(init.size, Image.Resampling.BILINEAR)
            z0, noise, timesteps_dd, num_steps = self._prepare_diffdiff_noise(
                init, max(1, steps), generator
            )
            # Area-downsample strength to latent res (VAE pooling analogue).
            strength_px = np.asarray(cmap, dtype=np.float32) / 255.0
            vae_scale = int(getattr(self.pipe, "vae_scale_factor", 8))
            map_np = to_latent_strength(strength_px, scale=vae_scale)
            map_t = torch.from_numpy(map_np)[None, None].to(
                device=z0.device, dtype=z0.dtype
            )
            # Research §4.2: thresh descends 1→0; high strength released first.
            # active keeps denoising trajectory; inactive pinned to re-noised z0.

            def _diff_callback(pipe, step_index, timestep, callback_kwargs):  # noqa: ARG001
                latents = callback_kwargs["latents"]
                next_i = int(step_index) + 1
                if next_i < num_steps:
                    thresh = 1.0 - (next_i / float(num_steps))
                    active = (map_t > thresh).to(dtype=latents.dtype)
                    t_ref = timesteps_dd[next_i]
                    t_batch = t_ref.expand(z0.shape[0])
                    z_ref = pipe.scheduler.add_noise(z0, noise, t_batch)
                    latents = active * latents + (1.0 - active) * z_ref
                    callback_kwargs["latents"] = latents
                return callback_kwargs

            call_kwargs["callback_on_step_end"] = _diff_callback
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        if self.optimization_report.compiled:
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except Exception:  # noqa: BLE001
                pass
        with torch.inference_mode():
            try:
                result = self.pipe(**call_kwargs)
            except TypeError:
                # Older signatures / no callback support — fall back without diff.
                if use_diff:
                    logger.warning(
                        "Pipeline rejected Differential Diffusion callback; "
                        "falling back to global strength img2img"
                    )
                    call_kwargs.pop("callback_on_step_end", None)
                    call_kwargs.pop("callback_on_step_end_tensor_inputs", None)
                result = self.pipe(
                    prompt=prompt,
                    image=init,
                    strength=strength,
                    num_inference_steps=max(1, steps),
                    guidance_scale=guidance,
                    generator=generator,
                    **({"negative_prompt": negative_prompt} if negative_prompt else {}),
                )

        out = result.images[0]
        return out.convert("RGB")

    def unload(self) -> None:
        pipe = self.pipe
        self.pipe = None
        self.loaded_loras.clear()
        self.loaded_embeddings.clear()
        self.optimization_report = OptimizationReport(
            component="sdxl",
            requested=self.profile,
            applied=f"unloaded (next: {profile_label(self.profile)})",
        )
        hard_release(pipe, reset_compiler=True)
