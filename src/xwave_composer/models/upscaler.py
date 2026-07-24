"""Final 2x export upscaler. Prefers SeedVR2; falls back to Real-ESRGAN or bicubic."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, gpu_summary

logger = logging.getLogger(__name__)


class ImageUpscaler:
    """2x high-quality upscale for export."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.backend: str | None = None
        self._pipe: Any = None

    def load(self, force: bool = False) -> str:
        if self._pipe is not None and not force:
            return f"Upscaler already loaded: {self.backend}"

        preferred = str(self.config.get("export", "upscaler", default="seedvr2")).lower()
        errors: list[str] = []

        if preferred == "seedvr2":
            try:
                return self._load_seedvr2()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"seedvr2: {exc}")
                logger.warning("SeedVR2 load failed: %s", exc)
                if not bool(self.config.get("export", "allow_fallback", default=False)):
                    raise

        if preferred in ("seedvr2", "realesrgan", "auto"):
            try:
                return self._load_realesrgan()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"realesrgan: {exc}")
                logger.warning("Real-ESRGAN load failed: %s", exc)

        # Always available fallback
        self.backend = "bicubic"
        self._pipe = "bicubic"
        return (
            "Using bicubic 2x fallback (install SeedVR2 or Real-ESRGAN for higher quality). "
            f"Errors: {errors}" if errors else "Using bicubic 2x upscaler."
        )

    def _load_seedvr2(self) -> str:
        """Configure the maintained SeedVR2 single-image CLI.

        SeedVR2 is not a Diffusers pipeline.  Keeping it in a short-lived
        subprocess gives final export exclusive access to VRAM and guarantees
        that every model allocation is released when the export completes.
        """
        runtime = self.config.path(
            "export", "seedvr2_runtime_dir", default="models/seedvr2-runtime"
        )
        cli = runtime / "inference_cli.py"
        if not cli.is_file():
            raise RuntimeError(
                f"SeedVR2 runtime not found at {cli}. Run scripts/setup_seedvr2.py."
            )
        self._pipe = cli
        self.backend = "seedvr2"
        model = self.config.get(
            "export",
            "seedvr2_model",
            default="seedvr2_ema_7b_fp16.safetensors",
        )
        return f"SeedVR2 export runtime ready ({model}); model loads only during export."

    def _load_realesrgan(self) -> str:
        """Real-ESRGAN via spandrel or basicsr if installed; else raise."""
        try:
            from realesrgan import RealESRGANer  # type: ignore
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
            )
            # Weights are downloaded by the user or package; path configurable later
            upsampler = RealESRGANer(
                scale=2,
                model_path=None,
                model=model,
                device=self.device,
            )
            self._pipe = upsampler
            self.backend = "realesrgan"
            return f"Real-ESRGAN loaded | {gpu_summary()}"
        except Exception as exc:
            raise RuntimeError(f"Real-ESRGAN unavailable: {exc}") from exc

    def ensure_loaded(self) -> None:
        if self._pipe is None:
            self.load()

    def upscale_2x(
        self,
        image: Image.Image,
        steps: int | None = None,
        seedvr2_options: dict[str, object] | None = None,
    ) -> Image.Image:
        """Upscale by 2. steps is used when the backend supports diffusion steps."""
        self.ensure_loaded()
        rgb = image.convert("RGB")
        factor = int(self.config.get("export", "factor", default=2))

        if self.backend == "bicubic" or self._pipe == "bicubic":
            w, h = rgb.size
            return rgb.resize((w * factor, h * factor), Image.Resampling.LANCZOS)

        if self.backend == "seedvr2":
            return self._run_seedvr2(
                rgb,
                steps=steps,
                options=seedvr2_options,
            )

        if self.backend == "realesrgan":
            return self._run_realesrgan(rgb)

        # Unknown backend → LANCZOS
        w, h = rgb.size
        return rgb.resize((w * factor, h * factor), Image.Resampling.LANCZOS)

    def _run_seedvr2(
        self,
        image: Image.Image,
        steps: int | None = None,
        options: dict[str, object] | None = None,
    ) -> Image.Image:
        del steps  # SeedVR2 is a distilled one-step restoration model.
        options = options or {}
        cli = Path(self._pipe)
        runtime = cli.parent
        factor = int(self.config.get("export", "factor", default=2))
        model = str(
            options.get(
                "model",
                self.config.get(
                "export",
                "seedvr2_model",
                default="seedvr2_ema_7b_fp16.safetensors",
                ),
            )
        )
        model_dir = self.config.path(
            "export",
            "seedvr2_weights_dir",
            default="models/seedvr2-runtime/weights",
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        target_short = min(image.size) * factor
        target_long = max(image.size) * factor

        with tempfile.TemporaryDirectory(prefix="xwave_seedvr2_") as tmp:
            input_path = Path(tmp) / "input.png"
            output_path = Path(tmp) / "output.png"
            image.save(input_path, format="PNG")
            command = [
                sys.executable,
                str(cli),
                str(input_path),
                "--output",
                str(output_path),
                "--output_format",
                "png",
                "--model_dir",
                str(model_dir),
                "--dit_model",
                model,
                "--resolution",
                str(target_short),
                "--max_resolution",
                str(target_long),
                "--batch_size",
                "1",
                "--seed",
                str(int(options.get("seed", 42))),
                "--color_correction",
                str(options.get("color_correction", "lab")),
                "--input_noise_scale",
                str(float(options.get("input_noise_scale", 0.0))),
                "--latent_noise_scale",
                str(float(options.get("latent_noise_scale", 0.0))),
                "--attention_mode",
                str(self.config.get("export", "seedvr2_attention", default="sdpa")),
                "--vae_encode_tiled",
                "--vae_decode_tiled",
                "--vae_encode_tile_size",
                str(self.config.get("export", "seedvr2_tile_size", default=1024)),
                "--vae_decode_tile_size",
                str(self.config.get("export", "seedvr2_tile_size", default=1024)),
            ]
            logger.info("Starting export-only SeedVR2 process: %s", " ".join(command))
            result = subprocess.run(
                command,
                cwd=runtime,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=int(
                    self.config.get("export", "seedvr2_timeout_s", default=1800)
                ),
                check=False,
            )
            if result.returncode != 0 or not output_path.is_file():
                details = result.stdout[-6000:] if result.stdout else "no process output"
                raise RuntimeError(
                    f"SeedVR2 export failed (exit {result.returncode}):\n{details}"
                )
            with Image.open(output_path) as restored:
                return restored.convert("RGB").copy()

    def _run_realesrgan(self, image: Image.Image) -> Image.Image:
        import numpy as np

        upsampler = self._pipe
        arr = np.array(image)[:, :, ::-1]  # RGB -> BGR often expected
        try:
            output, _ = upsampler.enhance(arr, outscale=2)
            output = output[:, :, ::-1]
            return Image.fromarray(output)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Real-ESRGAN enhance failed (%s); bicubic fallback.", exc)
            w, h = image.size
            return image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)

    def save_export(
        self,
        image: Image.Image,
        stem: str = "export",
        seedvr2_options: dict[str, object] | None = None,
    ) -> Path:
        """Upscale and write to the export directory. Returns file path."""
        from datetime import datetime

        out_dir = self.config.path("export", "output_dir", default="exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{stem}_{ts}.jpg"
        up = self.upscale_2x(image, seedvr2_options=seedvr2_options)
        up.convert("RGB").save(
            path,
            format="JPEG",
            quality=int(self.config.get("export", "jpeg_quality", default=95)),
            subsampling=0,
            optimize=True,
        )
        return path

    def unload(self) -> None:
        self._pipe = None
        self.backend = None
        empty_cache()
