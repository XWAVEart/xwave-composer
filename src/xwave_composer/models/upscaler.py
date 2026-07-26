"""Final 2x export upscaler. Prefers SeedVR2; falls back to Real-ESRGAN or bicubic."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, gpu_summary

logger = logging.getLogger(__name__)

# Product model allowlist (subset of upstream SeedVR2 registry).
SEEDVR2_MODEL_CHOICES: list[tuple[str, str]] = [
    ("7B FP16 — highest fidelity", "seedvr2_ema_7b_fp16.safetensors"),
    ("7B Sharp FP16 — enhanced detail", "seedvr2_ema_7b_sharp_fp16.safetensors"),
    (
        "7B FP8 mixed — lower VRAM",
        "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    ),
    (
        "7B Sharp FP8 mixed",
        "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
    ),
    ("3B FP16 — balanced", "seedvr2_ema_3b_fp16.safetensors"),
    ("3B FP8 — fast / low VRAM", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"),
]

SEEDVR2_MODEL_VALUES = {value for _, value in SEEDVR2_MODEL_CHOICES}
DEFAULT_SEEDVR2_MODEL = "seedvr2_ema_7b_fp16.safetensors"

# Named packs for the Gradio preset dropdown. Values map 1:1 into CLI options.
SEEDVR2_PRESETS: dict[str, dict[str, Any]] = {
    "quality": {
        "label": "Quality — 7B FP16",
        "model": "seedvr2_ema_7b_fp16.safetensors",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "dit_offload_device": "none",
        "vae_offload_device": "none",
        "compile_dit": False,
    },
    "balanced": {
        "label": "Balanced — 7B FP8",
        "model": "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "dit_offload_device": "none",
        "vae_offload_device": "none",
        "compile_dit": False,
    },
    "low_vram": {
        "label": "Low VRAM — 3B FP8 + BlockSwap",
        "model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "blocks_to_swap": 16,
        "swap_io_components": True,
        "dit_offload_device": "cpu",
        "vae_offload_device": "cpu",
        "compile_dit": False,
    },
    "fast": {
        "label": "Fast — 3B FP8 + compile",
        "model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "dit_offload_device": "none",
        "vae_offload_device": "none",
        "compile_dit": True,
    },
}

SEEDVR2_PRESET_CHOICES: list[tuple[str, str]] = [
    (meta["label"], key) for key, meta in SEEDVR2_PRESETS.items()
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_seedvr2_option(
    options: dict[str, object] | None,
    config: AppConfig,
    key: str,
    *,
    default: Any,
    cast: Callable[[Any], Any] | None = None,
    config_key: str | None = None,
) -> Any:
    """Prefer UI/export options, then ``export.seedvr2_*`` YAML, then default."""
    cast = cast or (lambda x: x)
    opts = options or {}
    if key in opts and opts[key] is not None:
        return cast(opts[key])
    cfg_key = config_key or f"seedvr2_{key}"
    raw = config.get("export", cfg_key, default=default)
    if raw is None:
        raw = default
    return cast(raw)


def build_seedvr2_command(
    *,
    cli: Path,
    input_path: Path,
    output_path: Path,
    model_dir: Path,
    target_short: int,
    target_long: int,
    config: AppConfig,
    options: dict[str, object] | None = None,
) -> list[str]:
    """Build the SeedVR2 ``inference_cli.py`` argv (no subprocess)."""
    model = str(
        resolve_seedvr2_option(
            options, config, "model", default=DEFAULT_SEEDVR2_MODEL
        )
    )
    if model not in SEEDVR2_MODEL_VALUES:
        logger.warning(
            "SeedVR2 model %s is outside the product allowlist; passing through.",
            model,
        )

    seed = int(resolve_seedvr2_option(options, config, "seed", default=42, cast=int))
    color = str(
        resolve_seedvr2_option(
            options, config, "color_correction", default="lab"
        )
    )
    input_noise = float(
        resolve_seedvr2_option(
            options, config, "input_noise_scale", default=0.0, cast=float
        )
    )
    latent_noise = float(
        resolve_seedvr2_option(
            options, config, "latent_noise_scale", default=0.0, cast=float
        )
    )
    attention = str(
        resolve_seedvr2_option(
            options, config, "attention", default="sdpa"
        )
    )
    tile = int(
        resolve_seedvr2_option(
            options, config, "tile_size", default=1024, cast=int
        )
    )
    blocks = int(
        resolve_seedvr2_option(
            options, config, "blocks_to_swap", default=0, cast=int
        )
    )
    swap_io = _as_bool(
        resolve_seedvr2_option(
            options, config, "swap_io_components", default=False, cast=_as_bool
        )
    )
    dit_offload = str(
        resolve_seedvr2_option(
            options, config, "dit_offload_device", default="none"
        )
    ).strip().lower()
    vae_offload = str(
        resolve_seedvr2_option(
            options, config, "vae_offload_device", default="none"
        )
    ).strip().lower()
    compile_dit = _as_bool(
        resolve_seedvr2_option(
            options, config, "compile_dit", default=False, cast=_as_bool
        )
    )
    compile_mode = str(
        resolve_seedvr2_option(
            options, config, "compile_mode", default="default"
        )
    )

    # BlockSwap / I/O swap require an offload device in upstream CLI.
    if (blocks > 0 or swap_io) and dit_offload in ("", "none", "null"):
        dit_offload = "cpu"

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
        str(int(target_short)),
        "--max_resolution",
        str(int(target_long)),
        "--batch_size",
        "1",
        "--seed",
        str(seed),
        "--color_correction",
        color,
        "--input_noise_scale",
        str(input_noise),
        "--latent_noise_scale",
        str(latent_noise),
        "--attention_mode",
        attention,
        "--dit_offload_device",
        dit_offload,
        "--vae_offload_device",
        vae_offload,
        "--blocks_to_swap",
        str(max(0, blocks)),
        "--vae_encode_tiled",
        "--vae_decode_tiled",
        "--vae_encode_tile_size",
        str(tile),
        "--vae_decode_tile_size",
        str(tile),
    ]
    if swap_io:
        command.append("--swap_io_components")
    if compile_dit:
        command.extend(["--compile_dit", "--compile_mode", compile_mode])
    return command


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
            default=DEFAULT_SEEDVR2_MODEL,
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
        cli = Path(self._pipe)
        runtime = cli.parent
        factor = int(self.config.get("export", "factor", default=2))
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
            command = build_seedvr2_command(
                cli=cli,
                input_path=input_path,
                output_path=output_path,
                model_dir=model_dir,
                target_short=target_short,
                target_long=target_long,
                config=self.config,
                options=options,
            )
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
