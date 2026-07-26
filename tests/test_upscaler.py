from pathlib import Path

from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.models.upscaler import (
    DEFAULT_SEEDVR2_MODEL,
    ImageUpscaler,
    SEEDVR2_PRESETS,
    build_seedvr2_command,
)


def test_saved_export_is_high_quality_jpeg(tmp_path):
    config = AppConfig(
        raw={
            "device": "cpu",
            "export": {
                "upscaler": "bicubic",
                "factor": 2,
                "jpeg_quality": 95,
                "output_dir": "exports",
            },
        },
        root=tmp_path,
    )
    path = ImageUpscaler(config).save_export(
        Image.new("RGB", (16, 16), "blue"),
        stem="jpeg_export",
    )

    assert path.suffix == ".jpg"
    with Image.open(path) as exported:
        assert exported.format == "JPEG"
        assert exported.size == (32, 32)


def _cfg(tmp_path, **export_extra) -> AppConfig:
    export = {
        "upscaler": "seedvr2",
        "factor": 2,
        "seedvr2_model": DEFAULT_SEEDVR2_MODEL,
        "seedvr2_seed": 7,
        "seedvr2_color_correction": "lab",
        "seedvr2_input_noise_scale": 0.0,
        "seedvr2_latent_noise_scale": 0.0,
        "seedvr2_attention": "sdpa",
        "seedvr2_tile_size": 1024,
        "seedvr2_blocks_to_swap": 0,
        "seedvr2_swap_io_components": False,
        "seedvr2_dit_offload_device": "none",
        "seedvr2_vae_offload_device": "none",
        "seedvr2_compile_dit": False,
        "output_dir": "exports",
    }
    export.update(export_extra)
    return AppConfig(raw={"device": "cpu", "export": export}, root=tmp_path)


def test_build_seedvr2_command_defaults_and_yaml_fallback(tmp_path):
    config = _cfg(tmp_path, seedvr2_seed=99)
    cmd = build_seedvr2_command(
        cli=Path("/tmp/inference_cli.py"),
        input_path=Path("/tmp/in.png"),
        output_path=Path("/tmp/out.png"),
        model_dir=Path("/tmp/weights"),
        target_short=2048,
        target_long=2048,
        config=config,
        options=None,
    )
    assert "--dit_model" in cmd
    assert DEFAULT_SEEDVR2_MODEL in cmd
    assert cmd[cmd.index("--seed") + 1] == "99"
    assert "--blocks_to_swap" in cmd
    assert cmd[cmd.index("--blocks_to_swap") + 1] == "0"
    assert "--compile_dit" not in cmd
    assert "--swap_io_components" not in cmd


def test_build_seedvr2_command_blockswap_forces_cpu_offload(tmp_path):
    config = _cfg(tmp_path)
    cmd = build_seedvr2_command(
        cli=Path("/tmp/inference_cli.py"),
        input_path=Path("/tmp/in.png"),
        output_path=Path("/tmp/out.png"),
        model_dir=Path("/tmp/weights"),
        target_short=1024,
        target_long=1024,
        config=config,
        options={
            "model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "blocks_to_swap": 16,
            "swap_io_components": True,
            "dit_offload_device": "none",
            "compile_dit": True,
        },
    )
    assert "seedvr2_ema_3b_fp8_e4m3fn.safetensors" in cmd
    assert cmd[cmd.index("--blocks_to_swap") + 1] == "16"
    assert cmd[cmd.index("--dit_offload_device") + 1] == "cpu"
    assert "--swap_io_components" in cmd
    assert "--compile_dit" in cmd
    assert cmd[cmd.index("--compile_mode") + 1] == "default"


def test_low_vram_preset_shape():
    meta = SEEDVR2_PRESETS["low_vram"]
    assert "3b" in meta["model"]
    assert meta["blocks_to_swap"] > 0
    assert meta["dit_offload_device"] == "cpu"
    assert meta["swap_io_components"] is True
