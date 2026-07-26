"""Day-2 export / VRAM choreography tests (no real GPU required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession


class FakeSDXL:
    def __init__(self) -> None:
        self.init_image = None
        self.ready = True
        self.unload = Mock()

    def refine(self, *, init_image, **kwargs):
        self.init_image = init_image
        return Image.new("RGB", init_image.size, "green")

    def load(self) -> str:
        self.ready = True
        return "loaded"


def _session(tmp_path: Path) -> ComposerSession:
    config = AppConfig(
        raw={
            "device": "cpu",
            "paths": {
                "models_cache": "models",
                "layers_dir": "layers",
                "workspace_dir": "workspace",
            },
            "style": {"lora_dir": "loras", "embedding_dir": "embeddings"},
            "export": {"upscaler": "bicubic", "factor": 2, "output_dir": "exports"},
        },
        root=tmp_path,
    )
    return ComposerSession(config)


def test_export_refine_steps_uses_last_output_not_work(tmp_path):
    session = _session(tmp_path)
    fake = FakeSDXL()
    session.sdxl = fake
    work = Image.new("RGB", (32, 32), "red")
    output = Image.new("RGB", (32, 32), "blue")
    session.last_work = work
    session.last_output = output
    session.last_output_prompt = "accepted output"
    session.output_settings.seed = 7
    session.output_settings.denoise = 0.25

    session.upscaler = Mock()
    session.upscaler.backend = "bicubic"
    out_path = tmp_path / "exports" / "xwave_export.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "yellow").save(out_path)
    session.upscaler.save_export.return_value = out_path

    session.export_final(refine_steps=8)

    assert fake.init_image is not None
    assert fake.init_image.getpixel((0, 0)) == output.getpixel((0, 0))
    assert fake.init_image.getpixel((0, 0)) != work.getpixel((0, 0))


def test_seedvr2_export_unloads_isolator(tmp_path):
    session = _session(tmp_path)
    session.config.raw["export"]["upscaler"] = "seedvr2"
    session.sdxl = FakeSDXL()
    session.last_output = Image.new("RGB", (16, 16), "blue")
    session.flux = Mock(ready=True)
    session.flux.unload = Mock()
    session.llm = Mock()
    session.llm.unload = Mock()
    session.isolator = Mock()
    session.isolator.unload = Mock()

    session.upscaler = Mock()
    session.upscaler.backend = "seedvr2"
    out_path = tmp_path / "exports" / "seed.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(out_path)
    session.upscaler.save_export.return_value = out_path
    session.upscaler.unload = Mock()

    session.export_final()

    session.flux.unload.assert_called()
    session.sdxl.unload.assert_called()
    session.llm.unload.assert_called()
    session.isolator.unload.assert_called()


def test_build_prompt_holds_lock_during_llm_rewrite(tmp_path):
    session = _session(tmp_path)
    session.output_settings.use_llm_rewrite = True
    session.doc.background_prompt = "studio"
    held = []

    def rewrite_side_effect(concat, **kwargs):
        held.append(session._lock._is_owned())  # type: ignore[attr-defined]
        return "rewritten studio"

    session.flux = Mock(ready=True)
    session.llm = Mock()
    session.llm.rewrite.side_effect = rewrite_side_effect
    session.last_work = Image.new("RGB", (32, 32), "white")

    assert session.build_prompt() == "rewritten studio"
    assert held == [True]
    session.flux.unload.assert_called_once()
