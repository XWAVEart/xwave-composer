from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession


class FakeSDXL:
    def __init__(self) -> None:
        self.init_image = None

    def refine(self, *, init_image, **kwargs):
        self.init_image = init_image
        return Image.new("RGB", init_image.size, "green")


def test_final_refine_uses_output_without_replacing_work(tmp_path):
    config = AppConfig(
        raw={
            "device": "cpu",
            "paths": {
                "models_cache": "models",
                "layers_dir": "layers",
                "workspace_dir": "workspace",
            },
            "style": {
                "lora_dir": "loras",
                "embedding_dir": "embeddings",
            },
            "export": {"output_dir": "exports"},
        },
        root=tmp_path,
    )
    session = ComposerSession(config)
    fake_sdxl = FakeSDXL()
    session.sdxl = fake_sdxl
    work = Image.new("RGB", (32, 32), "red")
    output = Image.new("RGB", (32, 32), "blue")
    session.last_work = work
    session.last_output = output
    session.last_output_prompt = "preserve the accepted output"
    session.output_settings.seed = 42

    refined = session.refine_final(steps=8, denoise=0.2)

    # Refine snapshots OUTPUT (copy) so WORK edits cannot mutate the init mid-flight.
    assert fake_sdxl.init_image is not None
    assert fake_sdxl.init_image.size == output.size
    assert fake_sdxl.init_image.getpixel((0, 0)) == output.getpixel((0, 0))
    assert session.last_work is work
    assert session.last_output is refined
