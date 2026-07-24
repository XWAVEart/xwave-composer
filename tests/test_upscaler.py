from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.models.upscaler import ImageUpscaler


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
