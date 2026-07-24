"""Run a small end-to-end SeedVR2 export smoke test."""

from __future__ import annotations

from PIL import Image, ImageDraw

from xwave_composer.config import AppConfig
from xwave_composer.models.upscaler import ImageUpscaler


def main() -> None:
    image = Image.new("RGB", (256, 256), (24, 32, 48))
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 224, 224), outline=(230, 180, 80), width=8)
    draw.ellipse((72, 72, 184, 184), fill=(70, 140, 220))

    upscaler = ImageUpscaler(AppConfig.load())
    path = upscaler.save_export(image, stem="seedvr2_smoke")
    with Image.open(path) as result:
        print(f"SeedVR2 smoke test passed: {result.size[0]}x{result.size[1]} -> {path}")


if __name__ == "__main__":
    main()
