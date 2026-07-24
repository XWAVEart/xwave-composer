"""Install the export-only SeedVR2 runtime used by xwave-composer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "models" / "seedvr2-runtime"
REPOSITORY = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
DEPENDENCIES = [
    "omegaconf>=2.3.0",
    "rotary-embedding-torch>=0.5.3",
    "opencv-python-headless>=4.9.0",
    "gguf",
    "matplotlib",
]


def main() -> None:
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    if not (RUNTIME / "inference_cli.py").is_file():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPOSITORY, str(RUNTIME)],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *DEPENDENCIES],
        check=True,
    )
    print(f"SeedVR2 runtime ready at {RUNTIME}")
    print("Weights download automatically on the first final export.")


if __name__ == "__main__":
    main()
