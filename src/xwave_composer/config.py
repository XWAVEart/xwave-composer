"""Load and hold application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

# Project root: .../xwave-composer
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def pick_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name.lower(), torch.bfloat16)


def pick_device(preferred: str = "cuda") -> str:
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return preferred if preferred != "cuda" else "cuda"
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class AppConfig:
    """Typed view over config.yaml with path resolution."""

    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path
        raw: dict[str, Any] = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        return cls(raw=raw, root=PROJECT_ROOT)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def device(self) -> str:
        return pick_device(str(self.get("device", default="cuda")))

    @property
    def dtype(self) -> torch.dtype:
        return pick_dtype(str(self.get("dtype", default="bfloat16")))

    @property
    def canvas_size(self) -> tuple[int, int]:
        w = int(self.get("canvas", "width", default=1024))
        h = int(self.get("canvas", "height", default=1024))
        return w, h

    def ensure_dirs(self) -> None:
        """Create workspace and model directories if missing."""
        for key in ("models_cache", "layers_dir", "workspace_dir"):
            p = _resolve_path(str(self.get("paths", key, default=key)), self.root)
            p.mkdir(parents=True, exist_ok=True)
        export_dir = _resolve_path(
            str(self.get("export", "output_dir", default="exports")), self.root
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        for key in ("lora_dir", "embedding_dir"):
            p = _resolve_path(str(self.get("style", key, default=key)), self.root)
            p.mkdir(parents=True, exist_ok=True)
        phrases = _resolve_path(
            str(self.get("style", "phrases_file", default="data/styles.json")), self.root
        )
        phrases.parent.mkdir(parents=True, exist_ok=True)

    def path(self, *keys: str, default: str = ".") -> Path:
        return _resolve_path(str(self.get(*keys, default=default)), self.root)

    def hf_cache_dir(self) -> str | None:
        env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if env:
            return env
        cache = self.path("paths", "models_cache", default="models")
        return str(cache)
