"""Config loading tests."""

from xwave_composer.config import AppConfig, PROJECT_ROOT


def test_load_default_config():
    cfg = AppConfig.load()
    assert cfg.root == PROJECT_ROOT
    w, h = cfg.canvas_size
    assert w == 1024 and h == 1024
    assert cfg.get("flux", "model_id") is not None
    assert cfg.get("sdxl_hyper", "default_denoise") is not None


def test_ensure_dirs(tmp_path, monkeypatch):
    cfg = AppConfig.load()
    # Point workspace under tmp by overriding raw paths
    cfg.raw.setdefault("paths", {})
    cfg.raw["paths"]["workspace_dir"] = str(tmp_path / "ws")
    cfg.raw["paths"]["layers_dir"] = str(tmp_path / "ws" / "layers")
    cfg.raw["paths"]["models_cache"] = str(tmp_path / "models")
    cfg.raw.setdefault("export", {})["output_dir"] = str(tmp_path / "exports")
    cfg.raw.setdefault("style", {})["lora_dir"] = str(tmp_path / "loras")
    cfg.raw["style"]["embedding_dir"] = str(tmp_path / "emb")
    cfg.raw["style"]["phrases_file"] = str(tmp_path / "styles.json")
    cfg.raw["paths"]["library_dir"] = str(tmp_path / "library")
    cfg.ensure_dirs()
    assert (tmp_path / "ws" / "layers").exists() or cfg.path("paths", "layers_dir").exists()
    assert cfg.path("paths", "library_dir").exists()
