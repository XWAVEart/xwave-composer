"""H4: WORK scene pixels are served as Gradio file URLs, not base64 data-URLs."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from PIL import Image

from xwave_composer.canvas.layers import ObjectLayer
from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession
from xwave_composer.ui import gradio_app as ui


def _session(tmp_path) -> ComposerSession:
    config = AppConfig(
        raw={
            "device": "cpu",
            "sdxl_hyper": {"default_denoise": 0.3, "default_steps": 6},
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
    return ComposerSession(config)


def test_scene_dict_uses_file_urls_not_data_uris(tmp_path):
    session = _session(tmp_path)
    session.doc.background = Image.new("RGB", (64, 64), (20, 30, 40))
    layer = ObjectLayer(
        name="Subject",
        prompt="a cat",
        image=Image.new("RGBA", (32, 32), (255, 0, 0, 200)),
    )
    session.doc.add_object(layer)

    scene = ui._scene_dict(session)
    layer_url = scene["layers"][0]["data_url"]
    bg_url = scene["bg_data_url"]

    assert layer_url and layer_url.startswith("/gradio_api/file=")
    assert bg_url and bg_url.startswith("/gradio_api/file=")
    assert "base64," not in layer_url
    assert "data:image" not in layer_url
    assert Path(layer_url.split("file=", 1)[1]).is_file()
    assert Path(bg_url.split("file=", 1)[1]).is_file()

    html = ui.render_work_html(scene)
    assert "data:image" not in html
    decoded = json.loads(
        base64.b64decode(html.split('data-scene="', 1)[1].split('"', 1)[0])
    )
    assert decoded["layers"][0]["data_url"].startswith("/gradio_api/file=")
    assert len(html) < 8_000


def test_launch_kwargs_allows_scene_cache(tmp_path):
    session = _session(tmp_path)
    demo = type("Demo", (), {})()
    kwargs = ui._launch_kwargs(demo, session.config)
    cache = str(ui._scene_cache_dir(session.config))
    assert cache in kwargs["allowed_paths"]
    assert all(str(p).startswith("/") for p in kwargs["allowed_paths"])
