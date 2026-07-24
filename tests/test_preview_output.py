"""The fast preview must never be what gets exported."""

from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession


class RecordingSDXL:
    """Records the size and step count of every refine call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def ready(self) -> bool:
        return True

    def refine(self, *, init_image, steps, **kwargs):
        self.calls.append({"size": init_image.size, "steps": steps})
        return Image.new("RGB", init_image.size, "green")


def _session(tmp_path) -> ComposerSession:
    config = AppConfig(
        raw={
            "device": "cpu",
            "canvas": {"width": 1024, "height": 1024},
            "sdxl_hyper": {
                "default_denoise": 0.3,
                "default_steps": 6,
                "preview_scale": 0.625,
                "preview_steps": 4,
            },
            "paths": {
                "models_cache": "models",
                "layers_dir": "layers",
                "workspace_dir": "workspace",
            },
            "style": {"lora_dir": "loras", "embedding_dir": "embeddings"},
            "export": {"output_dir": "exports"},
        },
        root=tmp_path,
    )
    session = ComposerSession(config)
    session.sdxl = RecordingSDXL()
    session.doc.background = Image.new("RGB", (1024, 1024), "white")
    return session


def test_preview_renders_smaller_and_shorter(tmp_path):
    session = _session(tmp_path)
    work = Image.new("RGB", (1024, 1024), "white")

    session.run_output(work, preview=True)

    call = session.sdxl.calls[-1]
    assert call["size"] == (640, 640)  # 1024 * 0.625, on an 8px grid
    assert call["steps"] == 4
    assert session.output_is_preview is True


def test_preview_is_scaled_back_to_canvas_size(tmp_path):
    """The OUTPUT pane must not resize under the user mid-drag."""
    session = _session(tmp_path)
    work = Image.new("RGB", (1024, 1024), "white")

    out = session.run_output(work, preview=True)

    assert out.size == (1024, 1024)


def test_full_pass_uses_the_real_settings(tmp_path):
    session = _session(tmp_path)
    work = Image.new("RGB", (1024, 1024), "white")

    session.run_output(work)

    call = session.sdxl.calls[-1]
    assert call["size"] == (1024, 1024)
    assert call["steps"] == 6
    assert session.output_is_preview is False


def test_ensure_full_output_rerenders_a_preview(tmp_path):
    session = _session(tmp_path)
    work = Image.new("RGB", (1024, 1024), "white")
    session.run_output(work, preview=True)
    assert session.output_is_preview

    session.ensure_full_output()

    assert session.output_is_preview is False
    assert session.sdxl.calls[-1]["size"] == (1024, 1024)
    assert session.sdxl.calls[-1]["steps"] == 6


def test_ensure_full_output_leaves_a_real_output_alone(tmp_path):
    session = _session(tmp_path)
    session.run_output(Image.new("RGB", (1024, 1024), "white"))
    before = len(session.sdxl.calls)

    session.ensure_full_output()

    assert len(session.sdxl.calls) == before  # no wasted second pass


def test_refine_final_never_builds_on_a_preview(tmp_path):
    session = _session(tmp_path)
    session.run_output(Image.new("RGB", (1024, 1024), "white"), preview=True)

    session.refine_final(steps=8, denoise=0.3)

    # A full pass must have happened before the final refine.
    sizes = [c["size"] for c in session.sdxl.calls]
    assert (1024, 1024) in sizes[1:]
    assert session.output_is_preview is False
