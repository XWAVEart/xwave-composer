from PIL import Image

from xwave_composer.canvas.layers import ObjectLayer
from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession


class FakeSDXL:
    def __init__(self, ready: bool = True) -> None:
        self._ready = ready
        self.init_image = None
        self.calls = 0

    @property
    def ready(self) -> bool:
        return self._ready

    def refine(self, *, init_image, **kwargs):
        self.calls += 1
        self.init_image = init_image
        return Image.new("RGB", init_image.size, "green")

    def load(self) -> str:
        self._ready = True
        return "loaded"

    def unload(self) -> None:
        self._ready = False


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


def test_run_output_preserves_accepted_output_when_sdxl_unloaded(tmp_path):
    session = _session(tmp_path)
    fake = FakeSDXL(ready=False)
    session.sdxl = fake
    work = Image.new("RGB", (32, 32), "red")
    accepted = Image.new("RGB", (32, 32), "blue")
    session.last_work = work
    session.last_output = accepted

    result = session.run_output(work)

    assert result is accepted
    assert session.last_output is accepted
    assert fake.calls == 0


def test_live_transform_defers_compose_until_final(tmp_path):
    session = _session(tmp_path)
    layer = ObjectLayer(
        name="Move me",
        prompt="subject",
        image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
    )
    session.doc.add_object(layer)
    session.doc.selected_id = layer.id
    before = session.refresh_work()

    session.update_transform_by_id(layer.id, x=80.0, y=90.0, compose=False)

    assert session._work_stale is True
    assert layer.transform.x == 80.0
    assert session.last_work is before

    composed = session.ensure_work_composed()
    assert session._work_stale is False
    assert composed is not before


def test_run_output_releases_lock_during_refine(tmp_path):
    """Canvas transforms must be able to take the session lock while SDXL runs."""
    session = _session(tmp_path)
    acquired_during_refine = []

    class LockCheckingSDXL(FakeSDXL):
        def refine(self, *, init_image, **kwargs):
            got = session._lock.acquire(blocking=False)
            acquired_during_refine.append(got)
            if got:
                session._lock.release()
            return super().refine(init_image=init_image, **kwargs)

    session.sdxl = LockCheckingSDXL(ready=True)
    session.last_work = Image.new("RGB", (32, 32), "red")
    session.output_settings.seed = 1

    out = session.run_output()

    assert acquired_during_refine == [True]
    assert out is not None
    assert session.output_rev == 1


def test_mute_excludes_prompt_from_concatenation(tmp_path):
    session = _session(tmp_path)
    session.doc.background_prompt = "studio backdrop"
    session.doc.add_object(
        ObjectLayer(name="Subject", prompt="red sports car")
    )
    session.doc.selected_id = session.doc.objects[0].id

    before = session.build_prompt()
    assert "red sports car" in before

    enabled = session.toggle_selected_prompt()
    after = session.build_prompt()

    assert enabled is False
    assert "red sports car" not in after
    assert "studio backdrop" in after


def test_duplicate_selected_copies_image(tmp_path):
    session = _session(tmp_path)
    image = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    session.doc.add_object(
        ObjectLayer(name="Hero", prompt="hero figure", image=image)
    )
    session.doc.selected_id = session.doc.objects[0].id

    session.duplicate_selected()

    assert len(session.doc.objects) == 2
    copy = session.doc.objects[1]
    assert copy.id != session.doc.objects[0].id
    assert copy.prompt == "hero figure"
    assert copy.image is not None
    assert copy.image.size == image.size
    assert copy.image.getpixel((0, 0)) == image.getpixel((0, 0))
    assert session.doc.selected_id == copy.id


def test_reset_workspace_clears_layers_and_output(tmp_path):
    session = _session(tmp_path)
    session.doc.background = Image.new("RGB", (32, 32), "white")
    session.doc.background_prompt = "bg"
    session.doc.add_object(ObjectLayer(name="A", prompt="object a"))
    session.last_output = Image.new("RGB", (32, 32), "blue")
    session.last_output_prompt = "stale"

    session.reset_workspace()

    assert session.doc.background is None
    assert session.doc.objects == []
    assert session.last_output is None
    assert session.last_output_prompt == ""
    assert session.output_settings.denoise == 0.3
