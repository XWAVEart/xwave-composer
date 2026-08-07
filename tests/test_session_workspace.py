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
    origin_id = session.doc.objects[0].id

    session.duplicate_selected()

    assert len(session.doc.objects) == 2
    copy = session.doc.objects[1]
    assert copy.id != origin_id
    assert copy.origin_id == origin_id
    assert copy.prompt == "hero figure"
    assert copy.image is not None
    assert copy.image.size == image.size
    assert copy.image.getpixel((0, 0)) == image.getpixel((0, 0))
    assert session.doc.selected_id == copy.id


def test_roll_all_skips_imports_rerolls_muted_and_duplicates(tmp_path, monkeypatch):
    session = _session(tmp_path)
    fake_sdxl = FakeSDXL(ready=True)
    session.sdxl = fake_sdxl
    gen_calls: list[tuple[str, int, bool]] = []

    def fake_bg(prompt: str, seed: int = -1):
        raise AssertionError("imported background must not regenerate")

    def fake_gen(prompt, isolation_prompt="", seed=-1, isolate=True):
        gen_calls.append((str(prompt), int(seed), bool(isolate)))
        obj = session.doc.selected()
        assert obj is not None
        color = (0, 255, 0, 255) if len(gen_calls) == 1 else (0, 0, 255, 255)
        obj.image = Image.new("RGBA", (16, 16), color)
        obj.raw_image = Image.new("RGB", (16, 16), color[:3])
        obj.source = "generated"
        obj.cutout = isolate
        session.last_work = Image.new("RGB", (32, 32), "gray")
        return session.last_work

    monkeypatch.setattr(session, "generate_background", fake_bg)
    monkeypatch.setattr(session, "generate_selected", fake_gen)

    session.doc.background = Image.new("RGB", (32, 32), "white")
    session.doc.background_prompt = "sky"
    session.doc.background_source = "imported"
    session.pending_raw = Image.new("RGB", (8, 8), "black")
    session.pending_prompt = "pending sam"

    origin = ObjectLayer(
        name="Hero",
        prompt="hero figure",
        image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
        source="generated",
        cutout=True,
    )
    session.doc.add_object(origin)
    session.doc.selected_id = origin.id
    session.duplicate_selected()
    dup = session.doc.objects[1]
    dup.transform.x = 220.0
    dup.transform.y = 40.0
    dup.transform.rotation = 33.0
    dup.transform.flip_x = True
    dup.transform.flip_y = True

    imported = ObjectLayer(
        name="Photo",
        prompt="imported image",
        image=Image.new("RGBA", (16, 16), (10, 20, 30, 255)),
        source="imported",
    )
    imported_px = imported.image.getpixel((0, 0))
    session.doc.add_object(imported)

    muted = ObjectLayer(
        name="Prop",
        prompt="muted prop",
        prompt_enabled=False,
        image=Image.new("RGBA", (16, 16), (9, 9, 9, 255)),
        source="generated",
        cutout=False,
    )
    session.doc.add_object(muted)

    old_seed = session.output_settings.seed
    session.roll_all()

    assert session.pending_raw is None
    assert session.pending_prompt == ""
    # Origin once + muted once (duplicate is not a separate generate).
    assert len(gen_calls) == 2
    assert gen_calls[0][0] == "hero figure"
    assert gen_calls[1][0] == "muted prop"
    assert gen_calls[1][2] is False

    origin = session.doc.objects[0]
    dup = session.doc.objects[1]
    assert origin.image.getpixel((0, 0)) == (0, 255, 0, 255)
    assert dup.image.getpixel((0, 0)) == (0, 255, 0, 255)
    assert dup.transform.x == 220.0
    assert dup.transform.y == 40.0
    assert dup.transform.rotation == 33.0
    assert dup.transform.flip_x is True
    assert dup.transform.flip_y is True

    assert session.doc.objects[2].image.getpixel((0, 0)) == imported_px
    assert session.doc.objects[3].image.getpixel((0, 0)) == (0, 0, 255, 255)
    assert session.output_settings.seed != old_seed
    assert fake_sdxl.calls >= 1


def test_toggle_selected_visibility_hides_from_work_keeps_prompt(tmp_path):
    session = _session(tmp_path)
    session.sdxl = FakeSDXL(ready=True)
    layer = ObjectLayer(
        name="Hero",
        prompt="red sports car",
        image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
    )
    session.doc.add_object(layer)
    session.doc.selected_id = layer.id
    session.doc.background_prompt = "studio backdrop"
    session.refresh_work()
    before = session.build_prompt()
    assert "red sports car" in before

    visible = session.toggle_selected_visibility()
    assert visible is False
    assert layer.transform.visible is False
    assert "red sports car" in session.build_prompt()
    # Hidden layers are skipped by the compositor.
    from xwave_composer.canvas.compositor import transform_object_layer

    assert transform_object_layer(layer, session.doc.width, session.doc.height) is None

    visible = session.toggle_selected_visibility()
    assert visible is True
    assert layer.transform.visible is True
    assert transform_object_layer(layer, session.doc.width, session.doc.height) is not None


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
