"""Edit-tab document: SAM2 cuts, Base punch, composite (no GPU)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from xwave_composer.models.isolation import refine_sam_mask, tf_sam2_prompts
from xwave_composer.config import AppConfig
from xwave_composer.pipeline.edit import BASE_ID, PREVIEW_BG, EditSession
from xwave_composer.pipeline.session import ComposerSession


class FakeIsolator:
    sam2_ready = True

    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[float, float]], list[int]]] = []
        self.set_count = 0
        self.reset_count = 0

    def load_sam2(self) -> str:
        return "fake"

    def set_image(self, image: Image.Image) -> None:
        self.set_count += 1

    def reset_image(self) -> None:
        self.reset_count += 1

    def predict_mask(self, image, points, labels):
        self.calls.append((list(points), list(labels)))
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, : w // 2] = 255
        return mask


def _split_rgb(size=(64, 32)) -> Image.Image:
    img = Image.new("RGB", size, (0, 0, 255))
    left = Image.new("RGB", (size[0] // 2, size[1]), (255, 0, 0))
    img.paste(left, (0, 0))
    return img


def test_tf_sam2_prompts_are_four_level():
    pts, labs = tf_sam2_prompts([(12.0, 34.0), (8.0, 9.0)], [1, 0])
    assert pts == [[[[12.0, 34.0], [8.0, 9.0]]]]
    assert labs == [[[1, 0]]]


def test_refine_sam_mask_drops_distant_specks():
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[8:32, 8:36] = 255
    mask[2, 70] = 255
    mask[18:20, 74:76] = 255
    mask[36, 60] = 255
    cleaned = refine_sam_mask(mask, points=[(20.0, 20.0)], labels=[1], image_size=(80, 40))
    assert int(cleaned[20, 20]) == 255
    assert int(cleaned[2, 70]) == 0
    assert int(cleaned[19, 75]) == 0
    assert int(cleaned[36, 60]) == 0
    assert int((cleaned > 0).sum()) == int((mask[8:32, 8:36] > 0).sum())


def test_refine_sam_mask_keeps_small_object_under_cursor():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    mask[10:30, 10:30] = 255
    cleaned = refine_sam_mask(mask, points=[(3.0, 3.0)], labels=[1], image_size=(32, 32))
    assert int(cleaned[3, 3]) == 255
    assert int(cleaned[20, 20]) == 0


def test_refine_sam_mask_finds_object_near_missed_click():
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    cleaned = refine_sam_mask(mask, points=[(7.0, 12.0)], labels=[1], image_size=(24, 24))
    assert int(cleaned[12, 12]) == 255
    assert int((cleaned > 0).sum()) == 64


def test_load_creates_base_only():
    iso = FakeIsolator()
    session = EditSession(isolator=iso)
    src = _split_rgb()
    session.load(src, name="hero")
    assert session.has_image
    assert len(session.doc.layers) == 1
    base = session.doc.layers[0]
    assert base.id == BASE_ID
    assert base.is_base
    assert base.image is not None
    assert base.image.size == src.size
    assert iso.set_count == 0
    assert iso.reset_count == 1
    flat = session.flatten()
    assert flat.size == src.size
    assert flat.getpixel((4, 4)) == (255, 0, 0)
    assert flat.getpixel((src.width - 4, 4)) == (0, 0, 255)


def test_commit_punches_base_and_reconstructs(tmp_path):
    iso = FakeIsolator()
    session = EditSession(isolator=iso)
    src = _split_rgb((80, 40))
    session.load(src)
    session.set_mode("include")
    layer = session.add_point(10, 10)
    assert layer is not None
    assert session.doc.draft_mask is None
    assert session.doc.draft_points == []
    assert iso.calls
    assert len(session.doc.layers) == 2

    base = session.doc.base()
    assert base is not None and base.mask is not None
    base_a = np.array(base.mask)
    cut_a = np.array(layer.mask)
    assert base_a[5, 5] == 0
    assert cut_a[5, 5] == 255
    assert base_a[5, 70] == 255
    assert cut_a[5, 70] == 0

    flat = session.flatten()
    assert flat.size == src.size
    assert flat.getpixel((4, 4)) == (255, 0, 0)
    assert flat.getpixel((70, 4)) == (0, 0, 255)


def test_hidden_cut_shows_base_hole():
    iso = FakeIsolator()
    session = EditSession(isolator=iso)
    src = _split_rgb((64, 32))
    session.load(src)
    session.add_point(8, 8)
    cut = session.doc.cuts()[0]
    session.set_visible(cut.id, False)
    flat = session.flatten()
    assert flat.getpixel((4, 4)) == PREVIEW_BG
    assert flat.getpixel((60, 4)) == (0, 0, 255)


def test_delete_cut_restores_base():
    iso = FakeIsolator()
    session = EditSession(isolator=iso)
    src = _split_rgb((64, 32))
    session.load(src)
    session.add_point(8, 8)
    cut = session.doc.cuts()[0]
    session.delete_layer(cut.id)
    assert len(session.doc.layers) == 1
    assert session.doc.layers[0].is_base
    flat = session.flatten()
    assert flat.getpixel((4, 4)) == (255, 0, 0)
    assert flat.getpixel((60, 4)) == (0, 0, 255)
    session.delete_layer(BASE_ID)
    assert session.doc.layers[0].is_base


def test_include_click_then_exclude_trims_selected():
    iso = FakeIsolator()
    session = EditSession(isolator=iso)
    session.load(_split_rgb())
    session.set_mode("include")
    layer = session.add_point(12, 8)
    assert layer is not None
    assert len(iso.calls) == 1
    points, labels = iso.calls[0]
    assert points == [(12.0, 8.0)]
    assert labels == [1]
    session.set_mode("exclude")
    session.add_point(40, 8)
    assert len(iso.calls) == 2
    points, labels = iso.calls[-1]
    assert points == [(40.0, 8.0)]
    assert labels == [1]
    session.undo_point()
    assert session.doc.cuts() == []


def test_exclude_without_include_is_ignored():
    iso = FakeIsolator()
    session = EditSession(isolator=iso)
    session.load(_split_rgb())
    session.set_mode("exclude")
    session.add_point(10, 10)
    assert session.doc.draft_points == []
    assert iso.calls == []


def test_hover_glows_then_click_commits_without_second_predict():
    iso = FakeIsolator()
    session = EditSession(isolator=iso)
    src = _split_rgb((80, 40))
    session.load(src)
    session.hover_at(10, 10)
    assert session.doc.draft_mask is not None
    assert len(session.doc.layers) == 1
    glow = session.glow_overlay()
    assert glow is not None
    assert glow.mode == "RGBA"
    assert glow.size == src.size
    layer = session.cut_at(12, 11)
    assert layer is not None
    assert len(session.doc.layers) == 2
    assert len(iso.calls) == 1


def test_preview_keeps_size_with_draft_overlay():
    session = EditSession(isolator=FakeIsolator())
    src = _split_rgb((48, 24))
    session.load(src)
    session.add_point(6, 6)
    prev = session.preview()
    assert prev.size == src.size
    assert prev.mode == "RGB"


def test_enter_edit_mode_flag(tmp_path):
    cfg = AppConfig(
        raw={
            "device": "cpu",
            "paths": {
                "models_cache": str(tmp_path / "models"),
                "layers_dir": str(tmp_path / "layers"),
                "workspace_dir": str(tmp_path / "ws"),
            },
            "export": {"output_dir": str(tmp_path / "exports")},
        },
        root=tmp_path,
    )
    composer = ComposerSession(cfg)
    msg = composer.enter_edit_mode()
    assert composer.edit_active
    assert "Edit mode" in msg
    composer.enter_compose_mode()
    assert composer.compose_active
