"""Disk-backed image library store tests (no GPU)."""

from __future__ import annotations

from PIL import Image

from xwave_composer.library.render import pack_library_views, pager_label
from xwave_composer.library.store import PAGE_SIZE, ImageLibrary, THUMB_MAX_SIDE
from xwave_composer.pipeline.session import ComposerSession
from xwave_composer.config import AppConfig


def _rgb(color, size=(64, 48)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_add_page_persist_delete(tmp_path):
    lib = ImageLibrary(tmp_path / "library")
    first = lib.add(_rgb("red", (80, 40)), "compose", name="Red still")
    second = lib.add(_rgb("blue", (32, 32)), "infinite_canvas")
    assert lib.count() == 2
    assert first.name == "Red still"
    assert first.width == 80 and first.height == 40
    page = lib.page(0, limit=1)
    assert page.total == 2
    assert page.items[0].id == second.id
    assert page.page_count == 2

    thumb = Image.open(lib.thumb_path(first))
    assert max(thumb.size) <= THUMB_MAX_SIDE
    full = lib.open_full(first.id)
    assert full is not None
    assert full.size == (80, 40)

    reopened = ImageLibrary(tmp_path / "library")
    assert reopened.count() == 2
    assert reopened.get(first.id).name == "Red still"

    assert reopened.delete(first.id)
    assert reopened.count() == 1
    assert reopened.get(first.id) is None
    assert not lib.full_path(first).exists()


def test_page_does_not_load_full_bytes(tmp_path):
    lib = ImageLibrary(tmp_path / "library")
    for i in range(3):
        lib.add(_rgb((i * 40, 10, 10), (400, 300)), "compose", name=f"n{i}")
    page = lib.page(0, limit=2)
    assert len(page.items) == 2
    assert page.offset == 0
    assert "2 of 3" in pager_label(page) or "1–2 of 3" in pager_label(page)


def test_set_name_and_roundtrip_pil(tmp_path):
    lib = ImageLibrary(tmp_path / "library")
    item = lib.add(_rgb("green", (16, 16)), "compose")
    updated = lib.set_name(item.id, "  forest  canopy  ")
    assert updated is not None
    assert updated.name == "forest canopy"
    loaded = lib.open_full(item.id)
    assert loaded.getpixel((0, 0)) == (0, 128, 0) or loaded.getpixel((0, 0))[1] > 0


def test_pack_views_empty_and_selected(tmp_path):
    lib = ImageLibrary(tmp_path / "library")
    html, pager, *_ = pack_library_views(lib)
    assert "empty" in html.lower() or "No images" in pager
    item = lib.add(_rgb("white"), "compose", "hero")
    html, pager, compose, _, ic, _, edit, _ = pack_library_views(lib, selected_id=item.id)
    assert item.id in html
    assert "is-selected" in html
    assert "compose" in compose
    assert "infinite" in ic
    assert "edit" in edit


def test_compose_import_from_library_path(tmp_path):
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
    session = ComposerSession(cfg)
    session.doc.selected_id = "__bg__"
    lib = ImageLibrary(tmp_path / "library")
    item = lib.add(_rgb("navy", (32, 24)), "compose")
    img = lib.open_full(item.id)
    session.import_into_selected(img, cutout="none", prompt="from library")
    assert session.doc.background is not None
    assert session.doc.background.size == (session.doc.width, session.doc.height)
    assert session.doc.background_source == "imported"


def test_page_size_constant():
    assert PAGE_SIZE == 24
