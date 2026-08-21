"""CPU glitch effects on Edit layers (no GPU)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from xwave_composer.glitch.registry import coerce_params, get_effect, list_effects, run_effect
from xwave_composer.glitch.utils.helpers import generate_noise_map
from xwave_composer.pipeline.edit import EditLayer, EditSession
from tests.test_edit import FakeIsolator, _split_rgb


def test_registry_includes_xlitch_catalog():
    ids = {spec.id for spec in list_effects()}
    assert "posterize" in ids
    assert "advanced_pixel_sorting" in ids
    assert "slice_block_manipulation" in ids
    assert "double_expose" in ids
    assert "masked_merge" in ids
    assert "geometric_distortion" not in ids
    poster = get_effect("posterize")
    assert poster.params[0].default == 4
    offset = get_effect("offset")
    assert offset.warp is True
    assert coerce_params(poster, {})["levels"] == 4
    chunk = get_effect("advanced_pixel_sorting")
    coerced = coerce_params(chunk, {})
    assert coerced["sorting_method"] == "chunk"
    assert coerced["chunk_width"] == 48
    assert coerced["reverse_sort"] is True
    drift = get_effect("pixel_drift")
    assert coerce_params(drift, {})["direction"] == "right"
    assert coerce_params(drift, {})["intensity"] == 4.0


def test_generate_noise_map_is_vectorized_and_seeded():
    a = generate_noise_map((24, 32), scale=12, octaves=2, base=7)
    b = generate_noise_map((24, 32), scale=12, octaves=2, base=7)
    c = generate_noise_map((24, 32), scale=12, octaves=2, base=8)
    assert a.shape == (24, 32)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert float(a.min()) >= -1.2
    assert float(a.max()) <= 1.2


def test_posterize_and_offset_smoke():
    src = _split_rgb((32, 24))
    poster = run_effect("posterize", src, {"levels": 4})
    assert poster.size == src.size
    shifted = run_effect("offset", src, {"x_value": 8, "y_value": 0, "x_unit": "pixels", "y_unit": "pixels"})
    assert shifted.size == src.size
    sorted_im = run_effect(
        "advanced_pixel_sorting",
        src,
        {"sorting_method": "chunk", "chunk_width": 8, "chunk_height": 8, "sort_by": "brightness"},
    )
    assert sorted_im.size == src.size
    sliced = run_effect(
        "slice_block_manipulation",
        src,
        {"manipulation_type": "slice_shuffle", "slice_count": 4, "orientation": "rows", "seed": 1},
    )
    assert sliced.size == src.size


def test_apply_keeps_alpha_unless_warped():
    session = EditSession(isolator=FakeIsolator())
    src = _split_rgb((48, 32))
    session.load(src)
    session.add_point(8, 8)
    layer = session.selected()
    assert layer is not None and not layer.is_base
    before = np.array(layer.image.convert("RGBA").getchannel("A"))
    session.apply_effect("posterize", {"levels": 3})
    after = np.array(layer.image.convert("RGBA").getchannel("A"))
    assert np.array_equal(before, after)
    session.apply_effect(
        "offset",
        {"x_value": 6, "y_value": 0, "x_unit": "pixels", "y_unit": "pixels"},
        warp_alpha=True,
    )
    warped = np.array(layer.image.convert("RGBA").getchannel("A"))
    assert not np.array_equal(before, warped)


def test_double_expose_rest_only_changes_selected_layer():
    session = EditSession(isolator=FakeIsolator())
    src = _split_rgb((40, 24))
    session.load(src)
    session.add_point(6, 6)
    cut = session.selected()
    base = session.doc.base()
    assert cut is not None and base is not None
    base_before = np.array(base.image.convert("RGB"))
    cut_before = np.array(cut.image.convert("RGB"))
    session.apply_effect(
        "double_expose",
        {"blend_mode": "classic", "opacity": 0.5},
        secondary_key="rest",
    )
    base_after = np.array(session.doc.base().image.convert("RGB"))
    cut_after = np.array(cut.image.convert("RGB"))
    assert np.array_equal(base_before, base_after)
    assert not np.array_equal(cut_before, cut_after)
    session.undo_point()
    restored = np.array(cut.image.convert("RGB"))
    assert np.array_equal(cut_before, restored)


def test_manual_cut_layer_alpha_roundtrip():
    session = EditSession()
    src = Image.new("RGB", (20, 16), (10, 20, 30))
    session.load(src)
    mask = Image.new("L", src.size, 0)
    mask.paste(255, (2, 2, 10, 12))
    rgba = src.convert("RGBA")
    rgba.putalpha(mask)
    cut = EditLayer(name="Cut 1", kind="cut", image=rgba, mask=mask)
    session.doc.layers.append(cut)
    session.doc.selected_id = cut.id
    session.apply_effect("posterize", {"levels": 2})
    assert np.array_equal(np.array(mask), np.array(cut.image.getchannel("A")))
