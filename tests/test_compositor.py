"""Unit tests for WORK canvas composition (no GPU required)."""

from __future__ import annotations

from PIL import Image

from xwave_composer.canvas.layers import LayerTransform, ObjectLayer, WorkDocument
from xwave_composer.canvas.compositor import compose_work_image, transform_object_layer
from xwave_composer.style.style_manager import StylePreset, build_output_prompt


def test_compose_background_only():
    doc = WorkDocument(width=128, height=128)
    doc.background = Image.new("RGB", (128, 128), (10, 20, 30))
    out = compose_work_image(doc)
    assert out.size == (128, 128)
    assert out.getpixel((0, 0)) == (10, 20, 30)


def test_compose_background_flip_x():
    doc = WorkDocument(width=64, height=32)
    # Left half red, right half blue — flip X should swap them.
    bg = Image.new("RGB", (64, 32), (0, 0, 255))
    for x in range(32):
        for y in range(32):
            bg.putpixel((x, y), (255, 0, 0))
    doc.background = bg
    doc.bg_flip_x = True
    out = compose_work_image(doc)
    assert out.getpixel((8, 16))[2] > 200  # blue on left after flip
    assert out.getpixel((56, 16))[0] > 200  # red on right after flip


def test_object_layer_flip_x():
    # Left half red, right half blue — flip X should swap them around center.
    rgba = Image.new("RGBA", (32, 16), (0, 0, 255, 255))
    for x in range(16):
        for y in range(16):
            rgba.putpixel((x, y), (255, 0, 0, 255))
    layer = ObjectLayer(
        name="flip",
        image=rgba,
        transform=LayerTransform(x=16, y=8, scale_x=1.0, scale_y=1.0, flip_x=True),
    )
    placed = transform_object_layer(layer, 32, 16)
    assert placed is not None
    assert placed.getpixel((4, 8))[2] > 200  # blue on left after flip
    assert placed.getpixel((28, 8))[0] > 200  # red on right after flip


def test_feather_alpha_inward_softens_edge():
    from xwave_composer.canvas.compositor import feather_alpha_inward

    # Opaque disc on transparent canvas — feather should clear the rim.
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for y in range(64):
        for x in range(64):
            if (x - 32) ** 2 + (y - 32) ** 2 <= 20**2:
                rgba.putpixel((x, y), (255, 0, 0, 255))
    soft = feather_alpha_inward(rgba, radius=6)
    # Near the old disc rim should be much more transparent.
    assert soft.getpixel((32, 12))[3] < 80
    # Deep interior should remain opaque.
    assert soft.getpixel((32, 32))[3] > 200


def test_feather_opaque_image_insets_frame():
    from xwave_composer.canvas.compositor import feather_alpha_inward

    rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    soft = feather_alpha_inward(rgba, radius=8)
    assert soft.getpixel((0, 0))[3] < 40
    assert soft.getpixel((32, 32))[3] > 200


def test_compose_background_scale_zoom():
    doc = WorkDocument(width=64, height=64)
    doc.background = Image.new("RGB", (64, 64), (12, 34, 56))
    doc.bg_scale = 2.0
    out = compose_work_image(doc)
    assert out.size == (64, 64)
    assert out.getpixel((32, 32)) == (12, 34, 56)


def test_compose_background_offset():
    doc = WorkDocument(width=64, height=64)
    # Solid red; after +16 X offset the left edge should show the blank fill.
    doc.background = Image.new("RGB", (64, 64), (200, 10, 10))
    doc.bg_offset_x = 16
    out = compose_work_image(doc)
    assert out.getpixel((0, 32)) != (200, 10, 10)
    assert out.getpixel((48, 32)) == (200, 10, 10)


def test_blend_multiply_darkens():
    from xwave_composer.canvas.compositor import composite_layer

    base = Image.new("RGBA", (32, 32), (200, 200, 200, 255))
    over = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
    out = composite_layer(base, over, "multiply")
    r, g, b, a = out.getpixel((16, 16))
    assert a == 255
    assert r < 200 and g < 200 and b < 200


def test_compose_object_centered():
    doc = WorkDocument(width=200, height=200)
    doc.background = Image.new("RGB", (200, 200), (0, 0, 0))
    rgba = Image.new("RGBA", (40, 40), (255, 0, 0, 255))
    obj = ObjectLayer(
        name="red",
        prompt="red square",
        image=rgba,
        transform=LayerTransform(x=100, y=100, scale_x=1.0, scale_y=1.0, rotation=0),
    )
    doc.add_object(obj)
    out = compose_work_image(doc)
    # Center should be red-ish
    r, g, b = out.getpixel((100, 100))
    assert r > 200 and g < 50 and b < 50


def test_stretch_and_rotate():
    rgba = Image.new("RGBA", (50, 20), (0, 255, 0, 255))
    obj = ObjectLayer(
        image=rgba,
        transform=LayerTransform(x=64, y=64, scale_x=2.0, scale_y=0.5, rotation=45),
    )
    placed = transform_object_layer(obj, 128, 128)
    assert placed is not None
    assert placed.size == (128, 128)
    # Some green pixels should exist
    # Sample a grid of pixels for green content
    found = False
    for y in range(0, 128, 4):
        for x in range(0, 128, 4):
            p = placed.getpixel((x, y))
            if p[1] > 200 and p[3] > 0:
                found = True
                break
        if found:
            break
    assert found


def test_build_output_prompt_order():
    doc = WorkDocument()
    doc.background_prompt = "forest"
    doc.objects.append(ObjectLayer(prompt="fox"))
    doc.objects.append(ObjectLayer(prompt="cabin"))
    preset = StylePreset(
        name="Test",
        family="Art",
        cfg=1.8,
        denoise=0.7,
        eta=0.5,
        prefix="oil painting of ",
        suffix="rich textures, painterly",
        negative="photo",
    )
    p = build_output_prompt(doc, preset)
    assert p == "oil painting of forest, fox, cabin, rich textures, painterly"

    p_psc = build_output_prompt(doc, preset, order="psc")
    assert p_psc == "oil painting of, rich textures, painterly, forest, fox, cabin"

    p_cps = build_output_prompt(doc, preset, order="cps")
    assert p_cps == "forest, fox, cabin, oil painting of, rich textures, painterly"

    p_spc = build_output_prompt(doc, preset, order="spc")
    assert p_spc == "rich textures, painterly, oil painting of, forest, fox, cabin"


def test_build_output_prompt_manual_style():
    doc = WorkDocument()
    doc.background_prompt = "forest"
    p = build_output_prompt(
        doc, None, manual_prefix="watercolor", manual_suffix="paper texture"
    )
    assert p == "watercolor forest, paper texture"


def test_build_output_prompt_no_style():
    doc = WorkDocument()
    doc.background_prompt = "forest"
    doc.objects.append(ObjectLayer(prompt="fox"))
    assert build_output_prompt(doc, None) == "forest, fox"


def test_build_output_prompt_strips_embedding_tokens():
    doc = WorkDocument()
    doc.background_prompt = "city"
    preset = StylePreset(
        name="Tokens",
        family="Render",
        cfg=1.8,
        denoise=0.7,
        eta=0.5,
        prefix="render of ",
        suffix="octane render, <3D>, sharp",
        negative="<DEETS>, blurry",
    )
    p = build_output_prompt(doc, preset)
    assert "<3D>" not in p
    assert "octane render" in p and "sharp" in p


def test_layer_reorder():
    doc = WorkDocument()
    a = ObjectLayer(name="A", prompt="a")
    b = ObjectLayer(name="B", prompt="b")
    doc.add_object(a)
    doc.add_object(b)
    assert doc.objects[0].id == a.id
    doc.reorder(a.id, "up")
    assert doc.objects[-1].id == a.id


def test_object_appears_on_work_over_background():
    """Regression: isolated object must be visible in composed WORK image."""
    doc = WorkDocument(width=256, height=256)
    doc.background = Image.new("RGB", (256, 256), (0, 0, 40))
    # Opaque red square as isolated layer
    rgba = Image.new("RGBA", (80, 80), (255, 0, 0, 255))
    obj = ObjectLayer(
        name="red",
        prompt="red box",
        image=rgba,
        transform=LayerTransform(x=128, y=128, scale_x=1.0, scale_y=1.0),
    )
    doc.add_object(obj)
    out = compose_work_image(doc)
    r, g, b = out.getpixel((128, 128))
    assert r > 200 and g < 40 and b < 80, f"object not visible on WORK, got {(r, g, b)}"
    # Background corner still blue-ish
    r0, g0, b0 = out.getpixel((0, 0))
    assert b0 > 20
