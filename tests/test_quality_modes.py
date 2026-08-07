"""Tests for Fast/Quality OUTPUT packs and preview sizing."""

from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.pipeline.quality_modes import (
    normalize_quality_mode,
    quality_mode_pack,
)
from xwave_composer.pipeline.session import ComposerSession


def test_normalize_quality_mode():
    assert normalize_quality_mode("Fast") == "fast"
    assert normalize_quality_mode("QUALITY") == "quality"
    assert normalize_quality_mode("hq") == "quality"
    assert normalize_quality_mode(None) == "quality"


def test_quality_mode_pack_defaults():
    cfg = AppConfig(raw={}, root=AppConfig.load().root)
    fast = quality_mode_pack(cfg, "fast")
    quality = quality_mode_pack(cfg, "quality")
    assert fast["denoise"] == 0.30
    assert fast["steps"] == 6
    assert quality["denoise"] == 0.55
    assert quality["steps"] == 10
    assert int(quality["preview_steps"]) >= int(fast["preview_steps"])


def test_quality_mode_pack_config_override():
    cfg = AppConfig(
        raw={
            "sdxl_hyper": {
                "quality_modes": {
                    "fast": {"denoise": 0.25, "steps": 4, "preview_steps": 2},
                }
            }
        },
        root=AppConfig.load().root,
    )
    pack = quality_mode_pack(cfg, "fast")
    assert pack == {"denoise": 0.25, "steps": 4, "preview_steps": 2}


def test_apply_quality_mode_updates_settings():
    session = ComposerSession(config=AppConfig.load())
    s = session.apply_quality_mode("fast")
    assert session.quality_mode == "fast"
    assert s.denoise == 0.30
    assert s.steps == 6
    s = session.apply_quality_mode("quality")
    assert session.quality_mode == "quality"
    assert abs(s.denoise - 0.55) < 1e-6
    assert s.steps == 10


def test_run_output_preview_resizes_and_flags(monkeypatch):
    session = ComposerSession(config=AppConfig.load())
    work = Image.new("RGB", (1024, 1024), (20, 40, 60))
    session.last_work = work
    session.doc.width = 1024
    session.doc.height = 1024

    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        init = kwargs["init_image"]
        return Image.new("RGB", init.size, (90, 100, 110))

    session.sdxl = MagicMock()
    session.sdxl.ready = True
    session.sdxl.refine = fake_refine
    # Pretend core is ready without loading real models.
    monkeypatch.setattr(
        type(session),
        "core_ready",
        property(lambda self: True),
    )
    monkeypatch.setattr(session, "build_prompt", lambda: "a test prompt")
    monkeypatch.setattr(session, "_schedule_flux_reload", lambda: None)

    out = session.run_output(work, preview=True)
    assert session.output_is_preview is True
    assert out.size == (1024, 1024)
    assert len(calls) == 1
    init = calls[0]["init_image"]
    # Default preview_scale is 1.0 — same canvas size, fewer steps only.
    assert init.size == (1024, 1024)
    assert calls[0]["steps"] == session.preview_steps_for_mode()
    assert calls[0]["steps"] < session.output_settings.steps

    out2 = session.run_output(work, preview=False)
    assert session.output_is_preview is False
    assert out2.size == (1024, 1024)
    assert calls[1]["steps"] == session.output_settings.steps
    assert calls[1]["init_image"].size == (1024, 1024)


def test_ensure_full_output_rerenders_preview(monkeypatch):
    session = ComposerSession(config=AppConfig.load())
    session.last_output = Image.new("RGB", (64, 64), (1, 2, 3))
    session.output_is_preview = True
    ran = {"n": 0}

    def fake_run(*, preview=False):
        ran["n"] += 1
        session.output_is_preview = preview
        session.last_output = Image.new("RGB", (64, 64), (9, 9, 9))
        return session.last_output

    monkeypatch.setattr(session, "run_output", fake_run)
    out = session.ensure_full_output()
    assert ran["n"] == 1
    assert session.output_is_preview is False
    assert out is session.last_output

    # Already full — no re-render.
    out2 = session.ensure_full_output()
    assert ran["n"] == 1
    assert out2 is session.last_output
