"""Tests for style family loading and filtering."""

from __future__ import annotations

from pathlib import Path

from xwave_composer.config import AppConfig
from xwave_composer.style.style_manager import (
    STYLE_FAMILIES,
    StyleManager,
    normalize_family,
)


def test_normalize_family():
    assert normalize_family("art") == "Art"
    assert normalize_family("PHOTO") == "Photo"
    assert normalize_family("") == "Other"
    assert normalize_family("Custom") == "Custom"


def test_csv_loads_families():
    root = Path(__file__).resolve().parents[1]
    mgr = StyleManager(AppConfig(root=root))
    assert len(mgr.presets) >= 100
    assert all(p.family for p in mgr.presets.values())
    present = set(mgr.families_present())
    for fam in STYLE_FAMILIES:
        assert fam in present
    assert "Food Photo" in mgr.presets
    assert mgr.presets["Food Photo"].family == "Other"
    assert mgr.presets["Van Gogh"].family == "Art"
    assert mgr.presets["3D Render"].family == "Render"
    assert mgr.presets["Photoreal"].family == "Photo"
    assert mgr.presets["Claymation"].family == "Sculpture"
    assert mgr.presets["Mushroom Kingdom"].family == "Render"
    assert mgr.presets["Zeldascape"].family == "Render"
    assert mgr.presets["Stained Glass"].family == "Sculpture"
    art = mgr.names(["Art"])
    assert "Van Gogh" in art
    assert "3D Render" not in art
    assert set(mgr.names(["Art", "Photo"])).issuperset(set(art))
