"""Unit tests for style flipbook sampling and MP4 assembly helpers."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
from PIL import Image

from xwave_composer.pipeline.style_flipbook import (
    assemble_flipbook_stills,
    flipbook_seeds_for_frames,
    flipbook_total_frames,
    sample_style_names,
    styles_to_generate,
    write_flipbook_mp4,
)


def test_sample_includes_current_when_count_allows():
    names = [f"s{i}" for i in range(10)]
    rng = random.Random(1)
    picked = sample_style_names(names, "s3", 4, rng)
    assert len(picked) == 4
    assert "s3" in picked
    assert len(set(picked)) == 4


def test_sample_all_when_count_zero_or_large():
    names = ["a", "b", "c"]
    assert sorted(sample_style_names(names, "a", 0, random.Random(0))) == names
    assert sorted(sample_style_names(names, "a", 99, random.Random(1))) == names


def test_sample_without_current_just_samples():
    names = ["a", "b", "c", "d"]
    picked = sample_style_names(names, "missing", 2, random.Random(2))
    assert len(picked) == 2
    assert all(p in names for p in picked)


def test_styles_to_generate_skips_current():
    assert styles_to_generate(["a", "b", "c"], "b") == ["a", "c"]
    assert styles_to_generate(["a", "b"], None) == ["a", "b"]


def test_assemble_keeps_frame0_first():
    frame0 = Image.new("RGB", (8, 8), (255, 0, 0))
    others = [
        Image.new("RGB", (8, 8), (0, 255, 0)),
        Image.new("RGB", (8, 8), (0, 0, 255)),
    ]
    stills = assemble_flipbook_stills(frame0, others, random.Random(5))
    assert len(stills) == 3
    assert stills[0].getpixel((0, 0)) == (255, 0, 0)
    # Remaining are the two generated colors (order shuffled).
    rest = {stills[1].getpixel((0, 0)), stills[2].getpixel((0, 0))}
    assert rest == {(0, 255, 0), (0, 0, 255)}


def test_assemble_can_keep_generation_order():
    frame0 = Image.new("RGB", (8, 8), (1, 0, 0))
    others = [
        Image.new("RGB", (8, 8), (2, 0, 0)),
        Image.new("RGB", (8, 8), (3, 0, 0)),
    ]
    stills = assemble_flipbook_stills(frame0, others, shuffle_rest=False)
    assert [s.getpixel((0, 0))[0] for s in stills] == [1, 2, 3]


def test_flipbook_seeds_increment_and_random():
    assert flipbook_seeds_for_frames(10, 3, increment=True) == [11, 12, 13]
    assert flipbook_seeds_for_frames(0, 0, increment=True) == []
    rnd = flipbook_seeds_for_frames(10, 5, increment=False, rng=random.Random(0))
    assert len(rnd) == 5
    assert len(set(rnd)) == 5
    assert 10 not in rnd  # frame0 seed is not reused in the generated list


def test_flipbook_total_frames():
    assert flipbook_total_frames(10, 8) == 80
    assert flipbook_total_frames(0, 8) == 0
    assert flipbook_total_frames(3, 0) == 3  # hold clamped to 1


def test_write_flipbook_mp4(tmp_path: Path):
    stills = [
        Image.new("RGB", (64, 48), (10, 20, 30)),
        Image.new("RGB", (64, 48), (200, 100, 50)),
    ]
    path = tmp_path / "flip.mp4"
    write_flipbook_mp4(stills, path, fps=24, frames_per_image=2)
    assert path.exists() and path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == flipbook_total_frames(2, 2)
