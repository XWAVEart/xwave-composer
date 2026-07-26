"""Style flipbook helpers: sample styles and encode stills to MP4."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


def sample_style_names(
    all_names: Sequence[str],
    current_name: str | None,
    count: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Pick ``count`` style names, always including ``current_name`` when present.

    ``count <= 0`` or ``count >= len(all_names)`` returns a shuffled copy of all.
    If ``current_name`` is in the pool, it is forced into the result; remaining
    slots are filled by random sample from the rest.
    """
    names = [n for n in all_names if n]
    if not names:
        return []
    rng = rng or random.Random()
    n = int(count)
    if n <= 0 or n >= len(names):
        out = list(names)
        rng.shuffle(out)
        return out

    current = current_name if current_name in names else None
    if current is None:
        return rng.sample(names, n)

    others = [x for x in names if x != current]
    need = n - 1
    if need <= 0:
        return [current]
    if need >= len(others):
        picked = list(others)
    else:
        picked = rng.sample(others, need)
    out = [current, *picked]
    rng.shuffle(out)
    return out


def styles_to_generate(
    selected: Sequence[str],
    current_name: str | None,
) -> list[str]:
    """Styles that need a new refine — skip current (represented by frame 0)."""
    if current_name and current_name in selected:
        return [s for s in selected if s != current_name]
    return list(selected)


def flipbook_seeds_for_frames(
    base_seed: int,
    num_generate: int,
    *,
    increment: bool,
    rng: random.Random | None = None,
) -> list[int]:
    """Seeds for generated stills (excluding frame 0).

    ``increment=True`` → ``base+1 … base+N`` (wrapped to int32 range).
    ``increment=False`` → ``N`` independent random seeds.
    """
    n = max(0, int(num_generate))
    if n <= 0:
        return []
    base = int(base_seed) % 2_147_483_648
    if increment:
        return [(base + i) % 2_147_483_648 for i in range(1, n + 1)]
    rng = rng or random.Random()
    return [rng.randint(0, 2_147_483_647) for _ in range(n)]


def assemble_flipbook_stills(
    frame0: Image.Image,
    generated: Sequence[Image.Image],
    rng: random.Random | None = None,
    *,
    shuffle_rest: bool = True,
) -> list[Image.Image]:
    """Put ``frame0`` first; optionally shuffle the rest."""
    rest = [im.copy() for im in generated]
    if shuffle_rest:
        rng = rng or random.Random()
        rng.shuffle(rest)
    return [frame0.copy(), *rest]


def flipbook_total_frames(num_stills: int, frames_per_image: int) -> int:
    hold = max(1, int(frames_per_image))
    return max(0, int(num_stills)) * hold


def write_flipbook_mp4(
    stills: Sequence[Image.Image],
    path: Path,
    *,
    fps: int = 30,
    frames_per_image: int = 8,
) -> Path:
    """Write stills to MP4; each still is held for ``frames_per_image`` frames."""
    if not stills:
        raise ValueError("No stills to encode")
    fps = int(fps)
    if fps not in (24, 30, 48, 60):
        raise ValueError(f"Unsupported FPS: {fps}")
    hold = max(1, int(frames_per_image))

    first = stills[0].convert("RGB")
    w, h = first.size
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    try:
        for still in stills:
            rgb = still.convert("RGB")
            if rgb.size != (w, h):
                rgb = rgb.resize((w, h), Image.Resampling.LANCZOS)
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            for _ in range(hold):
                writer.write(bgr)
    finally:
        writer.release()
    return path
