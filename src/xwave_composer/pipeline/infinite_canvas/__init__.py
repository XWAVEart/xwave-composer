"""Infinite Canvas engine helpers — strength maps, priming, fusion, photometric.

Import submodules directly to avoid circular imports with region_fill, e.g.:

    from xwave_composer.pipeline.infinite_canvas.strength import build_strength_map
    from xwave_composer.pipeline.infinite_canvas.engine import CanvasEngine
"""

from xwave_composer.pipeline.infinite_canvas.blueprint import make_blueprint
from xwave_composer.pipeline.infinite_canvas.photometric import (
    match_stats_in_band,
    seam_energy,
)
from xwave_composer.pipeline.infinite_canvas.priming import prime_canvas
from xwave_composer.pipeline.infinite_canvas.strength import (
    build_strength_map,
    soft_stamp_mask,
    strength_to_change_map,
    to_latent_strength,
)

__all__ = [
    "build_strength_map",
    "make_blueprint",
    "match_stats_in_band",
    "prime_canvas",
    "seam_energy",
    "soft_stamp_mask",
    "strength_to_change_map",
    "to_latent_strength",
]
