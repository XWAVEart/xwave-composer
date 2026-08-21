"""Map UI params onto Xlitch effect callables."""

from __future__ import annotations

from typing import Any, Callable

from PIL import Image

from .effects.blend import double_expose
from .effects.color import (
    chromatic_aberration,
    color_channel_manipulation,
    color_filter,
    color_shift_expansion,
    curved_hue_shift,
    gaussian_blur,
    histogram_glitch,
    noise_effect,
    posterize,
    sharpen_effect,
    simulate_jpeg_artifacts,
    split_and_shift_channels,
    vhs_effect,
)
from .effects.consolidated import advanced_pixel_sorting, slice_block_manipulation
from .effects.contour import contour_effect
from .effects.distortion import (
    offset_effect,
    perlin_noise_displacement,
    pixel_drift,
    pixel_scatter,
    ripple_effect,
    wave_distortion,
)
from .effects.glitch import bit_manipulation, data_mosh_blocks, databend_image
from .effects.patterns import concentric_shapes, masked_merge
from .effects.pixelate import pixelate_by_attribute, voronoi_pixelate


def _g(params: dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in params or params[key] in ("", None):
        return default
    return params[key]


def _seed(params: dict[str, Any]) -> int | None:
    raw = _g(params, "seed")
    if raw in ("", None):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _odd_kernel(value: Any, default: int) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        size = default
    if size < 1:
        size = default
    if size % 2 == 0:
        size += 1
    return size


def _color_channel(image: Image.Image, params: dict[str, Any], secondary: Image.Image | None = None) -> Image.Image:
    kind = str(_g(params, "manipulation_type", "swap"))
    if kind == "swap":
        return color_channel_manipulation(image, kind, _g(params, "swap_choice", "red-green"))
    if kind == "invert":
        return color_channel_manipulation(image, kind, _g(params, "invert_choice", "red"))
    if kind == "negative":
        return color_channel_manipulation(image, kind, None)
    return color_channel_manipulation(
        image,
        "adjust",
        _g(params, "adjust_choice", "red"),
        _g(params, "intensity_factor", 1.5),
    )


def _channel_shift(image: Image.Image, params: dict[str, Any], secondary: Image.Image | None = None) -> Image.Image:
    mode = str(_g(params, "mode", "shift"))
    centered = _g(params, "center_channel", "green")
    if mode == "mirror":
        return split_and_shift_channels(image, 0, "horizontal", centered, mode)
    return split_and_shift_channels(
        image,
        int(_g(params, "shift_amount", 42)),
        _g(params, "direction", "horizontal"),
        centered,
        mode,
    )


def _ripple(image: Image.Image, params: dict[str, Any], secondary: Image.Image | None = None) -> Image.Image:
    distortion_type = str(_g(params, "distortion_type", "color_shift"))
    distortion_params: dict[str, Any] = {}
    if distortion_type == "color_shift":
        distortion_params = {
            "factor_r": _g(params, "color_r_factor", 0.8),
            "factor_g": _g(params, "color_g_factor", 1.0),
            "factor_b": _g(params, "color_b_factor", 1.2),
        }
    elif distortion_type == "pixelation":
        distortion_params = {
            "scale": _g(params, "pixelation_scale", 10),
            "max_mag": _g(params, "pixelation_magnitude", 10.0),
        }
    return ripple_effect(
        image,
        int(_g(params, "num_droplets", 9)),
        float(_g(params, "amplitude", 45.0)),
        float(_g(params, "frequency", 0.4)),
        float(_g(params, "decay", 0.004)),
        distortion_type,
        distortion_params,
        seed=_seed(params),
    )


def _masked_merge(image: Image.Image, params: dict[str, Any], secondary: Image.Image | None = None) -> Image.Image:
    if secondary is None:
        raise ValueError("Masked Merge needs a secondary image.")
    mask_type = str(_g(params, "mask_type", "checkerboard"))
    width = _g(params, "mask_width", 32)
    if mask_type == "concentric_rectangles":
        width = _g(params, "rectangle_band_width", 16)
    elif mask_type == "concentric_circles":
        width = _g(params, "circle_band_width", 16)
    return masked_merge(
        image,
        secondary,
        mask_type,
        width=int(width),
        height=int(_g(params, "mask_height", 32)),
        random_seed=_seed(params),
        stripe_width=int(_g(params, "stripe_width", 16)),
        stripe_angle=int(_g(params, "stripe_angle", 45)),
        gradient_direction=_g(params, "gradient_direction", "up"),
        perlin_noise_scale=float(_g(params, "perlin_noise_scale", 0.01)),
        threshold=float(_g(params, "perlin_threshold", 0.5)),
        perlin_octaves=int(_g(params, "perlin_octaves", 1)),
        voronoi_cells=int(_g(params, "voronoi_num_cells", 50)),
        circle_origin=_g(params, "circle_origin", "center"),
        triangle_size=int(_g(params, "triangle_size", 50)),
    )


def _double_expose(image: Image.Image, params: dict[str, Any], secondary: Image.Image | None = None) -> Image.Image:
    if secondary is None:
        raise ValueError("Double Expose needs a secondary image.")
    return double_expose(
        image,
        secondary,
        blend_mode=_g(params, "blend_mode", "classic"),
        opacity=float(_g(params, "opacity", 0.5)),
    )


def _advanced_sort(image: Image.Image, params: dict[str, Any], secondary: Image.Image | None = None) -> Image.Image:
    kwargs = dict(params)
    method = str(kwargs.pop("sorting_method", "chunk"))
    kwargs["seed"] = _seed(params)
    return advanced_pixel_sorting(image, method, **kwargs)


def _slice_block(image: Image.Image, params: dict[str, Any], secondary: Image.Image | None = None) -> Image.Image:
    kwargs = dict(params)
    kind = str(kwargs.pop("manipulation_type", "slice_shuffle"))
    kwargs["seed"] = _seed(params)
    return slice_block_manipulation(image, kind, **kwargs)


RUNNERS: dict[str, Callable[..., Image.Image]] = {
    "color_filter": lambda im, p, s=None: color_filter(
        im,
        filter_type=_g(p, "filter_type", "solid"),
        color=_g(p, "color", "#FF0000"),
        blend_mode=_g(p, "blend_mode", "overlay"),
        opacity=float(_g(p, "opacity", 0.5)),
        gradient_color2=_g(p, "gradient_color2", "#0000FF"),
        gradient_angle=int(_g(p, "gradient_angle", 0)),
    ),
    "color_channel": _color_channel,
    "channel_shift": _channel_shift,
    "curved_hue_shift": lambda im, p, s=None: curved_hue_shift(
        im, float(_g(p, "curve_value", 180)), float(_g(p, "shift_amount", 45.0))
    ),
    "color_shift_expansion": lambda im, p, s=None: color_shift_expansion(
        im,
        num_points=int(_g(p, "num_points", 7)),
        shift_amount=int(_g(p, "shift_amount", 20)),
        expansion_type=_g(p, "expansion_type", "circle"),
        mode="xtreme",
        saturation_boost=float(_g(p, "saturation_boost", 0.5)),
        value_boost=float(_g(p, "value_boost", 0.0)),
        pattern_type=_g(p, "pattern_type", "random"),
        color_theme=_g(p, "color_theme", "full-spectrum"),
        decay_factor=float(_g(p, "decay_factor", 0.2)),
        seed=_seed(p),
    ),
    "histogram_glitch": lambda im, p, s=None: histogram_glitch(
        im,
        _g(p, "r_mode", "solarize"),
        _g(p, "g_mode", "solarize"),
        _g(p, "b_mode", "solarize"),
        float(_g(p, "r_freq", 1.0)),
        float(_g(p, "r_phase", 0.4)),
        float(_g(p, "g_freq", 1.0)),
        float(_g(p, "g_phase", 0.3)),
        float(_g(p, "b_freq", 1.0)),
        float(_g(p, "b_phase", 0.2)),
        float(_g(p, "gamma_value", 0.5)),
    ),
    "posterize": lambda im, p, s=None: posterize(im, int(_g(p, "levels", 4))),
    "advanced_pixel_sorting": _advanced_sort,
    "pixelate": lambda im, p, s=None: pixelate_by_attribute(
        im,
        pixel_width=int(_g(p, "width", 16)),
        pixel_height=int(_g(p, "height", 16)),
        attribute=_g(p, "attribute", "hue"),
        num_bins=int(_g(p, "bins", 200)),
    ),
    "voronoi_pixelate": lambda im, p, s=None: voronoi_pixelate(
        im,
        num_cells=int(_g(p, "num_cells", 50)),
        attribute=_g(p, "attribute", "hue"),
        seed=_seed(p),
    ),
    "gaussian_blur": lambda im, p, s=None: gaussian_blur(
        im, float(_g(p, "radius", 5.0)), _g(p, "sigma")
    ),
    "sharpen_effect": lambda im, p, s=None: sharpen_effect(
        im,
        method=_g(p, "method", "unsharp_mask"),
        intensity=float(_g(p, "intensity", 1.0)),
        radius=float(_g(p, "radius", 1.0) or 1.0),
        threshold=int(_g(p, "threshold", 0) or 0),
        edge_enhancement=float(_g(p, "edge_enhancement", 0.0) or 0.0),
        high_pass_radius=float(_g(p, "high_pass_radius", 3.0) or 3.0),
        custom_kernel=_g(p, "custom_kernel", "default"),
    ),
    "vhs_effect": lambda im, p, s=None: vhs_effect(
        im,
        quality_preset=_g(p, "quality_preset", "medium"),
        scan_line_intensity=float(_g(p, "scan_line_intensity", 0.3) or 0.3),
        scan_line_spacing=int(_g(p, "scan_line_spacing", 2) or 2),
        static_intensity=float(_g(p, "static_intensity", 0.2) or 0.2),
        static_type=_g(p, "static_type", "white"),
        vertical_hold_frequency=float(_g(p, "vertical_hold_frequency", 0.1) or 0.1),
        vertical_hold_intensity=float(_g(p, "vertical_hold_intensity", 5.0) or 5.0),
        color_bleeding=float(_g(p, "color_bleeding", 0.3) or 0.3),
        chroma_shift=float(_g(p, "chroma_shift", 0.2) or 0.2),
        tracking_errors=float(_g(p, "tracking_errors", 0.15) or 0.15),
        tape_wear=float(_g(p, "tape_wear", 0.1) or 0.1),
        head_switching_noise=float(_g(p, "head_switching_noise", 0.1) or 0.1),
        color_desaturation=float(_g(p, "color_desaturation", 0.3) or 0.3),
        brightness_variation=float(_g(p, "brightness_variation", 0.2) or 0.2),
        seed=_seed(p),
    ),
    "concentric_shapes": lambda im, p, s=None: concentric_shapes(
        im,
        int(_g(p, "num_points", 7)),
        _g(p, "shape_type", "triangle"),
        int(_g(p, "thickness", 2)),
        int(_g(p, "spacing", 20)),
        int(_g(p, "rotation_angle", 9)),
        int(_g(p, "darken_step", 0)),
        color_shift=int(_g(p, "color_shift_amount", 15)),
        seed=_seed(p),
    ),
    "contour": lambda im, p, s=None: contour_effect(
        im,
        num_levels=int(_g(p, "num_levels", 10)),
        noise_std=int(_g(p, "noise_std", 5)),
        smooth_sigma=int(_g(p, "smooth_sigma", 16)),
        line_thickness=int(_g(p, "line_thickness", 1)),
        grad_threshold=int(_g(p, "grad_threshold", 28)),
        min_distance=int(_g(p, "min_distance", 3)),
        max_line_length=int(_g(p, "max_line_length", 256)),
        blur_kernel_size=_odd_kernel(_g(p, "blur_kernel_size", 5), 5),
        sobel_kernel_size=_odd_kernel(_g(p, "sobel_kernel_size", 15), 15),
        seed=_seed(p),
    ),
    "pixel_drift": lambda im, p, s=None: pixel_drift(
        im,
        _g(p, "direction", "right"),
        int(_g(p, "bands", 12)),
        float(_g(p, "intensity", 4.0)),
    ),
    "perlin_displacement": lambda im, p, s=None: perlin_noise_displacement(
        im,
        float(_g(p, "scale", 44)),
        int(_g(p, "intensity", 25)),
        int(_g(p, "octaves", 3)),
        seed=_seed(p),
    ),
    "wave_distortion": lambda im, p, s=None: wave_distortion(
        im,
        wave_type=_g(p, "wave_type", "horizontal"),
        amplitude=float(_g(p, "amplitude", 20.0)),
        frequency=float(_g(p, "frequency", 0.02)),
        phase=float(_g(p, "phase", 0.0) or 0.0),
        secondary_wave=bool(_g(p, "secondary_wave", False)),
        secondary_amplitude=float(_g(p, "secondary_amplitude", 10.0) or 10.0),
        secondary_frequency=float(_g(p, "secondary_frequency", 0.05) or 0.05),
        secondary_phase=float(_g(p, "secondary_phase", 90.0) or 90.0),
        blend_mode=_g(p, "blend_mode", "add"),
        edge_behavior=_g(p, "edge_behavior", "wrap"),
        interpolation=_g(p, "interpolation", "bilinear"),
    ),
    "ripple": _ripple,
    "pixel_scatter": lambda im, p, s=None: pixel_scatter(
        im,
        _g(p, "direction", "horizontal"),
        _g(p, "select_by", "red"),
        float(_g(p, "min_value", 180)),
        float(_g(p, "max_value", 360)),
    ),
    "offset": lambda im, p, s=None: offset_effect(
        im,
        offset_x=float(_g(p, "x_value", 0.0) or 0.0),
        offset_y=float(_g(p, "y_value", 0.0) or 0.0),
        unit_x=_g(p, "x_unit", "pixels"),
        unit_y=_g(p, "y_unit", "pixels"),
    ),
    "slice_block_manipulation": _slice_block,
    "bit_manipulation": lambda im, p, s=None: bit_manipulation(
        im,
        chunk_size=int(_g(p, "chunk_size", 24)),
        offset=int(_g(p, "offset", 1)),
        xor_value=int(_g(p, "xor_value", 255)),
        skip_pattern=_g(p, "skip_pattern", "alternate"),
        manipulation_type=_g(p, "manipulation_type", "xor"),
        bit_shift=int(_g(p, "shift_amount", 1)),
        randomize=bool(_g(p, "randomize_effect", False)),
        random_seed=_seed(p),
    ),
    "data_mosh_blocks": lambda im, p, s=None: data_mosh_blocks(
        im,
        num_operations=int(_g(p, "operations", 69)),
        max_block_size=int(_g(p, "block_size", 128)),
        block_movement=_g(p, "movement", "swap"),
        color_swap=_g(p, "color_swap", "random"),
        invert=_g(p, "invert_colors", "random"),
        shift=_g(p, "shift_values", "random"),
        flip=_g(p, "flip_blocks", "random"),
        seed=_seed(p),
    ),
    "databend": lambda im, p, s=None: databend_image(
        im,
        float(_g(p, "intensity", 0.1)),
        bool(_g(p, "preserve_header", True)),
        _seed(p),
    ),
    "jpeg_artifacts": lambda im, p, s=None: simulate_jpeg_artifacts(
        im, float(_g(p, "intensity", 1.0))
    ),
    "noise_effect": lambda im, p, s=None: noise_effect(
        im,
        noise_type=_g(p, "noise_type", "film_grain"),
        intensity=float(_g(p, "intensity", 0.3)),
        grain_size=float(_g(p, "grain_size", 1.0)),
        color_variation=float(_g(p, "color_variation", 0.2)),
        noise_color=_g(p, "noise_color", "#FFFFFF"),
        blend_mode=_g(p, "blend_mode", "overlay"),
        pattern=_g(p, "pattern", "random"),
        seed=_seed(p),
    ),
    "chromatic_aberration": lambda im, p, s=None: chromatic_aberration(
        im,
        intensity=float(_g(p, "intensity", 5.0)),
        pattern=_g(p, "pattern", "radial"),
        red_shift_x=float(_g(p, "red_shift_x", 0.0) or 0.0),
        red_shift_y=float(_g(p, "red_shift_y", 0.0) or 0.0),
        blue_shift_x=float(_g(p, "blue_shift_x", 0.0) or 0.0),
        blue_shift_y=float(_g(p, "blue_shift_y", 0.0) or 0.0),
        center_x=float(_g(p, "center_x", 0.5) or 0.5),
        center_y=float(_g(p, "center_y", 0.5) or 0.5),
        falloff=_g(p, "falloff", "quadratic"),
        edge_enhancement=float(_g(p, "edge_enhancement", 0.0) or 0.0),
        color_boost=float(_g(p, "color_boost", 1.0) or 1.0),
        seed=_seed(p),
    ),
    "double_expose": _double_expose,
    "masked_merge": _masked_merge,
}
