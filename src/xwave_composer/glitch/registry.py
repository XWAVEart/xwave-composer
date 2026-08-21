"""Effect catalog: Xlitch dropdown groups plus Flask form defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image

from .apply import RUNNERS


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str  # int, float, bool, choice, color, seed
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, str], ...] = ()
    visible_when: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class EffectSpec:
    id: str
    group: str
    label: str
    two_image: bool = False
    warp: bool = False
    params: tuple[Param, ...] = ()


def _c(*pairs: tuple[str, str]) -> tuple[tuple[str, str], ...]:
    return pairs


def P(
    key: str,
    label: str,
    kind: str,
    default: Any,
    *,
    min: float | None = None,
    max: float | None = None,
    step: float | None = None,
    choices: tuple[tuple[str, str], ...] = (),
    when: dict[str, tuple[str, ...]] | None = None,
) -> Param:
    return Param(
        key=key,
        label=label,
        kind=kind,
        default=default,
        min=min,
        max=max,
        step=step,
        choices=choices,
        visible_when=when or {},
    )


SORT_BY = _c(
    ("color", "Color (R+G+B)"),
    ("brightness", "Brightness"),
    ("hue", "Hue"),
    ("red", "Red Channel"),
    ("green", "Green Channel"),
    ("blue", "Blue Channel"),
    ("saturation", "Saturation"),
    ("luminance", "Luminance"),
    ("contrast", "Contrast"),
)
SEED = P("seed", "Random seed", "seed", None, min=1, max=99999)
HIST_MODE = _c(("solarize", "Solarize"), ("log", "Log"), ("gamma", "Gamma"), ("normal", "Normal"))
PIX_ATTR = _c(
    ("color", "Color (Most Common)"),
    ("brightness", "Brightness"),
    ("hue", "Hue"),
    ("saturation", "Saturation"),
    ("luminance", "Luminance"),
)

EFFECT_GROUPS: tuple[tuple[str, str], ...] = (
    ("color", "Color and tone"),
    ("sorting", "Pixel sorting"),
    ("stylize", "Pixelation and stylization"),
    ("distort", "Distortion"),
    ("slice", "Slice and block"),
    ("glitch", "Glitch"),
    ("blend", "Blend"),
)

EFFECTS: tuple[EffectSpec, ...] = (
    EffectSpec(
        "color_filter",
        "color",
        "Color Filter",
        params=(
            P("filter_type", "Filter type", "choice", "solid", choices=_c(("solid", "Solid Color"), ("gradient", "Gradient"))),
            P("blend_mode", "Blend mode", "choice", "overlay", choices=_c(("overlay", "Overlay"), ("soft_light", "Soft Light"))),
            P("opacity", "Filter opacity", "float", 0.5, min=0.0, max=1.0, step=0.05),
            P("color", "Filter color", "color", "#FF0000"),
            P("gradient_color2", "Gradient color 2", "color", "#0000FF", when={"filter_type": ("gradient",)}),
            P("gradient_angle", "Gradient angle", "int", 0, min=0, max=360, when={"filter_type": ("gradient",)}),
        ),
    ),
    EffectSpec(
        "color_channel",
        "color",
        "Color Channel Manipulation",
        params=(
            P(
                "manipulation_type",
                "Manipulation type",
                "choice",
                "swap",
                choices=_c(("swap", "Swap Channels"), ("invert", "Invert Channel"), ("adjust", "Adjust Channel Intensity"), ("negative", "Negative")),
            ),
            P("swap_choice", "Swap channels", "choice", "red-green", choices=_c(("red-green", "Red-Green"), ("red-blue", "Red-Blue"), ("green-blue", "Green-Blue")), when={"manipulation_type": ("swap",)}),
            P("invert_choice", "Invert channel", "choice", "red", choices=_c(("red", "Red"), ("green", "Green"), ("blue", "Blue")), when={"manipulation_type": ("invert",)}),
            P("adjust_choice", "Adjust channel", "choice", "red", choices=_c(("red", "Red"), ("green", "Green"), ("blue", "Blue")), when={"manipulation_type": ("adjust",)}),
            P("intensity_factor", "Intensity factor", "float", 1.5, min=0.1, max=10.0, step=0.1, when={"manipulation_type": ("adjust",)}),
        ),
    ),
    EffectSpec(
        "channel_shift",
        "color",
        "RGB Channel Shift",
        params=(
            P("mode", "Mode", "choice", "shift", choices=_c(("shift", "Shift Channels"), ("mirror", "Mirror"))),
            P("shift_amount", "Shift amount", "int", 42, min=1, max=500, when={"mode": ("shift",)}),
            P("direction", "Shift direction", "choice", "horizontal", choices=_c(("horizontal", "Horizontal"), ("vertical", "Vertical")), when={"mode": ("shift",)}),
            P("center_channel", "Centered channel", "choice", "green", choices=_c(("red", "Red"), ("green", "Green"), ("blue", "Blue"))),
        ),
    ),
    EffectSpec(
        "curved_hue_shift",
        "color",
        "Hue Skrift",
        params=(
            P("curve_value", "Curve value", "float", 180, min=1, max=360, step=1),
            P("shift_amount", "Shift amount", "float", 45.0, min=-360.0, max=360.0, step=1),
        ),
    ),
    EffectSpec(
        "color_shift_expansion",
        "color",
        "Color Shift Expansion",
        params=(
            P("num_points", "Seed points", "int", 7, min=1, max=100),
            P("shift_amount", "Shift amount", "int", 20, min=1, max=50),
            P("expansion_type", "Expansion type", "choice", "circle", choices=_c(("square", "Square"), ("diamond", "Diamond"), ("circle", "Circle"))),
            P("pattern_type", "Pattern", "choice", "random", choices=_c(("random", "Random"), ("grid", "Grid"), ("edges", "Edges"))),
            P("color_theme", "Color theme", "choice", "full-spectrum", choices=_c(("full-spectrum", "Full spectrum"), ("warm", "Warm"), ("cool", "Cool"), ("pastel", "Pastel"))),
            P("saturation_boost", "Saturation boost", "float", 0.5, min=0.0, max=1.0, step=0.05),
            P("value_boost", "Value boost", "float", 0.0, min=0.0, max=1.0, step=0.05),
            P("decay_factor", "Decay", "float", 0.2, min=0.0, max=1.0, step=0.05),
            SEED,
        ),
    ),
    EffectSpec(
        "histogram_glitch",
        "color",
        "Histogram Glitch",
        params=(
            P("r_mode", "Red treatment", "choice", "solarize", choices=HIST_MODE),
            P("g_mode", "Green treatment", "choice", "solarize", choices=HIST_MODE),
            P("b_mode", "Blue treatment", "choice", "solarize", choices=HIST_MODE),
            P("r_freq", "Red solarize frequency", "float", 1.0, min=0.1, max=10.0, step=0.1),
            P("r_phase", "Red solarize phase", "float", 0.4, min=0.0, max=6.28, step=0.05),
            P("g_freq", "Green solarize frequency", "float", 1.0, min=0.1, max=10.0, step=0.1),
            P("g_phase", "Green solarize phase", "float", 0.3, min=0.0, max=6.28, step=0.05),
            P("b_freq", "Blue solarize frequency", "float", 1.0, min=0.1, max=10.0, step=0.1),
            P("b_phase", "Blue solarize phase", "float", 0.2, min=0.0, max=6.28, step=0.05),
            P("gamma_value", "Gamma value", "float", 0.5, min=0.1, max=3.0, step=0.05),
        ),
    ),
    EffectSpec(
        "posterize",
        "color",
        "Posterfy",
        params=(P("levels", "Levels", "int", 4, min=2, max=32),),
    ),
    EffectSpec(
        "advanced_pixel_sorting",
        "sorting",
        "Advanced Pixel Sorting",
        params=(
            P(
                "sorting_method",
                "Sorting method",
                "choice",
                "chunk",
                choices=_c(
                    ("chunk", "Chunk-Based Sorting"),
                    ("full_frame", "Full Frame Sorting"),
                    ("polar", "Polar Sorting"),
                    ("spiral", "Spiral Sorting"),
                    ("voronoi", "Voronoi-Based Sorting"),
                    ("perlin_noise", "Perlin Noise Sorting"),
                    ("perlin_full_frame", "Perlin Full Frame Sorting"),
                    ("wrapped", "Wrapped Sort"),
                ),
            ),
            P("sort_by", "Sort by", "choice", "brightness", choices=SORT_BY),
            P("reverse_sort", "Descending sort", "bool", True),
            P("chunk_width", "Chunk width", "int", 48, min=2, max=2048, when={"sorting_method": ("chunk",)}),
            P("chunk_height", "Chunk height", "int", 48, min=2, max=2048, when={"sorting_method": ("chunk",)}),
            P("sort_mode", "Sort mode", "choice", "horizontal", choices=_c(("horizontal", "Horizontal"), ("vertical", "Vertical"), ("diagonal", "Diagonal")), when={"sorting_method": ("chunk",)}),
            P("starting_corner", "Starting corner", "choice", "top-left", choices=_c(("top-left", "Top-Left"), ("top-right", "Top-Right"), ("bottom-left", "Bottom-Left"), ("bottom-right", "Bottom-Right")), when={"sorting_method": ("chunk",)}),
            P("direction", "Direction", "choice", "horizontal", choices=_c(("horizontal", "Horizontal"), ("vertical", "Vertical")), when={"sorting_method": ("full_frame", "perlin_noise", "wrapped")}),
            P("chunk_size", "Chunk size", "int", 64, min=8, max=128, when={"sorting_method": ("polar", "spiral")}),
            P("polar_sort_by", "Polar sort by", "choice", "radius", choices=_c(("angle", "Angle"), ("radius", "Radius")), when={"sorting_method": ("polar",)}),
            P("num_cells", "Number of cells", "int", 69, min=10, max=1000, when={"sorting_method": ("voronoi",)}),
            P("size_variation", "Size variation", "float", 0.8, min=0.0, max=1.0, step=0.05, when={"sorting_method": ("voronoi",)}),
            P("sort_order", "Sort order", "choice", "clockwise", choices=_c(("clockwise", "Clockwise"), ("counter-clockwise", "Counter-clockwise")), when={"sorting_method": ("voronoi",)}),
            P("voronoi_orientation", "Line orientation", "choice", "spiral", choices=_c(("horizontal", "Horizontal"), ("vertical", "Vertical"), ("radial", "Radial"), ("spiral", "Spiral")), when={"sorting_method": ("voronoi",)}),
            P("start_position", "Start position", "choice", "center", choices=_c(("center", "Center"), ("top-left", "Top-Left"), ("top-right", "Top-Right"), ("bottom-left", "Bottom-Left"), ("bottom-right", "Bottom-Right")), when={"sorting_method": ("voronoi",)}),
            P("noise_scale", "Noise scale", "float", 0.008, min=0.001, max=0.1, step=0.001, when={"sorting_method": ("perlin_noise", "perlin_full_frame")}),
            P("pattern_width", "Pattern width", "int", 1, min=1, max=8, when={"sorting_method": ("perlin_full_frame",)}),
            P("perlin_chunk_width", "Perlin chunk width", "int", 120, min=8, max=1024, when={"sorting_method": ("perlin_noise",)}),
            P("perlin_chunk_height", "Perlin chunk height", "int", 1024, min=8, max=1024, when={"sorting_method": ("perlin_noise",)}),
            P("wrapped_chunk_width", "Wrapped chunk width", "int", 12, min=1, max=500, when={"sorting_method": ("wrapped",)}),
            P("wrapped_chunk_height", "Wrapped chunk height", "int", 123, min=1, max=500, when={"sorting_method": ("wrapped",)}),
            P("wrapped_starting_corner", "Wrapped starting corner", "choice", "top-left", choices=_c(("top-left", "Top-Left"), ("top-right", "Top-Right"), ("bottom-left", "Bottom-Left"), ("bottom-right", "Bottom-Right")), when={"sorting_method": ("wrapped",)}),
            P("wrapped_flow_direction", "Chunk flow", "choice", "primary", choices=_c(("primary", "Primary"), ("secondary", "Secondary")), when={"sorting_method": ("wrapped",)}),
            SEED,
        ),
    ),
    EffectSpec(
        "pixelate",
        "stylize",
        "Pixelate",
        params=(
            P("width", "Pixel width", "int", 16, min=2, max=64),
            P("height", "Pixel height", "int", 16, min=2, max=64),
            P("attribute", "Attribute", "choice", "hue", choices=PIX_ATTR),
            P("bins", "Number of bins", "int", 200, min=10, max=1000),
        ),
    ),
    EffectSpec(
        "voronoi_pixelate",
        "stylize",
        "Voronoi Pixelate",
        params=(
            P("num_cells", "Number of cells", "int", 50, min=10, max=500),
            P("attribute", "Attribute", "choice", "hue", choices=PIX_ATTR),
            SEED,
        ),
    ),
    EffectSpec(
        "gaussian_blur",
        "stylize",
        "Gaussian Blur",
        params=(
            P("radius", "Blur radius", "float", 5.0, min=0.1, max=50.0, step=0.1),
            P("sigma", "Sigma", "float", None, min=0.1, max=20.0, step=0.1),
        ),
    ),
    EffectSpec(
        "sharpen_effect",
        "stylize",
        "Sharpen",
        params=(
            P("method", "Method", "choice", "unsharp_mask", choices=_c(("unsharp_mask", "Unsharp Mask"), ("high_pass", "High-Pass"), ("edge_enhance", "Edge Enhancement"), ("custom", "Custom Kernel"))),
            P("intensity", "Intensity", "float", 1.0, min=0.0, max=5.0, step=0.1),
            P("radius", "Blur radius", "float", 1.0, min=0.1, max=10.0, step=0.1, when={"method": ("unsharp_mask",)}),
            P("threshold", "Threshold", "int", 0, min=0, max=255, when={"method": ("unsharp_mask",)}),
            P("high_pass_radius", "High-pass radius", "float", 3.0, min=1.0, max=10.0, step=0.1, when={"method": ("high_pass",)}),
            P("custom_kernel", "Custom kernel", "choice", "default", choices=_c(("default", "Default Sharpen"), ("laplacian", "Laplacian"), ("sobel", "Sobel"), ("prewitt", "Prewitt")), when={"method": ("custom",)}),
            P("edge_enhancement", "Extra edge enhancement", "float", 0.0, min=0.0, max=2.0, step=0.05),
        ),
    ),
    EffectSpec(
        "vhs_effect",
        "stylize",
        "VHS",
        params=(
            P("quality_preset", "Quality preset", "choice", "medium", choices=_c(("high", "High Quality"), ("medium", "Medium Quality"), ("low", "Low Quality"), ("damaged", "Damaged"))),
            P("scan_line_intensity", "Scan line intensity", "float", 0.3, min=0.0, max=1.0, step=0.05),
            P("scan_line_spacing", "Scan line spacing", "int", 2, min=1, max=5),
            P("static_intensity", "Static intensity", "float", 0.2, min=0.0, max=1.0, step=0.05),
            P("static_type", "Static type", "choice", "white", choices=_c(("white", "White"), ("colored", "Colored"), ("mixed", "Mixed"))),
            P("vertical_hold_frequency", "Vertical hold issues", "float", 0.1, min=0.0, max=1.0, step=0.05),
            P("vertical_hold_intensity", "Vertical hold intensity", "float", 5.0, min=0.0, max=20.0, step=0.5),
            P("color_bleeding", "Color bleeding", "float", 0.3, min=0.0, max=1.0, step=0.05),
            P("chroma_shift", "Chroma/luma separation", "float", 0.2, min=0.0, max=1.0, step=0.05),
            P("tracking_errors", "Tracking errors", "float", 0.15, min=0.0, max=1.0, step=0.05),
            P("tape_wear", "Tape wear", "float", 0.1, min=0.0, max=1.0, step=0.05),
            P("head_switching_noise", "Head switching noise", "float", 0.1, min=0.0, max=1.0, step=0.05),
            P("color_desaturation", "Color desaturation", "float", 0.3, min=0.0, max=1.0, step=0.05),
            P("brightness_variation", "Brightness variation", "float", 0.2, min=0.0, max=1.0, step=0.05),
            SEED,
        ),
    ),
    EffectSpec(
        "concentric_shapes",
        "stylize",
        "Concentric Shapes",
        params=(
            P("num_points", "Seed points", "int", 7, min=1, max=100),
            P("shape_type", "Shape", "choice", "triangle", choices=_c(("square", "Square"), ("circle", "Circle"), ("hexagon", "Hexagon"), ("triangle", "Triangle"))),
            P("thickness", "Thickness", "int", 2, min=1, max=10),
            P("spacing", "Spacing", "int", 20, min=1, max=50),
            P("rotation_angle", "Rotation", "int", 9, min=0, max=360),
            P("darken_step", "Darken step", "int", 0, min=0, max=255),
            P("color_shift_amount", "Color shift", "int", 15, min=0, max=360),
            SEED,
        ),
    ),
    EffectSpec(
        "contour",
        "stylize",
        "Contour",
        params=(
            P("num_levels", "Contour levels", "int", 10, min=5, max=30),
            P("noise_std", "Noise amount", "int", 5, min=0, max=10),
            P("smooth_sigma", "Smoothness", "int", 16, min=1, max=20),
            P("line_thickness", "Line thickness", "int", 1, min=1, max=5),
            P("grad_threshold", "Gradient threshold", "int", 28, min=1, max=100),
            P("min_distance", "Minimum distance", "int", 3, min=1, max=20),
            P("max_line_length", "Max line length", "int", 256, min=50, max=500),
            P("blur_kernel_size", "Blur kernel", "int", 5, min=3, max=33),
            P("sobel_kernel_size", "Edge kernel", "int", 15, min=3, max=33),
            SEED,
        ),
    ),
    EffectSpec(
        "pixel_drift",
        "distort",
        "Pixel Drift",
        warp=True,
        params=(
            P("direction", "Drift direction", "choice", "right", choices=_c(("up", "Up"), ("down", "Down"), ("left", "Left"), ("right", "Right"))),
            P("bands", "Number of bands", "int", 12, min=1, max=48),
            P("intensity", "Drift intensity", "float", 4.0, min=0.1, max=10.0, step=0.1),
        ),
    ),
    EffectSpec(
        "perlin_displacement",
        "distort",
        "Perlin Displacement",
        warp=True,
        params=(
            P("scale", "Noise scale", "int", 44, min=10, max=500),
            P("intensity", "Displacement intensity", "int", 25, min=1, max=100),
            P("octaves", "Octaves", "int", 3, min=1, max=10),
            SEED,
        ),
    ),
    EffectSpec(
        "wave_distortion",
        "distort",
        "Wave Distortion",
        warp=True,
        params=(
            P("wave_type", "Wave type", "choice", "horizontal", choices=_c(("horizontal", "Horizontal"), ("vertical", "Vertical"), ("both", "Both"), ("diagonal", "Diagonal"), ("radial", "Radial"))),
            P("amplitude", "Amplitude", "float", 20.0, min=0.0, max=100.0, step=0.5),
            P("frequency", "Frequency", "float", 0.02, min=0.001, max=0.1, step=0.001),
            P("phase", "Phase", "float", 0.0, min=0.0, max=360.0, step=1),
            P("secondary_wave", "Secondary wave", "bool", False),
            P("secondary_amplitude", "Secondary amplitude", "float", 10.0, min=0.0, max=100.0, step=0.5, when={"secondary_wave": ("true",)}),
            P("secondary_frequency", "Secondary frequency", "float", 0.05, min=0.001, max=0.1, step=0.001, when={"secondary_wave": ("true",)}),
            P("secondary_phase", "Secondary phase", "float", 90.0, min=0.0, max=360.0, step=1, when={"secondary_wave": ("true",)}),
            P("blend_mode", "Blend", "choice", "add", choices=_c(("add", "Add"), ("multiply", "Multiply"), ("max", "Max"), ("interference", "Interference"))),
            P("edge_behavior", "Edge behavior", "choice", "wrap", choices=_c(("wrap", "Wrap"), ("clamp", "Clamp"), ("reflect", "Reflect"))),
            P("interpolation", "Interpolation", "choice", "bilinear", choices=_c(("nearest", "Nearest"), ("bilinear", "Bilinear"), ("bicubic", "Bicubic"))),
        ),
    ),
    EffectSpec(
        "ripple",
        "distort",
        "Ripple",
        warp=True,
        params=(
            P("num_droplets", "Droplets", "int", 9, min=1, max=20),
            P("amplitude", "Amplitude", "float", 45.0, min=1.0, max=50.0, step=0.5),
            P("frequency", "Frequency", "float", 0.4, min=0.01, max=1.0, step=0.01),
            P("decay", "Decay", "float", 0.004, min=0.001, max=0.1, step=0.001),
            P("distortion_type", "Distortion type", "choice", "color_shift", choices=_c(("color_shift", "Color Shift"), ("pixelation", "Pixelation"), ("none", "None"))),
            P("color_r_factor", "Red factor", "float", 0.8, min=0.5, max=1.5, step=0.05, when={"distortion_type": ("color_shift",)}),
            P("color_g_factor", "Green factor", "float", 1.0, min=0.5, max=1.5, step=0.05, when={"distortion_type": ("color_shift",)}),
            P("color_b_factor", "Blue factor", "float", 1.2, min=0.5, max=1.5, step=0.05, when={"distortion_type": ("color_shift",)}),
            P("pixelation_scale", "Pixelation scale", "int", 10, min=2, max=20, when={"distortion_type": ("pixelation",)}),
            P("pixelation_magnitude", "Pixelation magnitude", "float", 10.0, min=1.0, max=20.0, step=0.5, when={"distortion_type": ("pixelation",)}),
            SEED,
        ),
    ),
    EffectSpec(
        "pixel_scatter",
        "distort",
        "Pixel Scatter",
        warp=True,
        params=(
            P("direction", "Direction", "choice", "horizontal", choices=_c(("horizontal", "Horizontal"), ("vertical", "Vertical"))),
            P("select_by", "Select by", "choice", "red", choices=_c(("brightness", "Brightness"), ("red", "Red"), ("green", "Green"), ("blue", "Blue"), ("hue", "Hue"), ("saturation", "Saturation"), ("luminance", "Luminance"), ("contrast", "Contrast"))),
            P("min_value", "Minimum value", "float", 180, min=0, max=360, step=1),
            P("max_value", "Maximum value", "float", 360, min=0, max=360, step=1),
        ),
    ),
    EffectSpec(
        "offset",
        "distort",
        "Offset",
        warp=True,
        params=(
            P("x_value", "Horizontal offset", "float", 0.0, min=-2048, max=2048, step=1),
            P("x_unit", "Horizontal unit", "choice", "pixels", choices=_c(("pixels", "Pixels"), ("percentage", "Percentage"))),
            P("y_value", "Vertical offset", "float", 0.0, min=-2048, max=2048, step=1),
            P("y_unit", "Vertical unit", "choice", "pixels", choices=_c(("pixels", "Pixels"), ("percentage", "Percentage"))),
        ),
    ),
    EffectSpec(
        "slice_block_manipulation",
        "slice",
        "Slice and Block Manipulation",
        warp=True,
        params=(
            P(
                "manipulation_type",
                "Manipulation type",
                "choice",
                "slice_shuffle",
                choices=_c(("slice_shuffle", "Slice Shuffle"), ("slice_offset", "Slice Offset"), ("slice_reduction", "Slice Reduction"), ("block_shuffle", "Block Shuffle")),
            ),
            P("orientation", "Orientation", "choice", "rows", choices=_c(("rows", "Rows"), ("columns", "Columns")), when={"manipulation_type": ("slice_shuffle", "slice_offset", "slice_reduction")}),
            P("slice_count", "Slice count", "int", 16, min=4, max=256, when={"manipulation_type": ("slice_shuffle", "slice_offset", "slice_reduction")}),
            P("max_offset", "Maximum offset", "int", 50, min=1, max=512, when={"manipulation_type": ("slice_offset",)}),
            P("offset_mode", "Offset pattern", "choice", "random", choices=_c(("random", "Random"), ("sine", "Sine")), when={"manipulation_type": ("slice_offset",)}),
            P("frequency", "Sine frequency", "float", 0.1, min=0.01, max=1.0, step=0.01, when={"manipulation_type": ("slice_offset",)}),
            P("reduction_value", "Reduction value", "int", 2, min=2, max=8, when={"manipulation_type": ("slice_reduction",)}),
            P("block_width", "Block width", "int", 32, min=2, max=512, when={"manipulation_type": ("block_shuffle",)}),
            P("block_height", "Block height", "int", 32, min=2, max=512, when={"manipulation_type": ("block_shuffle",)}),
            SEED,
        ),
    ),
    EffectSpec(
        "bit_manipulation",
        "glitch",
        "Bit Manipulation",
        params=(
            P("chunk_size", "Chunk size", "int", 24, min=8, max=128),
            P("offset", "Byte offset", "int", 1, min=0, max=1000000),
            P("xor_value", "XOR value", "int", 255, min=0, max=255),
            P("skip_pattern", "Skip pattern", "choice", "alternate", choices=_c(("alternate", "Every Other Chunk"), ("every_third", "Every Third"), ("every_fourth", "Every Fourth"), ("random", "Random"))),
            P("manipulation_type", "Manipulation type", "choice", "xor", choices=_c(("xor", "XOR"), ("invert", "Invert"), ("shift", "Bit Shift"), ("swap", "Swap Chunks"))),
            P("shift_amount", "Bit shift amount", "int", 1, min=-7, max=7),
            P("randomize_effect", "Add randomness", "bool", False),
            SEED,
        ),
    ),
    EffectSpec(
        "data_mosh_blocks",
        "glitch",
        "Data Mosh Blocks",
        warp=True,
        params=(
            P("operations", "Operations", "int", 69, min=1, max=256),
            P("block_size", "Max block size", "int", 128, min=1, max=500),
            P("movement", "Block movement", "choice", "swap", choices=_c(("swap", "Swap"), ("in_place", "In Place"))),
            P("color_swap", "Color channel swap", "choice", "random", choices=_c(("never", "Never"), ("always", "Always"), ("random", "Random"))),
            P("invert_colors", "Color inversion", "choice", "random", choices=_c(("never", "Never"), ("always", "Always"), ("random", "Random"))),
            P("shift_values", "Channel value shift", "choice", "random", choices=_c(("never", "Never"), ("always", "Always"), ("random", "Random"))),
            P("flip_blocks", "Block flipping", "choice", "random", choices=_c(("never", "Never"), ("vertical", "Vertical"), ("horizontal", "Horizontal"), ("random", "Random"))),
            SEED,
        ),
    ),
    EffectSpec(
        "databend",
        "glitch",
        "Databending",
        params=(
            P("intensity", "Intensity", "float", 0.1, min=0.1, max=1.0, step=0.05),
            P("preserve_header", "Preserve header", "bool", True),
            SEED,
        ),
    ),
    EffectSpec(
        "jpeg_artifacts",
        "glitch",
        "JPEG Artifacts",
        params=(P("intensity", "Intensity", "float", 1.0, min=0.0, max=1.0, step=0.05),),
    ),
    EffectSpec(
        "noise_effect",
        "glitch",
        "Noise",
        params=(
            P("noise_type", "Noise type", "choice", "film_grain", choices=_c(("film_grain", "Film Grain"), ("digital", "Digital"), ("colored", "Colored"), ("salt_pepper", "Salt & Pepper"), ("gaussian", "Gaussian"))),
            P("intensity", "Intensity", "float", 0.3, min=0.0, max=1.0, step=0.05),
            P("grain_size", "Grain size", "float", 1.0, min=0.5, max=5.0, step=0.1),
            P("color_variation", "Color variation", "float", 0.2, min=0.0, max=1.0, step=0.05),
            P("noise_color", "Noise color", "color", "#FFFFFF"),
            P("blend_mode", "Blend mode", "choice", "overlay", choices=_c(("overlay", "Overlay"), ("add", "Add"), ("multiply", "Multiply"), ("screen", "Screen"))),
            P("pattern", "Pattern", "choice", "random", choices=_c(("random", "Random"), ("perlin", "Perlin-like"), ("cellular", "Cellular"))),
            SEED,
        ),
    ),
    EffectSpec(
        "chromatic_aberration",
        "glitch",
        "Chromatic Aberration",
        warp=True,
        params=(
            P("intensity", "Intensity", "float", 5.0, min=0.0, max=50.0, step=0.5),
            P("pattern", "Pattern", "choice", "radial", choices=_c(("radial", "Radial"), ("linear", "Linear"), ("barrel", "Barrel"), ("custom", "Custom"))),
            P("red_shift_x", "Red X shift", "float", 0.0, min=-20.0, max=20.0, step=0.5),
            P("red_shift_y", "Red Y shift", "float", 0.0, min=-20.0, max=20.0, step=0.5),
            P("blue_shift_x", "Blue X shift", "float", 0.0, min=-20.0, max=20.0, step=0.5),
            P("blue_shift_y", "Blue Y shift", "float", 0.0, min=-20.0, max=20.0, step=0.5),
            P("center_x", "Center X", "float", 0.5, min=0.0, max=1.0, step=0.01),
            P("center_y", "Center Y", "float", 0.5, min=0.0, max=1.0, step=0.01),
            P("falloff", "Falloff", "choice", "quadratic", choices=_c(("linear", "Linear"), ("quadratic", "Quadratic"), ("cubic", "Cubic"))),
            P("edge_enhancement", "Edge enhancement", "float", 0.0, min=0.0, max=1.0, step=0.05),
            P("color_boost", "Saturation boost", "float", 1.0, min=0.5, max=2.0, step=0.05),
            SEED,
        ),
    ),
    EffectSpec(
        "double_expose",
        "blend",
        "Double Expose",
        two_image=True,
        params=(
            P(
                "blend_mode",
                "Blend mode",
                "choice",
                "classic",
                choices=_c(
                    ("classic", "Classic Blend"),
                    ("screen", "Screen"),
                    ("multiply", "Multiply"),
                    ("overlay", "Overlay"),
                    ("hard_light", "Hard Light"),
                    ("difference", "Difference"),
                    ("exclusion", "Exclusion"),
                    ("add", "Add"),
                    ("subtract", "Subtract"),
                    ("darken_only", "Darken Only"),
                    ("lighten_only", "Lighten Only"),
                    ("color_dodge", "Color Dodge"),
                    ("burn", "Color Burn"),
                ),
            ),
            P("opacity", "Opacity", "float", 0.5, min=0.0, max=1.0, step=0.05),
        ),
    ),
    EffectSpec(
        "masked_merge",
        "blend",
        "Masked Merge",
        two_image=True,
        params=(
            P(
                "mask_type",
                "Mask type",
                "choice",
                "checkerboard",
                choices=_c(
                    ("checkerboard", "Checkerboard"),
                    ("random_checkerboard", "Random Checkerboard"),
                    ("striped", "Striped"),
                    ("gradient_striped", "Gradient Striped"),
                    ("linear_gradient_striped", "Linear Gradient Striped"),
                    ("perlin", "Perlin Noise"),
                    ("voronoi", "Voronoi Cells"),
                    ("concentric_rectangles", "Concentric Rectangles"),
                    ("concentric_circles", "Concentric Circles"),
                    ("random_triangles", "Random Triangles"),
                ),
            ),
            P("mask_width", "Mask width", "int", 32, min=2, max=512, when={"mask_type": ("checkerboard", "random_checkerboard")}),
            P("mask_height", "Mask height", "int", 32, min=2, max=512, when={"mask_type": ("checkerboard", "random_checkerboard")}),
            P("stripe_width", "Stripe width", "int", 16, min=1, max=200, when={"mask_type": ("striped", "gradient_striped", "linear_gradient_striped")}),
            P("stripe_angle", "Stripe angle", "int", 45, min=0, max=180, when={"mask_type": ("striped", "gradient_striped", "linear_gradient_striped")}),
            P("gradient_direction", "Gradient direction", "choice", "up", choices=_c(("up", "Up"), ("down", "Down")), when={"mask_type": ("linear_gradient_striped",)}),
            P("perlin_noise_scale", "Perlin scale", "float", 0.01, min=0.001, max=0.1, step=0.001, when={"mask_type": ("perlin",)}),
            P("perlin_threshold", "Perlin threshold", "float", 0.5, min=0.0, max=1.0, step=0.05, when={"mask_type": ("perlin",)}),
            P("perlin_octaves", "Perlin octaves", "int", 1, min=1, max=8, when={"mask_type": ("perlin",)}),
            P("voronoi_num_cells", "Voronoi cells", "int", 50, min=10, max=500, when={"mask_type": ("voronoi",)}),
            P("rectangle_band_width", "Rectangle band width", "int", 16, min=2, max=100, when={"mask_type": ("concentric_rectangles",)}),
            P("circle_band_width", "Circle band width", "int", 16, min=2, max=100, when={"mask_type": ("concentric_circles",)}),
            P("circle_origin", "Circle origin", "choice", "center", choices=_c(("center", "Center"), ("top-left", "Top-Left"), ("top-right", "Top-Right"), ("bottom-left", "Bottom-Left"), ("bottom-right", "Bottom-Right")), when={"mask_type": ("concentric_circles",)}),
            P("triangle_size", "Triangle size", "int", 50, min=4, max=256, when={"mask_type": ("random_triangles",)}),
            SEED,
        ),
    ),
)

_BY_ID = {spec.id: spec for spec in EFFECTS}


def get_effect(effect_id: str) -> EffectSpec:
    spec = _BY_ID.get(str(effect_id or ""))
    if spec is None:
        raise KeyError(f"Unknown effect: {effect_id}")
    return spec


def list_effects() -> tuple[EffectSpec, ...]:
    return EFFECTS


def effects_schema() -> dict[str, Any]:
    groups = [{"id": gid, "label": label} for gid, label in EFFECT_GROUPS]
    effects = []
    for spec in EFFECTS:
        params = []
        for param in spec.params:
            item = asdict(param)
            item["choices"] = [{"id": cid, "label": clabel} for cid, clabel in param.choices]
            item["visible_when"] = {k: list(v) for k, v in param.visible_when.items()}
            params.append(item)
        effects.append(
            {
                "id": spec.id,
                "group": spec.group,
                "label": spec.label,
                "two_image": spec.two_image,
                "warp": spec.warp,
                "params": params,
            }
        )
    return {"groups": groups, "effects": effects}


def coerce_params(spec: EffectSpec, raw: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(raw or {})
    out: dict[str, Any] = {}
    for param in spec.params:
        value = src.get(param.key, param.default)
        if value in ("", None) and param.kind in ("seed", "float", "int") and param.default is None:
            out[param.key] = None
            continue
        if value in ("", None):
            out[param.key] = param.default
            continue
        if param.kind == "int":
            out[param.key] = int(float(value))
        elif param.kind == "float":
            out[param.key] = float(value)
        elif param.kind == "bool":
            if isinstance(value, str):
                out[param.key] = value.strip().lower() in ("1", "true", "yes", "on")
            else:
                out[param.key] = bool(value)
        elif param.kind == "seed":
            try:
                out[param.key] = int(float(value))
            except (TypeError, ValueError):
                out[param.key] = None
        else:
            out[param.key] = value
    return out


def run_effect(
    effect_id: str,
    image: Image.Image,
    params: dict[str, Any] | None = None,
    secondary: Image.Image | None = None,
) -> Image.Image:
    spec = get_effect(effect_id)
    runner = RUNNERS[spec.id]
    coerced = coerce_params(spec, params)
    result = runner(image.convert("RGB"), coerced, secondary)
    if result.mode not in ("RGB", "RGBA"):
        result = result.convert("RGB")
    return result
