"""Blackwell compute profiles, quantization, and regional compilation.

The helpers in this module keep TorchAO optional. Model loaders can always
fall back to the BF16 path when a package, kernel, or model shape is not
compatible with the requested profile.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch

from xwave_composer.config import AppConfig

logger = logging.getLogger(__name__)

ComputeProfile = Literal["bf16", "mxfp8", "nvfp4"]
PROFILE_LABELS: dict[str, str] = {
    "bf16": "BF16",
    "mxfp8": "MXFP8 Balanced",
    "nvfp4": "NVFP4 Maximum",
}
PROFILE_CHOICES = list(PROFILE_LABELS.values())
_LABEL_TO_PROFILE = {label: key for key, label in PROFILE_LABELS.items()}


def normalize_profile(value: object, default: ComputeProfile = "mxfp8") -> ComputeProfile:
    """Normalize config/CLI/UI names to a supported compute profile."""
    raw = str(value or "").strip()
    key = _LABEL_TO_PROFILE.get(raw, raw.lower().replace("-", "").replace(" ", ""))
    aliases = {
        "none": "bf16",
        "off": "bf16",
        "balanced": "mxfp8",
        "fp8": "mxfp8",
        "maximum": "nvfp4",
        "fp4": "nvfp4",
    }
    key = aliases.get(key, key)
    return key if key in PROFILE_LABELS else default


def profile_label(profile: str) -> str:
    return PROFILE_LABELS.get(normalize_profile(profile), str(profile))


def configured_profile(config: AppConfig) -> ComputeProfile:
    return normalize_profile(config.get("optimization", "profile", default="mxfp8"))


@dataclass(frozen=True)
class ComputeCapabilities:
    cuda: bool
    device_name: str
    capability: tuple[int, int]
    blackwell: bool
    torchao: bool
    mslk: bool

    @classmethod
    def detect(cls) -> "ComputeCapabilities":
        cuda = torch.cuda.is_available()
        capability = torch.cuda.get_device_capability(0) if cuda else (0, 0)
        device_name = torch.cuda.get_device_name(0) if cuda else "CPU"
        return cls(
            cuda=cuda,
            device_name=device_name,
            capability=capability,
            blackwell=cuda and capability >= (10, 0),
            torchao=importlib.util.find_spec("torchao") is not None,
            mslk=importlib.util.find_spec("mslk") is not None,
        )

    def supports(self, profile: ComputeProfile) -> tuple[bool, str]:
        if profile == "bf16":
            return True, ""
        if not self.cuda:
            return False, "CUDA is unavailable"
        if not self.blackwell:
            return False, f"compute capability {self.capability} is not Blackwell"
        if not self.torchao:
            return False, "TorchAO is not installed"
        if profile == "nvfp4" and not self.mslk:
            return False, "MSLK is not installed"
        return True, ""

    def summary(self) -> str:
        cc = f"sm_{self.capability[0]}{self.capability[1]}"
        return (
            f"{self.device_name} ({cc}) | TorchAO "
            f"{'✓' if self.torchao else '✗'} · MSLK {'✓' if self.mslk else '✗'}"
        )


@dataclass
class OptimizationReport:
    component: str
    requested: ComputeProfile
    applied: str = "bf16"
    quantized_layers: int = 0
    compiled: bool = False
    fallback_reason: str = ""

    def status(self) -> str:
        text = f"{self.component}: {self.applied}"
        if self.quantized_layers:
            text += f" ({self.quantized_layers} linear layers)"
        if self.compiled:
            text += " + compiled"
        if self.fallback_reason:
            text += f" [fallback: {self.fallback_reason}]"
        return text


def configure_blackwell_runtime(config: AppConfig) -> ComputeCapabilities:
    """Enable safe CUDA runtime defaults and return detected capabilities."""
    caps = ComputeCapabilities.detect()
    if not caps.cuda:
        return caps
    allow_tf32 = bool(config.get("optimization", "allow_tf32", default=True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = bool(
        config.get("optimization", "cudnn_benchmark", default=True)
    )
    torch.set_float32_matmul_precision("high")
    return caps


_SENSITIVE_NAMES = (
    "embed",
    "embedding",
    "norm",
    "proj_out",
    "out_proj",
    "time_text",
    "context_embedder",
)


def selective_linear_filter(
    module: torch.nn.Module,
    fqn: str,
    *,
    min_features: int = 1024,
    require_divisible_16: bool = False,
) -> bool:
    """Select large BF16 Linear layers that benefit from Blackwell kernels."""
    if not isinstance(module, torch.nn.Linear):
        return False
    lowered = fqn.lower()
    if any(part in lowered for part in _SENSITIVE_NAMES):
        return False
    weight = getattr(module, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
        return False
    if module.in_features < min_features or module.out_features < min_features:
        return False
    if require_divisible_16 and (
        module.in_features % 16 != 0 or module.out_features % 16 != 0
    ):
        return False
    return True


def _eligible_layers(
    module: torch.nn.Module,
    *,
    min_features: int,
    require_divisible_16: bool,
) -> int:
    return sum(
        selective_linear_filter(
            child,
            fqn,
            min_features=min_features,
            require_divisible_16=require_divisible_16,
        )
        for fqn, child in module.named_modules()
    )


def quantize_component(
    module: torch.nn.Module,
    profile: ComputeProfile,
    component: Literal["flux", "sdxl"],
    config: AppConfig,
) -> OptimizationReport:
    """Quantize a component in place and return the applied recipe."""
    report = OptimizationReport(component=component, requested=profile)
    if profile == "bf16":
        return report

    caps = ComputeCapabilities.detect()
    supported, reason = caps.supports(profile)
    if not supported:
        report.fallback_reason = reason
        return report

    min_features = int(
        config.get("optimization", "min_linear_features", default=1024)
    )
    try:
        from torchao.quantization import Float8WeightOnlyConfig, quantize_

        if profile == "nvfp4":
            from torchao.prototype.mx_formats.inference_workflow import (
                NVFP4DynamicActivationNVFP4WeightConfig,
            )

            # Adapters are loaded and fused before this function runs. Applying
            # FP8 unconditionally to SDXL made the "NVFP4 Maximum" selection a
            # false label and left its attention/FFN linears at twice the
            # requested weight precision. Optional LoRAs already rebuild the
            # pipeline before quantization, so SDXL can use the same real
            # Blackwell FP4 recipe as Flux.
            quant_config = NVFP4DynamicActivationNVFP4WeightConfig(
                use_dynamic_per_tensor_scale=True,
                use_triton_kernel=bool(
                    config.get("optimization", "use_mslk_kernel", default=True)
                ),
            )
            applied = "nvfp4"
            require_divisible_16 = True
        elif component == "sdxl":
            # Keep the balanced SDXL recipe weight-only so optional adapters
            # remain cheap to rebuild and quantize.
            quant_config = Float8WeightOnlyConfig()
            applied = "fp8-weight-only"
            require_divisible_16 = False
        else:
            from torchao.prototype.mx_formats.inference_workflow import (
                MXDynamicActivationMXWeightConfig,
            )

            quant_config = MXDynamicActivationMXWeightConfig(
                activation_dtype=torch.float8_e4m3fn,
                weight_dtype=torch.float8_e4m3fn,
            )
            applied = "mxfp8"
            require_divisible_16 = False

        eligible = _eligible_layers(
            module,
            min_features=min_features,
            require_divisible_16=require_divisible_16,
        )
        if eligible == 0:
            report.fallback_reason = "no eligible large BF16 linear layers"
            return report

        def filter_fn(child: torch.nn.Module, fqn: str) -> bool:
            return selective_linear_filter(
                child,
                fqn,
                min_features=min_features,
                require_divisible_16=require_divisible_16,
            )

        quantize_(module, config=quant_config, filter_fn=filter_fn)
        report.applied = applied
        report.quantized_layers = eligible
        return report
    except Exception as exc:
        report.fallback_reason = f"{type(exc).__name__}: {exc}"
        logger.exception("%s quantization failed", component)
        return report


def make_nvfp4_linear_inputs_contiguous(module: torch.nn.Module) -> int:
    """Guard TorchAO NVFP4 linears against non-contiguous SDXL activations.

    TorchAO 0.17's ``nvfp4_linear`` flattens its input with ``view`` rather
    than ``reshape``. SDXL attention commonly supplies permuted 3D tensors,
    for which that operation raises. Keep the dependency untouched and make
    only inputs to quantized Linear modules contiguous.
    """

    def contiguous_input(
        _child: torch.nn.Module, args: tuple[Any, ...]
    ) -> tuple[Any, ...] | None:
        if not args or not isinstance(args[0], torch.Tensor):
            return None
        if args[0].is_contiguous():
            return None
        return (args[0].contiguous(), *args[1:])

    hooked = 0
    for child in module.modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        weight = getattr(child, "weight", None)
        if type(weight).__name__ != "NVFP4Tensor":
            continue
        if getattr(child, "_xwave_nvfp4_contiguous_hook", False):
            continue
        child.register_forward_pre_hook(contiguous_input)
        child._xwave_nvfp4_contiguous_hook = True
        hooked += 1
    return hooked


def compile_component(
    module: torch.nn.Module,
    report: OptimizationReport,
    config: AppConfig,
) -> OptimizationReport:
    """Regionally compile repeated blocks, preserving a usable eager fallback."""
    enabled = bool(config.get("optimization", "compile", default=True))
    if not enabled:
        return report
    if report.component == "sdxl" and report.applied == "nvfp4":
        # Torch 2.13 regional compilation decomposes SDXL's batched Linear
        # into aten.expand(weight), which TorchAO NVFP4Tensor does not
        # implement. Eager Linear uses TorchAO's native NVFP4 kernel.
        return report
    mode = str(config.get("optimization", "compile_mode", default="default"))
    fullgraph = bool(config.get("optimization", "compile_fullgraph", default=True))
    try:
        compile_repeated = getattr(module, "compile_repeated_blocks", None)
        if callable(compile_repeated):
            compile_repeated(mode=mode, fullgraph=fullgraph)
        else:
            raise RuntimeError("regional compilation is not supported by this model")
        report.compiled = True
    except Exception as exc:
        detail = f"compile {type(exc).__name__}: {exc}"
        report.fallback_reason = (
            f"{report.fallback_reason}; {detail}" if report.fallback_reason else detail
        )
        logger.warning("%s compilation skipped: %s", report.component, exc)
    return report
