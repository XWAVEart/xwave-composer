"""CPU-safe tests for Blackwell compute-profile behavior."""

from __future__ import annotations

import threading
from unittest.mock import Mock

import torch
from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.models.flux_generator import FluxGenerator
from xwave_composer.models.llm_rewriter import PromptRewriter
from xwave_composer.optimization import (
    ComputeCapabilities,
    OptimizationReport,
    compile_component,
    make_nvfp4_linear_inputs_contiguous,
    normalize_profile,
    profile_label,
    quantize_component,
    selective_linear_filter,
)
from xwave_composer.pipeline.session import ComposerSession


def test_profile_names_and_labels():
    assert normalize_profile("MXFP8 Balanced") == "mxfp8"
    assert normalize_profile("fp4") == "nvfp4"
    assert normalize_profile("off") == "bf16"
    assert profile_label("nvfp4") == "NVFP4 Maximum"


def test_capability_reasons_are_explicit():
    cpu = ComputeCapabilities(False, "CPU", (0, 0), False, False, False)
    assert cpu.supports("bf16") == (True, "")
    supported, reason = cpu.supports("mxfp8")
    assert not supported and "CUDA" in reason

    no_torchao = ComputeCapabilities(True, "GPU", (12, 0), True, False, False)
    supported, reason = no_torchao.supports("nvfp4")
    assert not supported and "TorchAO" in reason


def test_selective_filter_skips_small_and_sensitive_layers():
    large = torch.nn.Linear(1024, 2048, dtype=torch.bfloat16)
    small = torch.nn.Linear(128, 128, dtype=torch.bfloat16)
    assert selective_linear_filter(large, "transformer.blocks.0.ff")
    assert not selective_linear_filter(small, "transformer.blocks.0.ff")
    assert not selective_linear_filter(large, "transformer.proj_out")


def test_bf16_quantization_is_a_noop():
    cfg = AppConfig(raw={"optimization": {}}, root=AppConfig.load().root)
    module = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
    report = quantize_component(module, "bf16", "flux", cfg)
    assert report.applied == "bf16"
    assert report.quantized_layers == 0


def test_sdxl_nvfp4_profile_applies_real_nvfp4(monkeypatch):
    cfg = AppConfig(
        raw={
            "optimization": {
                "min_linear_features": 16,
                "use_mslk_kernel": True,
            }
        },
        root=AppConfig.load().root,
    )
    module = torch.nn.Sequential(
        torch.nn.Linear(16, 16, dtype=torch.bfloat16),
        torch.nn.SiLU(),
    )
    caps = ComputeCapabilities(True, "GPU", (12, 0), True, True, True)
    monkeypatch.setattr(ComputeCapabilities, "detect", classmethod(lambda cls: caps))
    seen: dict[str, object] = {}

    def fake_quantize(target, *, config, filter_fn):
        seen["config"] = config
        seen["eligible"] = [
            name
            for name, child in target.named_modules()
            if filter_fn(child, name)
        ]

    monkeypatch.setattr("torchao.quantization.quantize_", fake_quantize)
    report = quantize_component(module, "nvfp4", "sdxl", cfg)

    assert report.applied == "nvfp4"
    assert report.quantized_layers == 1
    assert type(seen["config"]).__name__ == (
        "NVFP4DynamicActivationNVFP4WeightConfig"
    )
    assert seen["eligible"] == ["0"]


def test_nvfp4_linear_inputs_are_made_contiguous():
    class NVFP4Tensor(torch.nn.Parameter):
        pass

    linear = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
    linear.weight = NVFP4Tensor(linear.weight.detach(), requires_grad=False)
    linear.forward = lambda x: x.is_contiguous()
    source = torch.zeros((2, 3, 16), dtype=torch.bfloat16).transpose(0, 1)
    assert not source.is_contiguous()

    assert make_nvfp4_linear_inputs_contiguous(linear) == 1
    assert make_nvfp4_linear_inputs_contiguous(linear) == 0
    assert linear(source) is True


def test_sdxl_nvfp4_skips_incompatible_regional_compile():
    class Repeated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def compile_repeated_blocks(self, **_kwargs):
            self.called = True

    cfg = AppConfig(
        raw={"optimization": {"compile": True}},
        root=AppConfig.load().root,
    )
    module = Repeated()
    report = compile_component(
        module, OptimizationReport("sdxl", "nvfp4", applied="nvfp4"), cfg
    )
    assert not report.compiled
    assert not module.called


def test_regional_compile_hook_is_used(monkeypatch):
    called: dict[str, object] = {}

    class Repeated(torch.nn.Module):
        def compile_repeated_blocks(self, **kwargs):
            called.update(kwargs)

    cfg = AppConfig(
        raw={
            "optimization": {
                "compile": True,
                "compile_mode": "reduce-overhead",
                "compile_fullgraph": True,
            }
        },
        root=AppConfig.load().root,
    )
    report = compile_component(
        Repeated(), OptimizationReport("flux", "mxfp8"), cfg
    )
    assert report.compiled
    assert called == {"mode": "reduce-overhead", "fullgraph": True}


def test_flux_rebuilds_clean_pipeline_after_quantization_failure(monkeypatch):
    cfg = AppConfig(
        raw={"device": "cpu", "optimization": {"profile": "mxfp8", "compile": False}},
        root=AppConfig.load().root,
    )
    flux = FluxGenerator(cfg)
    pipes = []

    def build(_model_id):
        pipe = Mock()
        pipe.transformer = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
        pipes.append(pipe)
        return pipe

    monkeypatch.setattr(flux, "_build_pipeline", build)
    monkeypatch.setattr(
        "xwave_composer.models.flux_generator.quantize_component",
        lambda *_args, **_kwargs: OptimizationReport(
            "flux", "mxfp8", fallback_reason="forced failure"
        ),
    )
    monkeypatch.setattr(
        "xwave_composer.models.flux_generator.compile_component",
        lambda _module, report, _config: report,
    )
    flux._load_model("test/model")
    assert len(pipes) == 2
    assert flux.pipe is pipes[-1]
    assert flux.optimization_report.applied == "bf16"
    assert "forced failure" in flux.optimization_report.fallback_reason


def test_session_profile_switch_updates_both_models():
    cfg = AppConfig(
        raw={"optimization": {"profile": "bf16"}},
        root=AppConfig.load().root,
    )
    session = ComposerSession(cfg)
    session.last_work = None

    for model, component in ((session.flux, "flux"), (session.sdxl, "sdxl")):
        model.set_profile = Mock(side_effect=lambda profile, reload=False: profile)
        model.unload = Mock()
        model.load = Mock(return_value="loaded")
        model.optimization_report = OptimizationReport(component, "mxfp8", applied="mxfp8")

    status = session.switch_compute_profile("MXFP8 Balanced")
    assert cfg.get("optimization", "profile") == "mxfp8"
    session.flux.set_profile.assert_called_once_with("mxfp8", reload=False)
    session.sdxl.set_profile.assert_called_once_with("mxfp8", reload=False)
    assert "Performance profile ready" in status


def test_llm_rewrite_is_cached_until_work_or_style_changes():
    cfg = AppConfig(
        raw={"llm": {"enabled_by_default": True}},
        root=AppConfig.load().root,
    )
    session = ComposerSession(cfg)
    session.doc.background_prompt = "a red car"
    session.last_work = Image.new("RGB", (64, 64), "red")
    session._rewrite_with_vram_headroom = Mock(return_value="rewritten red car")

    assert session.build_prompt() == "rewritten red car"
    assert session.build_prompt() == "rewritten red car"
    session._rewrite_with_vram_headroom.assert_called_once()

    session.last_work = Image.new("RGB", (64, 64), "blue")
    assert session.build_prompt() == "rewritten red car"
    assert session._rewrite_with_vram_headroom.call_count == 2


def test_llm_headroom_unloads_flux_but_keeps_sdxl_resident():
    cfg = AppConfig(raw={"llm": {"keep_loaded": False}}, root=AppConfig.load().root)
    session = ComposerSession(cfg)
    session.last_work = Image.new("RGB", (64, 64), "white")
    session.flux = Mock(ready=True)
    session.llm = Mock()
    session.llm.rewrite.return_value = "rewritten"
    sdxl_pipe = object()
    session.sdxl.pipe = sdxl_pipe

    assert session._rewrite_with_vram_headroom("source") == "rewritten"
    session.flux.unload.assert_called_once()
    session.llm.unload.assert_called_once()
    assert session.sdxl.pipe is sdxl_pipe
    assert session._flux_reload_pending


def test_flux_is_reloaded_in_background_after_qwen():
    cfg = AppConfig(
        raw={"flux": {"keep_loaded": True}},
        root=AppConfig.load().root,
    )
    session = ComposerSession(cfg)
    loaded = threading.Event()
    session.flux = Mock(ready=False)
    session.flux.load.side_effect = loaded.set
    session.llm = Mock(ready=False)
    session._flux_reload_pending = True

    session._schedule_flux_reload()
    assert loaded.wait(timeout=1)
    session.flux.load.assert_called_once()
    assert not session._flux_reload_pending


def test_llm_retries_when_first_generation_copies_source():
    cfg = AppConfig(
        raw={"device": "cpu", "llm": {"min_new_tokens": 1, "max_new_tokens": 20}},
        root=AppConfig.load().root,
    )
    rewriter = PromptRewriter(cfg)
    processor = Mock()
    processor.apply_chat_template.return_value = "chat"
    processor.return_value = {"input_ids": torch.tensor([[1, 2]])}
    rewritten = " ".join(f"detail{i}" for i in range(40))
    processor.decode.side_effect = ["source prompt", rewritten]
    model = Mock()
    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    rewriter.processor = processor
    rewriter.model = model
    rewriter._runtime_device = "cpu"

    result = rewriter.rewrite(
        "source prompt",
        content_prompt="subject",
        style_name="Test style",
        style_suffix="cinematic",
    )
    assert result == rewritten
    assert model.generate.call_count == 2
