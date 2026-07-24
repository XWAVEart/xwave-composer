#!/usr/bin/env python3
"""Benchmark xwave-composer compute profiles on an RTX 5090.

This script intentionally runs outside the default test suite. It loads one
profile at a time, records cold/warm latency and peak CUDA memory, and saves
fixed-seed outputs plus a JSON report for visual/metric comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xwave_composer.config import AppConfig  # noqa: E402
from xwave_composer.models.flux_generator import FluxGenerator  # noqa: E402
from xwave_composer.models.sdxl_hyper import SDXLHyperPipeline  # noqa: E402
from xwave_composer.optimization import (  # noqa: E402
    ComputeCapabilities,
    configure_blackwell_runtime,
    normalize_profile,
)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def timed(call) -> tuple[Any, float, float]:
    reset_peak()
    synchronize()
    started = time.perf_counter()
    result = call()
    synchronize()
    return result, time.perf_counter() - started, peak_gb()


def image_metrics(baseline: Image.Image, candidate: Image.Image) -> dict[str, float]:
    a = np.asarray(baseline.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(candidate.convert("RGB").resize(baseline.size), dtype=np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    metrics = {"mse": mse, "psnr_db": psnr}
    try:
        import lpips

        metric = lpips.LPIPS(net="alex")
        ta = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        with torch.inference_mode():
            metrics["lpips_alex"] = float(metric(ta, tb).item())
    except ImportError:
        pass
    return metrics


def run_profile(
    profile: str,
    args: argparse.Namespace,
    output_dir: Path,
    shared_sdxl_init: Image.Image | None = None,
) -> tuple[dict[str, Any], dict[str, Image.Image]]:
    cfg = AppConfig.load(args.config)
    cfg.raw.setdefault("optimization", {})["profile"] = profile
    cfg.raw["optimization"]["compile"] = not args.no_compile

    record: dict[str, Any] = {"profile": profile}
    images: dict[str, Image.Image] = {}

    flux = FluxGenerator(cfg)
    _, load_s, load_peak = timed(flux.load)
    record["flux"] = {
        "load_seconds": load_s,
        "load_peak_allocated_gb": load_peak,
        "optimization": flux.optimization_report.__dict__,
    }
    generation = lambda: flux.generate(
        args.prompt,
        width=args.width,
        height=args.height,
        steps=args.flux_steps,
        seed=args.seed,
    )
    image, cold_s, cold_peak = timed(generation)
    warm, warm_s, warm_peak = timed(generation)
    images["flux"] = warm
    record["flux"].update(
        {
            "cold_seconds": cold_s,
            "cold_peak_allocated_gb": cold_peak,
            "warm_seconds": warm_s,
            "warm_peak_allocated_gb": warm_peak,
        }
    )
    warm.save(output_dir / f"{profile}_flux.png")
    flux.unload()

    sdxl = SDXLHyperPipeline(cfg)
    _, load_s, load_peak = timed(sdxl.load)
    record["sdxl"] = {
        "load_seconds": load_s,
        "load_peak_allocated_gb": load_peak,
        "optimization": sdxl.optimization_report.__dict__,
    }
    init_source = shared_sdxl_init or image
    init = init_source.resize((args.width, args.height), Image.Resampling.LANCZOS)
    refine = lambda: sdxl.refine(
        init_image=init,
        prompt=args.output_prompt or args.prompt,
        denoise=args.denoise,
        steps=args.sdxl_steps,
        seed=args.seed,
        guidance_scale=1.0,
    )
    output, cold_s, cold_peak = timed(refine)
    warm_output, warm_s, warm_peak = timed(refine)
    images["sdxl"] = warm_output
    record["sdxl"].update(
        {
            "cold_seconds": cold_s,
            "cold_peak_allocated_gb": cold_peak,
            "warm_seconds": warm_s,
            "warm_peak_allocated_gb": warm_peak,
        }
    )
    warm_output.save(output_dir / f"{profile}_sdxl.png")
    sdxl.unload()
    return record, images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--profiles", nargs="+", default=["bf16", "mxfp8", "nvfp4"])
    parser.add_argument("--prompt", default="a red vintage motorcycle in a neon-lit studio")
    parser.add_argument("--output-prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--flux-steps", type=int, default=4)
    parser.add_argument("--sdxl-steps", type=int, default=6)
    parser.add_argument("--denoise", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/blackwell")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig.load(args.config)
    caps = configure_blackwell_runtime(cfg)
    if not caps.blackwell:
        raise SystemExit(f"Blackwell GPU required; detected {caps.summary()}")

    results: dict[str, Any] = {
        "capabilities": caps.__dict__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "profiles": [],
    }
    profile_images: dict[str, dict[str, Image.Image]] = {}
    shared_sdxl_init: Image.Image | None = None
    for raw in args.profiles:
        profile = normalize_profile(raw)
        record, images = run_profile(
            profile, args, output_dir, shared_sdxl_init=shared_sdxl_init
        )
        results["profiles"].append(record)
        profile_images[profile] = images
        if profile == "bf16":
            shared_sdxl_init = images["flux"].copy()
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    baseline = profile_images.get("bf16")
    if baseline:
        for record in results["profiles"]:
            profile = record["profile"]
            if profile == "bf16":
                continue
            record["quality_vs_bf16"] = {
                stage: image_metrics(baseline[stage], profile_images[profile][stage])
                for stage in ("flux", "sdxl")
            }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
