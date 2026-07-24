"""GPU / memory helpers for RTX 5090 class devices."""

from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def empty_cache() -> None:
    """Free unused CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available; running on CPU."
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    free, total = torch.cuda.mem_get_info(idx)
    return (
        f"GPU: {name} | VRAM total {total_gb:.1f} GB | "
        f"free {free / (1024**3):.1f} / {total / (1024**3):.1f} GB"
    )


def vram_stats() -> tuple[float, float] | None:
    """(used_gb, total_gb) for the current CUDA device, or None on CPU."""
    if not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        return None
    return (total - free) / (1024**3), total / (1024**3)


def move_to_device(module: torch.nn.Module, device: str, dtype: torch.dtype | None = None):
    """Move a module to device with optional dtype cast."""
    if dtype is not None:
        return module.to(device=device, dtype=dtype)
    return module.to(device=device)
