"""
Device utilities — machine-agnostic helpers for CUDA / MPS / CPU,
with optional torchrun/DDP awareness.

Import pattern in every training/validation script:
    from utils.device import get_device, maybe_autocast, make_scaler, pin_memory_for
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_distributed_info() -> dict:
    """
    Return distributed launch metadata if running under torchrun.

    Returns:
        {
            "distributed": bool,
            "rank": int,
            "local_rank": int,
            "world_size": int,
        }
    """
    if not is_distributed():
        return {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        }

    return {
        "distributed": True,
        "rank": int(os.environ["RANK"]),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "world_size": int(os.environ["WORLD_SIZE"]),
    }


def get_device() -> torch.device:
    """
    Return the best available device.

    Under torchrun/DDP on CUDA, bind each process to its LOCAL_RANK GPU.
    Otherwise fall back to CUDA > MPS > CPU.
    """
    info = get_distributed_info()

    if info["distributed"]:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs.")
        return torch.device("cuda", info["local_rank"])

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pin_memory_for(device: torch.device) -> bool:
    """pin_memory is only beneficial (and supported) on CUDA."""
    return device.type == "cuda"


@contextmanager
def maybe_autocast(device: torch.device, enabled: bool):
    """
    AMP autocast that works on any device:
      - CUDA : uses float16 autocast
      - MPS  : no-op
      - CPU  : no-op
    """
    if enabled and device.type == "cuda":
        with torch.amp.autocast("cuda"):
            yield
    else:
        yield


def make_scaler(device: torch.device, enabled: bool):
    """
    GradScaler for loss scaling:
      - CUDA : real GradScaler
      - MPS / CPU : no-op shim
    """
    if enabled and device.type == "cuda":
        return torch.amp.GradScaler("cuda")

    class _NoOpScaler:
        def scale(self, loss):
            return loss

        def unscale_(self, opt):
            pass

        def step(self, opt):
            opt.step()

        def update(self):
            pass

    return _NoOpScaler()