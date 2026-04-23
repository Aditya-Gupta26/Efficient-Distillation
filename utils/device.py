"""
Device utilities — machine-agnostic helpers for CUDA / MPS / CPU.

Import pattern in every training/validation script:
    from utils.device import get_device, maybe_autocast, make_scaler, pin_memory_for
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
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
      - CUDA : uses float16 autocast (full AMP benefit)
      - MPS  : no-op  (MPS autocast exists but is unstable; skip for safety)
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
      - CUDA : real GradScaler (prevents underflow in float16)
      - MPS / CPU : no-op shim (AMP not used, no scaling needed)
    """
    if enabled and device.type == "cuda":
        return torch.amp.GradScaler("cuda")

    class _NoOpScaler:
        def scale(self, loss):        return loss
        def unscale_(self, opt):      pass
        def step(self, opt):          opt.step()
        def update(self):             pass

    return _NoOpScaler()
