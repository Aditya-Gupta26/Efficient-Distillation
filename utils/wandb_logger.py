"""
Weights & Biases initialisation helper.

Reads the ``wandb`` block from the config dict and returns a live W&B run
(or None if wandb is disabled / not installed).

Config schema (in distill_config.yaml):
    wandb:
      enabled:  true
      project:  "efficient-distillation"
      entity:   null          # wandb username / team, or null for default
      run_name: null          # null → wandb auto-generates a name
      tags:     ["swin", "knowledge-distillation", "coco"]
"""

from __future__ import annotations

from typing import Optional


def init_wandb(cfg: dict):
    """
    Initialise a W&B run from the ``wandb`` section of *cfg*.

    Args:
        cfg: Full config dict (as returned by ``load_config``).

    Returns:
        A ``wandb.Run`` object, or ``None`` if wandb is disabled / unavailable.
    """
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        print("[wandb_logger] wandb is not installed — skipping W&B logging.")
        return None

    # Use the mode from config; if not set, default to "online".
    mode = wcfg.get("mode", "online")

    run = wandb.init(
        project  = wcfg.get("project", "efficient-distillation"),
        entity   = wcfg.get("entity", None),
        name     = wcfg.get("run_name", None),
        tags     = wcfg.get("tags", []),
        config   = _flatten_cfg(cfg),
        resume   = "allow",
        mode     = mode,
    )
    print(f"[wandb_logger] Run initialised ({mode}) → {run.url}")
    return run


def _flatten_cfg(cfg: dict, prefix: str = "") -> dict:
    """Recursively flatten nested config dicts for wandb.config."""
    flat = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_cfg(v, prefix=f"{key}/"))
        else:
            flat[key] = v
    return flat
