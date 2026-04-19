"""
LoRADepthTrainer: fine-tunes LoRA adapters on the frozen student backbone
and the DPT depth head jointly on NYU-Depth V2.

Parameter groups:
  - LoRA A/B matrices  (student attention layers)
  - DPT head parameters (neck + prediction head)

The base student weights remain completely frozen throughout.

This module is intentionally thin — it inherits all loss/metric/checkpoint
logic from DepthTrainer and only overrides the parameter selection step
and adds LoRA-specific logging.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from distillation.depth_trainer import DepthTrainer
from models.lora import get_lora_params, count_lora_params
from utils.logger import setup_logger


class LoRADepthTrainer(DepthTrainer):
    """
    Identical to DepthTrainer but optimises LoRA params + DPT head.

    Assumes that apply_lora() has already been called on model.student
    before this trainer is instantiated, and that model.dpt_head is
    trainable (requires_grad=True).

    The optimiser uses two parameter groups so different LRs can be set
    (lora_lr for LoRA matrices, lr for DPT head).  Currently both use lr;
    override lora_lr in the config to decouple.
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          dict,
        device:       torch.device,
        wandb_run=None,
    ):
        # Build the trainer — base __init__ creates optimiser over all
        # requires_grad parameters, which now include LoRA + DPT head.
        super().__init__(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            cfg          = cfg,
            device       = device,
            wandb_run    = wandb_run,
        )
        self.logger = setup_logger("lora_depth_trainer")

        lora_n = count_lora_params(self.model.student)
        dpt_n  = sum(p.numel() for p in self.model.dpt_head.parameters() if p.requires_grad)
        self.logger.info(f"LoRA params: {lora_n:,}  |  DPT head params: {dpt_n:,}")

        # Override save_dir from config
        self.save_dir = Path(cfg.get("save_dir", "checkpoints/lora_depth"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Replace the single-group optimiser from parent with a two-group one
        # so LoRA matrices and DPT head can have independent learning rates.
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        lora_params = get_lora_params(self.model.student)
        dpt_params  = [p for p in self.model.dpt_head.parameters() if p.requires_grad]
        lora_lr     = cfg.get("lora_lr", cfg.get("lr", 5e-5))

        # Build parameter groups; skip empty groups (e.g. DPT frozen)
        param_groups = [{"params": lora_params, "lr": lora_lr}]
        if dpt_params:
            param_groups.append({"params": dpt_params, "lr": cfg.get("lr", 5e-5)})

        self.optimiser = AdamW(
            param_groups,
            weight_decay = cfg.get("weight_decay", 1e-2),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimiser,
            T_max   = cfg.get("epochs", 30),
            eta_min = cfg.get("lr_min", 1e-7),
        )
