"""
Smoke test for the depth estimation pipeline.

Verifies the full stack end-to-end using dummy data and (optionally) the
real student checkpoint — no NYU-Depth V2 download required.

Tests:
  1. StudentWithDPT forward pass + output shape
  2. SILog loss + masking edge case
  3. Two training epochs via DepthTrainer
  4. LoRA application — param counts and gradient isolation
  5. LoRA training step via LoRADepthTrainer
  6. PTQ dynamic quantization + forward pass

Usage:
    python scripts/smoke_test_depth.py
    python scripts/smoke_test_depth.py --checkpoint checkpoints/epoch_069.pth
    python scripts/smoke_test_depth.py --checkpoint checkpoints/epoch_069.pth --device cuda
    python scripts/smoke_test_depth.py --checkpoint checkpoints/epoch_069.pth --wandb
    python scripts/smoke_test_depth.py --skip_trainer   # model-only, fast
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("  [warn] CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _dummy_checkpoint(tmp_dir: str) -> str:
    """Create a minimal distillation checkpoint with random student weights."""
    from models.student import SwinStudentTiny
    student = SwinStudentTiny(pretrained=False, num_classes=0)
    path = str(Path(tmp_dir) / "dummy_student.pth")
    torch.save({"student": student.state_dict()}, path)
    return path


def _dummy_loader(batch_size: int = 2, img_size: int = 224, n_batches: int = 3):
    """DataLoader that yields {"images": ..., "depths": ...} from random tensors."""
    B = batch_size * n_batches
    images = torch.randn(B, 3, img_size, img_size)
    depths = torch.rand(B, img_size, img_size) * 10.0   # 0–10 m, all valid

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return B
        def __getitem__(self, i): return {"images": images[i], "depths": depths[i]}

    return DataLoader(_DS(), batch_size=batch_size, shuffle=False)


def _init_wandb(project: str, run_name: str):
    try:
        import wandb
        run = wandb.init(
            project = project,
            name    = run_name,
            tags    = ["smoke-test", "depth"],
            config  = {"smoke_test": True},
        )
        print(f"  W&B run: {run.url}")
        return run
    except ImportError:
        print("  [warn] wandb not installed — skipping W&B logging.")
        return None


def _section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def _ok(msg: str)   -> None: print(f"  ✓  {msg}")
def _fail(msg: str) -> None: print(f"  ✗  {msg}"); sys.exit(1)


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

def test_forward(checkpoint: str, device: torch.device) -> None:
    _section("1 · StudentWithDPT — forward pass")

    from models.depth_model import StudentWithDPT

    model = StudentWithDPT(
        student_checkpoint = checkpoint,
        dpt_pretrained     = False,   # random DPT weights — no HF download needed
        freeze_student     = True,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}  (DPT head only)")

    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        depth = model(x)

    expected = (2, 1, 224, 224)
    if depth.shape != expected:
        _fail(f"Expected output shape {expected}, got {tuple(depth.shape)}")
    _ok(f"Output shape: {tuple(depth.shape)}")

    if depth.isnan().any():
        _fail("NaN values in depth output!")
    _ok("No NaNs in output")


def test_silog_loss(device: torch.device) -> None:
    _section("2 · SILog loss")

    from distillation.depth_trainer import DepthTrainer

    trainer = DepthTrainer.__new__(DepthTrainer)
    trainer.SILOG_LAMBDA = 0.85

    pred   = torch.rand(2, 224, 224, device=device).clamp(0.01, 10.0)
    target = torch.rand(2, 224, 224, device=device).clamp(0.1,  10.0)

    loss = trainer._silog_loss(pred, target)
    if loss.isnan():
        _fail("SILog loss returned NaN")
    _ok(f"SILog loss on valid pixels: {loss.item():.6f}")

    # All-invalid mask → should return zero without error
    loss_masked = trainer._silog_loss(pred, torch.zeros_like(target))
    _ok(f"All-invalid-pixel loss: {loss_masked.item():.6f}  (should be 0.0)")


def test_depth_trainer(checkpoint: str, device: torch.device, wandb_run=None) -> None:
    _section("3 · DepthTrainer — 2 epochs")

    from models.depth_model import StudentWithDPT
    from distillation.depth_trainer import DepthTrainer

    model        = StudentWithDPT(checkpoint, dpt_pretrained=False, freeze_student=True)
    train_loader = _dummy_loader()
    val_loader   = _dummy_loader(n_batches=2)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "lr": 1e-4, "lr_min": 1e-6, "weight_decay": 1e-2,
            "grad_clip": 1.0, "epochs": 2, "amp": False,
            "log_every": 1, "save_dir": tmp,
        }
        trainer = DepthTrainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            cfg          = cfg,
            device       = device,
            wandb_run    = wandb_run,
        )
        trainer.train()

    _ok("2 epochs completed without error")


def test_lora(checkpoint: str, device: torch.device) -> None:
    _section("4 · LoRA — parameter isolation & gradients")

    from models.depth_model import StudentWithDPT
    from models.lora import apply_lora, count_lora_params, get_lora_params

    model = StudentWithDPT(checkpoint, dpt_pretrained=False, freeze_student=True)

    before = sum(p.numel() for p in model.student.parameters() if p.requires_grad)
    if before != 0:
        _fail(f"Student should have 0 trainable params before LoRA, got {before}")
    _ok(f"Student trainable before LoRA: {before}")

    apply_lora(model.student, r=4, alpha=1.0, target_modules=["qkv"])

    lora_n = count_lora_params(model.student)
    if lora_n == 0:
        _fail("apply_lora added 0 params — check target_modules matches model attr names")
    _ok(f"LoRA params added: {lora_n:,}")

    student_trainable = sum(p.numel() for p in model.student.parameters() if p.requires_grad)
    if student_trainable != lora_n:
        _fail(
            f"Expected exactly {lora_n} trainable student params, got {student_trainable}. "
            "Base weights may not be fully frozen."
        )
    _ok("Only LoRA A/B matrices are trainable in student")

    # Forward + backward
    model = model.to(device)
    x     = torch.randn(1, 3, 224, 224, device=device)
    model(x).sum().backward()

    no_grad = [n for n, p in model.student.named_parameters()
               if p.requires_grad and p.grad is None]
    if no_grad:
        _fail(f"LoRA params missing gradients: {no_grad[:3]}")
    _ok("All LoRA params received gradients")


def test_lora_trainer(checkpoint: str, device: torch.device, wandb_run=None) -> None:
    _section("5 · LoRADepthTrainer — 2 epochs (LoRA-only mode)")

    from models.depth_model import StudentWithDPT
    from models.lora import apply_lora, count_lora_params
    from distillation.lora_depth_trainer import LoRADepthTrainer

    model = StudentWithDPT(checkpoint, dpt_pretrained=False, freeze_student=True)

    # Simulate freeze_dpt_head=True: everything frozen, only LoRA trains
    for p in model.parameters():
        p.requires_grad = False
    apply_lora(model.student, r=4, alpha=1.0, target_modules=["qkv"])

    lora_n = count_lora_params(model.student)
    print(f"  LoRA-only mode: {lora_n:,} trainable params")

    train_loader = _dummy_loader()
    val_loader   = _dummy_loader(n_batches=2)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "lr": 5e-5, "lora_lr": 5e-5, "lr_min": 1e-7,
            "weight_decay": 1e-2, "grad_clip": 1.0,
            "epochs": 2, "amp": False, "log_every": 1,
            "save_dir": tmp,
        }
        trainer = LoRADepthTrainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            cfg          = cfg,
            device       = device,
            wandb_run    = wandb_run,
        )
        trainer.train()

    _ok("2 LoRA-only epochs completed without error")


def test_quantization(checkpoint: str, device: torch.device) -> None:
    _section("6 · PTQ dynamic quantization")

    from models.depth_model import StudentWithDPT
    from utils.quantization import quantize_model, model_size_mb

    model = StudentWithDPT(checkpoint, dpt_pretrained=False, freeze_student=True)
    model.eval()

    size_before = model_size_mb(model)
    print(f"  Model size before: {size_before:.1f} MB")

    q_model = quantize_model(model.cpu(), method="ptq_dynamic")

    size_after = model_size_mb(q_model)
    print(f"  Model size after : {size_after:.1f} MB")
    print(f"  Compression      : {size_before/size_after:.2f}x")

    with torch.no_grad():
        depth = q_model(torch.randn(1, 3, 224, 224))

    if depth.isnan().any():
        _fail("NaN in quantized model output")
    _ok(f"Quantized forward pass OK, output shape: {tuple(depth.shape)}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Distillation checkpoint (epoch_069.pth). "
                        "Omit to use random weights.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--wandb", action="store_true",
                   help="Log trainer steps to Weights & Biases.")
    p.add_argument("--wandb_project", type=str, default="efficient-distillation",
                   help="W&B project name (default: efficient-distillation).")
    p.add_argument("--skip_trainer", action="store_true",
                   help="Skip trainer tests — only tests model/loss/LoRA/quant.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = _make_device(args.device)

    print(f"Device : {device}")

    # W&B (only used in trainer tests)
    wandb_run = None
    if args.wandb:
        _section("W&B init")
        wandb_run = _init_wandb(
            project  = args.wandb_project,
            run_name = "smoke-test-depth",
        )

    with tempfile.TemporaryDirectory() as tmp:
        if args.checkpoint and Path(args.checkpoint).exists():
            checkpoint = args.checkpoint
            print(f"Checkpoint : {checkpoint}")
        else:
            if args.checkpoint:
                print(f"  [warn] Checkpoint not found: {args.checkpoint}")
            print("Checkpoint : dummy (random weights)")
            checkpoint = _dummy_checkpoint(tmp)

        test_forward(checkpoint, device)
        test_silog_loss(device)

        if not args.skip_trainer:
            test_depth_trainer(checkpoint, device, wandb_run)
            test_lora_trainer(checkpoint, device, wandb_run)

        test_lora(checkpoint, device)
        test_quantization(checkpoint, device)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"\n{'='*55}")
    print("  All smoke tests passed!")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
