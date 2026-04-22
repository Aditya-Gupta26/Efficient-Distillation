"""
Replay training metrics from a SLURM log file into a WandB run.

Parses epoch summaries logged by DistillationTrainer and uploads them
to a new (or resumed) WandB run — useful when WandB crashed mid-training.

Usage:
    # Create a brand-new run with the replayed data:
    python scripts/replay_wandb.py --log logs/distill_5450439.out

    # Resume the original crashed run (get run ID from wandb.ai dashboard):
    python scripts/replay_wandb.py --log logs/distill_5450439.out --run_id wnenlnkj
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# The log lines wrap at the terminal width, so a single epoch summary may
# be split across two physical lines.  We join the whole file into one string
# and use a multiline regex so \s+ absorbs the newline + any ANSI padding.
_EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?"   # timestamp
    r"Epoch \[(\d+)/(\d+)\]\s+Loss:\s*([\d.]+)\s+"      # epoch, total loss
    r"\(feat=([\d.]+),\s*at=([\d.]+),\s*kd=([\d.]+)\)\s*\|"  # component losses
    r"\s*Student Top-1:\s*([\d.]+)\s+Top-5:\s*([\d.]+)" # student metrics
    r"\s*\(Teacher:\s*([\d.]+)\)",                       # teacher top-1
    re.DOTALL,
)


def parse_log(log_path: str) -> list[dict]:
    text = Path(log_path).read_text(errors="replace")
    # Collapse line-wrapping: join lines that don't start a new log entry
    # (i.e. don't start with '[20')
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[20") or not cleaned_lines:
            cleaned_lines.append(stripped)
        else:
            cleaned_lines[-1] += " " + stripped
    cleaned = "\n".join(cleaned_lines)

    records = []
    for m in _EPOCH_RE.finditer(cleaned):
        ts, epoch, total_epochs, loss, feat, at, kd, top1, top5, teacher_top1 = m.groups()
        records.append({
            "timestamp":    datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
            "epoch":        int(epoch),
            "total_epochs": int(total_epochs),
            "train_loss_total": float(loss),
            "train_loss_feat":  float(feat),
            "train_loss_at":    float(at),
            "train_loss_kd":    float(kd),
            "val_student_top1": float(top1),
            "val_student_top5": float(top5),
            "val_teacher_top1": float(teacher_top1),
        })
    return records


# ---------------------------------------------------------------------------
# WandB upload
# ---------------------------------------------------------------------------

def replay(records: list[dict], run_id: str | None, project: str, entity: str | None,
           run_name: str | None, batches_per_epoch: int) -> None:
    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Run: pip install wandb")
        sys.exit(1)

    init_kwargs = dict(
        project = project,
        entity  = entity,
        name    = run_name,
        tags    = ["replayed", "swin", "knowledge-distillation", "imagenet"],
    )
    if run_id:
        init_kwargs["id"]     = run_id
        init_kwargs["resume"] = "must"
        print(f"Resuming WandB run: {run_id}")
    else:
        print("Creating a new WandB run for replayed data...")

    run = wandb.init(**init_kwargs)
    print(f"Run URL: {run.url}")

    for r in records:
        step = r["epoch"] * batches_per_epoch
        payload = {
            "epoch":                    r["epoch"],
            "epoch/train_loss_total":   r["train_loss_total"],
            "epoch/train_loss_feat":    r["train_loss_feat"],
            "epoch/train_loss_at":      r["train_loss_at"],
            "epoch/train_loss_kd":      r["train_loss_kd"],
            "epoch/val_student_top1":   r["val_student_top1"],
            "epoch/val_student_top5":   r["val_student_top5"],
            "epoch/val_teacher_top1":   r["val_teacher_top1"],
        }
        run.log(payload, step=step)
        print(f"  Logged epoch {r['epoch']:3d}/{r['total_epochs']}  "
              f"loss={r['train_loss_total']:.4f}  "
              f"top1={r['val_student_top1']:.4f}  "
              f"(step={step})")

    run.finish()
    print(f"\nDone — {len(records)} epochs uploaded to {run.url}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Replay WandB metrics from a SLURM log.")
    p.add_argument("--log",     required=True,  help="Path to the .out log file")
    p.add_argument("--run_id",  default=None,   help="Crashed WandB run ID to resume (optional)")
    p.add_argument("--project", default="efficient-distillation")
    p.add_argument("--entity",  default="ag11023-new-york-university")
    p.add_argument("--run_name",default="replayed-distill-5450439")
    p.add_argument("--batches_per_epoch", type=int, default=5005,
                   help="Number of training batches per epoch (1,281,166 imgs / bs=256 ≈ 5005)")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    records = parse_log(args.log)

    if not records:
        print("No epoch records found in the log. Check the log path / format.")
        sys.exit(1)

    print(f"Parsed {len(records)} epoch records from {args.log}")
    for r in records:
        print(f"  Epoch {r['epoch']:3d} | loss={r['train_loss_total']:.4f} | "
              f"top1={r['val_student_top1']:.4f} | top5={r['val_student_top5']:.4f}")

    print()
    replay(records, args.run_id, args.project, args.entity, args.run_name,
           args.batches_per_epoch)
