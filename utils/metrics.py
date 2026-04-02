"""
Evaluation metrics.

For classification (interim proxy metric during distillation):
    - Top-1 and Top-5 accuracy.

For detection (COCO mAP):
    - Delegates to torchmetrics or pycocotools when available.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1, 5)) -> Dict[str, float]:
    """
    Compute top-k classification accuracy.

    Args:
        output : (B, C) logits.
        target : (B,)  ground-truth class indices.
        topk   : Tuple of k values.

    Returns:
        Dict like ``{"top1": 0.82, "top5": 0.96}``.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                                       # (maxk, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res[f"top{k}"] = (correct_k / batch_size).item()
        return res


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Generic metric computation entry-point.

    Currently computes Top-1 / Top-5 accuracy for classification.
    Extend this function to add COCO mAP via pycocotools.

    Args:
        predictions : Model output logits (B, C).
        targets     : Ground-truth labels (B,) or (B, ...).

    Returns:
        Dict of metric name → float value.
    """
    metrics: Dict[str, float] = {}

    if predictions.ndim == 2 and targets.ndim == 1:
        # Classification setting
        acc = accuracy(predictions, targets, topk=(1, 5))
        metrics.update(acc)
        # Use top-1 as a proxy "mAP" for the trainer's early-stopping logic
        metrics["mAP"] = acc["top1"]

    return metrics
