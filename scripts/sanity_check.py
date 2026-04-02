"""
Quick sanity check: instantiate teacher, student, and adapters,
run a single forward pass, and print parameter counts + output shapes.

Usage:
    python scripts/sanity_check.py
"""

import torch
from models.teacher import SwinTeacher
from models.student import SwinStudentTiny
from models.adapters import FeatureAdapter
from distillation.losses import DistillationLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running sanity check on: {device}\n")

    # ---- Models -------------------------------------------------------
    print("Loading teacher (swin_large, pretrained=False for speed)...")
    teacher = SwinTeacher(variant="swin_large", pretrained=False, num_classes=80).to(device)
    teacher.eval()

    print("Loading student (swin_tiny, pretrained=False)...")
    student = SwinStudentTiny(pretrained=False, num_classes=80).to(device)

    print(f"Teacher params : {teacher.num_parameters:,}")
    print(f"  trainable    : {teacher.num_trainable_parameters:,}")
    print(f"Student params : {student.num_parameters:,}\n")

    # ---- Adapter ------------------------------------------------------
    adapter = FeatureAdapter(
        student_channels=student.stage_channels,
        teacher_channels=teacher.stage_channels,
    ).to(device)
    print(f"Adapter params : {adapter.num_parameters:,}\n")

    # ---- Forward pass -------------------------------------------------
    x = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad():
        t_feats, t_logits = teacher(x)
        s_feats, s_logits = student(x)
        adapted = adapter(s_feats, t_feats)

    print("Feature map shapes (teacher | student | adapted):")
    for i, (tf, sf, af) in enumerate(zip(t_feats, s_feats, adapted)):
        print(f"  Stage {i}: teacher={tuple(tf.shape)}  student={tuple(sf.shape)}  adapted={tuple(af.shape)}")

    if t_logits is not None and s_logits is not None:
        print(f"\nLogits : teacher={tuple(t_logits.shape)}  student={tuple(s_logits.shape)}")

    # ---- Loss ---------------------------------------------------------
    loss_fn = DistillationLoss()
    losses  = loss_fn(
        adapted_student_feats=adapted,
        teacher_feats=t_feats,
        student_feats_raw=s_feats,
        student_logits=s_logits,
        teacher_logits=t_logits,
    )
    print("\nLoss breakdown:")
    for k, v in losses.items():
        print(f"  {k:8s}: {v.item():.6f}")

    print("\n✅  Sanity check passed!")


if __name__ == "__main__":
    main()
