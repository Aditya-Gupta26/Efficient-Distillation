"""
metrics_guide.py — Why we chose each evaluation metric for monocular depth estimation.

Run this script to print the full guide to stdout:
    python customization/metrics_guide.py

This is a reference document — it contains no training or inference code.
Every metric here is computed in customization/shared.py :: compute_depth_metrics().

Overview of the 7 metrics
─────────────────────────
  Lower is better  : RMSE, AbsRel, log-RMSE, SILog
  Higher is better : δ1, δ2, δ3 (threshold accuracy)

These 7 metrics are the standard suite used in every major monocular depth paper
since Eigen et al. (2014).  Using the same set makes our results directly
comparable to the literature without any conversion.

References
──────────
  [1] Eigen, D., Puhrsch, C., Fergus, R. (2014).
      "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network."
      NeurIPS 2014.  https://arxiv.org/abs/1406.2283
      — Introduced RMSE, AbsRel, SILog, and the δ threshold metrics.

  [2] Laina, I. et al. (2016).
      "Deeper Depth Prediction with Fully Convolutional Residual Networks."
      3DV 2016.  https://arxiv.org/abs/1606.00373
      — Added log-RMSE to the standard suite.

  [3] Ranftl, R. et al. (2021).
      "Vision Transformers for Dense Prediction."
      ICCV 2021.  https://arxiv.org/abs/2103.13413
      — DPT paper; uses the same 7 metrics on NYU Depth V2.

  [4] Bhat, S. et al. (2021).
      "AdaBins: Depth Estimation Using Adaptive Bins."
      CVPR 2021.  https://arxiv.org/abs/2011.14141
      — State-of-the-art on NYU; the SILog λ=0.85 convention comes from here.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Metric explanations — printed when this file is run directly
# ─────────────────────────────────────────────────────────────────────────────

METRICS = [
    {
        "name":    "RMSE  (Root Mean Squared Error)",
        "formula": "√[ mean( (pred - target)² ) ]",
        "unit":    "metres",
        "direction": "↓ lower is better",
        "why": """\
  RMSE is the most intuitive depth metric: it measures the average per-pixel
  error in metres.  Squaring the error before averaging means large mistakes
  (e.g. predicting 5 m instead of 1 m) are penalised much more than small ones,
  making RMSE sensitive to outlier predictions.

  Why we use it: It is the primary metric reported in virtually every depth
  estimation paper since Eigen et al. (2014), so it gives an apples-to-apples
  comparison with published results.  It is also the most interpretable number
  for end-users: "on average, our model is off by X metres."

  Typical range on NYU Depth V2:
    Excellent  : < 0.35 m   (state-of-the-art e.g. AdaBins, DepthFormer)
    Good       : 0.35–0.55 m
    Acceptable : 0.55–0.90 m
    Ours (30e) : ~1.00–1.18 m  (expected to improve significantly at 300 epochs)""",
    },
    {
        "name":    "AbsRel  (Mean Absolute Relative Error)",
        "formula": "mean( |pred - target| / target )",
        "unit":    "dimensionless ratio",
        "direction": "↓ lower is better",
        "why": """\
  AbsRel normalises the absolute error by the ground-truth depth, making it
  scale-invariant.  A pixel at 0.5 m and a pixel at 5 m contribute equally
  if the relative error is the same.  This removes the bias towards far-away
  regions that dominate RMSE.

  Why we use it: Depth in real scenes spans orders of magnitude (0.5 m to 10 m
  for NYU).  AbsRel is the fairest single-number summary of how well the model
  does across the whole depth range, not just for objects that happen to be far
  from the camera.

  Typical range on NYU Depth V2:
    Excellent  : < 0.080
    Good       : 0.080–0.120
    Acceptable : 0.12–0.18
    Ours (30e) : ~0.35–0.45  (high because we only trained 30 epochs)""",
    },
    {
        "name":    "log-RMSE  (Root Mean Squared Error in log space)",
        "formula": "√[ mean( (log(pred) - log(target))² ) ]",
        "unit":    "dimensionless (log scale)",
        "direction": "↓ lower is better",
        "why": """\
  By computing RMSE in logarithmic space, log-RMSE compresses the dynamic range
  so that relative errors matter more than absolute ones.  An error of ×2 at any
  depth contributes the same as any other ×2 error.

  Why we use it: It bridges RMSE (absolute, metre-scale) and AbsRel (relative,
  dimensionless).  It is particularly useful for detecting systematic biases
  (e.g. a model that over-predicts all depths by 20% will show low RMSE but
  still have elevated log-RMSE).  Also commonly used because it gives a smooth
  gradient during training (we use SILog as the loss, not log-RMSE).

  Relationship to SILog: log-RMSE = √ Var(d) + Mean(d)²  where d = log(p)-log(t).
  SILog downweights the Mean(d)² term with λ=0.85 to be more tolerant of
  global scale offsets.""",
    },
    {
        "name":    "SILog  (Scale-Invariant Logarithmic Error)",
        "formula": "√[ Var(d) + 0.15 · Mean(d)² ]  where d = log(pred) - log(target)",
        "unit":    "dimensionless",
        "direction": "↓ lower is better",
        "why": """\
  SILog was introduced by Eigen et al. (2014) and refined by AdaBins (λ=0.85).
  It is both our training loss AND one of our evaluation metrics, because depth
  estimation is inherently ambiguous about global scale — the network may learn
  a scene's structure perfectly but with a consistent scale offset.

  The variance term Var(d) captures pixel-to-pixel inconsistency (structural
  errors), while λ·Mean(d)² is a mild penalty for global scale offset.  Setting
  λ < 1.0 means perfect scale doesn't dominate — what matters is getting
  relative depths right.

  Why we use it as both loss and metric:
    As a loss: Differentiable, numerically stable (we add 1e-8 under the sqrt),
              directly optimises the scale-invariant structure of depth.
    As a metric: Lets us check that the training loss actually correlates with
              lower SILog on the held-out val set (which it should, by design).

  Typical range on NYU Depth V2:
    Excellent  : < 0.10
    Good       : 0.10–0.15
    Acceptable : 0.15–0.25""",
    },
    {
        "name":    "δ1  (Threshold Accuracy at 1.25)",
        "formula": "% pixels where max(pred/target, target/pred) < 1.25",
        "unit":    "fraction [0, 1]  (higher = better)",
        "direction": "↑ higher is better",
        "why": """\
  δ1 counts the fraction of pixels where the predicted depth is within 25% of
  the ground truth (either direction).  It is a "pass/fail" metric per pixel
  — a pixel either satisfies the threshold or it doesn't.

  Why we use it: Error metrics like RMSE and AbsRel are averages over all
  pixels, so they can be dominated by a small number of very bad predictions.
  δ1 tells you "what fraction of the scene is the model getting roughly right?"
  This is more meaningful for practical applications (robotics, AR) where you
  need a reliable reading for most of the image, not just a low average.

  The 1.25 threshold is industry standard (from Eigen 2014): not too strict
  (sub-5% error), not too lenient.  δ2 (1.25²=1.5625) and δ3 (1.25³≈1.95)
  extend this to successively looser thresholds.

  Typical range on NYU Depth V2:
    Excellent  : > 0.90
    Good       : 0.80–0.90
    Acceptable : 0.65–0.80
    Ours (30e) : ~0.35–0.50  (expected to improve significantly at 300 epochs)""",
    },
    {
        "name":    "δ2  (Threshold Accuracy at 1.25²)",
        "formula": "% pixels where max(pred/target, target/pred) < 1.5625",
        "unit":    "fraction [0, 1]  (higher = better)",
        "direction": "↑ higher is better",
        "why": """\
  δ2 is a more lenient version of δ1 that allows up to ~56% relative error.
  A model that is "in the right ballpark" will score well on δ2 even if it
  misses the tight δ1 threshold.

  Why we use it alongside δ1: The trio (δ1, δ2, δ3) together paint a picture
  of the model's error distribution.  If δ1 is low but δ2 and δ3 are high, the
  model is making many moderately wrong predictions.  If all three are low, the
  model has catastrophic failures on many pixels (e.g. background / sky).

  For a well-trained model we expect δ1 < δ2 < δ3 with diminishing gaps.""",
    },
    {
        "name":    "δ3  (Threshold Accuracy at 1.25³)",
        "formula": "% pixels where max(pred/target, target/pred) < 1.953",
        "unit":    "fraction [0, 1]  (higher = better)",
        "direction": "↑ higher is better",
        "why": """\
  δ3 is the loosest standard accuracy threshold: it tolerates up to ~95%
  relative error.  Essentially, it checks whether the model is at least
  getting depths in the right order of magnitude.

  Why we use it: It is almost always reported alongside δ1 and δ2 in the
  depth estimation literature for direct comparability.  It also acts as a
  sanity check — if δ3 is below 0.5, the model is producing fundamentally
  wrong depth maps (e.g. all outputs near zero or infinity), which helps
  diagnose training problems early.""",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Print guide
# ─────────────────────────────────────────────────────────────────────────────

def print_guide() -> None:
    """Print a formatted explanation of all 7 depth metrics."""
    print(__doc__)
    print("=" * 72)
    print("  DEPTH ESTIMATION METRICS — DETAILED GUIDE")
    print("=" * 72)

    for i, m in enumerate(METRICS, 1):
        print(f"\n{'─'*72}")
        print(f"  {i}. {m['name']}")
        print(f"{'─'*72}")
        print(f"  Formula   : {m['formula']}")
        print(f"  Unit      : {m['unit']}")
        print(f"  Direction : {m['direction']}")
        print()
        for line in m["why"].splitlines():
            print(f"  {line}")

    print(f"\n{'='*72}")
    print("  QUICK-REFERENCE TABLE")
    print(f"{'='*72}")
    print(f"  {'Metric':<12}  {'Formula':<45}  Direction")
    print(f"  {'-'*12}  {'-'*45}  {'-'*20}")
    for m in METRICS:
        short_name = m["name"].split("(")[0].strip()
        formula    = m["formula"][:44]
        direction  = m["direction"]
        print(f"  {short_name:<12}  {formula:<45}  {direction}")

    print(f"\n{'='*72}")
    print("  WHAT 'GOOD' LOOKS LIKE ON NYU DEPTH V2 (state-of-the-art ~2024)")
    print(f"{'='*72}")
    print("  RMSE    < 0.35 m       |  AbsRel  < 0.080")
    print("  log-RMSE < 0.12        |  SILog   < 0.100")
    print("  δ1      > 0.900        |  δ2      > 0.980   |  δ3 > 0.995")
    print()
    print("  Our 30-epoch results are a starting point — 300 epochs should")
    print("  bring RMSE into the 0.50–0.70 m range with δ1 > 0.75.")
    print()


if __name__ == "__main__":
    print_guide()
