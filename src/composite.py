# =============================================================================
# src/composite.py
# CS F429 — Track A: Dynamic Uncertainty-Aware Attribution
#
# PURPOSE: Combines multiple token-level metrics into a single composite
# hallucination score using variance-proportional weighting.
#
# KEY DESIGN DECISION — UNSUPERVISED WEIGHTING:
#   The rubric forbids supervised (logistic regression) weighting on test data.
#   We use variance-proportional weights instead: metrics with higher variance
#   across tokens have more discriminative power and receive higher weight.
#   This is computed purely from the metric score distributions — no labels used.
# =============================================================================

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def normalize_score(s: np.ndarray) -> np.ndarray:
    """
    Min-max normalises a score array to [0, 1].

    WHY WE NORMALISE:
      Different metrics have very different scales:
        - KL divergence: typically 0.0 – 2.0
        - Entropy: typically 0.5 – 6.0
        - Confidence Drop: typically -0.3 – 0.3
      Normalising puts all metrics on the same scale before combining.

    EDGE CASES:
      - NaN values replaced with 0.0 (safe neutral value)
      - Zero-variance arrays (all same value) return all zeros
        — a constant metric provides no discrimination signal
    """
    s = np.nan_to_num(np.array(s), nan=0.0)
    if s.size == 0:
        return np.array([])
    s_min, s_max = np.min(s), np.max(s)
    if abs(s_max - s_min) < 1e-9:
        # All values are identical — no signal in this metric
        return np.zeros_like(s)
    return (s - s_min) / (s_max - s_min)


def build_composite(all_metrics: dict, labels: np.ndarray,
                    mode: str = "variance_weight") -> np.ndarray:
    """
    Builds the composite hallucination score from multiple metrics.

    Args:
      all_metrics: dict of {metric_name: np.array of per-token scores}
                   All arrays must have the same length (one score per token).
      labels:      Binary ground-truth labels. Only used in logistic mode
                   (which is disabled for Track A — unsupervised only).
      mode:        "variance_weight" (default, unsupervised) or
                   "equal_weight" (used in the incremental E1+E2 table).

    HOW VARIANCE WEIGHTING WORKS:
      1. Each metric is cleaned (NaN→0, Inf→1) and min-max normalised.
      2. Zero-variance metrics are dropped (flat signal = useless).
      3. Each remaining metric's weight = its variance / total variance.
         Higher variance means the metric spreads tokens more widely
         across its score range → more discriminative → higher weight.
      4. Final composite = weighted sum of normalised metrics.

    WHY NOT SUPERVISED WEIGHTING:
      The rubric says "no re-training or re-calibration" on the test set.
      Logistic regression would learn weights from labels, violating this.
      Variance-based weighting uses only the metric distributions themselves.

    Returns:
      np.array of shape [n_tokens] with composite scores in [0, 1].
      Higher score = higher hallucination risk.
    """
    if not all_metrics:
        return np.zeros(len(labels))

    names = list(all_metrics.keys())

    # ── Step 1: Clean and normalise each metric ───────────────────────────────
    cleaned_cols = []
    valid_names  = []
    result = None

    for n in names:
        # Replace NaN and Inf before any computation
        col = np.nan_to_num(all_metrics[n], nan=0.0, posinf=1.0, neginf=0.0)

        lo, hi = np.min(col), np.max(col)
        if (hi - lo) < 1e-10:
            # Constant metric — skip entirely. It adds zero variance to the
            # composite and would just dilute the signal from useful metrics.
            # SemEnt often hits this case because it is a per-sample constant.
            continue

        normed = (col - lo) / (hi - lo + 1e-8)  # min-max to [0, 1]
        cleaned_cols.append(normed)
        valid_names.append(n)

    if not cleaned_cols:
        # All metrics were constant — return zeros (no hallucination signal)
        return np.zeros(len(labels))

    # ── Step 2: Stack into matrix [n_tokens, n_metrics] ──────────────────────
    X_norm = np.column_stack(cleaned_cols)

    # ── Step 3: Compute composite based on mode ───────────────────────────────
    if mode == "equal_weight":
        # Simple average — used in the incremental E1+E2 table to show how
        # each metric contributes when added one by one.
        result = np.mean(X_norm, axis=1)

    elif mode == "variance_weight":
        variances = np.var(X_norm, axis=0)   # variance of each metric column
        total_var = np.sum(variances)

        if total_var < 1e-10:
            # All normalised metrics are nearly constant — fall back to equal weight
            result = np.mean(X_norm, axis=1)
        else:
            # Weight proportional to variance: high variance = more discriminative
            weights     = variances / (total_var + 1e-12)
            weight_dict = {n: round(float(w), 3) for n, w in zip(valid_names, weights)}
            print(f"  [Composite] Active metrics & weights: {weight_dict}")
            result      = np.dot(X_norm, weights)  # weighted sum
    
    elif mode == "entropy_weight":
        # Entropy-based weighting (unsupervised)
        entropies = -np.sum(X_norm * np.log(X_norm + 1e-8), axis=0)
        weights = 1 / (entropies + 1e-8)
        weights = weights / weights.sum()

        weight_dict = {n: round(float(w), 3) for n, w in zip(valid_names, weights)}
        print(f"  [Composite-Entropy] weights: {weight_dict}")

        result = np.dot(X_norm, weights)


    elif mode == "logistic":
        # DISABLED FOR TRACK A — supervised mode not permitted on test set.
        # This would fit a logistic regression using ground-truth labels,
        # which constitutes training on test data and violates the rubric.
        raise ValueError("Logistic mode is supervised and not allowed for Track A")

    # ── Step 4: Final safety check ────────────────────────────────────────────
    # Replace any residual NaN or Inf before the result reaches AUROC computation
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)


def incremental_auroc_table(all_metrics: dict, labels: np.ndarray) -> list:
    """
    Builds the E1+E2 rubric table by adding metrics one at a time.

    The rubric requires showing AUROC as each metric is added incrementally:
      Row 1: Entropy-only alone
      Row 2: Entropy-only + Info Gain
      Row 3: Entropy-only + Info Gain + KL
      ...and so on.

    This demonstrates that each new metric provides additional signal
    beyond what the previous metrics already captured.
    Uses equal_weight mode so the table reflects pure metric contribution,
    not variance-based weighting artefacts.
    """
    rows         = []
    metric_names = list(all_metrics.keys())

    for i in range(1, len(metric_names) + 1):
        subset    = {k: all_metrics[k] for k in metric_names[:i]}
        composite = build_composite(subset, labels, mode="equal_weight")
        auroc     = roc_auc_score(labels, composite)
        rows.append({
            "metrics_included": metric_names[:i],
            "label":            " + ".join(metric_names[:i]),
            "AUROC":            round(auroc, 4)
        })

    return rows