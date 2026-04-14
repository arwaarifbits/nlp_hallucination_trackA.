# =============================================================================
# src/evaluate.py
# CS F429 — Track A: Dynamic Uncertainty-Aware Attribution
#
# PURPOSE: Evaluation utilities — AUROC, bootstrap CI, and E5 type breakdown.
#
# NOTE ON USAGE:
#   evaluate_metric() is a standalone evaluator for a single metric array.
#   bootstrap_ci() is called in run_all_experiments() for every table row.
#   auroc_by_haltype() is called in main() for E5.
# =============================================================================

import warnings
from sklearn.exceptions import UndefinedMetricWarning
# Suppress sklearn's warning when a bootstrap resample contains only one class.
# This happens legitimately with small datasets — not a code error.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import numpy as np
import ast
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu, pointbiserialr


def evaluate_metric(scores: np.ndarray, labels: np.ndarray,
                    metric_name: str = "IG") -> dict:
    """
    Full evaluation of a single metric array against ground-truth labels.

    Computes AUROC, AUPRC, Spearman correlation, and Mann-Whitney U test.

    AUROC (Area Under ROC Curve):
      Probability that a randomly chosen hallucinated token scores higher than
      a randomly chosen faithful token. 0.5 = random, 1.0 = perfect.
      This is the PRIMARY grading criterion in the rubric.

    AUPRC (Average Precision):
      Area under the precision-recall curve. More sensitive to class imbalance
      than AUROC. Reported as supplementary metric.

    Point-biserial correlation:
      Pearson correlation between continuous scores and binary labels.
      Equivalent to Spearman rho for binary labels.

    Mann-Whitney U:
      Non-parametric test of whether hallucinated token scores are
      stochastically larger than faithful token scores.
      p < 0.05 = statistically significant discrimination.
    """
    assert len(scores) == len(labels), "Length mismatch between scores and labels"
    assert labels.sum() > 0, "No positive labels found — check your data"

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    corr, pval_corr = pointbiserialr(labels, scores)

    hal_scores   = scores[labels == 1]
    faith_scores = scores[labels == 0]
    u_stat, pval_mw = mannwhitneyu(hal_scores, faith_scores, alternative='greater')

    return {
        "Metric":             metric_name,
        "AUROC":              round(auroc, 4),
        "AUPRC":              round(auprc, 4),
        "Pearson r":          round(corr, 4),
        "Corr p-val":         round(pval_corr, 5),
        "MannWhitney p-val":  round(pval_mw, 5),
        "N hallucinated":     int(labels.sum()),
        "N faithful":         int((labels == 0).sum()),
    }


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray,
                 n_bootstrap: int = 1000, metric: str = "auroc") -> tuple:
    """
    Computes 95% bootstrap confidence interval for AUROC or AUPRC.

    HOW BOOTSTRAP CI WORKS:
      1. Resample the data WITH REPLACEMENT n_bootstrap=1000 times.
      2. Compute the metric (AUROC or AUPRC) on each resample.
      3. Report [2.5th percentile, 97.5th percentile] of the 1000 values.

    WHY BOOTSTRAP CI:
      The rubric explicitly requires bootstrap CI alongside AUROC.
      CI shows the stability of the result — a narrow CI means the AUROC
      is reliable, a wide CI means more data would be needed.

    DEGENERATE SAMPLES:
      Some bootstrap resamples may contain only one class (all hallucinated
      or all faithful). AUROC is undefined in this case. We skip these
      resamples (try/except) rather than crashing or artificially adding zeros.

    Returns:
      (mean, lower_95, upper_95) — all floats
    """
    n           = len(labels)
    boot_scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)  # resample with replacement
        try:
            if metric == "auroc":
                s = roc_auc_score(labels[idx], scores[idx])
            elif metric == "auprc":
                s = average_precision_score(labels[idx], scores[idx])
            boot_scores.append(s)
        except Exception:
            pass  # skip degenerate bootstrap samples silently

    boot_scores = np.array(boot_scores)
    return (
        float(np.mean(boot_scores)),
        float(np.percentile(boot_scores, 2.5)),   # lower bound
        float(np.percentile(boot_scores, 97.5))   # upper bound
    )


def compile_results_table(all_results: list) -> pd.DataFrame:
    """
    Converts a list of result dicts into a formatted DataFrame for the report.
    Each dict in all_results is one row (one metric or composite configuration).
    """
    df = pd.DataFrame(all_results)
    df = df.set_index("Metric")
    return df


def auroc_by_haltype(ds_ragtruth, composite_scores_per_sample,
                     label_arrays_per_sample) -> dict:
    """
    E5: Computes composite AUROC broken down by hallucination type.

    RAGTruth labels each hallucinated span with a 'label_type':
      "Evident Conflict"     — response directly contradicts context
      "Evident Baseless Info" — response adds information not in context
      "Subtle Conflict"      — subtle contradiction with context
      "Subtle Baseless Info"  — subtle unsupported addition

    HOW IT WORKS:
      1. For each sample, parse the 'labels' string (JSON-like) to extract label_type.
      2. Group token-level composite scores and labels by hallucination type.
      3. Also collect all "No Hallucination" tokens as the clean baseline.
      4. For each type: combine its hallucinated tokens with ALL clean tokens,
         then compute AUROC. This gives a per-type detection difficulty score.

    WHY COMBINE WITH ALL CLEAN TOKENS:
      Each type needs both positive (hallucinated) and negative (clean) examples
      to compute AUROC. Using ALL clean tokens (not just from the same sample)
      gives a more stable estimate and follows standard evaluation practice.

    Args:
      ds_ragtruth:                 HuggingFace Dataset — the original RAGTruth data
                                   (needed to read 'labels' column with label_type)
      composite_scores_per_sample: List of np.arrays, one composite score array per sample
      label_arrays_per_sample:     List of np.arrays, one binary label array per sample

    Returns:
      dict: {label_type: {"auroc": float, "token_count": int}}
    """
    type_scores = {}

    print(f"  [DEBUG E5] Processing {len(composite_scores_per_sample)} samples...")

    for i, sample in enumerate(ds_ragtruth):
        if i >= len(composite_scores_per_sample):
            break

        # Parse the 'labels' column — stored as a string representation of a list of dicts.
        # Example: "[{'start': 10, 'end': 25, 'label_type': 'Evident Conflict', ...}]"
        raw_labels_attr = sample.get("labels", "[]")

        try:
            if isinstance(raw_labels_attr, str):
                label_list = ast.literal_eval(raw_labels_attr)  # string → Python list
            else:
                label_list = raw_labels_attr

            if isinstance(label_list, list) and len(label_list) > 0:
                # Use the type of the first span in this sample.
                # Most samples have only one hallucination type.
                hal_type = label_list[0].get("label_type", "Unknown")
            else:
                hal_type = "No Hallucination"  # clean sample, no spans
        except (ValueError, SyntaxError):
            hal_type = "Parsing Error"

        scores  = np.array(composite_scores_per_sample[i])
        labels  = np.array(label_arrays_per_sample[i])
        min_len = min(len(scores), len(labels))

        if hal_type not in type_scores:
            type_scores[hal_type] = {"scores": [], "labels": []}

        type_scores[hal_type]["scores"].extend(scores[:min_len].tolist())
        type_scores[hal_type]["labels"].extend(labels[:min_len].tolist())

    # Collect all clean (No Hallucination) tokens as the negative class for each type
    clean_data = type_scores.get("No Hallucination", {"scores": [], "labels": []})
    print(f"  [DEBUG E5] Clean tokens found: {len(clean_data['labels'])}")

    results = {}
    for t, data in type_scores.items():
        if t == "No Hallucination":
            continue  # skip — this is our negative class, not a result row

        # Combine: all tokens of this hallucination type + ALL clean tokens
        c_scores = np.array(data["scores"] + clean_data["scores"])
        c_labels = np.array(data["labels"] + clean_data["labels"])

        unique_labels = np.unique(c_labels)
        if len(unique_labels) > 1:
            auroc = roc_auc_score(c_labels, c_scores)
            results[t] = {"auroc": round(auroc, 4), "token_count": len(c_labels)}
        else:
            print(f"  [DEBUG E5] {t} failed: Unique labels in set: {unique_labels}")
            results[t] = {"auroc": "N/A", "token_count": len(c_labels)}

    return results