# src/evaluate.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu, pointbiserialr
import pandas as pd

def evaluate_metric(scores: np.ndarray, labels: np.ndarray, metric_name: str = "IG") -> dict:
    """
    scores: per-token hallucination scores (for IG: negate it, since low IG = hallucination)
    labels: binary per-token labels (1 = hallucinated)
    """
    assert len(scores) == len(labels), "Length mismatch between scores and labels"
    assert labels.sum() > 0, "No positive labels found — check your data"

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # Point-biserial correlation
    corr, pval_corr = pointbiserialr(labels, scores)

    # Mann-Whitney U: are hallucinated token scores higher than faithful ones?
    hal_scores = scores[labels == 1]
    faith_scores = scores[labels == 0]
    u_stat, pval_mw = mannwhitneyu(hal_scores, faith_scores, alternative='greater')

    results = {
        "Metric": metric_name,
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "Pearson r": round(corr, 4),
        "Corr p-val": round(pval_corr, 5),
        "MannWhitney p-val": round(pval_mw, 5),
        "N hallucinated": int(labels.sum()),
        "N faithful": int((labels == 0).sum()),
    }
    return results

def bootstrap_ci(scores: np.ndarray, labels: np.ndarray,
                 n_bootstrap: int = 1000, metric: str = "auroc") -> tuple:
    """
    Returns (mean, lower_95, upper_95) via bootstrapping.
    """
    n = len(labels)
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            if metric == "auroc":
                s = roc_auc_score(labels[idx], scores[idx])
            elif metric == "auprc":
                s = average_precision_score(labels[idx], scores[idx])
            boot_scores.append(s)
        except Exception:
            pass  # Skip degenerate bootstrap samples
    
    boot_scores = np.array(boot_scores)
    return (float(np.mean(boot_scores)),
            float(np.percentile(boot_scores, 2.5)),
            float(np.percentile(boot_scores, 97.5)))

def compile_results_table(all_results: list) -> pd.DataFrame:
    """Turn list of result dicts into a formatted DataFrame for the report."""
    df = pd.DataFrame(all_results)
    df = df.set_index("Metric")
    return df