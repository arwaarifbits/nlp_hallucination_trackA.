# src/evaluate.py
import numpy as np
import ast
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import mannwhitneyu, pointbiserialr


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


def auroc_by_haltype(ds_ragtruth, composite_scores_per_sample, label_arrays_per_sample):
    """
    Parses RAGTruth CSV 'labels' string to extract 'label_type' and compute AUROC per category.
    """
    type_scores = {} # {type_name: {"scores": [], "labels": []}}

    # DEBUG: Check how many samples we are actually processing
    print(f"  [DEBUG E5] Processing {len(composite_scores_per_sample)} samples...")

    for i, sample in enumerate(ds_ragtruth):
        if i >= len(composite_scores_per_sample):
            break

        # 1. Extract the hallucination type from the 'labels' column
        raw_labels_attr = sample.get("labels", "[]")
        
        try:
            # Handle string vs list formats
            if isinstance(raw_labels_attr, str):
                label_list = ast.literal_eval(raw_labels_attr)
            else:
                label_list = raw_labels_attr
            
            # Identify the hallucination type
            if isinstance(label_list, list) and len(label_list) > 0:
                # We take the primary label_type found in this sample
                hal_type = label_list[0].get("label_type", "Unknown")
            else:
                hal_type = "No Hallucination"
        except (ValueError, SyntaxError):
            hal_type = "Parsing Error"

        # 2. Alignment Safety & Data Aggregation
        scores = np.array(composite_scores_per_sample[i])
        labels = np.array(label_arrays_per_sample[i])
        min_len = min(len(scores), len(labels))
        
        if hal_type not in type_scores:
            type_scores[hal_type] = {"scores": [], "labels": []}

        type_scores[hal_type]["scores"].extend(scores[:min_len].tolist())
        type_scores[hal_type]["labels"].extend(labels[:min_len].tolist())

    
    # Before calculating results, add 'No Hallucination' tokens to every other category 
    # so there is a baseline of 0s to compare against the 1s.
    # Check if we actually found clean data
    clean_data = type_scores.get("No Hallucination", {"scores": [], "labels": []})
    print(f"  [DEBUG E5] Clean tokens found: {len(clean_data['labels'])}")
    
    
    # 3. Final AUROC calculation per type
    results = {}
    for t, data in type_scores.items():
        if t == "No Hallucination": continue
        
        # Combine category-specific hallucinations with all clean tokens
        c_scores = np.array(data["scores"] + clean_data["scores"])
        c_labels = np.array(data["labels"] + clean_data["labels"])
        
        # DEBUG: Check if we have both 0s and 1s
        unique_labels = np.unique(c_labels)
        if len(unique_labels) > 1:
            auroc = roc_auc_score(c_labels, c_scores)
            results[t] = {"auroc": round(auroc, 4), "token_count": len(c_labels)}
        else:
            print(f"  [DEBUG E5] {t} failed: Unique labels in set: {unique_labels}")
            results[t] = {"auroc": "N/A", "token_count": len(c_labels)}
        
    return results