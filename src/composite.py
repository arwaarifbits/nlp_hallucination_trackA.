# src/composite.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def normalize_score(s):
    s = np.nan_to_num(np.array(s), nan=0.0)
    if s.size == 0:
        return np.array([])
    s_min, s_max = np.min(s), np.max(s)
    if abs(s_max - s_min) < 1e-9:
        return np.zeros_like(s)
    return (s - s_min) / (s_max - s_min)

def build_composite(all_metrics: dict, labels: np.ndarray, mode: str = "variance_weight"):
    if not all_metrics:
        return np.zeros(len(labels))

    names = list(all_metrics.keys())
    
    # 1. PRE-CLEAN: Ensure all raw inputs are finite and non-NaN
    cleaned_cols = []
    valid_names = []
    
    for n in names:
        # Replace NaNs with 0 and Infs with 1.0 (assuming normalized context)
        col = np.nan_to_num(all_metrics[n], nan=0.0, posinf=1.0, neginf=0.0)
        
        # Check for variance (Min-Max)
        lo, hi = np.min(col), np.max(col)
        if (hi - lo) < 1e-10:
            continue # Drop metrics that provide no signal (all same value)
            
        # Normalize to [0, 1]
        normed = (col - lo) / (hi - lo + 1e-8) if (hi - lo) > 1e-8 else np.zeros_like(col)
        cleaned_cols.append(normed)
        valid_names.append(n)

    # If no metrics survived the variance check, return a flat zero array
    if not cleaned_cols:
        return np.zeros(len(labels))

    # 2. CONSTRUCT MATRIX
    X_norm = np.column_stack(cleaned_cols)

    if mode == "equal_weight":
        result = np.mean(X_norm, axis=1)

    elif mode == "variance_weight":
        variances = np.var(X_norm, axis=0)
        total_var = np.sum(variances)
        
        if total_var < 1e-10:
            result = np.mean(X_norm, axis=1)
        else:
            if total_var < 1e-10:
                result = np.mean(X_norm, axis=1)
                weights = np.ones_like(variances) / len(variances)
            else:
                weights = variances / (total_var + 1e-12)
            weight_dict = {n: round(float(w), 3) for n, w in zip(valid_names, weights)}
            print(f"  [Composite] Active metrics & weights: {weight_dict}")
            
            # Use np.dot for safer 1D/2D multiplication and wrap in nan_to_num
            result = np.dot(X_norm, weights)
            
    elif mode == "logistic":
        raise ValueError("Logistic mode is supervised and not allowed for Track A")

    # FINAL SAFETY: Ensure no NaNs or Infs leak into the final AUROC calculation
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)

def incremental_auroc_table(all_metrics: dict, labels: np.ndarray) -> list:
    """
    Build the E1+E2 table: add metrics one by one and report AUROC.
    This matches exactly the rubric's table format.
    """
    rows = []
    metric_names = list(all_metrics.keys())
    
    for i in range(1, len(metric_names) + 1):
        subset = {k: all_metrics[k] for k in metric_names[:i]}
        composite = build_composite(subset, labels, mode="equal_weight")
        auroc = roc_auc_score(labels, composite)
        rows.append({
            "metrics_included": metric_names[:i],
            "label": " + ".join(metric_names[:i]),
            "AUROC": round(auroc, 4)
        })
    
    return rows