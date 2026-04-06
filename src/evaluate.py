import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pointbiserialr, mannwhitneyu

def evaluate_metric(ig_scores, hallucination_labels):
    """
    Detailed statistical breakdown of the IG Metric.
    ig_scores: Array of raw IG values.
    hallucination_labels: Binary array (1 = hallucinated token).
    """
    # 1. Prepare scores for classification (Flip IG: lower IG = higher hallucination risk)
    hal_risk_score = -np.array(ig_scores) 
    y_true = np.array(hallucination_labels)

    # 2. Threshold-Independent Metrics
    auroc = roc_auc_score(y_true, hal_risk_score)
    auprc = average_precision_score(y_true, hal_risk_score)

    # 3. Correlation Analysis
    # Does a decrease in IG strongly correlate with the presence of a 1?
    corr, pval = pointbiserialr(y_true, hal_risk_score)

    # 4. Significance Testing (Non-parametric)
    # Testing the hypothesis: IG_hallucination < IG_faithful
    hal_ig = ig_scores[y_true == 1]
    faith_ig = ig_scores[y_true == 0]
    
    # Check if we actually have both classes before testing
    if len(hal_ig) > 0 and len(faith_ig) > 0:
        _, u_pval = mannwhitneyu(hal_ig, faith_ig, alternative='less')
    else:
        u_pval = 1.0

    return {
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "Point-Biserial Corr": round(corr, 4),
        "P-Value (Significance)": f"{u_pval:.2e}"
    }

def get_bootstrap_ci(ig_scores, labels, n_iterations=1000):
    """Computes the 95% Confidence Interval for AUROC."""
    stats = []
    ig_scores = -np.array(ig_scores)
    labels = np.array(labels)
    
    for _ in range(n_iterations):
        # Sample with replacement
        indices = np.random.randint(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue
        score = roc_auc_score(labels[indices], ig_scores[indices])
        stats.append(score)
        
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)