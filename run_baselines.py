import pandas as pd
import numpy as np
import ast
import torch
import warnings
from src.metric import InformationGainMetric
from src.baselines import SelfCheckBaseline, compute_entropy_baseline
from sklearn.metrics import roc_auc_score
from src.evaluate import evaluate_metric, get_bootstrap_ci

# Silence warnings for a cleaner report output
warnings.filterwarnings("ignore")

# Helper function for word/token alignment
def create_hallucination_mask(response, errors, tokenizer, ig_len):
    mask = np.zeros(ig_len)
    encoding = tokenizer(response, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    resp_offsets = offsets[-ig_len:] 
    for i, (start, end) in enumerate(resp_offsets):
        for error in errors:
            if start < error['end'] and end > error['start']:
                mask[i] = 1
    return mask

# 1. Setup - Scale to 50 for statistical significance if your Mac allows
SAMPLE_COUNT = 20 
df = pd.read_csv("data/ragtruth/ragtruth_final.csv").head(SAMPLE_COUNT)
metric = InformationGainMetric("facebook/opt-1.3b")
selfcheck = SelfCheckBaseline(device="mps")

# We will collect both response-level averages AND token-level raw data
# Token-level data is better for P-value significance testing
results = {"ig": [], "entropy": [], "selfcheck": [], "labels": []}
all_token_ig = []
all_token_labels = []

print(f"Starting Phase 5: Statistical Validation on {SAMPLE_COUNT} samples...")

for idx, row in df.iterrows():
    # A. Compute Information Gain & Entropy
    ig, h_no, h_with = metric.compute_information_gain(row['query'], row['context'], row['response'])
    
    # B. Compute SelfCheckGPT (Sentence Level)
    sentences = [s.strip() for s in row['response'].split('.') if len(s) > 5]
    if sentences:
        samples = selfcheck.get_stochastic_samples(metric.model, metric.tokenizer, row['query'], row['context'])
        sc_scores = selfcheck.compute_selfcheck(sentences, samples)
        avg_sc = np.mean(sc_scores)
    else:
        avg_sc = 0.5

    # C. Process Labels
    label_dicts = ast.literal_eval(row['labels'])
    binary_mask = create_hallucination_mask(row['response'], label_dicts, metric.tokenizer, len(ig))
    
    # D. Store Response-Level (for Baseline comparison)
    results["ig"].append(-np.mean(ig)) 
    results["entropy"].append(np.mean(h_with))
    results["selfcheck"].append(avg_sc)
    results["labels"].append(1 if sum(binary_mask) > 0 else 0)

    # E. Store Token-Level (for Significance Testing)
    all_token_ig.extend(ig)
    all_token_labels.extend(binary_mask)

    print(f"Sample {idx} processed. (Hallucinated: {sum(binary_mask) > 0})")

# 2. Advanced Statistical Evaluation
print("\n" + "="*40)
print("PHASE 5: FINAL RESEARCH RESULTS")
print("="*40)

# Run the detailed evaluation function from src/evaluate.py
stats = evaluate_metric(np.array(all_token_ig), np.array(all_token_labels))
low_ci, high_ci = get_bootstrap_ci(np.array(all_token_ig), np.array(all_token_labels))

print(f"Main AUROC:         {stats['AUROC']}")
print(f"95% Confidence Int: [{low_ci:.4f}, {high_ci:.4f}]")
print(f"AUPRC (Imbalance):  {stats['AUPRC']}")
print(f"Significance (P):   {stats['P-Value (Significance)']}")

print("\n--- BASELINE COMPARISON (Response Level) ---")
print(f"IG Metric AUROC:    {roc_auc_score(results['labels'], results['ig']):.4f}")
print(f"Entropy AUROC:      {roc_auc_score(results['labels'], results['entropy']):.4f}")
print(f"SelfCheckGPT AUROC: {roc_auc_score(results['labels'], results['selfcheck']):.4f}")
print("="*40)