# src/e6_generator_breakdown.py
import pickle
import pandas as pd
import ast
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import load_ragtruth

# Load dataset and checkpoint
ragtruth = load_ragtruth(max_samples=150)
with open("results/checkpoint_ragtruth.pkl", "rb") as f:
    rt_data = pickle.load(f)

comp_list  = rt_data["per_sample"]["composite"]
label_list = rt_data["per_sample"]["labels"]

# After confirming field name (replace "model" with actual field name):
generator_field = "model"   # ← confirm this from above output

gen_scores = {}   # {model_name: {"scores": [], "labels": []}}

for i, sample in enumerate(ragtruth):
    if i >= len(comp_list):
        break
    gen = sample.get(generator_field, "unknown")
    scores = np.array(comp_list[i])
    labels = np.array(label_list[i])
    min_len = min(len(scores), len(labels))
    
    if gen not in gen_scores:
        gen_scores[gen] = {"scores": [], "labels": []}
    gen_scores[gen]["scores"].extend(scores[:min_len].tolist())
    gen_scores[gen]["labels"].extend(labels[:min_len].tolist())

print("\n=== E6: AUROC by Generator ===")
rows = []
for gen, data in gen_scores.items():
    s = np.array(data["scores"])
    l = np.array(data["labels"])
    if len(np.unique(l)) > 1:
        auroc = roc_auc_score(l, s)
        rows.append({"Generator": gen, "AUROC": round(auroc, 4), 
                     "N_tokens": len(l), "N_hal": int(l.sum())})
        print(f"  {gen}: AUROC={auroc:.4f}, tokens={len(l)}, hal={int(l.sum())}")

pd.DataFrame(rows).to_csv("results/E6_generator_breakdown.csv", index=False)