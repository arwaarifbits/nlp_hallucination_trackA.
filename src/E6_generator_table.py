# src/E6_generator_table.py
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
import os
sys.path.insert(0, "src")
from utils import load_ragtruth

# Load dataset — use same max_samples as your checkpoint
ragtruth = load_ragtruth(max_samples=1252)

with open("results/checkpoint_ragtruth.pkl", "rb") as f:
    rt_data = pickle.load(f)

comp_list  = rt_data["per_sample"]["composite"]
label_list = rt_data["per_sample"]["labels"]

print(f"Dataset samples: {len(ragtruth)}")
print(f"Checkpoint samples: {len(comp_list)}")
print(f"Model values in dataset: {set(ragtruth['model'])}")

gen_scores = {}

for i, sample in enumerate(ragtruth):
    if i >= len(comp_list):
        break

    gen = sample["model"] if sample["model"] else "unknown"
    scores = np.array(comp_list[i])
    labels = np.array(label_list[i])
    min_len = min(len(scores), len(labels))

    if gen not in gen_scores:
        gen_scores[gen] = {"scores": [], "labels": []}

    gen_scores[gen]["scores"].extend(scores[:min_len].tolist())
    gen_scores[gen]["labels"].extend(labels[:min_len].tolist())

print("\n=== E6: AUROC by Generator Model ===")
rows = []
for gen, data in sorted(gen_scores.items()):
    s = np.array(data["scores"])
    l = np.array(data["labels"])
    n_tokens = len(l)
    n_hal    = int(l.sum())

    if len(np.unique(l)) > 1:
        auroc = roc_auc_score(l, s)
        status = f"AUROC={auroc:.4f}"
        rows.append({
            "Generator": gen,
            "AUROC": round(auroc, 4),
            "N_tokens": n_tokens,
            "N_hallucinated": n_hal,
            "Pct_hallucinated": round(100 * n_hal / n_tokens, 1)
        })
    else:
        status = f"AUROC=N/A (only one class: {np.unique(l)})"
        rows.append({
            "Generator": gen,
            "AUROC": "N/A",
            "N_tokens": n_tokens,
            "N_hallucinated": n_hal,
            "Pct_hallucinated": round(100 * n_hal / n_tokens, 1) if n_tokens > 0 else 0
        })

    print(f"  {gen:30s} {status}  tokens={n_tokens}  hal={n_hal}")

df = pd.DataFrame(rows)
df.to_csv("results/E6_generator_breakdown.csv", index=False)
print(f"\nSaved to results/E6_generator_breakdown.csv")
print(df.to_string(index=False))