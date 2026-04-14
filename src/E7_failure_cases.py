# src/E7_failure_cases.py
import pickle
import numpy as np
import sys
import os
sys.path.insert(0, "src")
from utils import load_ragtruth

# Load checkpoint and original dataset (to get response text)
with open("results/checkpoint_ragtruth.pkl", "rb") as f:
    rt_data = pickle.load(f)

ragtruth = load_ragtruth(max_samples=1252)

comp_scores  = rt_data["per_sample"]["composite"]
label_list   = rt_data["per_sample"]["labels"]
ent_list     = rt_data["per_sample"]["EntOnly"]
ig_list      = rt_data["per_sample"]["IG"]

print("=" * 65)
print("E7 FAILURE CASE ANALYSIS")
print("=" * 65)

# ── Failure Case 1: False Negatives ──────────────────────────────
# Samples that are hallucinated (label=1 tokens exist) but the
# composite scores them LOW (metric thinks they are faithful)
print("\n[FAILURE CASE 1] False Negatives — hallucinated but scored faithful")
print("Mechanism: model's parametric knowledge matches context,")
print("so IG and KL stay low even for hallucinated tokens.\n")

fc1_count = 0
for i in range(min(len(comp_scores), len(ragtruth))):
    labs  = np.array(label_list[i])
    comp  = np.array(comp_scores[i])
    n_hal = labs.sum()
    if n_hal == 0:
        continue
    # False negative: hallucinated tokens but composite score is low
    hal_mean_score   = comp[labs == 1].mean() if n_hal > 0 else 0
    faith_mean_score = comp[labs == 0].mean() if (labs == 0).sum() > 0 else 1
    if hal_mean_score < faith_mean_score and n_hal >= 3:
        sample = ragtruth[i]
        print(f"  Sample {i} | model={sample['model']}")
        print(f"  Response (first 120 chars): {sample['response'][:120]}...")
        print(f"  Hal tokens: {n_hal} | Mean composite on hal tokens: {hal_mean_score:.4f}")
        print(f"  Mean composite on faithful tokens: {faith_mean_score:.4f}")
        print(f"  Raw label entry: {str(sample['labels'])[:100]}")
        print()
        fc1_count += 1
        if fc1_count >= 2:
            break

if fc1_count == 0:
    print("  None found in this sample set. Try increasing max_samples.")

# ── Failure Case 2: Boundary Effect ──────────────────────────────
# Hallucination starts at token position 0 or 1 — no pre-span context
print("\n[FAILURE CASE 2] Boundary Effect — hallucination at sequence start")
print("Mechanism: temporal analysis requires t-3 to t-1 context.")
print("When span starts at position 0, all pre-span offsets are invalid.\n")

fc2_count = 0
for i in range(min(len(label_list), len(ragtruth))):
    labs = np.array(label_list[i])
    if len(labs) == 0:
        continue
    # Find first hallucinated position
    hal_positions = np.where(labs == 1)[0]
    if len(hal_positions) == 0:
        continue
    first_hal = hal_positions[0]
    if first_hal <= 2:   # t-3, t-2, t-1 all missing or incomplete
        sample = ragtruth[i]
        print(f"  Sample {i} | model={sample['model']}")
        print(f"  First hallucinated token at position: {first_hal}")
        print(f"  Total tokens: {len(labs)} | Hal tokens: {labs.sum()}")
        print(f"  Response (first 120 chars): {sample['response'][:120]}...")
        print()
        fc2_count += 1
        if fc2_count >= 2:
            break

if fc2_count == 0:
    print("  None found — all hallucinated spans start after position 2.")

# ── Failure Case 3: High-Entropy Faithful Tokens ─────────────────
# Function words (conjunctions, pronouns) have high entropy naturally
# — entropy-only baseline falsely flags them as hallucinations
print("\n[FAILURE CASE 3] High-entropy faithful tokens (false positives)")
print("Mechanism: function words like 'however', 'additionally' have")
print("naturally high predictive entropy regardless of faithfulness.\n")

fc3_count = 0
for i in range(min(len(ent_list), len(ragtruth))):
    labs = np.array(label_list[i])
    ent  = np.array(ent_list[i])
    min_len = min(len(labs), len(ent))
    labs, ent = labs[:min_len], ent[:min_len]

    faithful_mask   = labs == 0
    if faithful_mask.sum() < 5:
        continue

    threshold       = np.percentile(ent, 80)   # top 20% entropy
    high_ent_faith  = (faithful_mask) & (ent > threshold)

    if high_ent_faith.sum() >= 5:
        sample = ragtruth[i]
        # Decode the high-entropy faithful tokens
        token_ids  = sample["response"]
        false_pos_rate = high_ent_faith.sum() / faithful_mask.sum()

        print(f"  Sample {i} | model={sample['model']}")
        print(f"  Faithful tokens in top-20% entropy: {high_ent_faith.sum()}")
        print(f"  False positive rate on this sample: {false_pos_rate:.1%}")
        print(f"  Mean entropy (faithful): {ent[faithful_mask].mean():.4f}")
        print(f"  Mean entropy (hallucinated): {ent[labs==1].mean():.4f}" 
              if labs.sum() > 0 else "  No hallucinated tokens in this sample")
        print(f"  Response (first 120 chars): {sample['response'][:120]}...")
        print()
        fc3_count += 1
        if fc3_count >= 2:
            break

if fc3_count == 0:
    print("  None found — try lowering the percentile threshold to 70.")

# ── Summary stats across all samples ─────────────────────────────
print("\n=== Summary Statistics for Report ===")
all_hal_scores   = []
all_faith_scores = []
for comp, labs in zip(comp_scores, label_list):
    c = np.array(comp)
    l = np.array(labs)
    min_len = min(len(c), len(l))
    all_hal_scores.extend(c[:min_len][l[:min_len] == 1].tolist())
    all_faith_scores.extend(c[:min_len][l[:min_len] == 0].tolist())

if all_hal_scores and all_faith_scores:
    print(f"  Mean composite on hallucinated tokens: {np.mean(all_hal_scores):.4f}")
    print(f"  Mean composite on faithful tokens:     {np.mean(all_faith_scores):.4f}")
    print(f"  Std composite on hallucinated tokens:  {np.std(all_hal_scores):.4f}")
    print(f"  Std composite on faithful tokens:      {np.std(all_faith_scores):.4f}")
    overlap = min(np.mean(all_hal_scores) + np.std(all_hal_scores),
                  np.mean(all_faith_scores) + np.std(all_faith_scores))
    print(f"  Distribution overlap region: ~{overlap:.4f}")
    print(f"\n  Total hallucinated tokens analysed: {len(all_hal_scores)}")
    print(f"  Total faithful tokens analysed:     {len(all_faith_scores)}")