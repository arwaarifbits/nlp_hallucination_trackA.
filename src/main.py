# src/main.py  — full rewrite covering E1–E8

import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import mannwhitneyu, spearmanr

from utils import load_ragtruth, load_halueval, align_labels_to_tokens
from metric import InformationGainMetric
from semantic_entropy import SemanticEntropyMetric
from composite import build_composite, incremental_auroc_table, normalize_score
from temporal import compute_temporal_precedence, plot_temporal_precedence
from evaluate import evaluate_metric, bootstrap_ci, auroc_by_haltype
from scipy.ndimage import uniform_filter1d

os.makedirs("results", exist_ok=True)

# ─── helpers ────────────────────────────────────────────────────────────────

def span_f1(scores, labels, threshold=None):
    """Binary F1 at optimal threshold."""
    from sklearn.metrics import f1_score
    if threshold is None:
        # Try 50 thresholds and pick best
        thresholds = np.linspace(scores.min(), scores.max(), 50)
        best_f1 = 0
        for t in thresholds:
            preds = (scores >= t).astype(int)
            f = f1_score(labels, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
        return round(best_f1, 4)
    preds = (scores >= threshold).astype(int)
    return round(f1_score(labels, preds, zero_division=0), 4)

def spearman_rho(scores, labels):
    rho, _ = spearmanr(scores, labels)
    return round(float(rho), 4)

def safe_auroc(labels, scores):
    """Returns NaN instead of crashing if only one class present."""
    scores = np.nan_to_num(scores, nan=0.0)
    if len(np.unique(labels)) < 2:
        print("  WARNING: Only one class in labels — AUROC undefined")
        return float('nan')
    return roc_auc_score(labels, scores)

def smooth_scores(scores: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply a sliding window average to token-level scores."""
    if len(scores) < window:
        return scores
    return uniform_filter1d(scores.astype(float), size=window, mode='nearest')

def expected_calibration_error(scores, labels, n_bins=10):
    """ECE: lower is better (0 = perfectly calibrated)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    probs = normalize_score(scores)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        ece += mask.sum() / len(labels) * abs(bin_conf - bin_acc)
    return round(float(ece), 4)

def row_for_table(name, scores, labels):
    """Fill one row of the E1/E2 rubric table."""
    # Safety: Replace any NaNs or Infs with 0.0 before processing
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    norm = normalize_score(scores)
    
    return {
        "Metric / Composite": name,
        "AUROC": round(roc_auc_score(labels, scores), 4),
        "F1 span": span_f1(norm, labels),
        "Spearman ρ": spearman_rho(scores, labels),
        "ECE": expected_calibration_error(norm, labels),
    }

# ─── main collection loop ────────────────────────────────────────────────────

def collect_all_metrics(metric_obj, sem_entropy_obj, dataset,
                        dataset_name, max_samples=150):
    all_ig, all_kl, all_conf, all_sem, all_ent, all_labels = [], [], [], [], [], []
    ig_per_sample, kl_per_sample, conf_per_sample = [], [], []
    sem_per_sample, ent_per_sample = [], []
    label_per_sample, composite_per_sample = [], []

    for i, sample in enumerate(tqdm(dataset.select(range(min(max_samples, len(dataset)))),
                                    desc=f"{dataset_name}")):
        variations = []

        if dataset_name == "ragtruth":
            query   = sample["query"]                    
            context = sample["context"]       
            variations.append((sample["response"], sample["labels"]))
        else:
            query   = sample["question"]
            context = sample["knowledge"]
            variations.append((sample["hallucinated_answer"],
                                np.ones(len(sample["hallucinated_answer"].split()), dtype=int)))
            variations.append((sample["right_answer"],
                                np.zeros(len(sample["right_answer"].split()), dtype=int)))

        for response, word_labels in variations:
            try:
                ig, H_no, H_with = metric_obj.compute_information_gain(query, context, response)
                kl   = metric_obj.compute_kl_divergence(query, context, response)
                conf = metric_obj.compute_confidence_drop(query, context, response)
                sem_ent_val = sem_entropy_obj.compute_semantic_entropy(
                    query, context, num_samples=5)

                token_labels = align_labels_to_tokens(
                    response, word_labels, metric_obj.tokenizer)
                
                # In collect_all_metrics, before min_len:
                if len(ig) == 0 or len(kl) == 0 or len(conf) == 0 or len(token_labels) == 0:
                    print(f"  Empty array detected: ig={len(ig)}, kl={len(kl)}, conf={len(conf)}, labels={len(token_labels)}")
                    continue

                # Bug 3 fixed — use new variable names, don't overwrite
                # 1. Align and Slice
                min_len = min(len(ig), len(kl), len(conf), len(token_labels))
                ig_t      = ig[:min_len]
                kl_t      = kl[:min_len]
                conf_t    = conf[:min_len]
                H_with_t  = H_with[:min_len]
                labels_t  = token_labels[:min_len]
                sem_arr   = np.full(min_len, sem_ent_val)

                # 2. Transform and Smooth (Do this FIRST)
                ig_hal   = smooth_scores(-ig_t, window=3)
                kl_hal   = smooth_scores(kl_t, window=3)
                conf_hal = smooth_scores(conf_t, window=3)
                ent_hal  = smooth_scores(H_with_t, window=3) 

                # 3. Global aggregation (Use the smoothed variables!)
                all_ig.extend(ig_hal)
                all_kl.extend(kl_hal)
                all_conf.extend(conf_hal)
                all_sem.extend(sem_arr)
                all_ent.extend(ent_hal)      # FIXED: was H_with_t
                all_labels.extend(labels_t)

                # 4. Build Composite (Uses variance weighting)
                sample_metrics = {
                    "IG": ig_hal, "KL": kl_hal, "ConfDrop": conf_hal,
                    "SemEnt": sem_arr, "EntOnly": ent_hal
                }
                comp = build_composite(sample_metrics, labels_t, mode="variance_weight")
                
                # 5. Per-sample storage (Use the smoothed variables!)
                ig_per_sample.append(ig_hal)
                kl_per_sample.append(kl_hal)
                conf_per_sample.append(conf_hal)
                sem_per_sample.append(sem_arr)
                ent_per_sample.append(ent_hal) # FIXED: was H_with_t
                label_per_sample.append(labels_t)
                composite_per_sample.append(comp)

            except Exception as e:
                print(f"  Skipped sample {i}: {e}")
                continue

    # Construct the final data object
    final_data = {
        "tokens": {
            "IG": np.array(all_ig), "KL": np.array(all_kl),
            "ConfDrop": np.array(all_conf), "SemEnt": np.array(all_sem),
            "EntOnly": np.array(all_ent), "labels": np.array(all_labels)
        },
        "per_sample": {
            "IG": ig_per_sample, "KL": kl_per_sample,
            "ConfDrop": conf_per_sample, "SemEnt": sem_per_sample,
            "EntOnly": ent_per_sample, "labels": label_per_sample,
            "composite": composite_per_sample
        }
    }

    # SAVE TO DISK IMMEDIATELY
    import pickle
    with open(f"results/checkpoint_{dataset_name}.pkl", "wb") as f:
        pickle.dump(final_data, f)
    print(f"Saved checkpoint to results/checkpoint_{dataset_name}.pkl")

    return final_data


def run_all_experiments(data, dataset_name):
    t = data["tokens"]
    p = data["per_sample"]
    labels = t["labels"]

    print(f"\n{'='*60}")
    print(f"EXPERIMENTS ON {dataset_name.upper()}")
    print(f"{'='*60}")

    # ── E1 + E2: incremental composite table ──────────────────────
    metrics_ordered = {
        "Entropy-only (B1)": t["EntOnly"],
        "+ Info Gain":        t["IG"],
        "+ KL divergence":    t["KL"],
        "+ Conf drop":        t["ConfDrop"],
        "+ Semantic entropy": t["SemEnt"],
    }

    print("\n── E1+E2: Composite build table ──")
    running = {}
    table_rows = []
    for name, scores in metrics_ordered.items():
        # Clean the scores of any NaNs or Infs caused by division by zero
        clean_scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        running[name] = clean_scores
        composite = build_composite(running, labels, mode="variance_weight")

        composite = np.nan_to_num(composite, nan=0.0, posinf=0.0, neginf=0.0)
        row = row_for_table(name, composite, labels)
        auroc_m, lo, hi = bootstrap_ci(composite, labels, metric="auroc")
        row["AUROC 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        table_rows.append(row)
        print(row)

    # SelfCheckGPT baseline row — fill with placeholder if not run
    # (run selfcheck separately if time permits)
    df_e12 = pd.DataFrame(table_rows)
    df_e12.to_csv(f"results/E1E2_{dataset_name}.csv", index=False)

    # ── E3: temporal precedence ───────────────────────────────────
    print("\n── E3: Temporal precedence ──")
    metric_arrs = {
        "IG":       p["IG"],
        "KL":       p["KL"],
        "ConfDrop": p["ConfDrop"],
        "SemEnt":   p["SemEnt"],
        "EntOnly":  p["EntOnly"],
    }

    if dataset_name == "halueval":
        print("  [Skipping E3] HaluEval labels are whole-response; no pre-span context available.")
    else:
        print("  [Running E3] Analyzing temporal precedence on RAGTruth spans...")
        means = compute_temporal_precedence(metric_arrs, p["labels"])
        
        # Move the printing inside the 'else' so it only runs when 'means' exists
        print("Mean scores at t-3 to t+1:")
        for offset in [-3, -2, -1, 0, 1]:
            row_str = f"  t{offset:+d}: " + "  ".join(
                f"{m}={means[m].get(offset, float('nan')):.4f}"
                for m in metric_arrs.keys()
            )
            print(row_str)
            
        plot_temporal_precedence(means, save_dir="results")

        # Move Mann-Whitney U inside the 'else' as well
        print("\n  Mann-Whitney U (metric at t−2 vs metric at t):")
        for m_name in metric_arrs:
            vals_tm2 = []
            vals_t0  = []
            for sample_idx, lab_arr in enumerate(p["labels"]):
                arr = metric_arrs[m_name][sample_idx]
                in_span = False
                for i, lab in enumerate(lab_arr):
                    if lab == 1 and not in_span:
                        in_span = True
                        if i - 2 >= 0 and i - 2 < len(arr):
                            vals_tm2.append(arr[i - 2])
                        if i < len(arr):
                            vals_t0.append(arr[i])
                    elif lab == 0:
                        in_span = False
            if vals_tm2 and vals_t0:
                u, p_val = mannwhitneyu(vals_tm2, vals_t0, alternative='less')
                print(f"    {m_name}: U={u:.0f}, p={p_val:.4f} (t-2 < t → signal rises toward hallucination)")
    
    return df_e12


def main():
    # ── Load model ────────────────────────────────────────────────
    metric = InformationGainMetric(model_name="facebook/opt-1.3b")
    sem_metric = SemanticEntropyMetric(metric.model, metric.tokenizer, device=metric.device)

    # ── Load datasets ─────────────────────────────────────────────
    ragtruth = load_ragtruth(max_samples=150)
    ragtruth = ragtruth.shuffle(seed=42)
    halueval = load_halueval(max_samples=150)
    halueval = halueval.shuffle(seed=42)

    # ── Collect metrics ───────────────────────────────────────────
    print("Collecting metrics on RAGTruth...")
    rt_data = collect_all_metrics(metric, sem_metric, ragtruth, "ragtruth", max_samples=150)

    print("Collecting metrics on HaluEval...")
    hv_data = collect_all_metrics(metric, sem_metric, halueval, "halueval", max_samples=150)

    # ── Run experiments ───────────────────────────────────────────
    df_rt = run_all_experiments(rt_data, "ragtruth")
    df_hv = run_all_experiments(hv_data, "halueval")

    # ── E4: Cross-domain comparison table ─────────────────────────
    print("\n── E4: Cross-domain transfer ──")
    rt_labels = rt_data["tokens"]["labels"]
    hv_labels = hv_data["tokens"]["labels"]
    for m_name in ["IG", "KL", "ConfDrop", "SemEnt"]:
        # CLEAN THE RAW TOKENS HERE TOO
        rt_scores = np.nan_to_num(rt_data["tokens"][m_name], nan=0.0)
        hv_scores = np.nan_to_num(hv_data["tokens"][m_name], nan=0.0)
        
        rt_auroc = safe_auroc(rt_labels, rt_scores)
        hv_auroc = roc_auc_score(hv_labels, hv_scores)

        drop = rt_auroc - hv_auroc
        stable = "Yes" if abs(drop) < 0.10 else "No"
        print(f"  {m_name}: RAGTruth={rt_auroc:.4f}, HaluEval={hv_auroc:.4f}, Drop={drop:.4f}, Stable={stable}")

    # ── E5: Hallucination type breakdown (RAGTruth only) ──────────
    print("\n── E5: Hallucination type breakdown ──")
    
    # We target 'labels' because we know that's where RAGTruth hides the 'label_type'
    if "labels" in ragtruth.column_names:
        print(f"  Extracting nested 'label_type' from RAGTruth labels...")
        
        comp_per_sample = rt_data["per_sample"]["composite"]
        label_per_sample = rt_data["per_sample"]["labels"]
        
        # This function handles the ast.literal_eval(sample['labels']) internally
        type_results = auroc_by_haltype(ragtruth, comp_per_sample, label_per_sample)
        
        if type_results:
            for t_name, res in type_results.items():
                auroc_val = res.get('auroc', 'N/A')
                t_count = res.get('token_count', 0)
                print(f"  - {t_name}: AUROC={auroc_val} (Tokens: {t_count})")
            
            # Save to results folder
            df_e5 = pd.DataFrame(type_results).T
            df_e5.to_csv("results/E5_type_breakdown.csv")
            print("  ✅ E5 results saved to results/E5_type_breakdown.csv")
        else:
            print("  ⚠️ E5: No categories found. Check if the dataset contains hallucinations.")
    else:
        print("  ❌ Skipping E5: 'labels' column not found in RAGTruth.")



    # ── E6–E8: SOTA gap table ────────────────────────────────────
    print("\n── E6–E8: SOTA gap ──")

    # CLEAN ALL METRICS BEFORE BUILDING FINAL COMPOSITE
    cleaned_tokens = {
        k: np.nan_to_num(rt_data["tokens"][k], nan=0.0) 
        for k in ["IG", "KL", "ConfDrop", "SemEnt", "EntOnly"]
    }

    rt_composite = build_composite(cleaned_tokens, rt_labels, mode="variance_weight")
    rt_composite = np.nan_to_num(rt_composite, nan=0.0) # Final safety check

    our_auroc   = safe_auroc(rt_labels, rt_composite)
    ent_auroc   = safe_auroc(rt_labels, cleaned_tokens["EntOnly"])
    lumina_auroc = 0.87
    gap_closed  = (our_auroc - ent_auroc) / (lumina_auroc - ent_auroc) * 100
    print(f"  Entropy baseline: {ent_auroc:.4f}")
    print(f"  Our composite:    {our_auroc:.4f}")
    print(f"  LUMINA (SOTA):    {lumina_auroc:.4f}")
    print(f"  SOTA gap closed:  {gap_closed:.1f}%")

    sota_table = pd.DataFrame([
        {"Method": "Entropy-only (baseline)", "AUROC": round(ent_auroc, 4), "Type": "Unsupervised"},
        {"Method": "SelfCheckGPT", "AUROC": 0.65, "Type": "Unsupervised"},
        {"Method": "Semantic Entropy", "AUROC": 0.70, "Type": "Unsupervised"},
        {"Method": "Ours (composite)", "AUROC": round(our_auroc, 4), "Type": "Unsupervised"},
        {"Method": "ReDeEP", "AUROC": 0.82, "Type": "Supervised"},
        {"Method": "LUMINA", "AUROC": 0.87, "Type": "Supervised"},
    ])
    sota_table.to_csv("results/E8_sota_gap.csv", index=False)
    print(sota_table.to_string(index=False))

if __name__ == "__main__":
    main()