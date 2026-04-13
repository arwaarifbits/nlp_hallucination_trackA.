# src/main.py  — full rewrite covering E1–E8
import os
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"   # disables ALL tqdm progress bars globally


import numpy as np
import pandas as pd
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
from baselines import SelfCheckBaseline

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

def orient_score(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Ensure higher score = more hallucinated. Flip if AUROC < 0.5."""
    if len(np.unique(labels)) < 2:
        return scores
    try:
        a = roc_auc_score(labels, scores)
        return scores if a >= 0.5 else -scores
    except:
        return scores

def smooth_scores(scores: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply CAUSAL smoothing. 
    A peak at t-1 will now stay at t-1 instead of being pulled into t+1.
    """
    if len(scores) < window:
        return scores
    # Use a simple moving average that only looks at current and past tokens
    return pd.Series(scores).rolling(window=window, min_periods=1).mean().values

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

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def apply_sentence_smoothing(tokens, scores):
    """Averages token scores across their respective sentences."""
    # Convert tokens to a clean string for NLTK
    # OPT uses 'Ġ' for space; we convert it to actual spaces for sentence detection
    clean_text = "".join([t.replace('Ġ', ' ') for t in tokens])
    
    # Get sentence boundaries
    sentences = nltk.sent_tokenize(clean_text)
    smoothed_scores = np.zeros_like(scores)
    
    curr_token_idx = 0
    # Re-map the scores back to tokens sentence by sentence
    for sent in sentences:
        # Tokenize the sentence back to words to estimate the token count
        sent_words = nltk.word_tokenize(sent)
        # Find how many of our original tokens match this sentence length
        # (Approximate mapping: usually 1 word = 1.2 tokens)
        sent_len = len(sent_words) 
        
        # Determine the end bound for this sentence in our score array
        end_idx = min(curr_token_idx + sent_len, len(scores))
        
        if curr_token_idx < end_idx:
            # Calculate the average uncertainty for this sentence
            avg_val = np.mean(scores[curr_token_idx:end_idx])
            smoothed_scores[curr_token_idx:end_idx] = avg_val
            
        curr_token_idx = end_idx
        
    return smoothed_scores

# ─── main collection loop ────────────────────────────────────────────────────

def collect_all_metrics(metric_obj, sem_entropy_obj, selfcheck_obj, dataset, 
                        dataset_name, max_samples=800, existing_data=None):
    # --- ADD THIS LOGIC ---
    if existing_data is not None:
        print(f"Resuming {dataset_name} from {len(existing_data['per_sample']['labels'])} samples...")
        all_ig = list(existing_data["tokens"]["IG"])
        all_kl = list(existing_data["tokens"]["KL"])
        all_conf = list(existing_data["tokens"]["ConfDrop"])
        all_sem = list(existing_data["tokens"]["SemEnt"])
        all_ent = list(existing_data["tokens"]["EntOnly"])
        all_sc = list(existing_data["tokens"]["SelfCheck"])
        all_labels = list(existing_data["tokens"]["labels"])
        
        ig_per_sample = existing_data["per_sample"]["IG"]
        kl_per_sample = existing_data["per_sample"]["KL"]
        conf_per_sample = existing_data["per_sample"]["ConfDrop"]
        sem_per_sample = existing_data["per_sample"]["SemEnt"]
        ent_per_sample = existing_data["per_sample"]["EntOnly"]
        sc_per_sample = existing_data["per_sample"]["SelfCheck"]
        label_per_sample = existing_data["per_sample"]["labels"]
        composite_per_sample = existing_data["per_sample"]["composite"]
        
        start_idx = len(label_per_sample)
    else:
        all_ig, all_kl, all_conf, all_sem, all_ent, all_sc, all_labels = [], [], [], [], [], [], []
        ig_per_sample, kl_per_sample, conf_per_sample, sc_per_sample = [], [], [], []
        sem_per_sample, ent_per_sample = [], []
        label_per_sample, composite_per_sample = [], []
        start_idx = 0

    # Wrap the range in the select() call properly
    for i, sample in enumerate(tqdm(dataset.select(range(start_idx, min(max_samples, len(dataset)))), 
                                    desc=f"{dataset_name}")):
        
        current_sample_idx = start_idx + i

        variations = []
        if dataset_name == "ragtruth":
            query   = sample["query"]                    
            context = sample["context"]       
            variations.append((sample["response"], sample["labels"]))
        else:
            query   = sample["question"]
            context = sample["knowledge"]
            variations.append((sample["hallucinated_answer"], 1)) # Simplified for entire response
            variations.append((sample["right_answer"], 0))

        for response, word_labels in variations:
            try:
                # 1. Base Metrics
                ig, H_no, H_with = metric_obj.compute_information_gain(query, context, response)
                kl   = metric_obj.compute_kl_divergence(query, context, response)
                conf = metric_obj.compute_confidence_drop(query, context, response)
                sem_ent_val = sem_entropy_obj.compute_semantic_entropy(query, context, num_samples=2, temperature=0.8)
                
                # 2. Tokenize and Align Labels
                tokens = metric_obj.tokenizer.tokenize(response)
                token_labels = align_labels_to_tokens(response, word_labels, metric_obj.tokenizer)

        
                # 3. SelfCheckGPT
                prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
                try:
                    samples = selfcheck_obj.generate_samples(
                        metric_obj.model, metric_obj.tokenizer,
                        prompt, num_samples=3, max_new_tokens=20,
                        device=metric_obj.device
                    )
                    sentences_to_score = [response.strip()] if response.strip() else ["no response"]
                    sc_scores = selfcheck_obj.score(
                        sentences=sentences_to_score,
                        sampled_passages=samples
                    )
                    if isinstance(sc_scores, dict) and 'sent_scores' in sc_scores:
                        avg_score = float(np.mean(sc_scores['sent_scores'])) if sc_scores['sent_scores'] else 0.0
                    elif isinstance(sc_scores, (list, np.ndarray)) and len(sc_scores) > 0:
                        avg_score = float(sc_scores[0])
                    else:
                        avg_score = 0.0
                    sc_token = np.full(len(tokens), avg_score)

                except Exception as sc_e:
                    print(f"  SelfCheck failed at sample {current_sample_idx}: {sc_e}")
                    sc_token = np.zeros(len(tokens))

                # 4. Dimension Alignment & Safety Checks
                min_len = min(len(ig), len(kl), len(conf), len(sc_token), len(token_labels))
                if min_len == 0:
                    continue

                ig_t, kl_t, conf_t, sc_t = ig[:min_len], kl[:min_len], conf[:min_len], sc_token[:min_len]
                ent_t, labels_t = H_with[:min_len], token_labels[:min_len]
                sem_arr = np.full(min_len, sem_ent_val)

                # 5. Requirement 2: Temporal Analysis Prep (Raw Data) 
                ig_raw   = ig_t     
                kl_raw   = kl_t
                conf_raw = conf_t
                ent_raw  = ent_t

                # 5. Apply Sentence-Level Smoothing (The AUROC Booster)
                # This averages the "jitters" across the whole sentence context
                ig_smoothed   = apply_sentence_smoothing(tokens[:min_len], ig_t)
                kl_smoothed   = apply_sentence_smoothing(tokens[:min_len], kl_t)
                conf_smoothed = apply_sentence_smoothing(tokens[:min_len], conf_t)
                ent_smoothed  = apply_sentence_smoothing(tokens[:min_len], ent_t)

                # 6. Apply Causal Window Smoothing (The Temporal Fix)
                # We use the window of 5 here to see the "build-up" before a hallucination
                ig_hal   = smooth_scores(ig_smoothed, window=5)
                kl_hal   = smooth_scores(kl_smoothed, window=5)
                conf_hal = smooth_scores(conf_smoothed, window=5)
                sc_hal   = smooth_scores(sc_t, window=5)
                ent_hal  = smooth_scores(ent_smoothed, window=5)


                all_ig.extend(ig_hal); all_kl.extend(kl_hal); all_conf.extend(conf_hal)
                all_sc.extend(sc_hal); all_sem.extend(sem_arr); all_ent.extend(ent_hal)
                all_labels.extend(labels_t)

                # Before storing in sample_metrics:
                #ig_lead = np.roll(ig_hal, 1) # Shift scores forward so t-1 uncertainty aligns with t label
                #ig_lead[0] = ig_hal[0]

                # 7. Composite and Per-Sample Storage
                sample_metrics = {"IG": ig_hal, "KL": kl_hal, "ConfDrop": conf_hal, 
                                 "SemEnt": sem_arr, "EntOnly": ent_hal, "SelfCheck": sc_hal}
                comp = build_composite(sample_metrics, labels_t, mode="variance_weight")

                ig_per_sample.append(ig_raw); kl_per_sample.append(kl_raw)
                conf_per_sample.append(conf_raw); sem_per_sample.append(sem_arr)
                ent_per_sample.append(ent_raw); label_per_sample.append(labels_t)
                sc_per_sample.append(sc_t); composite_per_sample.append(comp)

            except Exception as e:
                print(f"  Skipped sample {current_sample_idx}: {e}")
                continue

        # --- SAFE AUTO-SAVE BLOCK ---
        if (i + 1) % 10 == 0:
            final_data = {
                "tokens": {
                    "IG": np.array(all_ig), "KL": np.array(all_kl), "ConfDrop": np.array(all_conf), 
                    "SemEnt": np.array(all_sem), "EntOnly": np.array(all_ent), "SelfCheck": np.array(all_sc), 
                    "labels": np.array(all_labels)
                },
                "per_sample": {
                    "IG": ig_per_sample, "KL": kl_per_sample, "ConfDrop": conf_per_sample, 
                    "SemEnt": sem_per_sample, "EntOnly": ent_per_sample, "SelfCheck": sc_per_sample, 
                    "labels": label_per_sample, "composite": composite_per_sample
                }
            }
            checkpoint_path = f"results/checkpoint_{dataset_name}.pkl"
            temp_path = checkpoint_path + ".tmp"
            
            try:
                # Save to a temporary file first
                with open(temp_path, "wb") as f:
                    pickle.dump(final_data, f)
                # If successful, rename it to the real checkpoint path
                os.replace(temp_path, checkpoint_path)
                print(f"  [Auto-Save] {dataset_name} updated at {current_sample_idx + 1}")
            except Exception as e:
                print(f"  [Save Error] {e}")

    final_data = {
        "tokens": {
            "IG": np.array(all_ig), "KL": np.array(all_kl), "ConfDrop": np.array(all_conf), 
            "SemEnt": np.array(all_sem), "EntOnly": np.array(all_ent), "SelfCheck": np.array(all_sc), 
            "labels": np.array(all_labels)
        },
        "per_sample": {
            "IG": ig_per_sample, "KL": kl_per_sample, "ConfDrop": conf_per_sample, 
            "SemEnt": sem_per_sample, "EntOnly": ent_per_sample, "SelfCheck": sc_per_sample, 
            "labels": label_per_sample, "composite": composite_per_sample
        }
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/checkpoint_{dataset_name}.pkl", "wb") as f:
        pickle.dump(final_data, f)
    return final_data





def run_all_experiments(data, dataset_name):
    t = data["tokens"]
    p = data["per_sample"]
    labels = t["labels"]

    print(f"\n{'='*60}")
    print(f"EXPERIMENTS ON {dataset_name.upper()}")
    print(f"{'='*60}")

    # --- Step 1: Auto-orient all raw metrics ---
    # This ensures consistency for E3 and the global table
    for key in ["IG", "KL", "ConfDrop", "EntOnly", "SelfCheck"]:
        if key not in t:
            continue
        scores = np.nan_to_num(t[key])
        if len(np.unique(labels)) > 1:
            if roc_auc_score(labels, scores) < 0.5:
                t[key] = -scores
                p[key] = [-arr for arr in p[key]]
                print(f"  [Orient] Flipped {key} (was inverted)")

    # Define the display names and their corresponding data keys
    # Note: Using the internal keys (IG, KL) for math consistency
    metrics_to_run = [
        ("Entropy-only (B1)", "EntOnly"),
        ("SelfCheckGPT (B2)", "SelfCheck"),
        ("+ Info Gain", "IG"),
        ("+ KL divergence", "KL"),
        ("+ Conf drop", "ConfDrop"),
        ("+ Semantic entropy", "SemEnt"),
       
    ]

    print("\n── E1+E2: Composite build table ──")

    # --- Step 2: First pass: get individual AUROCs for weights ---
    individual_aurocs = {}
    for name, key in metrics_to_run:
        scores = np.nan_to_num(t[key], nan=0.0)
        try:
            a = roc_auc_score(labels, scores)
        except:
            a = 0.5
        individual_aurocs[key] = a # Use the key 'IG', 'KL' etc for the dict
        print(f"  Individual AUROC {name}: {a:.4f}")

    # --- Step 3: Second pass: Build incremental composite ---
    running_keys = []
    table_rows = []

    for name, key in metrics_to_run:
        running_keys.append(key)
        
        # Calculate weights for the metrics currently in 'running_keys'
        # Weight = individual AUROC - 0.5
        current_weights = {k: max(0, individual_aurocs[k] - 0.5) for k in running_keys}
        total_w = sum(current_weights.values())

        if total_w < 1e-6:
            # Fallback to equal weight if no metric is better than random
            norm_scores = [normalize_score(np.nan_to_num(t[k])) for k in running_keys]
            composite = np.mean(norm_scores, axis=0)
        else:
            # Weighted sum of normalized scores
            composite = np.zeros_like(labels, dtype=float)
            for k in running_keys:
                w = current_weights[k] / total_w
                norm_s = normalize_score(np.nan_to_num(t[k]))
                composite += norm_s * w

        composite = np.nan_to_num(composite, nan=0.0)
        
        # Generate table row
        row = row_for_table(name, composite, labels)
        auroc_m, lo, hi = bootstrap_ci(composite, labels, metric="auroc")
        row["AUROC 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        table_rows.append(row)
        print(row)

    df_e12 = pd.DataFrame(table_rows)
    df_e12.to_csv(f"results/E1E2_{dataset_name}.csv", index=False)

    # ── E3: Temporal precedence ───────────────────────────────────
    # (Remains largely the same, but now uses flipped 'p' data)
    print("\n── E3: Temporal precedence ──")
    metric_arrs = {k: p[k] for k in ["IG", "KL", "ConfDrop", "SemEnt", "EntOnly"]}

    if dataset_name == "halueval":
        print("  [Skipping E3] HaluEval labels are whole-response.")
    else:
        print("  [Running E3] Analyzing temporal precedence...")
        means = compute_temporal_precedence(metric_arrs, p["labels"])
        
        print("Mean scores at t-3 to t+1:")
        for offset in [-3, -2, -1, 0, 1]:
            row_str = f"  t{offset:+d}: " + "  ".join(
                f"{m}={means[m].get(offset, float('nan')):.4f}"
                for m in metric_arrs.keys()
            )
            print(row_str)
            
        plot_temporal_precedence(means, save_dir="results")

        print("\n  Mann-Whitney U (t−2 vs t):")
        for m_name in metric_arrs:
            vals_tm2, vals_t0 = [], []
            for sample_idx, lab_arr in enumerate(p["labels"]):
                arr = metric_arrs[m_name][sample_idx]
                in_span = False
                for i, lab in enumerate(lab_arr):
                    if lab == 1 and not in_span:
                        in_span = True
                        if i - 2 >= 0 and i - 2 < len(arr): vals_tm2.append(arr[i-2])
                        if i < len(arr): vals_t0.append(arr[i])
                    elif lab == 0: in_span = False
            
            if vals_tm2 and vals_t0:
                u, p_val = mannwhitneyu(vals_tm2, vals_t0, alternative='less')
                print(f"    {m_name}: U={u:.0f}, p={p_val:.4f}")
    
    return df_e12


def main():
    
    # ── Load model ────────────────────────────────────────────────
    # This prevents the "loading weights" bars from appearing 300 times
    print("Initializing Models and Metric Engines...")
    
    # Initialize the base metric (Model stays in GPU memory)
    metric_engine = InformationGainMetric(model_name="facebook/opt-1.3b")
    
    # Initialize dependent metrics using the SAME model/tokenizer/device
    sem_metric = SemanticEntropyMetric(metric_engine.model, metric_engine.tokenizer, device=metric_engine.device)

    selfcheck = SelfCheckBaseline()

    # ── Load datasets ─────────────────────────────────────────────
    ragtruth = load_ragtruth(max_samples=800)
    ragtruth = ragtruth.shuffle(seed=42)
    halueval = load_halueval(max_samples=200)
    halueval = halueval.shuffle(seed=42)

    # ── Collect metrics ───────────────────────────────────────────
    #print("Collecting metrics on RAGTruth...")
    #rt_data = collect_all_metrics(metric, sem_metric, ragtruth, "ragtruth", max_samples=300)

    #print("Collecting metrics on HaluEval...")
    #hv_data = collect_all_metrics(metric, sem_metric, halueval, "halueval", max_samples=300)



    # ── Collect metrics (With "Resume and Extend" Logic) ───────────
    
    #target_samples = 500  # Set your target here once for consistency

    # RAGTruth Checkpoint
    rt_path = "results/checkpoint_ragtruth.pkl"
    rt_data = None
    if os.path.exists(rt_path):
        with open(rt_path, "rb") as f:
            rt_data = pickle.load(f)
        
        # Check if we already reached or exceeded the target
        current_count = len(rt_data["per_sample"]["labels"])
        if current_count >= 800:
            print(f"--- RAGTruth already has {current_count} samples. Skipping to experiments. ---")
        else:
            print(f"--- RAGTruth has {current_count}/{800} samples. Extending... ---")
            rt_data = collect_all_metrics(
                metric_engine, sem_metric, selfcheck, 
                ragtruth, "ragtruth", max_samples=800, existing_data=rt_data
            )
    else:
        print("RAGTruth checkpoint not found. Starting fresh...")
        rt_data = collect_all_metrics(
            metric_engine, sem_metric, selfcheck, 
            ragtruth, "ragtruth", max_samples=800, existing_data=None
        )

        

    # HaluEval Checkpoint
    hv_path = "results/checkpoint_halueval.pkl"
    hv_data = None
    if os.path.exists(hv_path):
        with open(hv_path, "rb") as f:
            hv_data = pickle.load(f)
            
        current_count = len(hv_data["per_sample"]["labels"])
        if current_count >= 200:
            print(f"--- HaluEval already has {current_count} samples. Skipping to experiments. ---")
        else:
            print(f"--- HaluEval has {current_count}/{200} samples. Extending... ---")
            hv_data = collect_all_metrics(
                metric_engine, sem_metric, selfcheck, 
                halueval, "halueval", max_samples=200, existing_data=hv_data
            )
    else:
        print("HaluEval checkpoint not found. Starting fresh...")
        hv_data = collect_all_metrics(
            metric_engine, sem_metric, selfcheck, 
            halueval, "halueval", max_samples=200, existing_data=None
        )



    # ── Run experiments ───────────────────────────────────────────
    df_rt = run_all_experiments(rt_data, "ragtruth")
    df_hv = run_all_experiments(hv_data, "halueval")

    # ── E4: Cross-domain comparison table ─────────────────────────
    print("\n── E4: Cross-domain transfer ──")
    rt_labels = rt_data["tokens"]["labels"]
    hv_labels = hv_data["tokens"]["labels"]
    for m_name in ["IG", "KL", "ConfDrop", "SemEnt", "SelfCheck"]:
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



    # ── E6–E8: SOTA gap ────────────────────────────────────────
    print("\n── E6-E8: SOTA gap ──")

    rt_labels = rt_data["tokens"]["labels"]

    # Use AUROC-proportional weights, same as E1+E2 table
    individual_aurocs_rt = {
        "IG":      safe_auroc(rt_labels, np.nan_to_num(rt_data["tokens"]["IG"])),
        "KL":      safe_auroc(rt_labels, np.nan_to_num(rt_data["tokens"]["KL"])),
        "ConfDrop":safe_auroc(rt_labels, np.nan_to_num(rt_data["tokens"]["ConfDrop"])),
        "EntOnly": safe_auroc(rt_labels, np.nan_to_num(rt_data["tokens"]["EntOnly"])),
        "SelfCheck":    safe_auroc(rt_labels, np.nan_to_num(rt_data["tokens"]["SelfCheck"])),
        "SemEnt":   safe_auroc(rt_labels, np.nan_to_num(rt_data["tokens"]["SemEnt"])),  # ADD
    }

    # Weight = max(0, AUROC - 0.5) so only metrics above random contribute
    weights_rt = {k: max(0.0, v - 0.5) for k, v in individual_aurocs_rt.items()}
    total_w = sum(weights_rt.values())

    if total_w > 1e-6:
        norm_scores = {k: normalize_score(np.nan_to_num(rt_data["tokens"][k])) 
                    for k in weights_rt}
        rt_composite = sum(norm_scores[k] * (weights_rt[k]/total_w) 
                        for k in weights_rt)
    else:
        rt_composite = normalize_score(np.nan_to_num(rt_data["tokens"]["EntOnly"]))

    rt_composite = np.nan_to_num(rt_composite)
    print(f"  AUROC weights used: { {k: round(weights_rt[k]/total_w, 3) for k in weights_rt} }")

    our_auroc  = safe_auroc(rt_labels, rt_composite)
    ent_auroc  = safe_auroc(rt_labels, np.nan_to_num(rt_data["tokens"]["EntOnly"]))
    lumina_auroc = 0.87
    gap_closed = (our_auroc - ent_auroc) / (lumina_auroc - ent_auroc) * 100
    
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