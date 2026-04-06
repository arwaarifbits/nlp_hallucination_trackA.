import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from utils import load_ragtruth, load_halueval, align_labels_to_tokens
from metric import InformationGainMetric
from temporal import (extract_temporal_features, analyze_precursor_patterns,
                      plot_ig_sequence, plot_precursor_distributions)
from baselines import entropy_only_hallucination_score
from evaluate import evaluate_metric, bootstrap_ci, compile_results_table

os.makedirs("results", exist_ok=True)

def run_on_dataset(metric, dataset, dataset_name, max_samples=40):
    all_ig, all_H_with_ctx, all_labels = [], [], []
    all_ig_arrays, all_label_arrays = [], []

    # Use min to avoid index errors if dataset is smaller than max_samples
    num_samples = min(max_samples, len(dataset))
    
    for i in tqdm(range(num_samples), desc=f"Processing {dataset_name}"):
        sample = dataset[i]
        
        try:
            if dataset_name == "ragtruth":
                query = sample["query"]
                context = sample["context"]
                response = sample["response"]
                # Use your existing aligner for RAGTruth
                word_labels = sample["labels"] 
                ig, _, h_with = metric.compute_information_gain(query, context, response)
                token_labels = align_labels_to_tokens(response, word_labels, metric.tokenizer)
                
                # Standard alignment for RAGTruth
                min_len = min(len(ig), len(token_labels))
                ig, h_with, token_labels = ig[:min_len], h_with[:min_len], token_labels[:min_len]
                
                all_ig.extend(ig)
                all_H_with_ctx.extend(h_with)
                all_labels.extend(token_labels)
                all_ig_arrays.append(ig)
                all_label_arrays.append(token_labels)
                
                # Plotting
                if i < 5:
                    tokens = metric.get_response_tokens(response)[:min_len]
                    plot_ig_sequence(ig, tokens, token_labels, sample_id=f"{dataset_name}_{i}")

            else:  # halueval
                query = sample["question"]
                context = sample["knowledge"]
                
                # 1. Process Hallucination (Class 1)
                res_h = sample["hallucinated_answer"]
                ig_h, _, h_h = metric.compute_information_gain(query, context, res_h)
                all_ig.extend(ig_h)
                all_H_with_ctx.extend(h_h)
                all_labels.extend([1] * len(ig_h))
                all_ig_arrays.append(ig_h) # For temporal analysis
                all_label_arrays.append(np.ones(len(ig_h)))

                # 2. Process Faithful Answer (Class 0)
                res_f = sample["right_answer"]
                ig_f, _, h_f = metric.compute_information_gain(query, context, res_f)
                all_ig.extend(ig_f)
                all_H_with_ctx.extend(h_f)
                all_labels.extend([0] * len(ig_f))
                all_ig_arrays.append(ig_f)
                all_label_arrays.append(np.zeros(len(ig_f)))

                # Plotting (Using the hallucinated version for the visual)
                if i < 5:
                    tokens = metric.get_response_tokens(res_h)
                    # Match lengths for plotting
                    p_len = min(len(ig_h), len(tokens))
                    plot_ig_sequence(ig_h[:p_len], tokens[:p_len], np.ones(p_len), 
                                     sample_id=f"{dataset_name}_{i}")

        except Exception as e:
            print(f"Error on {dataset_name} sample {i}: {e}")
            continue

    # Convert to numpy for evaluation
    all_ig = np.array(all_ig)
    all_labels = np.array(all_labels)
    all_H_with_ctx = np.array(all_H_with_ctx)
    
    # ... (Rest of your evaluation code: ig_results, ent_results, temporal analysis)

    print(f"\n=== Results on {dataset_name} ===")

    # --- IG Metric (low IG = hallucination, so negate for scoring) ---
    ig_results = evaluate_metric(-all_ig, all_labels, metric_name=f"IG ({dataset_name})")
    auroc_m, auroc_lo, auroc_hi = bootstrap_ci(-all_ig, all_labels, metric="auroc")
    ig_results["AUROC 95% CI"] = f"[{auroc_lo:.3f}, {auroc_hi:.3f}]"
    print(ig_results)

    # --- Entropy-Only Baseline ---
    ent_scores = entropy_only_hallucination_score(all_H_with_ctx)
    ent_results = evaluate_metric(ent_scores, all_labels,
                                  metric_name=f"Entropy-Only ({dataset_name})")
    ent_auroc_m, ent_lo, ent_hi = bootstrap_ci(ent_scores, all_labels, metric="auroc")
    ent_results["AUROC 95% CI"] = f"[{ent_lo:.3f}, {ent_hi:.3f}]"
    print(ent_results)

    # --- Temporal Analysis ---
    print(f"\n--- Temporal Analysis ({dataset_name}) ---")
    pre_hal, pre_faith, pval = analyze_precursor_patterns(
        all_ig_arrays, all_label_arrays, window=5, k=3)
    plot_precursor_distributions(pre_hal, pre_faith,
                                  save_dir="results")

    # --- Save results ---
    table = compile_results_table([ig_results, ent_results])
    table.to_csv(f"results/results_{dataset_name}.csv")
    print(table)

    return ig_results, ent_results

def main():
    # Start with small model for testing, upgrade later
    metric = InformationGainMetric(model_name="facebook/opt-1.3b")

    ragtruth = load_ragtruth(max_samples=40)
    halueval = load_halueval(max_samples=40)

    ig_rt, ent_rt = run_on_dataset(metric, ragtruth, "ragtruth")
    ig_hv, ent_hv = run_on_dataset(metric, halueval, "halueval")

    # Combined cross-dataset table
    all_results = [ig_rt, ent_rt, ig_hv, ent_hv]
    final_table = compile_results_table(all_results)
    final_table.to_csv("results/final_comparison_table.csv")
    print("\n=== FINAL RESULTS TABLE ===")
    print(final_table.to_string())

if __name__ == "__main__":
    main()