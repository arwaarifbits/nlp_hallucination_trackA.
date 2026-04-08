import pickle
import numpy as np

# 1. Load your existing RAGTruth results
checkpoint_path = "results/checkpoint_ragtruth.pkl"

with open(checkpoint_path, "rb") as f:
    rt_data = pickle.load(f)

# 2. Extract the necessary lists from your actual structure
comp_scores = rt_data["per_sample"]["composite"]
labels = rt_data["per_sample"]["labels"]

# Try to find where the text data is stored
# If it's not in 'metadata', it might be in 'texts' or 'samples'
# Based on your previous run, let's try to get it from the raw keys if they exist
# or handle the case where we just use the index to reference the original dataset
print("="*60)
print("SEARCHING FOR E7 FAILURE CASES (Confident Hallucinations)")
print("="*60)

failures_found = 0
for i in range(len(comp_scores)):
    avg_score = np.mean(comp_scores[i])
    hallucination_mask = np.array(labels[i]) == 1
    hallucination_count = np.sum(hallucination_mask)
    
    # Threshold for a "Confident Hallucination"
    if avg_score < 0.45 and hallucination_count >= 2:
        failures_found += 1
        
        print(f"\n[FAILURE CASE #{failures_found}]")
        print(f"Sample Index: {i}")
        print(f"Avg Uncertainty (Composite): {avg_score:.4f}")
        print(f"Hallucinated Tokens: {hallucination_count}")
        
        # Since 'metadata' key failed, let's look at what keys ARE available
        # so you can see where your text is hidden
        if i == 0:
            print(f"Available keys in per_sample: {rt_data['per_sample'].keys()}")
            
        print("-" * 30)

    if failures_found >= 5:
        break

if failures_found == 0:
    print("No cases found. Try increasing the avg_score threshold to 0.55.")