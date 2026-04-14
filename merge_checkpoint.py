import pickle
import numpy as np
import os


def merge_pkls(file1_path, file2_path, output_path):
    print(f"Loading {file1_path}...")
    with open(file1_path, "rb") as f:
        data1 = pickle.load(f)
    
    print(f"Loading {file2_path}...")
    with open(file2_path, "rb") as f:
        data2 = pickle.load(f)

    combined = {
        "tokens": {},
        "per_sample": {}
    }

    # 1. Merge the 'tokens' section (Numpy Arrays)
    # These are long 1D arrays of every token processed
    print("Merging token-level data...")
    for key in data1["tokens"].keys():
        combined["tokens"][key] = np.concatenate([
            data1["tokens"][key], 
            data2["tokens"][key]
        ])

    # 2. Merge the 'per_sample' section (Lists)
    # These are lists where each element is a sentence/sample
    print("Merging sample-level data...")
    for key in data1["per_sample"].keys():
        combined["per_sample"][key] = data1["per_sample"][key] + data2["per_sample"][key]

    # 3. Save the merged result
    with open(output_path, "wb") as f:
        pickle.dump(combined, f)
    
    print(f"\nSuccess! Combined file saved to: {output_path}")
    print(f"Total samples in merged file: {len(combined['per_sample']['labels'])}")

# Usage
merge_pkls("results/800/checkpoint_halueval.pkl", 
           "results/checkpoint_halueval.pkl", 
           "results/checkpoint_halueval_FINAL.pkl")