from datasets import Dataset
import numpy as np
import pickle, os
import pandas as pd
import ast

import pickle
import os

def load_ragtruth(max_samples=500):
    df_all = pd.read_csv("data/ragtruth/ragtruth.csv")

    # ─── STEP 0: GET CURRENT COUNTS FROM CHECKPOINT ───
    current_hal, current_clean = 0, 0
    checkpoint_path = "results/checkpoint_ragtruth.pkl"
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                ckpt_data = pickle.load(f)
            
            # CRITICAL FIX: Count sample-level labels, not token-level sums
            # We look at the 'per_sample' section which has 1 label per sentence
            if "per_sample" in ckpt_data and "labels" in ckpt_data["per_sample"]:
                labels = ckpt_data["per_sample"]["labels"]
                current_hal = sum(1 for l in labels if l == 1)
                current_clean = sum(1 for l in labels if l == 0)
            
        except Exception as e:
            print(f"  [Warning] Could not read checkpoint counts: {e}")
    
    # 1. Clean the labels immediately
    def is_hal(val):
        if pd.isna(val) or val == "[]" or val == "":
            return False
        try:
            # Parse the string representation of the list
            parsed = ast.literal_eval(val)
            return len(parsed) > 0
        except:
            return False

    # 2. Use the correct logic for the natural distribution stats
    hal_mask = df_all['labels'].apply(is_hal)

    
    print(f"  [Data Setup] Total dataset: {len(df_all)} samples.")
    print(f"  [Data Setup] Natural distribution: {hal_mask.sum()} hallucinated, {len(df_all) - hal_mask.sum()} clean.")

    # ─── BALANCE LOGIC ───────────────────────────────────────────
    # Calculate half of the requested total
    half_target = max_samples // 2
    
    # Select the samples dynamically
    # .head() ensures we take only up to half_target
    #df_hal = df_all[hal_mask].head(half_target)
    #df_clean = df_all[~hal_mask].head(half_target)

    # Calculate how many NEW samples we need to fetch
    needed_hal = max(0, half_target - current_hal)     # 250 - 135 = 115 more
    needed_clean = max(0, half_target - current_clean) # 250 - 165 = 85 more

    print(f"--- Balancing RAGTruth to {max_samples} samples ---")
    print(f"Current: {current_hal} Hal / {current_clean} Clean")
    print(f"Fetching: {needed_hal} new Hal / {needed_clean} new Clean")

    # TEMP- We already have 100 samples in the checkpoint (mostly clean).
    # To reach 150, we want to specifically fetch 50 hallucinated samples.
    #df_hal = df_all[hal_mask].iloc[0:150] # Skipping the 1st one you already have
    #df_clean = df_all[~hal_mask].iloc[0:150] # Your existing 99 + 1 more
    
    #print(f"  [Balance] Sampling {len(df_hal)} hallucinated and {len(df_clean)} clean samples.")

    # 2. Select NEW samples (skipping the ones already in your checkpoint)
    # We use iloc to skip the first N samples you already processed
    df_hal = df_all[hal_mask].iloc[current_hal : current_hal + needed_hal]
    df_clean = df_all[~hal_mask].iloc[current_clean : current_clean + needed_clean]


    # ─── MODIFIED BALANCE LOGIC ───────────────────────────────────────────
    # We target 115 Hallucinated samples and 85 Clean samples to reach 200 total.
    # Adjusting .iloc indices to skip what you've already processed.
    
    # Selecting 115 hallucinated samples (the 65 you have + 50 new ones)
    #df_hal = df_all[hal_mask].iloc[0:115] 
    
    # Selecting the 85 clean samples you already have
    #df_clean = df_all[~hal_mask].iloc[0:85] 
    #print(f"  [Balance] Current Checkpoint: 88 Hal / 112 Normal")
    #print(f"  [Balance] New Target: {len(df_hal)} Hallucinated and {len(df_clean)} Normal.")



    # Combine and shuffle
    final_df = pd.concat([df_hal, df_clean]).sample(frac=1, random_state=42)

    return Dataset.from_pandas(final_df.reset_index(drop=True))


def load_halueval(max_samples=None):
    """Loads HaluEval from local CSV and converts to HF Dataset."""
    path = "data/halueval/halueval.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
        
    df = pd.read_csv(path)
    
    if max_samples:
        df = df.head(max_samples)
        
    return Dataset.from_pandas(df)

def get_dataset_stats(ds, label_field="labels"):
    """Print basic stats about hallucination rate."""
    all_labels = []
    for sample in ds:
        label_val = sample[label_field]
        # Handle cases where labels are lists (RAGTruth) or single values (HaluEval)
        if isinstance(label_val, list):
            all_labels.extend(label_val)
        else:
            all_labels.append(label_val)
            
    all_labels = np.array(all_labels)
    print(f"Total entries/tokens: {len(all_labels)}")
    print(f"Hallucinated: {np.sum(all_labels)} ({100*np.mean(all_labels):.1f}%)")
    print(f"Faithful: {len(all_labels) - np.sum(all_labels)} ({100*(1-np.mean(all_labels)):.1f}%)")

def save_preprocessed(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_preprocessed(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
import ast
import numpy as np

def align_labels_to_tokens(response_text, labels_input, tokenizer):
    """
    Maps RAGTruth character offsets OR HaluEval binary arrays to subword tokens.
    """
    # 1. Tokenize with offsets
    encoding = tokenizer(
        response_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=1800
    )
    offsets = encoding.get('offset_mapping', [])
    n_tokens = len(offsets)
    token_labels = np.zeros(n_tokens, dtype=int)

    if n_tokens == 0:
        return token_labels

    # 2. Parse the labels_input
    # If it's a string like "[{'start': 679...}]", convert to list
    if isinstance(labels_input, str):
        try:
            # RAGTruth CSVs often have 'NaN' or empty strings
            if labels_input.strip() in ["[]", "", "nan", "NaN"]:
                return token_labels
            actual_labels = ast.literal_eval(labels_input)
        except (ValueError, SyntaxError):
            return token_labels
    else:
        actual_labels = labels_input

    # 3. Apply the character-to-token mapping (Case A: RAGTruth)
    if isinstance(actual_labels, list) and len(actual_labels) > 0:
        if isinstance(actual_labels[0], dict):
            for entry in actual_labels:
                # Use .get() to handle different versions of RAGTruth keys
                s = entry.get('start', entry.get('char_start'))
                e = entry.get('end',   entry.get('char_end'))
                
                if s is None or e is None: 
                    continue
                
                s, e = int(s), int(e)
                for i, (tok_s, tok_e) in enumerate(offsets):
                    # Flag token if it overlaps with the hallucination span
                    if tok_s < e and tok_e > s:
                        token_labels[i] = 1
            return token_labels

    # --- Case B: HaluEval (Numpy Array or list of ints) ---
    # Force convert to numpy array of floats/ints for math safety
    try:
        labels_arr = np.array(labels_input)
        if labels_arr.dtype.kind in 'if': # only if it's numeric
            unique = np.unique(labels_arr)
            
            # Subcase: Global label (all 0s or all 1s)
            if len(unique) <= 1:
                val = int(unique[0]) if len(unique) == 1 else 0
                return np.full(n_tokens, val, dtype=int)

            # Subcase: Word-level alignment
            words = response_text.split()
            if len(words) == len(labels_arr):
                char_to_word = {}
                curr_pos = 0
                for w_idx, word in enumerate(words):
                    start = response_text.find(word, curr_pos)
                    if start == -1: continue # Should not happen with split()
                    for c in range(start, start + len(word)):
                        char_to_word[c] = w_idx
                    curr_pos = start + len(word)

                for i, (tok_s, tok_e) in enumerate(offsets):
                    # Check characters in token; if any match a hallucinated word, mark token
                    for c in range(tok_s, tok_e):
                        if c in char_to_word:
                            token_labels[i] = int(labels_arr[char_to_word[c]])
                            if token_labels[i] == 1: break # Short circuit
                return token_labels
            
            # Fallback for length mismatch: check if the mean is > 0.5
            return np.full(n_tokens, int(np.mean(labels_arr) > 0.5), dtype=int)
    except (ValueError, TypeError):
        # This handles cases where labels_input might be an empty list or non-numeric
        pass

    # --- Case C: Scalar ---
    if isinstance(labels_input, (int, float, np.integer, np.floating)):
        return np.full(n_tokens, int(labels_input), dtype=int)

    return token_labels