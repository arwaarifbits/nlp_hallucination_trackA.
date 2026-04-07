from datasets import Dataset
import numpy as np
import pickle, os
import pandas as pd
import ast

def load_ragtruth(max_samples=None):
    # 1. Load the full dataset
    df_all = pd.read_csv("data/ragtruth/ragtruth.csv")

    # 2. Define "Hallucination" based on the 'labels' column content
    # If it's just '[]', the string length is 2. If it has data, it's > 2.
    is_hallucination = df_all['labels'].apply(lambda x: len(str(x)) > 2)

    hal_pool = df_all[is_hallucination]
    clean_pool = df_all[~is_hallucination]

    print(f"  [Data Setup] Found {len(hal_pool)} hallucinated and {len(clean_pool)} clean samples.")

    # 3. Create a balanced subset (e.g., 25 of each for a total of 50)
    num_per_side = min(5, len(hal_pool), len(clean_pool))
    ragtruth_balanced_df = pd.concat([
        hal_pool.sample(num_per_side, random_state=42),
        clean_pool.sample(num_per_side, random_state=42)
    ]).sample(frac=1).reset_index(drop=True) # Shuffle the final set

    # 4. Convert to HuggingFace Dataset for the rest of your pipeline
    ragtruth = Dataset.from_pandas(ragtruth_balanced_df)

    print(f"  [Data Setup] Balanced RAGTruth dataset created with {len(ragtruth)} samples.")

    return ragtruth


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