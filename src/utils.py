from datasets import Dataset
import numpy as np
import pickle, os
import pandas as pd
import ast

def load_ragtruth(max_samples=None):
    """Loads RAGTruth from local CSV and converts to HF Dataset."""
    # Ensure the path matches your project structure
    path = "data/ragtruth/ragtruth.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
        
    df = pd.read_csv(path)
    
    # Pre-parse stringified labels if they are stored as strings
    if 'labels' in df.columns and isinstance(df['labels'].iloc[0], str):
        df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x) if x.startswith('[') else x)

    if max_samples:
        df = df.head(max_samples)
        
    return Dataset.from_pandas(df)

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
    
def align_labels_to_tokens(response_text, word_labels, tokenizer):
    """
    Maps word-level labels to tokenizer subword tokens.
    Strategy: a token inherits the label of the word it belongs to.
    """
    # Simple check for binary/constant labels (used for HaluEval)
    if isinstance(word_labels, (int, float, np.integer)):
        full_encoding = tokenizer.encode(response_text, add_special_tokens=False)
        return np.array([word_labels] * len(full_encoding))

    words = response_text.split()
    
    # Fallback if label length doesn't match word count (common in noisy RAG data)
    if len(words) != len(word_labels):
        # Default to marking the whole thing if it's a known hallucination sample
        full_encoding = tokenizer.encode(response_text, add_special_tokens=False)
        return np.array([1 if any(word_labels) else 0] * len(full_encoding))
    
    token_labels = []
    for word, label in zip(words, word_labels):
        # Tokenize word with space prefix to respect subword logic (e.g., ' apple' vs 'apple')
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        token_labels.extend([label] * len(token_ids))
    
    return np.array(token_labels)