import pandas as pd
import ast
import os

def load_ragtruth(filepath="data/ragtruth_final.csv"):
    """Loads the merged RAGTruth dataset from local CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Run merge_ragtruth.py first! {filepath} missing.")
    
    df = pd.read_csv(filepath)
    
    # CRITICAL: CSVs store lists as strings (e.g., "[0, 1, 0]"). 
    # This converts them back to actual Python lists for your metrics.
    df['labels'] = df['labels'].apply(ast.literal_eval)
    
    print(f"Loaded RAGTruth with {len(df)} samples.")
    return df

def load_halueval(filepath="data/halueval/halueval.csv"):
    """Loads the HaluEval dataset from local CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File missing: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded HaluEval with {len(df)} samples.")
    return df