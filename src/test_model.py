from load_data import load_ragtruth, load_halueval, preprocess_dataframe

# Load RAGTruth
rag_df = load_ragtruth("data/raw/ragtruth.csv")
rag_df = preprocess_dataframe(rag_df)

# Load HaluEval
h_eval_df = load_halueval("data/raw/halueval.csv")
h_eval_df = preprocess_dataframe(h_eval_df)

# Convert span-based hallucination labels to binary 1/0
rag_df['label_binary'] = rag_df['label_hallucination'].apply(
    lambda x: 1 if x == 'hallucinated' else 0
)

# Convert HaluEval span-based hallucination labels to binary 0/1
h_eval_df['label_binary'] = h_eval_df['label_hallucination'].apply(
    lambda x: 1 if x == 'hallucinated' else 0
)

import pandas as pd
from compute_cig import run_cig_in_batches

# Run on small subset first for debug

print("Running debug on first 5 samples of RAGTruth...")
run_cig_in_batches(
    rag_df.head(2),
    batch_size=1,
    prompt_col="prompt",
    context_col="context",
    label_col="label_binary"   # numeric 0/1 needed
)

print("Running debug on first 5 samples of HaluEval...")
run_cig_in_batches(
    h_eval_df.head(2),         # first 5 samples for debug
    batch_size=1,              # small batch for debug
    prompt_col="prompt",       # original question/prompt
    context_col="context",     # will internally be replaced with original_prompt when dataset_name="halueval"
    label_col="label_binary",  # numeric 0/1 required for Track A AUROC
    dataset_name="halueval"    # triggers usage of original_prompt internally
)