import torch
import torch.nn.functional as F
import pandas as pd
from typer import prompt
from src.model_utils import load_model, get_logits
import os
import math

def compute_cig(logits_ctx, logits_noctx):
    """
    Compute token-level Contextual Information Gain (CIG)
    logits_ctx: [seq_len, vocab_size]
    logits_noctx: [seq_len, vocab_size]
    """
    probs_ctx = F.softmax(logits_ctx, dim=-1)
    probs_noctx = F.softmax(logits_noctx, dim=-1)

    pred_tokens = torch.argmax(probs_ctx, dim=-1)  # predicted tokens

    # Ensure we don't go out of bounds if any length mismatch
    seq_len = min(probs_ctx.shape[0], probs_noctx.shape[0])
    cig_scores = []

    for t in range(seq_len):
        token_id = pred_tokens[t].item()
        p_ctx = max(probs_ctx[t, token_id].item(), 1e-12)
        p_noctx = max(probs_noctx[t, token_id].item(), 1e-12)
        cig = torch.log(torch.tensor(p_ctx)) - torch.log(torch.tensor(p_noctx))
        cig_scores.append(cig.item())

    return cig_scores, pred_tokens[:seq_len]

def save_token_level_csv(tokenizer, cig_scores, pred_tokens, labels=None, filename="results/token_level.csv"):
    """
    Save token-level data for analysis
    """
    tokens = tokenizer.convert_ids_to_tokens(pred_tokens)
    
    data = {
        "token": tokens,
        "cig_score": cig_scores
    }
    
    if labels:
        # labels should match number of tokens
        data["hallucination_label"] = labels
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Token-level data saved to {filename}")


def run_cig_on_dataset(df, context_col="context", prompt_col="prompt", label_col="label_hallucination"):
    """
    Run CIG computation for entire dataset
    """
    tokenizer, model = load_model()
    
    all_results = []
    
    for i, row in df.iterrows():
        prompt = row[prompt_col]
        context = row[context_col] if context_col in row else None
        label = row[label_col] if label_col in row else None
        
        # Step 1: Generate logits with and without context
        logits_ctx = get_logits(prompt, context)
        logits_noctx = get_logits(prompt)

        # Step 2: Compute CIG
        cig_scores, pred_tokens = compute_cig(logits_ctx[0], logits_noctx[0])
        
        # Step 3: Save CSV per sample
        sample_filename = f"results/token_level_sample_{i}.csv"
        labels_list = [label]*len(pred_tokens) if label is not None else None
        save_token_level_csv(tokenizer, cig_scores, pred_tokens, labels_list, filename=sample_filename)
        
        all_results.append({
            "prompt": prompt,
            "context": context,
            "sample_csv": sample_filename
        })
    
    # Optionally save summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("results/summary.csv", index=False)
    print("Summary CSV saved to results/summary.csv")