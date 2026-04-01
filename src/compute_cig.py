import torch
import torch.nn.functional as F
import pandas as pd
from src.model_utils import load_model, get_logits, tokenizer
import os
import math

def compute_cig(logits_ctx, logits_noctx):
    """
    Computes token-level CIG: log(P(y|x,C)) - log(P(y|x))
    
    logits_ctx: [seq_len, vocab_size]
    logits_noctx: [seq_len, vocab_size]
    
    Returns: list of CIG values per token
    """
    # Convert logits → probabilities
    probs_ctx = F.softmax(logits_ctx, dim=-1)
    probs_noctx = F.softmax(logits_noctx, dim=-1)
    
    # Take the probability of each predicted token
    pred_tokens = torch.argmax(probs_ctx, dim=-1)  # predicted tokens by model
    
    cig_scores = []
    for t, token_id in enumerate(pred_tokens):
        p_ctx = probs_ctx[t, token_id].item()
        p_noctx = probs_noctx[t, token_id].item()
        # Avoid log(0)
        p_ctx = max(p_ctx, 1e-12)
        p_noctx = max(p_noctx, 1e-12)
        cig = math.log(p_ctx) - math.log(p_noctx)
        cig_scores.append(cig)
    return cig_scores, pred_tokens.tolist()

def save_token_level_csv(prompt, context, cig_scores, pred_tokens, labels=None, filename="results/token_level.csv"):
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
        logits_ctx = get_logits(prompt, tokenizer, model, context)
        logits_noctx = get_logits(prompt, tokenizer, model)
        
        # Step 2: Compute CIG
        cig_scores, pred_tokens = compute_cig(logits_ctx[0], logits_noctx[0])  # logits shape: [1, seq_len, vocab_size]
        
        # Step 3: Save CSV per sample
        sample_filename = f"results/token_level_sample_{i}.csv"
        labels_list = [label]*len(pred_tokens) if label is not None else None
        save_token_level_csv(prompt, context, cig_scores, pred_tokens, labels_list, filename=sample_filename)
        
        all_results.append({
            "prompt": prompt,
            "context": context,
            "sample_csv": sample_filename
        })
    
    # Optionally save summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("results/summary.csv", index=False)
    print("Summary CSV saved to results/summary.csv")