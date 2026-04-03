import torch
import torch.nn.functional as F
import pandas as pd
from src.model_utils import load_model, get_logits
import os
import math
import ast
import json

def compute_cig(logits_ctx, logits_noctx):
    """
    Compute token-level Contextual Information Gain (CIG)
    Equation: CIG = log P(y|ctx) - log P(y|no_ctx)
    """
    probs_ctx = F.softmax(logits_ctx, dim=-1)
    probs_noctx = F.softmax(logits_noctx, dim=-1)

    pred_tokens = torch.argmax(probs_ctx, dim=-1) 

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
    tokens = tokenizer.convert_ids_to_tokens(pred_tokens)
    min_len = min(len(tokens), len(cig_scores))
    
    data = {
        "token": tokens[:min_len],
        "cig_score": cig_scores[:min_len]
    }
    
    if labels is not None:
        data["hallucination_label"] = labels[:min_len]
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Saved {len(df)} tokens to {filename}")

def run_cig_in_batches(df, dataset_name=None, batch_size=10, prompt_col="prompt", context_col="context", label_col="hallucination_labels"):
    tokenizer, model = load_model()
    all_results = []
    n_batches = math.ceil(len(df) / batch_size)
    
    for b in range(n_batches):
        batch_df = df.iloc[b*batch_size : (b+1)*batch_size]
        print(f"Processing batch {b+1}/{n_batches}")
        
        for i, row in batch_df.iterrows():
            # --- AUTO-MAPPING LOGIC FOR HALUEVAL ---
            # If the user didn't rename columns, we find them automatically
            current_answer = row.get("model_answer", row.get("answer", None))
            current_labels = row.get(label_col, row.get("labels", None))
            current_context = row.get(context_col, row.get("original_prompt", row.get("context", "")))
            
            if pd.notna(current_answer):
                answer_text = str(current_answer)
            else:
                print(f"[WARNING] No answer found for sample {i}. Generating...")
                answer_text = generate_answer(row[prompt_col], current_context, tokenizer, model)

            # Compute Logits
            logits_ctx = get_logits(answer_text, tokenizer, model, current_context)
            logits_noctx = get_logits(answer_text, tokenizer, model)
            
            cig_scores, pred_tokens = compute_cig(logits_ctx[0], logits_noctx[0])

            # Label Parsing
            label = None
            if isinstance(current_labels, str):
                try:
                    label = ast.literal_eval(current_labels)
                except:
                    label = 1 if str(current_labels).lower() in ['hallucinated', '1'] else 0
            else:
                label = current_labels

            # Save per-sample results
            os.makedirs(f"results/{dataset_name}", exist_ok=True)
            sample_filename = f"results/{dataset_name}/token_level_sample_{i}.csv"
            
            labels_list = None
            if label is not None:
                if isinstance(label, list): 
                    encoding = tokenizer(answer_text, return_offsets_mapping=True, return_tensors="pt")
                    offsets = encoding["offset_mapping"][0].tolist()
                    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
                    labels_list = convert_spans_to_token_labels(offsets, label, tokens)
                else:
                    labels_list = [label] * len(pred_tokens)

            save_token_level_csv(tokenizer, cig_scores, pred_tokens, labels_list, filename=sample_filename)

            all_results.append({
                "index": i,
                "prompt": str(row.get(prompt_col, ""))[:100],
                "sample_csv": sample_filename
            })
    
    summary_df = pd.DataFrame(all_results)
    summary_path = f"results/{dataset_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Workflow Complete. Summary: {summary_path}")

def convert_spans_to_token_labels(offsets, spans, tokens=None):
    if isinstance(spans, str):
        if spans.lower() == 'hallucinated': return [1] * len(offsets)
        if spans == "[]" or spans.lower() == 'faithful': return [0] * len(offsets)
        
        try:
            safe_ns = {"false": False, "true": True, "null": None}
            try:
                spans = ast.literal_eval(spans)
            except:
                spans = eval(spans, {"__builtins__": {}}, safe_ns)
        except:
            try: spans = json.loads(spans)
            except: return [0] * len(offsets)

    token_labels = []
    for i, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            token_labels.append(0)
            continue
            
        label = 0
        if isinstance(spans, list):
            for span in spans:
                s_start = span.get('start', 0)
                s_end = span.get('end', 0)
                if not (end <= s_start or start >= s_end):
                    label = 1
                    break
        token_labels.append(label)
    return token_labels

def generate_answer(prompt, context, tokenizer, model, max_new_tokens=100):
    input_text = (str(context) + "\n" if context else "") + str(prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text[len(input_text):].strip() if generated_text.startswith(input_text) else generated_text