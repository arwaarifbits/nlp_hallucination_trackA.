import torch
import torch.nn.functional as F
import pandas as pd
from typer import prompt
from src.model_utils import load_model, get_logits
import os
import math
import ast


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

        # 🔥 DEBUG HERE
        if t < 5:
            print(f"[DEBUG] Token {t}: p_ctx={p_ctx:.6f}, p_noctx={p_noctx:.6f}, CIG={cig.item():.4f}")

    return cig_scores, pred_tokens[:seq_len]

def save_token_level_csv(tokenizer, cig_scores, pred_tokens, labels=None, filename="results/token_level.csv"):
    tokens = tokenizer.convert_ids_to_tokens(pred_tokens)
    
    min_len = min(len(tokens), len(cig_scores))
    if labels is not None:
        min_len = min(min_len, len(labels))
    
    data = {
        "token": tokens[:min_len],
        "cig_score": cig_scores[:min_len]
    }
    
    # Using 'is not None' ensures that 0 is saved correctly
    if labels is not None:
        data["hallucination_label"] = labels[:min_len]
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"[SUCCESS] Saved {len(df)} tokens to {filename}")


def run_cig_on_dataset(df, dataset_name=None, context_col="context", prompt_col="prompt", label_col="label_hallucination"):
    """
    Run CIG computation for entire dataset.
    For HaluEval, original_prompt is used as context.
    """
    tokenizer, model = load_model()
    
    all_results = []

    for i, row in df.iterrows():
        prompt = row[prompt_col]
        # Use original_prompt for HaluEval
        if dataset_name == "halueval":
            context = row.get("original_prompt", None)
        else:
            context = row.get(context_col, None)
        
        #label extraction
        raw_label = row.get(label_col, None)
        if isinstance(raw_label, str):
            try:
                label = ast.literal_eval(raw_label)
            except:
                label = None
        else:
            label = raw_label
        
        # Step 1: Generate answer using context
        answer_text = generate_answer(prompt, context, tokenizer, model)

        if len(answer_text.strip()) == 0:
            print(f"[WARNING] Empty generation at sample {i}")
            continue

        # Step 2: Compute logits on GENERATED answer
        logits_ctx = get_logits(answer_text, tokenizer, model, context)
        logits_noctx = get_logits(answer_text, tokenizer, model)

        # Step 3: Compute CIG
        cig_scores, pred_tokens = compute_cig(logits_ctx[0], logits_noctx[0])
        
        # Step 4: Save CSV per sample
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
    summary_df.to_csv(f"results/{dataset_name}_summary.csv", index=False)
    print(f"Summary CSV saved to results/{dataset_name}_summary.csv")

def run_cig_in_batches(df, dataset_name=None, batch_size=10, prompt_col="prompt", context_col="context", label_col="label_binary"):
    tokenizer, model = load_model()
    all_results = []
    n_batches = math.ceil(len(df) / batch_size)
    
    for b in range(n_batches):
        batch_df = df.iloc[b*batch_size : (b+1)*batch_size]
        print(f"Processing batch {b+1}/{n_batches}")
        
        for i, row in batch_df.iterrows():
            prompt = row[prompt_col]
            
            # THE SMART CONTEXT PICKER
            if dataset_name == "halueval":
                context = row.get("original_prompt", None)
            else:
                context = row.get(context_col, None)
                if context is None and "context" in row:
                    context = row["context"]

            # GENERATION
            answer_text = generate_answer(prompt, context, tokenizer, model, dataset_name=dataset_name)
            if not answer_text or len(answer_text.strip()) == 0:
                answer_text = prompt if prompt else "N/A"

            # LOGITS & CIG
            logits_ctx = get_logits(answer_text, tokenizer, model, context)
            logits_noctx = get_logits(answer_text, tokenizer, model)
            cig_scores, pred_tokens = compute_cig(logits_ctx[0], logits_noctx[0])

            # LABELS
            raw_label = row.get(label_col, None)
            # Add logic here to ensure label is never None if it exists in the row
            label = raw_label if raw_label is not None else None

            # FOLDER & FILE PATH
            os.makedirs(f"results/{dataset_name}", exist_ok=True)
            sample_filename = f"results/{dataset_name}/token_level_sample_{i}.csv"
            
            # PROCESS LABELS (Span vs Binary)
            labels_list = None
            if label is not None:
                if isinstance(label, list): 
                    encoding = tokenizer(answer_text, return_offsets_mapping=True, return_tensors="pt")
                    offsets = encoding["offset_mapping"][0].tolist()
                    labels_list = convert_spans_to_token_labels(offsets, label)
                else:
                    labels_list = [label] * len(pred_tokens)

            # SAVE
            save_token_level_csv(tokenizer, cig_scores, pred_tokens, labels_list, filename=sample_filename)

            all_results.append({
                "prompt": prompt[:100],
                "sample_csv": sample_filename
            })
    
    # FINAL SUMMARY
    summary_df = pd.DataFrame(all_results)
    summary_path = f"results/{dataset_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Workflow Complete. Summary saved to {summary_path}")



def generate_answer(prompt, context, tokenizer, model, max_new_tokens=100, dataset_name=None):
    """
    Generate answer using model.
    For HaluEval, automatically splits 'Knowledge:' from the question.
    """
    if dataset_name == "halueval" and context:
        # Split context into knowledge and question
        if context.startswith("Knowledge:"):
            split_pos = context.rfind("Question:")
            if split_pos != -1:
                knowledge = context[len("Knowledge:"):split_pos].strip()
                input_text = knowledge + "\n" + prompt
            else:
                input_text = context + "\n" + prompt
        else:
            input_text = context + "\n" + prompt
    else:
        input_text = (context + "\n" if context else "") + prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the generated part
    if generated_text.startswith(input_text):
        answer_text = generated_text[len(input_text):]
    else:
        answer_text = generated_text

    return answer_text.strip()


def convert_spans_to_token_labels(offsets, spans):
    """
    Convert character-level spans to token-level labels
    offsets: list of (start, end) for each token
    spans: list of dicts with 'start', 'end'
    """
    token_labels = []

    for (token_start, token_end) in offsets:
        label = 0

        for span in spans:
            span_start = span['start']
            span_end = span['end']

            # Check overlap
            if not (token_end <= span_start or token_start >= span_end):
                label = 1
                break

        token_labels.append(label)

    return token_labels