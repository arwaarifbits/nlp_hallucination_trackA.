# =============================================================================
# src/utils.py
# CS F429 — Track A: Dynamic Uncertainty-Aware Attribution
#
# PURPOSE: Dataset loading and label alignment utilities.
#
# TWO DATASETS:
#   1. RAGTruth — span-level hallucination annotations
#      Each sample has: query, context, response, labels (list of char-offset dicts)
#      Labels store the EXACT character offsets of hallucinated spans + label_type.
#
#   2. HaluEval — response-level hallucination labels
#      Each sample has: question, knowledge, hallucinated_answer, right_answer
#      No span-level labels — entire response is either hallucinated or faithful.
#
# LABEL ALIGNMENT CHALLENGE:
#   Our metrics operate at the SUBWORD TOKEN level (BPE tokenisation).
#   RAGTruth labels are at the CHARACTER offset level.
#   align_labels_to_tokens() bridges this gap using offset_mapping from the tokenizer.
# =============================================================================

from datasets import Dataset
import numpy as np
import pickle, os
import pandas as pd
import ast


def load_ragtruth(max_samples=1252):
    """
    Loads and balances the RAGTruth dataset.

    BALANCING STRATEGY:
      RAGTruth has 1550 hallucinated and 2008 clean samples (unbalanced).
      We balance to ~50% hallucinated for unbiased AUROC estimation.
      An unbalanced dataset would make AUROC optimistic for the majority class.

    RESUME-AWARE LOADING:
      If a checkpoint exists, this function reads how many hallucinated and clean
      samples are already in the checkpoint, then fetches only the ADDITIONAL
      samples needed to reach the target (max_samples // 2 of each class).

      This implements the "extend" part of the checkpoint resume/extend system.
      By tracking current_hal and current_clean separately, we can add more
      data incrementally without re-processing existing samples.

    DATASET COLUMNS:
      - query:    The user's question (string)
      - context:  Retrieved document(s) (string)
      - response: The LLM's answer (string)
      - labels:   JSON-like string. Either "[]" (clean) or a list of dicts:
                  [{"start": 10, "end": 25, "label_type": "Evident Conflict", ...}]
      - model:    Which LLM generated this response (GPT-3.5, GPT-4, Llama-2-7b, etc.)
                  Used in E6 generator analysis.

    Args:
      max_samples: Target total samples (split 50/50 hallucinated/clean)

    Returns:
      HuggingFace Dataset object, shuffled with seed=42 in the calling code.
    """
    df_all = pd.read_csv("data/ragtruth/ragtruth.csv")

    # ── Count what's already in the checkpoint ────────────────────────────────
    # This allows incremental extension: if we have 100 samples and want 150,
    # we fetch 25 more hallucinated and 25 more clean (assuming 50/50 target).
    current_hal, current_clean = 0, 0
    checkpoint_path = "results/checkpoint_ragtruth.pkl"

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                ckpt_data = pickle.load(f)
            # Count sample-level labels (per_sample["labels"] is a list of arrays).
            # We count label arrays that are entirely 1 (hallucinated) or 0 (clean).
            if "per_sample" in ckpt_data and "labels" in ckpt_data["per_sample"]:
                labels      = ckpt_data["per_sample"]["labels"]
                current_hal   = sum(1 for l in labels if l == 1)
                current_clean = sum(1 for l in labels if l == 0)
        except Exception as e:
            print(f"  [Warning] Could not read checkpoint counts: {e}")

    # ── Identify hallucinated vs clean samples ────────────────────────────────
    def is_hal(val):
        """
        Returns True if this sample has at least one hallucination span.
        "[]" or NaN or "" → clean sample (no spans).
        A non-empty list of span dicts → hallucinated sample.
        """
        if pd.isna(val) or val == "[]" or val == "":
            return False
        try:
            parsed = ast.literal_eval(val)
            return len(parsed) > 0
        except:
            return False

    hal_mask = df_all['labels'].apply(is_hal)

    print(f"  [Data Setup] Total dataset: {len(df_all)} samples.")
    print(f"  [Data Setup] Natural distribution: {hal_mask.sum()} hallucinated, {len(df_all) - hal_mask.sum()} clean.")

    # ── Select new samples to add ─────────────────────────────────────────────
    half_target   = max_samples // 2
    needed_hal    = max(0, half_target - current_hal)
    needed_clean  = max(0, half_target - current_clean)

    print(f"--- Balancing RAGTruth to {max_samples} samples ---")
    print(f"Current: {current_hal} Hal / {current_clean} Clean")
    print(f"Fetching: {needed_hal} new Hal / {needed_clean} new Clean")

    # iloc slicing: skip the first current_hal rows (already processed)
    # and take the next needed_hal rows. This ensures we never process
    # the same sample twice, even across multiple resume/extend cycles.
    df_hal   = df_all[hal_mask].iloc[current_hal : current_hal + needed_hal]
    df_clean = df_all[~hal_mask].iloc[current_clean : current_clean + needed_clean]

    # Combine and shuffle for the current batch
    final_df = pd.concat([df_hal, df_clean]).sample(frac=1, random_state=42)
    return Dataset.from_pandas(final_df.reset_index(drop=True))


def load_halueval(max_samples=None):
    """
    Loads the HaluEval QA dataset.

    HaluEval STRUCTURE (per sample):
      - question:           The user's question
      - knowledge:          Retrieved context (like RAGTruth's 'context')
      - hallucinated_answer: A factually incorrect answer
      - right_answer:        The correct answer

    In collect_all_metrics(), each HaluEval sample produces TWO variations:
      (hallucinated_answer, label=1) and (right_answer, label=0).
    This is how we get response-level binary labels for HaluEval.

    NO BALANCING NEEDED: HaluEval has exactly 1 hallucinated and 1 faithful
    response per sample, so it's inherently balanced.
    """
    path = "data/halueval/halueval.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")
    df = pd.read_csv(path)
    if max_samples:
        df = df.head(max_samples)
    return Dataset.from_pandas(df)


def get_dataset_stats(ds, label_field="labels"):
    """
    Prints basic hallucination rate statistics for a dataset.
    Used for sanity-checking the dataset balance before running experiments.
    """
    all_labels = []
    for sample in ds:
        label_val = sample[label_field]
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


def align_labels_to_tokens(response_text, labels_input, tokenizer) -> np.ndarray:
    """
    Maps hallucination labels to subword token positions.

    THIS IS THE MOST CRITICAL DATA PREPARATION FUNCTION.
    All metric computations produce token-level scores. The evaluation
    (AUROC, F1) requires token-level binary labels. This function creates
    those labels by mapping the original annotations to token positions.

    HANDLES TWO CASES:

    Case A — RAGTruth: character-offset spans
      labels_input = "[{'start': 10, 'end': 25, 'label_type': 'Evident Conflict'}]"
      Strategy: use tokenizer's offset_mapping to find which tokens overlap
      with each annotated character span.
      Token is marked 1 if ANY character in the token falls within a span.

    Case B — HaluEval: scalar integer
      labels_input = 0 (faithful) or 1 (hallucinated)
      Strategy: broadcast the scalar to all token positions.
      Every token in the response gets the same label.

    HOW OFFSET MAPPING WORKS:
      tokenizer(..., return_offsets_mapping=True) returns alongside input_ids
      a list of (start_char, end_char) tuples for each token.
      Token i spans characters [tok_s, tok_e) in the original text.
      We check overlap: if tok_s < span_end AND tok_e > span_start, mark token.

    Args:
      response_text: The raw response string
      labels_input:  Either a JSON string (RAGTruth) or scalar 0/1 (HaluEval)
      tokenizer:     The same tokenizer used in InformationGainMetric

    Returns:
      np.array of shape [n_tokens] with binary labels (0 = faithful, 1 = hallucinated)
    """
    # Tokenise with character offset mapping
    encoding = tokenizer(
        response_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=1800
    )
    offsets       = encoding.get('offset_mapping', [])
    n_tokens      = len(offsets)
    token_labels  = np.zeros(n_tokens, dtype=int)  # default: all faithful

    if n_tokens == 0:
        return token_labels

    # ── Parse labels_input ────────────────────────────────────────────────────
    if isinstance(labels_input, str):
        try:
            if labels_input.strip() in ["[]", "", "nan", "NaN"]:
                return token_labels  # clean sample — all zeros
            actual_labels = ast.literal_eval(labels_input)
        except (ValueError, SyntaxError):
            return token_labels  # unparseable — treat as clean
    else:
        actual_labels = labels_input

    # ── Case A: RAGTruth character-offset spans ───────────────────────────────
    if isinstance(actual_labels, list) and len(actual_labels) > 0:
        if isinstance(actual_labels[0], dict):
            for entry in actual_labels:
                # Handle different versions of RAGTruth key names
                s = entry.get('start', entry.get('char_start'))
                e = entry.get('end',   entry.get('char_end'))
                if s is None or e is None:
                    continue
                s, e = int(s), int(e)
                for i, (tok_s, tok_e) in enumerate(offsets):
                    # Overlap check: token [tok_s, tok_e) overlaps span [s, e)
                    # iff tok_s < e AND tok_e > s
                    if tok_s < e and tok_e > s:
                        token_labels[i] = 1
            return token_labels

    # ── Case B: HaluEval numeric labels ──────────────────────────────────────
    try:
        labels_arr = np.array(labels_input)
        if labels_arr.dtype.kind in 'if':  # numeric types only
            unique = np.unique(labels_arr)

            # Scalar 0 or 1: broadcast to all tokens
            if len(unique) <= 1:
                val = int(unique[0]) if len(unique) == 1 else 0
                return np.full(n_tokens, val, dtype=int)

            # Word-level labels: map words to tokens via character offsets
            # (not used in current implementation but handles edge cases)
            words = response_text.split()
            if len(words) == len(labels_arr):
                char_to_word = {}
                curr_pos = 0
                for w_idx, word in enumerate(words):
                    start = response_text.find(word, curr_pos)
                    if start == -1:
                        continue
                    for c in range(start, start + len(word)):
                        char_to_word[c] = w_idx
                    curr_pos = start + len(word)

                for i, (tok_s, tok_e) in enumerate(offsets):
                    for c in range(tok_s, tok_e):
                        if c in char_to_word:
                            token_labels[i] = int(labels_arr[char_to_word[c]])
                            if token_labels[i] == 1:
                                break
                return token_labels

            # Fallback: if word count doesn't match, use mean threshold
            return np.full(n_tokens, int(np.mean(labels_arr) > 0.5), dtype=int)

    except (ValueError, TypeError):
        pass

    # ── Case C: Plain scalar ──────────────────────────────────────────────────
    if isinstance(labels_input, (int, float, np.integer, np.floating)):
        return np.full(n_tokens, int(labels_input), dtype=int)

    return token_labels  # default: all zeros if nothing matched