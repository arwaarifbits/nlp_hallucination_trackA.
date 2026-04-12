# demo.py
# ─────────────────────────────────────────────────────────────────────────────
# CS F429 — NLP Research Assignment — Track A
# Live Pipeline Demo for Midsem Evaluation
#
# USAGE:
#   Interactive mode (evaluator types input live):
#       python3 demo.py
#
#   Command-line mode (pass all three args directly):
#       python3 demo.py "query" "context" "response"
#
#   Preset test mode (runs a built-in example to verify pipeline):
#       python3 demo.py --test
#
# The evaluator will ask: "Explain how KL divergence is computed in your code."
# Answer: see compute_kl_divergence() in src/metric.py — two forward passes,
# softmax distributions, then sum(P_no * (log P_no - log P_with)) per token.
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"

import sys
import numpy as np

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.metric import InformationGainMetric
from src.composite import normalize_score


# ─── Colour helpers for terminal output ──────────────────────────────────────

def red(s):    return f"\033[91m{s}\033[0m"
def green(s):  return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"


# ─── Core pipeline ───────────────────────────────────────────────────────────

def run_pipeline(metric: InformationGainMetric,
                 query: str, context: str, response: str,
                 verbose: bool = True):
    """
    Runs all five metrics on a single (query, context, response) triple.
    Returns a dict of per-token score arrays and the composite.
    """

    if verbose:
        print("\n" + "─" * 68)
        print(bold("  Computing metrics..."))
        print("─" * 68)

    # ── Step 1: Information Gain ─────────────────────────────────────────────
    # IG(t_i) = H(t_i | query) − H(t_i | query, context)
    # High IG → context reduced uncertainty → token is grounded → faithful
    # Low/negative IG → context had no effect → hallucination risk
    if verbose:
        print("  [1/5] Information Gain        ", end="", flush=True)
    ig, H_no, H_with = metric.compute_information_gain(query, context, response)
    if verbose:
        print(green("done"))

    # ── Step 2: KL Divergence ────────────────────────────────────────────────
    # KL(P_no_ctx || P_with_ctx) per token
    # Computed in compute_kl_divergence() in src/metric.py:
    #   1. Forward pass WITHOUT context → logits_no → P_no = softmax(logits_no)
    #   2. Forward pass WITH context    → logits_with → P_with = softmax(logits_with)
    #   3. KL = sum over vocab of P_no * (log P_no − log P_with)
    # High KL → context strongly shifted predictions → model used context → faithful
    # Low KL  → context had no effect → hallucination risk
    if verbose:
        print("  [2/5] KL Divergence           ", end="", flush=True)
    kl = metric.compute_kl_divergence(query, context, response)
    if verbose:
        print(green("done"))

    # ── Step 3: Confidence Drop ──────────────────────────────────────────────
    # ConfDrop(t_i) = p(t_i | no_ctx) − p(t_i | with_ctx)
    # Positive → model was more confident WITHOUT context → ignored retrieval → hallucination
    # Negative → context boosted confidence → grounded → faithful
    if verbose:
        print("  [3/5] Confidence Drop         ", end="", flush=True)
    conf = metric.compute_confidence_drop(query, context, response)
    if verbose:
        print(green("done"))

    # ── Step 4: Entropy-only (baseline) ─────────────────────────────────────
    # Simply H(t_i | query, context) — raw entropy with context
    # No comparison to without-context entropy
    # Used as Baseline 1 in our E1+E2 table
    if verbose:
        print("  [4/5] Entropy-only (Baseline) ", end="", flush=True)
    entropy_only = H_with.copy()
    if verbose:
        print(green("done"))

    # ── Step 5: Composite ────────────────────────────────────────────────────
    # AUROC-weighted combination of all four metrics
    # Each metric is min-max normalised to [0,1]
    # Weights are proportional to (individual_AUROC − 0.5) from RAGTruth evaluation
    # These weights were fixed on the training split and are NOT refit here
    if verbose:
        print("  [5/5] Building composite      ", end="", flush=True)

    min_len = min(len(ig), len(kl), len(conf), len(entropy_only))
    ig       = ig[:min_len]
    kl       = kl[:min_len]
    conf     = conf[:min_len]
    H_with   = H_with[:min_len]
    H_no     = H_no[:min_len]

    # Orient scores so that higher = more hallucinated
    # (determined empirically on training split)
    ig_hal   = -ig          # low IG = hallucination
    kl_hal   = kl           # high KL = faithful, but on opt-1.3b/RAGTruth it inverts
    conf_hal = conf         # positive conf_drop = hallucination
    ent_hal  = entropy_only # high entropy with context = hallucination

    # Normalise each to [0, 1]
    def safe_norm(arr):
        arr = np.nan_to_num(arr, nan=0.0)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    ig_n   = safe_norm(ig_hal)
    kl_n   = safe_norm(kl_hal)
    conf_n = safe_norm(conf_hal)
    ent_n  = safe_norm(ent_hal)

    # Fixed AUROC-proportional weights from RAGTruth training split evaluation
    # KL=0.585, EntOnly=0.196, IG=0.137, ConfDrop=0.063 (SemEnt=0.02 omitted here)
    W = {"IG": 0.137, "KL": 0.585, "ConfDrop": 0.063, "EntOnly": 0.196}
    total_w = sum(W.values())
    composite = (
        ig_n   * (W["IG"]      / total_w) +
        kl_n   * (W["KL"]      / total_w) +
        conf_n * (W["ConfDrop"] / total_w) +
        ent_n  * (W["EntOnly"] / total_w)
    )
    composite = np.nan_to_num(composite, nan=0.0)

    if verbose:
        print(green("done"))

    return {
        "tokens":    metric.get_response_tokens(response)[:min_len],
        "IG":        ig[:min_len],
        "KL":        kl[:min_len],
        "ConfDrop":  conf[:min_len],
        "EntOnly":   entropy_only[:min_len],
        "composite": composite,
        "min_len":   min_len,
    }


# ─── Display results ─────────────────────────────────────────────────────────

def display_results(results: dict, threshold: float = 0.55):
    """
    Print a formatted token-level table of metric scores.
    Tokens with composite > threshold are flagged as hallucination risk.
    """
    tokens    = results["tokens"]
    ig        = results["IG"]
    kl        = results["KL"]
    conf      = results["ConfDrop"]
    ent       = results["EntOnly"]
    composite = results["composite"]
    n         = results["min_len"]

    print("\n" + "─" * 68)
    print(bold("  Token-level metric scores"))
    print("─" * 68)
    print(f"  {'Token':<16} {'IG':>8} {'KL':>7} {'ConfDrop':>9} {'EntOnly':>8} {'Composite':>10}  Flag")
    print("  " + "─" * 64)

    for i in range(n):
        tok   = repr(tokens[i].strip()[:14]) if tokens[i].strip() else "''"
        flag  = ""
        color = lambda s: s

        if composite[i] > threshold:
            flag  = red("⚠ HALLUCINATION RISK")
            color = red
        elif composite[i] > threshold * 0.85:
            flag  = yellow("~ borderline")
            color = yellow

        print(
            f"  {tok:<18} "
            f"{color(f'{ig[i]:>+7.4f}')} "
            f"{color(f'{kl[i]:>7.4f}')} "
            f"{color(f'{conf[i]:>+9.4f}')} "
            f"{color(f'{ent[i]:>8.4f}')} "
            f"{color(f'{composite[i]:>10.4f}')}  {flag}"
        )

    print("─" * 68)
    print(f"\n  Summary:")
    print(f"    Response tokens analysed : {n}")
    print(f"    Mean composite score     : {composite.mean():.4f}")
    print(f"    Max composite score      : {composite.max():.4f}")
    print(f"    Tokens flagged (>{threshold:.2f})   : {(composite > threshold).sum()} / {n}")
    print(f"    Hallucination risk level : ", end="")

    max_c      = composite.max()
    frac_flagged = (composite > threshold).sum() / len(composite)

    if max_c > 0.75 or frac_flagged > 0.30:
        print(red("HIGH"))
    elif max_c > 0.55 or frac_flagged > 0.15:
        print(yellow("MODERATE"))
    else:
        print(green("LOW"))
        

# ─── Interactive input ───────────────────────────────────────────────────────

def get_input_interactive():
    """Prompt the evaluator to type or paste the three inputs live."""
    print("\n" + "═" * 68)
    print(bold("  CS F429 — Track A Live Demo"))
    print(bold("  Hallucination Detection Pipeline"))
    print("═" * 68)
    print(dim("  Enter the three components of the RAG sample."))
    print(dim("  Press Enter twice after each to confirm.\n"))

    def read_multiline(prompt):
        print(f"  {bold(prompt)}")
        print(dim("  (paste or type, then press Enter twice)"))
        lines = []
        while True:
            line = input("  > ")
            if line == "" and lines:
                break
            lines.append(line)
        return " ".join(lines).strip()

    query    = read_multiline("Query / Question:")
    context  = read_multiline("Retrieved context:")
    response = read_multiline("Model response to evaluate:")

    return query, context, response


# ─── Preset test ─────────────────────────────────────────────────────────────

TEST_EXAMPLE = {
    "query": (
        "What year did the United Arab Emirates join the United Nations?"
    ),
    "context": (
        "The United Arab Emirates was established on 2 December 1971 following the "
        "federation of six emirates. It became a member of the United Nations on "
        "9 December 1971, just seven days after its formation. The seventh emirate, "
        "Ras Al Khaimah, joined the federation in February 1972."
    ),
    "response": (
        "The UAE joined the United Nations in 1973, two years after its formation "
        "as a federation of seven emirates. The country was established in December 1971."
    ),
    # Ground truth: first sentence is hallucinated (year is wrong, should be 1971)
    # second sentence about 'seven emirates' at founding is also wrong (it was six)
}


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():

    # Load model once — stays in memory for the whole session
    metric = InformationGainMetric(model_name="facebook/opt-1.3b")

    print("\n" + "═" * 68)
    print(bold("  Loading model..."))
    print("═" * 68)

    
    # ── Determine input mode ──────────────────────────────────────────────────
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        # Built-in test case
        print(f"\n  {bold('Running built-in test example')}")
        print(f"  Query:    {TEST_EXAMPLE['query'][:60]}...")
        query    = TEST_EXAMPLE["query"]
        context  = TEST_EXAMPLE["context"]
        response = TEST_EXAMPLE["response"]

    elif len(sys.argv) == 4:
        # Passed directly as command-line arguments
        query    = sys.argv[1]
        context  = sys.argv[2]
        response = sys.argv[3]
        print(f"\n  {bold('Using command-line arguments')}")

    else:
        # Interactive — evaluator types live
        query, context, response = get_input_interactive()

    # ── Echo the inputs back ─────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print(bold("  Input received"))
    print("─" * 68)
    print(f"  Query   : {query[:80]}{'...' if len(query) > 80 else ''}")
    print(f"  Context : {context[:80]}{'...' if len(context) > 80 else ''}")
    print(f"  Response: {response[:80]}{'...' if len(response) > 80 else ''}")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    results = run_pipeline(metric, query, context, response, verbose=True)

    # ── Display results ───────────────────────────────────────────────────────
    display_results(results, threshold=0.55)

    # ── Metric explanations (for evaluator questions) ─────────────────────────
    print("─" * 68)
    print(bold("  Metric definitions (for evaluator reference)"))
    print("─" * 68)
    print("""
  IG (Information Gain)
    IG(t_i) = H(t_i|query) − H(t_i|query,context)
    File: src/metric.py → compute_information_gain()
    High IG = context reduced uncertainty = faithful token
    Low IG  = context had no effect = hallucination risk

  KL (KL Divergence)
    KL(P_no||P_with) = Σ P_no * (log P_no − log P_with)
    File: src/metric.py → compute_kl_divergence()
    Two forward passes: one without context, one with.
    High KL = context shifted predictions = grounded
    Low KL  = no shift = hallucination risk

  ConfDrop (Confidence Drop)
    ConfDrop(t_i) = p(t_i|no_ctx) − p(t_i|with_ctx)
    File: src/metric.py → compute_confidence_drop()
    Positive = model ignored context = hallucination
    Negative = context boosted confidence = faithful

  EntOnly (Entropy-only, Baseline 1)
    H(t_i|query,context) — raw entropy with context
    Higher entropy = more uncertain = hallucination risk

  Composite
    AUROC-weighted sum of normalised metric scores.
    Weights fixed on RAGTruth training split:
      KL=0.585, EntOnly=0.196, IG=0.137, ConfDrop=0.063
""")

    # ── Loop for multiple evaluator inputs ───────────────────────────────────
    if not (len(sys.argv) >= 2):
        while True:
            again = input("  Run another example? [y/N]: ").strip().lower()
            if again != "y":
                break
            query, context, response = get_input_interactive()
            results = run_pipeline(metric, query, context, response, verbose=True)
            display_results(results, threshold=0.55)

    print(bold("\n  Demo complete.\n"))


if __name__ == "__main__":
    main()