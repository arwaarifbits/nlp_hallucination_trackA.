# demo.py — run this live during the midsem
import sys
from metric import InformationGainMetric
from semantic_entropy import SemanticEntropyMetric
from composite import build_composite, normalize_score
import numpy as np

def demo(query: str, context: str, response: str):
    print("\nLoading model...")
    m = InformationGainMetric("facebook/opt-1.3b")
    s = SemanticEntropyMetric(m.model, m.tokenizer, device=m.device)

    print("Computing metrics...")
    ig,  H_no, H_with = m.compute_information_gain(query, context, response)
    kl  = m.compute_kl_divergence(query, context, response)
    conf = m.compute_confidence_drop(query, context, response)
    sem  = s.compute_semantic_entropy(query, context)

    min_len = min(len(ig), len(kl), len(conf))
    tokens = m.get_response_tokens(response)[:min_len]

    metrics = {
        "IG":      -ig[:min_len],
        "KL":      -kl[:min_len],
        "ConfDrop": conf[:min_len],
        "SemEnt":   np.full(min_len, sem),
        "EntOnly":  H_with[:min_len],
    }
    composite = build_composite(metrics, np.zeros(min_len), mode="equal_weight")

    print("\n{'Token':<20} {'IG':>8} {'KL':>8} {'ConfDrop':>10} {'Composite':>10}")
    print("-" * 60)
    for i, tok in enumerate(tokens):
        flag = " <-- HALLUCINATION RISK" if composite[i] > 0.6 else ""
        print(f"{tok:<20} {metrics['IG'][i]:>8.4f} {metrics['KL'][i]:>8.4f} "
              f"{metrics['ConfDrop'][i]:>10.4f} {composite[i]:>10.4f}{flag}")

if __name__ == "__main__":
    # Default test — replace with evaluator's input
    query   = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    context = sys.argv[2] if len(sys.argv) > 2 else "France is a country in Europe. Its capital is Paris."
    response = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is Lyon, a major city in the south."
    demo(query, context, response)