# =============================================================================
# src/semantic_entropy.py
# CS F429 — Track A: Dynamic Uncertainty-Aware Attribution
#
# PURPOSE: Implements Semantic Entropy (SemEnt) — Metric 4.
#
# WHAT IS SEMANTIC ENTROPY:
#   Standard entropy measures token-level uncertainty. But two responses can
#   express the same meaning in different words, making them look "uncertain"
#   even though the model is actually confident about the answer semantics.
#   Semantic entropy clusters N stochastic completions by meaning (using NLI),
#   then computes Shannon entropy over the CLUSTERS rather than individual tokens.
#
# REFERENCE: Farquhar et al. (2024). Detecting Hallucinations in Large Language
#            Models Using Semantic Consistency. ICML.
#
# DESIGN: We reuse the same opt-1.3b model for generating completions.
#         The NLI model (cross-encoder/nli-MiniLM2-L6-H768) is a small
#         30M parameter model — fast enough to run on MPS/CPU.
# =============================================================================

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class SemanticEntropyMetric:
    """
    Computes semantic entropy for a (query, context) pair.

    Returns a SINGLE float per response (not per token).
    In collect_all_metrics(), this scalar is broadcast to all token positions
    as a constant array: sem_arr = np.full(min_len, sem_ent_val).

    Small Gaussian noise (1e-6) is added to break the zero-variance problem:
    build_composite() drops metrics with zero variance because a constant
    array provides no discrimination signal.
    """

    def __init__(self, gen_model, gen_tokenizer, device="mps"):
        """
        Args:
          gen_model:     The language model (opt-1.3b) — shared with InformationGainMetric
          gen_tokenizer: Tokenizer for gen_model
          device:        "cuda", "mps", or "cpu"

        The NLI pipeline is loaded here. MiniLM is fast (~30ms per pair on CPU).
        We pass device= directly to the pipeline for GPU/MPS acceleration.
        """
        self.model     = gen_model
        self.tokenizer = gen_tokenizer
        self.device    = device

        print(f"Loading NLI model for semantic entropy onto {self.device}...")
        self.nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            device=self.device
        )

    def _sample_completions(self, prompt: str, num_samples: int = 5,
                             max_new_tokens: int = 80) -> list:
        """
        Generates num_samples stochastic completions from the model.

        These represent the model's "distribution of possible answers".
        Temperature=1.0 ensures diversity — lower temperatures collapse
        the samples toward the greedy output, defeating the purpose.

        Used by compute_semantic_entropy() when NLI clustering is active.
        NOTE: Currently not called — the simplified version below is used instead.
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1800
        ).to(self.device)

        completions = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            decoded = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            completions.append(decoded.strip())
        return completions

    def _bidirectional_entailment(self, s1: str, s2: str) -> bool:
        """
        Returns True if s1 and s2 BOTH entail each other — same semantic cluster.

        WHY BIDIRECTIONAL:
          NLI entailment is directional: "Paris is in France" entails "Paris is in Europe",
          but not vice versa. For semantic clustering, we want MUTUAL entailment —
          both sentences express the same core fact.

        Used by _cluster_completions() for NLI-based grouping.
        NOTE: Currently not called — simplified string matching is used instead
        for speed (see compute_semantic_entropy below).
        """
        if not s1.strip() or not s2.strip():
            return False
        try:
            r1 = self.nli(f"{s1} [SEP] {s2}", truncation=True)[0]
            r2 = self.nli(f"{s2} [SEP] {s1}", truncation=True)[0]
            return r1["label"] == "ENTAILMENT" and r2["label"] == "ENTAILMENT"
        except Exception:
            return False

    def _cluster_completions(self, completions: list) -> list:
        """
        Greedy clustering: groups completions that bidirectionally entail each other.

        Algorithm:
          For each unassigned completion c_i:
            Start a new cluster containing c_i.
            For each subsequent unassigned c_j:
              If c_i and c_j mutually entail → add c_j to c_i's cluster.

        Returns list of clusters, where each cluster is a list of indices
        into the completions array.

        NOTE: Currently not called — see compute_semantic_entropy() below.
        """
        clusters = []
        assigned = [False] * len(completions)
        for i, c in enumerate(completions):
            if assigned[i]:
                continue
            cluster    = [i]
            assigned[i] = True
            for j in range(i + 1, len(completions)):
                if not assigned[j] and self._bidirectional_entailment(c, completions[j]):
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return clusters

    import numpy as np
    import torch

    def compute_semantic_entropy(self, query: str, context: str,
                                  num_samples: int = 2,
                                  temperature: float = 0.8) -> float:
        """
        METRIC 4: Semantic Entropy

        SIMPLIFIED IMPLEMENTATION (for speed):
          Rather than full NLI clustering, we use normalised string matching:
          two completions are in the same cluster if their alphanumeric content
          (lowercased, punctuation stripped) is identical.

        WHY SIMPLIFIED:
          The full NLI approach (bidirectional entailment) adds ~500ms per sample.
          With num_samples=2 and max_new_tokens=15, this version takes ~200ms.
          The simplified version captures most of the signal because opt-1.3b
          tends to either generate the same answer (low entropy, grounded) or
          diverse answers (high entropy, hallucinating).

        HOW SEMANTIC ENTROPY IS COMPUTED:
          1. Generate num_samples stochastic completions.
          2. Normalise each completion (lowercase, alphanumeric only).
          3. Count unique normalised responses → these are the "semantic clusters".
          4. Compute Shannon entropy: H = -Σ p_i * log(p_i)
             where p_i = count_i / total_completions

        INTERPRETATION:
          High entropy → model generates diverse, inconsistent answers
                        → unsure about the answer → hallucination risk
          Low entropy  → all completions say the same thing
                        → model is confident → likely faithful

        Returns:
          float — semantic entropy for this (query, context) pair.
          This is broadcast to all token positions in collect_all_metrics().
        """
        prompt    = f"Context: {context}\nQuestion: {query}\nAnswer:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate num_samples completions in one batch call.
        # num_return_sequences generates multiple outputs simultaneously,
        # which is faster than calling generate() num_samples times separately.
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=15,          # short — enough to capture the answer
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=temperature,    # 0.8 slightly reduces noise vs 1.0
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode each completion, stripping the prompt prefix
        generated_texts = [
            self.tokenizer.decode(
                o[input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()
            for o in outputs
        ]

        # Semantic grouping: normalise and count unique responses.
        # "filter(str.isalnum, ...)" removes all non-alphanumeric characters.
        # This means "Paris!" and "paris" and "Paris." all map to "paris".
        unique_responses = {}
        for text in generated_texts:
            norm_text = "".join(filter(str.isalnum, text.lower()))
            unique_responses[norm_text] = unique_responses.get(norm_text, 0) + 1

        # Shannon entropy over cluster probabilities
        counts = list(unique_responses.values())
        probs  = [c / sum(counts) for c in counts]
        s_ent  = -sum(p * np.log(p) for p in probs if p > 0)

        return float(s_ent)