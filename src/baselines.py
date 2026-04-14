# =============================================================================
# src/baselines.py
# CS F429 — Track A: Dynamic Uncertainty-Aware Attribution
#
# PURPOSE: Implements the two baselines required by the rubric.
#   Baseline 1 — Entropy-only: raw entropy H(t_i | query, context)
#                No comparison to without-context entropy.
#                Simplest possible hallucination signal.
#   Baseline 2 — SelfCheckGPT: generates N stochastic samples and measures
#                consistency. Inconsistent = hallucinated.
#                Reference: Manakul et al., EMNLP 2023.
#
# WHY TWO BASELINES:
#   The rubric requires both baselines to be beaten by the composite.
#   Beating entropy-only shows our metrics add value beyond simple uncertainty.
#   Beating SelfCheckGPT shows we improve on the published SOTA baseline.
# =============================================================================

import os
import logging
import numpy as np
import torch

# ── Suppress logging BEFORE any imports that trigger model loading ────────────
# These environment variables and log level settings prevent transformers,
# bert_score, and sentence_transformers from printing progress bars and
# info messages during weight loading. Must be set before the imports below.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import transformers
transformers.utils.logging.set_verbosity_error()

# ── Patch tqdm to silence SelfCheckGPT's internal progress bars ──────────────
# selfcheckgpt uses tqdm internally when calling bert_score.
# We replace tqdm with a version that has disable=True by default.
# This must happen BEFORE importing selfcheckgpt.
from tqdm import tqdm as _original_tqdm
import tqdm as _tqdm_module
_tqdm_module.tqdm = lambda *a, **kw: _original_tqdm(*a, **{**kw, "disable": True})

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore


def entropy_only_hallucination_score(H_with_ctx: np.ndarray) -> np.ndarray:
    """
    BASELINE 1: Entropy-only

    Simply returns H(t_i | query, context) as the hallucination score.
    No subtraction, no context comparison — just raw predictive entropy.

    Higher entropy → model is more uncertain → token may be hallucinated.

    This is the weakest baseline: it doesn't use the retrieved context at all
    to determine whether the model was grounded. It is included to show that
    our information-theoretic metrics (which DO compare against context) provide
    additional discriminative power beyond raw uncertainty.
    """
    return H_with_ctx  # used directly as the hallucination score


class SelfCheckBaseline:
    """
    BASELINE 2: SelfCheckGPT (BERTScore variant)

    HOW IT WORKS:
      1. Generate N stochastic completions from the same model with temperature > 0.
         These are sampled passages that represent what the model "thinks" the answer is.
      2. Score the original response sentence against these sampled passages using
         BERTScore (contextual embedding similarity).
      3. Low consistency between the original response and the samples → hallucination.

    WHY BERTSCORE VARIANT:
      BERTScore uses BERT embeddings to compare sentences semantically rather than
      at the word level. This is more robust to paraphrasing — "Paris is the capital"
      and "The capital is Paris" would score as consistent.

    REFERENCE: Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box
               Hallucination Detection for Generative LLMs. EMNLP.
    """

    def __init__(self):
        """
        Initialises the BERTScore checker and warms it up.

        WARMUP REASON:
          bert_score loads bert-base-uncased lazily — it reloads the weights
          from disk on every call to .predict() unless we force it to cache
          the model. The warmup call forces the cache to be populated, so
          subsequent calls reuse the in-memory model instead of hitting disk.
          Without warmup, you see "Loading weights: 100%|..." printed hundreds
          of times during a 150-sample run.

        BERT MODEL REFERENCE:
          We hold self._bert_model directly to prevent Python's garbage
          collector from evicting bert-base-uncased between predict() calls.
          bert_score stores the model in a module-level cache that can be GC'd.
        """
        self.checker = SelfCheckBERTScore(rescale_with_baseline=True)

        # Warmup: force bert-base-uncased to load and stay in memory
        try:
            self.checker.predict(
                sentences=["Warming up."],
                sampled_passages=["This warms up the scorer.", "Second warmup passage here."]
            )
        except Exception:
            pass  # warmup failure is non-fatal — scorer will still work

        # Hold a direct reference to prevent garbage collection
        try:
            from bert_score.utils import get_model, get_tokenizer
            self._bert_model     = get_model("bert-base-uncased", num_layers=9, all_layers=False)
            self._bert_tokenizer = get_tokenizer("bert-base-uncased", use_fast=False)
            self._bert_model.eval()
        except Exception:
            self._bert_model = None  # fallback: scorer will reload as needed

        print("SelfCheck-BERTScore ready.")

    def generate_samples(self, model, tokenizer, prompt: str,
                         num_samples: int = 3, max_new_tokens: int = 20,
                         device: str = "mps") -> list:
        """
        Generates N stochastic completions of the prompt using temperature sampling.

        These samples represent the model's "distribution of beliefs" about the answer.
        If the model is confident and grounded, all samples will be similar.
        If the model is hallucinating, samples will be inconsistent with each other.

        PARAMETERS:
          num_samples=3: we use 3 instead of the original paper's 5-20 for speed.
                         3 samples still provides a meaningful consistency signal.
          max_new_tokens=20: short generations — enough to capture the key answer
                             tokens without spending too much compute time.
          temperature=1.0: full temperature for diversity. Lower temperature would
                           make all samples identical (defeating the purpose).
          top_p=0.9: nucleus sampling — prevents very low-probability tokens
                     while still allowing diverse outputs.

        MINIMUM LENGTH GUARD:
          SelfCheckGPT internally uses NLTK's sent_tokenize() on each sampled passage.
          If a passage is fewer than 5 words, NLTK may return an empty list, which
          causes a "list index out of range" crash inside bert_score. We pad short
          outputs to ensure NLTK always finds at least one sentence.
        """
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1500  # leave room for the generated tokens
        ).to(device)

        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,      # stochastic sampling (not greedy)
                    temperature=1.0,     # full diversity
                    top_p=0.9,           # nucleus sampling
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],  # slice off the prompt
                skip_special_tokens=True
            ).strip()

            # Minimum length guard for NLTK's sentence tokeniser
            if len(decoded.split()) >= 5:
                samples.append(decoded)
            else:
                samples.append(decoded + " The answer is based on the provided context.")

        if not samples:
            # Emergency fallback — should never be reached in practice
            samples = ["The answer could not be determined from the given context."]

        return samples

    def score(self, sentences: list, sampled_passages: list) -> np.ndarray:
        """
        Scores the consistency of sentences against sampled passages.

        Uses BERTScore to compare each sentence in the original response
        against all sampled passages. Low BERTScore = low consistency = hallucination.

        Args:
          sentences:       List of sentence strings from the original response.
                           We pass [full_response] as one sentence for simplicity.
          sampled_passages: List of full sampled completions from generate_samples().

        Returns:
          np.array of shape [len(sentences)] with hallucination scores.
          Higher score = more hallucinated (less consistent with samples).

        SAFETY:
          Empty sentences are replaced with "No content." to prevent bert_score
          from crashing on empty inputs.
        """
        # Replace empty strings with a safe placeholder
        clean_sentences = [
            s.strip() if (s and len(s.strip()) >= 3) else "No content."
            for s in sentences
        ]

        if not sampled_passages:
            return np.zeros(len(clean_sentences))

        try:
            scores = self.checker.predict(
                sentences=clean_sentences,
                sampled_passages=sampled_passages
            )
            if scores is not None:
                return np.array(scores).flatten()
            return np.zeros(len(clean_sentences))
        except Exception as e:
            print(f"  SelfCheck scorer error: {e}")
            return np.zeros(len(clean_sentences))