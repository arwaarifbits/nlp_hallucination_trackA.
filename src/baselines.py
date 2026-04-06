# src/baselines.py
import numpy as np
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import torch

def entropy_only_hallucination_score(H_with_ctx: np.ndarray) -> np.ndarray:
    """
    Baseline: raw entropy WITH context as hallucination signal.
    Higher entropy → more likely hallucinated.
    No comparison to without-context entropy.
    """
    return H_with_ctx  # used directly as the score


class SelfCheckBaseline:
    """
    SelfCheckGPT: generate N samples stochastically, check if they are
    consistent with each other. Inconsistent = hallucinated.
    """
    def __init__(self):
        self.checker = SelfCheckBERTScore(rescale_with_baseline=True)

    def generate_samples(self, model, tokenizer, prompt: str, 
                         num_samples: int = 5, max_new_tokens: int = 100,
                         device: str = "mps"):
        """Generate N diverse responses using temperature sampling."""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9
                )
            decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            samples.append(decoded)
        return samples

    def score(self, sentences: list, sampled_passages: list) -> np.ndarray:
        """
        sentences: list of sentence strings from the primary response
        sampled_passages: list of full stochastic sample strings
        Returns: array of per-sentence hallucination scores
        """
        scores = self.checker.predict(
            sentences=sentences,
            sampled_passages=sampled_passages
        )
        return np.array(scores)