# src/baselines.py
import numpy as np
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
import torch

def entropy_only_hallucination_score(H_with_ctx: np.ndarray) -> np.ndarray:
    return H_with_ctx

class SelfCheckBaseline:
    def __init__(self):
        self.checker = SelfCheckBERTScore(rescale_with_baseline=True)
        print("SelfCheck-BERTScore initialized")

    def generate_samples(self, model, tokenizer, prompt: str,
                         num_samples: int = 5, max_new_tokens: int = 100,
                         min_new_tokens: int = 0, device: str = "mps"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            samples.append(decoded.strip())
        return samples

    def score(self, sentences: list, sampled_passages: list) -> np.ndarray:
        """Returns np.array of shape [n_sentences] with hallucination scores."""
        # Safety: Ensure sentences isn't empty and has valid strings
        processed_sentences = [s.strip() if (s and s.strip()) else "." for s in sentences]
        
        if not sampled_passages:
            return np.zeros(len(processed_sentences))

        try:
            # predict() returns the BERTScore 'sent_scores'
            scores = self.checker.predict(
                sentences=processed_sentences,
                sampled_passages=sampled_passages
            )
            
            # SelfCheckBERTScore usually returns a list or a 1D array of scores per sentence
            if scores is not None:
                return np.array(scores).flatten()
            return np.zeros(len(processed_sentences))

        except Exception as e:
            # If the internal library splitter still fails, return zeros
            print(f"  SelfCheck scorer error: {e}")
            return np.zeros(len(processed_sentences))