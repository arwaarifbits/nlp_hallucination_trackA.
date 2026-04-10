# src/baselines.py
import os
import logging
import numpy as np
import torch

# Suppress ALL logging before any imports that trigger loading
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import transformers
transformers.utils.logging.set_verbosity_error()

# Patch tqdm BEFORE importing selfcheckgpt so its internal progress bars are silent
from tqdm import tqdm as _original_tqdm
import tqdm as _tqdm_module
_tqdm_module.tqdm = lambda *a, **kw: _original_tqdm(*a, **{**kw, "disable": True})

from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore


def entropy_only_hallucination_score(H_with_ctx: np.ndarray) -> np.ndarray:
    return H_with_ctx


class SelfCheckBaseline:
    def __init__(self):
        self.checker = SelfCheckBERTScore(rescale_with_baseline=True)
        
        # Warmup to force load
        try:
            self.checker.predict(
                sentences=["Warming up."],
                sampled_passages=["This warms up the scorer.", "Second warmup passage here."]
            )
        except Exception:
            pass

        # CRITICAL: Hold a direct reference to the internal scorer model
        # so Python's garbage collector does not evict it between calls.
        # bert_score stores its model inside a cached function — grab it here.
        try:
            import bert_score
            # Force the scorer to cache by calling get_model directly
            from bert_score.utils import get_model, get_tokenizer
            self._bert_model = get_model("bert-base-uncased", num_layers=9, all_layers=False)
            self._bert_tokenizer = get_tokenizer("bert-base-uncased", use_fast=False)
            self._bert_model.eval()
        except Exception:
            self._bert_model = None
        
        print("SelfCheck-BERTScore ready.")

    def generate_samples(self, model, tokenizer, prompt: str,
                         num_samples: int = 5, max_new_tokens: int = 40,
                         device: str = "mps"):
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1500
        ).to(device)

        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # SelfCheckGPT internally uses NLTK sent_tokenize on sampled_passages.
            # If a passage has fewer than 5 words, NLTK may return an empty list
            # causing "list index out of range" inside bert_score internals.
            if len(decoded.split()) >= 5:
                samples.append(decoded)
            else:
                # Pad to ensure NLTK produces at least one sentence
                samples.append(decoded + " The answer is based on the provided context.")

        if not samples:
            samples = ["The answer could not be determined from the given context."]

        return samples

    def score(self, sentences: list, sampled_passages: list) -> np.ndarray:
        """Returns np.array of shape [n_sentences] with hallucination scores."""
        # Ensure no empty strings reach the scorer
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