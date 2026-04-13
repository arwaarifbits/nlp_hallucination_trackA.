# src/semantic_entropy.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class SemanticEntropyMetric:
    def __init__(self, gen_model, gen_tokenizer, device="mps"):
        self.model = gen_model
        self.tokenizer = gen_tokenizer
        self.device = device

        print(f"Loading NLI model for semantic entropy onto {self.device}...")
        # pipeline 'device' argument accepts strings like "mps" or "cuda:0"
        self.nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            device=self.device 
        )

    def _sample_completions(self, prompt: str, num_samples: int = 5,
                             max_new_tokens: int = 80) -> list:
        """Sample N stochastic completions."""
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=1800).to(self.device)
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
        """Return True if s1 and s2 entail each other (same semantic cluster)."""
        if not s1.strip() or not s2.strip():
            return False
        try:
            r1 = self.nli(f"{s1} [SEP] {s2}", truncation=True)[0]
            r2 = self.nli(f"{s2} [SEP] {s1}", truncation=True)[0]
            return r1["label"] == "ENTAILMENT" and r2["label"] == "ENTAILMENT"
        except Exception:
            return False

    def _cluster_completions(self, completions: list) -> list:
        """Greedy cluster: merge completions that bidirectionally entail each other."""
        clusters = []
        assigned = [False] * len(completions)
        for i, c in enumerate(completions):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, len(completions)):
                if not assigned[j] and self._bidirectional_entailment(c, completions[j]):
                    cluster.append(j)
                    assigned[j] = True
            clusters.append(cluster)
        return clusters

    import numpy as np
    import torch

    # In semantic_entropy.py, replace compute_semantic_entropy with:
    def compute_semantic_entropy(self, query: str, context: str, 
                                num_samples: int = 2, 
                                temperature: float = 0.8) -> float:
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 1. Generate samples
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=15, # Keep it short for speed!
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = [
            self.tokenizer.decode(o[input_ids.shape[-1]:], skip_special_tokens=True).strip() 
            for o in outputs
        ]

        # 2. Simple Semantic Grouping (Replacing the missing NLI method)
        # We group identical or near-identical strings
        unique_responses = {}
        for text in generated_texts:
            # Simple normalization: lowercase and strip punctuation
            norm_text = "".join(filter(str.isalnum, text.lower()))
            unique_responses[norm_text] = unique_responses.get(norm_text, 0) + 1
        
        # 3. Calculate Entropy
        counts = list(unique_responses.values())
        probs = [c / sum(counts) for c in counts]
        s_ent = -sum(p * np.log(p) for p in probs if p > 0)
        
        return float(s_ent)