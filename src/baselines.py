import numpy as np
import torch
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

# --- Baseline 1: Entropy-Only ---
def compute_entropy_baseline(h_with_ctx):
    """
    Standard Baseline: Just uses the uncertainty of the model 
    when it has access to the context.
    """
    # We return it directly. For AUROC, higher entropy should 
    # correlate with higher hallucination probability.
    return np.array(h_with_ctx)

# --- Baseline 2: SelfCheckGPT (Consistency) ---
class SelfCheckBaseline:
    def __init__(self, device="mps"):
        # BERTScore is used to check if the response is consistent 
        # with other stochastic samples from the same model.
        print("Initializing SelfCheckGPT (BERTScore variant)...")
        self.selfcheck = SelfCheckBERTScore(rescale_with_baseline=True)
        self.device = device

    def get_stochastic_samples(self, model, tokenizer, query, context, num_samples=3):
        """
        Generates 'N' alternative answers to see if the model 
        contradicts itself.
        """
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        samples = []
        for _ in range(num_samples):
            # Use high temperature (0.7 - 1.0) to get diverse samples
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True, 
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
            # Decode only the generated part
            gen_text = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            samples.append(gen_text)
        return samples

    def compute_selfcheck(self, sentences, sampled_passages):
        """
        Compares the original sentences against the stochastic samples.
        Low similarity = Hallucination.
        """
        sent_scores = self.selfcheck.predict(
            sentences=sentences,
            sampled_passages=sampled_passages
        )
        return sent_scores # Higher score = More likely to be a hallucination