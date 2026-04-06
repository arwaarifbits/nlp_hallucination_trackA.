import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class InformationGainMetric:
    def __init__(self, model_name="facebook/opt-1.3b"):
        print(f"Initializing {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"Using device: {self.device}")
        
        # Use float32 on Mac to avoid NaN errors in entropy
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32 
        ).to(self.device)
        self.model.eval()

    def get_token_probs(self, prompt: str, response: str):
        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits # [1, seq_len, vocab_size]

        # Shift: logits[i] predicts token[i+1]
        response_logits = logits[0, prompt_len - 1 : -1, :] 
        
        # Force float32 before Softmax to prevent NaNs
        probs = F.softmax(response_logits.to(torch.float32), dim=-1)
        return probs

    def token_entropy(self, probs: torch.Tensor):
        # We add a small epsilon and clamp to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Clean up any residual NaNs
        entropy = torch.nan_to_num(entropy, nan=0.0)
        return entropy.cpu().numpy()

    def compute_information_gain(self, query: str, context: str, response: str):
        # 1. Baseline: H(Token | Query)
        prompt_no_ctx = f"Question: {query}\nAnswer:"
        probs_no_ctx = self.get_token_probs(prompt_no_ctx, response)
        H_no_ctx = self.token_entropy(probs_no_ctx)

        # 2. Conditioned: H(Token | Query + Context)
        prompt_with_ctx = f"Context: {context}\nQuestion: {query}\nAnswer:"
        probs_with_ctx = self.get_token_probs(prompt_with_ctx, response)
        H_with_ctx = self.token_entropy(probs_with_ctx)

        min_len = min(len(H_no_ctx), len(H_with_ctx))
        ig = H_no_ctx[:min_len] - H_with_ctx[:min_len]
        
        return ig, H_no_ctx[:min_len], H_with_ctx[:min_len]