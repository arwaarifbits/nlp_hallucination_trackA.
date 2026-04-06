import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class InformationGainMetric:
    def __init__(self, model_name="facebook/opt-1.3b"):
        """
        Recommended models (in order of preference by resource):
        - "facebook/opt-1.3b"    (fast, ~5GB RAM, good for testing)
        - "facebook/opt-2.7b"    (better quality)
        - "meta-llama/Llama-2-7b-hf"  (best, needs HuggingFace token)
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use MPS on Mac, fall back to CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)
        self.model.eval()
        self.model_name = model_name

    def _get_logits(self, prompt: str, response: str):
        """Forward pass. Returns logits only for response token positions."""
        full_text = prompt + response
        
        # Tokenize separately to know prompt length
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
        
        prompt_len = prompt_ids.shape[1]
        
        # Truncate if too long (model max is usually 2048)
        max_len = 1800
        if full_ids.shape[1] > max_len:
            # Keep full prompt + truncate response
            response_ids = self.tokenizer.encode(response, return_tensors="pt")
            allowed_resp = max_len - prompt_len
            response_ids = response_ids[:, :allowed_resp]
            full_ids = torch.cat([prompt_ids, response_ids], dim=1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # Logit at position i predicts token i+1
        # Response token predictions start at index prompt_len-1
        resp_logits = logits[0, prompt_len - 1 : full_ids.shape[1] - 1, :]
        return resp_logits  # [resp_len, vocab_size]

    def _entropy(self, logits: torch.Tensor) -> np.ndarray:
        """Shannon entropy from logits."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.float().cpu().numpy()

    def compute_information_gain(self, query: str, context: str, response: str):
        """
        Main metric computation.
        
        Returns:
            ig: np.array [resp_tokens] — information gain per token
            H_no_ctx: np.array — entropy without context
            H_with_ctx: np.array — entropy with context
        """
        # WITHOUT context
        prompt_no_ctx = f"Question: {query}\nAnswer:"
        logits_no_ctx = self._get_logits(prompt_no_ctx, response)
        H_no_ctx = self._entropy(logits_no_ctx)

        # WITH context
        prompt_with_ctx = f"Context: {context}\nQuestion: {query}\nAnswer:"
        logits_with_ctx = self._get_logits(prompt_with_ctx, response)
        H_with_ctx = self._entropy(logits_with_ctx)

        # Align lengths (may differ by 1 due to tokenization edge cases)
        min_len = min(len(H_no_ctx), len(H_with_ctx))
        ig = H_no_ctx[:min_len] - H_with_ctx[:min_len]

        return ig, H_no_ctx[:min_len], H_with_ctx[:min_len]

    def get_response_tokens(self, response: str):
        """Decode response into individual token strings (for visualization)."""
        ids = self.tokenizer.encode(response, add_special_tokens=False)
        return [self.tokenizer.decode([i]) for i in ids]