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

        # Check for CUDA (NVIDIA), then MP(Apple), then fall back to CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Device: {self.device}")
        
        # Set precision: float16 for GPU (CUDA/MPS), float32 for CPU
        # Note: CUDA also supports bfloat16 if the hardware is modern (A100/H100/L4)
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(self.device)
        
        self.model.eval()
        self.model_name = model_name

    def _get_logits(self, prompt: str, response: str):
        """Forward pass. Prioritizes keeping the response and truncating the prompt."""
    
        # 1. Tokenize separately
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        response_ids = self.tokenizer.encode(response, add_special_tokens=False, return_tensors="pt")
    
        # 2. Define safety limits (OPT max is 2048, 1800 is a safe buffer)
        max_total = 1800 
        res_len = response_ids.shape[1]
    
        # 3. Truncate prompt from the LEFT if total > max_total
        # This keeps the 'Query' (usually at the end of the prompt) and trims the 'Context'
        if (prompt_ids.shape[1] + res_len) > max_total:
            allowed_prompt_len = max_total - res_len
            if allowed_prompt_len > 0:
                prompt_ids = prompt_ids[:, -allowed_prompt_len:] # Keep the last N tokens
            else:
                # Emergency fallback: if response is somehow > 1800 tokens
                response_ids = response_ids[:, :max_total-10]
                prompt_ids = prompt_ids[:, :10] 

        # 4. Construct full sequence
        full_ids = torch.cat([prompt_ids, response_ids], dim=1).to(self.device)
        prompt_len = prompt_ids.shape[1]
    
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]
    
        # 5. Extract logits that predict the response tokens
        # Logit at index (prompt_len - 1) predicts the first token of the response
        # We stop at full_ids.shape[1] - 1 because we don't need the logit for the last token's successor
        resp_logits = logits[0, prompt_len - 1 : full_ids.shape[1] - 1, :]
    
        # Safety check for empty tensors
        if resp_logits.shape[0] == 0:
            raise ValueError("Response logits are empty. Check truncation logic.")
        
        return resp_logits.float()

    def _entropy(self, logits: torch.Tensor) -> np.ndarray:
        """Shannon entropy from logits."""
        logits = logits.float()
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
    
    def compute_kl_divergence(self, query: str, context: str, response: str):
        """
        KL(P_no_ctx || P_with_ctx) per token.
        Measures how much the context SHIFTS the distribution, not just reduces entropy.
        High KL = context strongly changed predictions = model is anchored to context = faithful.
        Low KL = context had no effect = hallucination risk.
        """
        prompt_no_ctx = f"Question: {query}\nAnswer:"
        logits_no = self._get_logits(prompt_no_ctx, response)
        P_no = torch.softmax(logits_no, dim=-1)          # [T, V]

        prompt_with_ctx = f"Context: {context}\nQuestion: {query}\nAnswer:"
        logits_with = self._get_logits(prompt_with_ctx, response)
        P_with = torch.softmax(logits_with, dim=-1)       # [T, V]

        min_len = min(P_no.shape[0], P_with.shape[0])
        P_no = P_no[:min_len]
        P_with = P_with[:min_len]

        # KL(P_no || P_with) = sum P_no * log(P_no / P_with)
        kl = (P_no * (torch.log(P_no + 1e-10) - torch.log(P_with + 1e-10))).sum(dim=-1)
        return kl.float().cpu().numpy()   # [T]

    def compute_confidence_drop(self, query: str, context: str, response: str):
        """
        Confidence drop = p(actual token | no_ctx) - p(actual token | with_ctx).
        Positive = model was MORE confident without context (ignoring retrieval = hallucination).
        Negative = context boosted confidence in this token = faithful.
        We return NEGATIVE of this so higher score = hallucination.
        """
        prompt_no_ctx = f"Question: {query}\nAnswer:"
        logits_no = self._get_logits(prompt_no_ctx, response)
        P_no = torch.softmax(logits_no, dim=-1)

        prompt_with_ctx = f"Context: {context}\nQuestion: {query}\nAnswer:"
        logits_with = self._get_logits(prompt_with_ctx, response)
        P_with = torch.softmax(logits_with, dim=-1)

        # Get the actual token ids of the response
        resp_ids = self.tokenizer.encode(response, add_special_tokens=False)

        min_len = min(P_no.shape[0], P_with.shape[0], len(resp_ids))
    
        conf_drop = []
        for i in range(min_len):
            tok_id = resp_ids[i]
            p_no   = P_no[i, tok_id].item()
            p_with = P_with[i, tok_id].item()
            # Positive value = context REDUCED confidence = likely hallucinated
            conf_drop.append(p_no - p_with)

        return np.array(conf_drop)   # [T], positive = hallucination signal
    

    
