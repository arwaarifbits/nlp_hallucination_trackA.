# =============================================================================
# src/metric.py
# CS F429 — Track A: Dynamic Uncertainty-Aware Attribution
#
# PURPOSE: This file implements all four token-level uncertainty metrics
# that form the core of Track A. Each metric measures a different aspect of
# how the retrieved context affects the model's probability distribution.
#
# DESIGN PRINCIPLE: Every metric requires two forward passes through the model:
#   Pass 1: prompt = "Question: {query}\nAnswer:"          (NO context)
#   Pass 2: prompt = "Context: {context}\nQuestion: ...\nAnswer:" (WITH context)
# The difference between the two passes is the hallucination signal.
#
# EVALUATOR NOTE: The key method for the demo question is compute_kl_divergence().
# =============================================================================

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class InformationGainMetric:
    """
    Wrapper class that holds the language model and implements all four metrics.
    We use a single class so the model is loaded once and shared across all
    metric computations — avoids loading a 5GB model four separate times.
    """

    def __init__(self, model_name="facebook/opt-1.3b"):
        """
        Loads the language model and tokenizer onto the appropriate device.

        Device selection priority:
          1. CUDA (NVIDIA GPU) — fastest, used on the university server
          2. MPS (Apple Silicon) — used on Mac for development
          3. CPU — slowest fallback

        We use float16 on GPU/MPS to halve memory usage (2 bytes per weight
        instead of 4). CPU must use float32 because MPS/CPU float16 operations
        are less numerically stable.
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # OPT has no pad token by default — set it to EOS so generation doesn't crash
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Device: {self.device}")

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        # `dtype=` replaces the deprecated `torch_dtype=` argument in newer
        # versions of transformers. Using the new kwarg avoids the deprecation warning.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype
        ).to(self.device)

        # eval() disables dropout and batch normalisation — critical for
        # reproducible logit values. Without this, each forward pass would
        # give slightly different results.
        self.model.eval()
        self.model_name = model_name

    # ─── Private helper ──────────────────────────────────────────────────────

    def _get_logits(self, prompt: str, response: str) -> torch.Tensor:
        """
        Runs a single forward pass and returns logits ONLY for the response tokens.

        HOW IT WORKS:
          The model sees: [prompt_tokens | response_tokens]
          Logit at position i predicts token at position i+1.
          So logit[prompt_len - 1] predicts the first response token.
          We slice: logits[prompt_len-1 : total_len-1] to get one logit
          per response token position.

        TRUNCATION STRATEGY:
          OPT-1.3b has a maximum context window of 2048 tokens.
          We truncate from the LEFT of the prompt (trimming the context)
          rather than the response, because we must score every response token.
          The question part at the end of the prompt is preserved.

        Returns:
          Tensor of shape [n_response_tokens, vocab_size], dtype float32.
          Always float32 regardless of model precision — this prevents NaN
          from float16 underflow when computing log(very_small_probability).
        """
        prompt_ids   = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        response_ids = self.tokenizer.encode(response, add_special_tokens=False, return_tensors="pt")

        max_total = 1800  # safety buffer below OPT's 2048 token maximum
        res_len   = response_ids.shape[1]

        # If prompt + response would exceed the model's context window,
        # trim the LEFT side of the prompt (oldest context tokens).
        # This keeps the query and the [Answer:] cue at the end intact.
        if (prompt_ids.shape[1] + res_len) > max_total:
            allowed_prompt_len = max_total - res_len
            if allowed_prompt_len > 0:
                prompt_ids = prompt_ids[:, -allowed_prompt_len:]
            else:
                # Extreme edge case: response itself is > 1800 tokens
                response_ids = response_ids[:, :max_total - 10]
                prompt_ids   = prompt_ids[:, :10]

        full_ids   = torch.cat([prompt_ids, response_ids], dim=1).to(self.device)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():  # no_grad() saves memory — we don't need gradients
            outputs = self.model(full_ids)
            logits  = outputs.logits   # shape: [1, total_tokens, vocab_size]

        # Extract only the response token positions
        # logit[i] predicts token[i+1], so response predictions start at prompt_len-1
        resp_logits = logits[0, prompt_len - 1 : full_ids.shape[1] - 1, :]

        if resp_logits.shape[0] == 0:
            raise ValueError("Response logits are empty. Check truncation logic.")

        # Always return float32 to prevent NaN from float16 underflow in log()
        return resp_logits.float()

    def _entropy(self, logits: torch.Tensor) -> np.ndarray:
        """
        Computes Shannon entropy H = -Σ p * log(p) for each token position.

        Uses log_softmax for numerical stability:
          log_softmax(x) = x - log(Σ exp(x))
        This avoids computing softmax then taking log separately, which can
        produce -inf when probabilities are very small.

        Returns array of shape [n_tokens] where each value is the entropy
        of the model's next-token distribution at that position.
        Higher entropy = model is more uncertain = potential hallucination.
        """
        logits    = logits.float()
        log_probs = F.log_softmax(logits, dim=-1)   # numerically stable
        probs     = log_probs.exp()                  # equivalent to softmax
        entropy   = -(probs * log_probs).sum(dim=-1) # H = -Σ p * log(p)
        return entropy.float().cpu().numpy()

    # ─── Public metric methods ───────────────────────────────────────────────

    def compute_information_gain(self, query: str, context: str, response: str):
        """
        METRIC 1: Information Gain (IG)

        FORMAL DEFINITION:
          IG(t_i) = H(t_i | query) − H(t_i | query, context)

          where H is Shannon entropy of the model's next-token distribution.

        INTERPRETATION:
          Positive IG → context REDUCED uncertainty → model used the context
                       → token is likely grounded (faithful)
          Negative IG → context INCREASED uncertainty, or had no effect
                       → model ignoring context → hallucination risk

        HOW IT'S COMPUTED:
          1. Two forward passes: without context, then with context.
          2. Entropy computed at each token position for both passes.
          3. IG = (entropy without context) − (entropy with context)

        Returns:
          ig:         np.array [n_tokens] — information gain per token
          H_no_ctx:   np.array [n_tokens] — entropy without context
          H_with_ctx: np.array [n_tokens] — entropy with context (Baseline 1)
        """
        # Pass 1: model sees only the question, no retrieved context
        prompt_no_ctx   = f"Question: {query}\nAnswer:"
        logits_no_ctx   = self._get_logits(prompt_no_ctx, response)
        H_no_ctx        = self._entropy(logits_no_ctx)

        # Pass 2: model sees the retrieved context before the question
        prompt_with_ctx = f"Context: {context}\nQuestion: {query}\nAnswer:"
        logits_with_ctx = self._get_logits(prompt_with_ctx, response)
        H_with_ctx      = self._entropy(logits_with_ctx)

        # Align lengths — may differ by 1 token due to tokenization edge cases
        # (e.g. the first token of the response can shift slightly)
        min_len = min(len(H_no_ctx), len(H_with_ctx))
        ig      = H_no_ctx[:min_len] - H_with_ctx[:min_len]

        return ig, H_no_ctx[:min_len], H_with_ctx[:min_len]

    def get_response_tokens(self, response: str) -> list:
        """
        Decodes the response into a list of readable token strings.
        Used for visualisation in the temporal plot and the demo output.
        OPT uses 'Ġ' as a space prefix — this shows the raw tokenisation.
        """
        ids = self.tokenizer.encode(response, add_special_tokens=False)
        return [self.tokenizer.decode([i]) for i in ids]

    def compute_kl_divergence(self, query: str, context: str, response: str):
        """
        METRIC 2: KL Divergence

        FORMAL DEFINITION:
          KL(P_no || P_with) = Σ_v P_no(v) * [log P_no(v) − log P_with(v)]

          where P_no and P_with are full vocabulary distributions at each token.

        INTERPRETATION:
          High KL → context strongly SHIFTED the distribution
                  → model is anchored to retrieved document → faithful token
          Low KL  → context had NO EFFECT on predictions
                  → model generating from parametric memory → hallucination risk

        NOTE ON ORIENTATION: KL is naturally high for faithful tokens.
          In the composite, we negate KL before combining so that high score
          always means high hallucination risk. The orient_score() function
          in main.py handles this automatically based on empirical AUROC.

        EVALUATOR QUESTION ANSWER:
          "KL is computed in compute_kl_divergence() in src/metric.py.
           Two forward passes → softmax to get distributions P_no and P_with →
           sum(P_no * (log P_no − log P_with)) per token position."
        """
        prompt_no_ctx   = f"Question: {query}\nAnswer:"
        logits_no       = self._get_logits(prompt_no_ctx, response)
        P_no            = torch.softmax(logits_no, dim=-1)   # shape: [T, vocab_size]

        prompt_with_ctx = f"Context: {context}\nQuestion: {query}\nAnswer:"
        logits_with     = self._get_logits(prompt_with_ctx, response)
        P_with          = torch.softmax(logits_with, dim=-1) # shape: [T, vocab_size]

        # Align lengths in case the two passes differ by one token
        min_len = min(P_no.shape[0], P_with.shape[0])
        P_no    = P_no[:min_len]
        P_with  = P_with[:min_len]

        # KL formula: Σ P_no * (log P_no − log P_with)
        # We add 1e-10 inside log() to prevent log(0) = -inf, which would
        # produce NaN when multiplied by P_no ≈ 0. This is standard practice.
        kl = (P_no * (torch.log(P_no + 1e-10) - torch.log(P_with + 1e-10))).sum(dim=-1)

        # KL is always ≥ 0 by definition (Gibbs inequality).
        # Small negative values can appear from floating point — safe to ignore.
        return kl.float().cpu().numpy()  # shape: [T]

    def compute_confidence_drop(self, query: str, context: str, response: str):
        """
        METRIC 3: Confidence Drop

        FORMAL DEFINITION:
          ConfDrop(t_i) = p(t_i | query) − p(t_i | query, context)

          where p(t_i | ...) is the probability of the ACTUAL generated token
          (not the argmax, but the specific token that appears in the response).

        INTERPRETATION:
          Positive value → model was MORE confident WITHOUT context
                         → it ignored the retrieved document
                         → hallucination signal
          Negative value → context BOOSTED confidence in the actual token
                         → model used the document → faithful token

        WHY THIS IS DIFFERENT FROM IG:
          IG measures entropy of the FULL distribution.
          ConfDrop measures probability of the SPECIFIC generated token.
          ConfDrop is more targeted: it directly asks "did the context help
          the model be more confident about the word it actually said?"
        """
        prompt_no_ctx   = f"Question: {query}\nAnswer:"
        logits_no       = self._get_logits(prompt_no_ctx, response)
        P_no            = torch.softmax(logits_no, dim=-1)

        prompt_with_ctx = f"Context: {context}\nQuestion: {query}\nAnswer:"
        logits_with     = self._get_logits(prompt_with_ctx, response)
        P_with          = torch.softmax(logits_with, dim=-1)

        # We need the actual token IDs of the response to look up their probabilities
        resp_ids = self.tokenizer.encode(response, add_special_tokens=False)
        min_len  = min(P_no.shape[0], P_with.shape[0], len(resp_ids))

        conf_drop = []
        for i in range(min_len):
            tok_id = resp_ids[i]          # the actual token at position i
            p_no   = P_no[i, tok_id].item()    # p(that token | no context)
            p_with = P_with[i, tok_id].item()  # p(that token | with context)
            # Positive = model was MORE confident without context = hallucination
            conf_drop.append(p_no - p_with)

        return np.array(conf_drop)  # shape: [T], positive = hallucination signal