# src/model_utils.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B"

def load_model():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Llama-3.2 requires a pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

def get_logits(answer_text, tokenizer, model, context=None):
    """
    Correctly slices logits to include the entire answer by calculating 
    the offset after the context.
    """
    if context:
        full_text = context + "\n" + answer_text
        inputs_full = tokenizer(full_text, return_tensors="pt").to(device)
        
        # Calculate length of context including the newline
        inputs_context = tokenizer(context + "\n", return_tensors="pt")
        context_len = inputs_context["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model(**inputs_full)
        
        # Take everything FROM the end of context TO the end of the sequence
        # This ensures we get every token in the answer, not just the last one
        return outputs.logits[:, context_len:, :]
    else:
        inputs_answer = tokenizer(answer_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs_answer)
        return outputs.logits