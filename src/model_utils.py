# src/model_utils.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "meta-llama/Llama-3.2-1B"
print(f"Loading model {MODEL_NAME} on {device}...")

def load_model():
    """
    Load tokenizer and model
    """
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)

    print(f"Loading model {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    print("Model loaded successfully!")

    return tokenizer, model


def get_logits(prompt, tokenizer, model, context=None, max_length=100):
    """
    Returns logits for each token of the prompt.
    If context is provided, prepend it to the prompt.
    """
    if context:
        input_text = context + "\n" + prompt
    else:
        input_text = prompt

    inputs_full = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_full = model(**inputs_full)

     # Tokenize prompt alone to slice logits
    inputs_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs_prompt["input_ids"].shape[1]

    # Slice last `prompt_len` tokens
    prompt_logits = outputs_full.logits[:, -prompt_len:, :]
    return prompt_logits