# src/model_utils.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "meta-llama/Llama-3.2-1B"


def load_model():
    """
    Load tokenizer and model
    """
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)

    print(f"Loading model {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    print("Model loaded successfully!")

    return tokenizer, model


def get_logits(prompt, tokenizer, model, context=None):
    """
    Returns logits for each token.
    """
    input_text = (context + "\n" + prompt) if context else prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits