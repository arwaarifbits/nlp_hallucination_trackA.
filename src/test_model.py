# src/test_model.py
from model_utils import get_logits

prompt = "The capital of France is"
context = "Geography knowledge: France is a country in Europe."

logits = get_logits(prompt, context)
print("Logits shape:", logits.shape)