# temp_check.py
from src.utils import load_ragtruth, align_labels_to_tokens
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
ds = load_ragtruth(max_samples=5)

print("Columns:", ds.column_names)
for i in range(min(3, len(ds))):
    s = ds[i]
    print(f"\n--- Sample {i} ---")
    print("Labels raw:", s["labels"])
    response = s["response"]
    token_labels = align_labels_to_tokens(response, s["labels"], tok)
    print("Token labels sum:", token_labels.sum(), "/ total:", len(token_labels))
    print("Has hallucinated tokens:", token_labels.sum() > 0)