from datasets import load_dataset

# Load HaluEval normalized span-level dataset
halueval = load_dataset("llm-semantic-router/halueval-spans-normalized")

# Save split(s) to CSV
halueval["train"].to_csv("data/raw/halueval_train.csv", index=False)

print("Downloaded and saved HaluEval locally!")