import pickle
from datasets import load_dataset

# Load your specific RAGTruth split
dataset = load_dataset("data/ragtruth/ragtruth.csv") # or your local path
print("Columns available:", dataset['test'].column_names)
print("Sample model name:", dataset['test'][0].get('model_name', 'Not Found'))