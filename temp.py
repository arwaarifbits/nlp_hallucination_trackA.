import pandas as pd
import ast

df = pd.read_csv("data/ragtruth/ragtruth_final.csv")

def process_ragtruth_labels(label_entry):
    # Convert string representation of list to actual list
    if isinstance(label_entry, str):
        label_list = ast.literal_eval(label_entry)
    else:
        label_list = label_entry
        
    # If the list is empty, it's faithful (all 0s)
    if not label_list:
        return 0
    
    # In RAGTruth, if the list contains dictionaries, it means there ARE hallucinations
    # We return the count of hallucination dictionaries found
    return len(label_list)

# Create a 'has_hallucination' flag
df['hal_count'] = df['labels'].apply(process_ragtruth_labels)

# Find samples that actually have hallucinations
hallucinated_samples = df[df['hal_count'] > 0].head(5)

print("Sample Indices with Hallucinations:")
print(hallucinated_samples.index.tolist())

# Let's look at the first one's response to be sure
idx = hallucinated_samples.index[0]
print(f"\nTesting Index {idx}")
print(f"Response: {df.loc[idx, 'response'][:100]}...")