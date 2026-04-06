#The Qualitative Case Study: Use this for your report's "Analysis" section. It generates that beautiful graph of Sample 0 showing exactly where the hallucination happened. It proves the mechanics work.

import pandas as pd
import numpy as np
import ast
from src.metric import InformationGainMetric
from src.temporal import extract_temporal_features, plot_ig_with_labels

# 1. Load Data
df = pd.read_csv("data/ragtruth/ragtruth_final.csv")
row = df.iloc[0]

# Ensure labels are a list of dictionaries
label_dicts = ast.literal_eval(row['labels']) if isinstance(row['labels'], str) else row['labels']

# 2. Init Metric
metric = InformationGainMetric("facebook/opt-1.3b")

# 3. Compute IG
ig, h_no, h_with = metric.compute_information_gain(
    query=row['query'], 
    context=row['context'], 
    response=row['response']
)

# 4. ALIGNMENT LOGIC: Map character-level errors to tokens
def create_hallucination_mask(response, errors, tokenizer, ig_len):
    mask = np.zeros(ig_len)
    # Get character offsets for every token in the full text
    encoding = tokenizer(response, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    
    # We only care about the last 'ig_len' tokens (the response)
    resp_offsets = offsets[-ig_len:] 

    for i, (start, end) in enumerate(resp_offsets):
        for error in errors:
            # Check for overlap between token (start, end) and error (error['start'], error['end'])
            if start < error['end'] and end > error['start']:
                mask[i] = 1
    return mask

binary_labels = create_hallucination_mask(row['response'], label_dicts, metric.tokenizer, len(ig))

# 5. Temporal Analysis
temporal_feats = extract_temporal_features(ig, window=5)
print(f"Response Snippet: {row['response'][:50]}...")
print(f"Mean IG Score: {np.mean(ig):.4f}")
print(f"Hallucination Tokens Found: {int(sum(binary_labels))}")

# 6. Generate Plot
# Passing the binary_labels ensures the red spans now appear
plot_ig_with_labels(ig, None, binary_labels, title="Sample_0_Grounded_Analysis")