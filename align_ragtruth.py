import json
import numpy as np

def align_ragtruth_labels(response_text, labels_json, tokenizer):
    """
    Converts RAGTruth JSON spans into a binary mask the same length as tokens.
    """
    # Parse the string if it's not already a list
    if isinstance(labels_json, str):
        labels = json.loads(labels_json.replace("'", '"').replace('True', 'true').replace('False', 'false'))
    else:
        labels = labels_json

    # Encode response to get token offsets
    encoding = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding.input_ids
    offsets = encoding.offset_mapping  # List of (start, end) for each token
    
    binary_mask = np.zeros(len(tokens))
    
    for lab in labels:
        l_start = lab['start']
        l_end = lab['end']
        
        for i, (t_start, t_end) in enumerate(offsets):
            # If the token overlaps with the hallucination span, mark it 1
            if max(l_start, t_start) < min(l_end, t_end):
                binary_mask[i] = 1
                
    return binary_mask