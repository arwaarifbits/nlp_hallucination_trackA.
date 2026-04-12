import pickle
import numpy as np

def check_hallucination_split(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        raw_labels = data['per_sample']['labels']
        
        hallucinated_count = 0
        normal_count = 0
        
        for l in raw_labels:
            # Convert to a numpy array to handle Tensors/Lists/Arrays the same way
            arr = np.array(l)
            
            # Logic: If ANY value in the sample is 1, the sample is hallucinated.
            # This is how RAGTruth/HaluEval AUROC is typically calculated.
            if np.any(arr == 1):
                hallucinated_count += 1
            else:
                normal_count += 1
        
        total = len(raw_labels)
        
        print(f"--- {filepath} ---")
        print(f"  Total Samples:    {total}")
        print(f"  Hallucinated (1): {hallucinated_count}")
        print(f"  Normal (0):       {normal_count}")
        print(f"  Ratio:            {(hallucinated_count/total)*100:.1f}% Hallucination")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

check_hallucination_split('results/checkpoint_halueval.pkl')
check_hallucination_split('results/checkpoint_ragtruth.pkl')

