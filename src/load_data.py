# src/load_data.py
import ast
import pandas as pd
from sklearn.model_selection import train_test_split


COLUMN_MAP_HALUEVAL = {
    "prompt": "prompt",
    "answer": "model_answer",
    "labels": "label_hallucination"
}

def load_ragtruth(path="data/raw/ragtruth.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    # Rename columns
    df = df.rename(columns={
        "query": "prompt",
        "context": "context",
        "output": "model_answer",
        "hallucination_labels_processed": "label_hallucination"
    })
    
    # Create gold_answer column same as model_answer (if you don't have separate gold answers)
    if "gold_answer" not in df.columns:
        df["gold_answer"] = df["model_answer"]
    
    # Check required columns
    required_cols = ['prompt', 'context', 'gold_answer', 'model_answer', 'label_hallucination']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} missing in RAGTruth dataset")
    
    print(f"RAGTruth loaded: {len(df)} rows")
    return df

def load_halueval(path="data/raw/halueval.csv"):
    """
    Load HaluEval dataset and simplify the label_hallucination column.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_MAP_HALUEVAL)

    # HaluEval doesn’t have context, so fill it with empty strings
    if 'context' not in df.columns:
        df['context'] = ""

    # Convert span-based labels (list of dicts) to simple label
    def simplify_label(label_column):
        simplified = []
        for item in label_column:
            if pd.isna(item) or item == "[]" or item == "":
                simplified.append("non-hallucinated")
            else:
                try:
                    spans = ast.literal_eval(item)
                    if len(spans) > 0:
                        simplified.append("hallucinated")
                    else:
                        simplified.append("non-hallucinated")
                except:
                    simplified.append("non-hallucinated")
        return simplified

    df['label_hallucination'] = simplify_label(df['label_hallucination'])

    # Check required columns
    required_cols = ['prompt', 'model_answer', 'label_hallucination', 'context']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} missing in HaluEval dataset")

    print(f"HaluEval loaded: {len(df)} rows")
    return df


def preprocess_text(text):
    """
    Basic text preprocessing: lowercase, strip spaces
    Extend this function if needed
    """
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

def preprocess_dataframe(df):
    """
    Apply preprocessing to all text columns
    """
    for col in ['prompt', 'context', 'gold_answer', 'model_answer']:
        if col in df.columns:
            df[col] = df[col].apply(preprocess_text)
    return df

def train_test_split_dataset(df, test_size=0.2, random_state=42):
    """
    Split dataset into train and test
    """
    if df['label_hallucination'].nunique() > 1:
        stratify_col = df['label_hallucination']
    else:
        stratify_col = None

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df

if __name__ == "__main__":
    # Example usage
    rag_df = load_ragtruth()
    rag_df = preprocess_dataframe(rag_df)
    train_df, test_df = train_test_split_dataset(rag_df)

    halueval_df = load_halueval()
    halueval_df = preprocess_dataframe(halueval_df)
    train_h, test_h = train_test_split_dataset(halueval_df)