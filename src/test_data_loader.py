# src/test_data_loader.py
from load_data import load_ragtruth, load_halueval, preprocess_dataframe, train_test_split_dataset

# -----------------------------
# Load RAGTruth
# -----------------------------
rag_df = load_ragtruth()
print("RAGTruth sample before preprocessing:")
print(rag_df.head())

# Preprocess
rag_df = preprocess_dataframe(rag_df)
print("RAGTruth sample after preprocessing:")
print(rag_df.head())

# Train/Test split
train_rag, test_rag = train_test_split_dataset(rag_df)
print("RAGTruth Train sample:")
print(train_rag.head())
print("RAGTruth Test sample:")
print(test_rag.head())

# -----------------------------
# Load HaluEval
# -----------------------------
halueval_df = load_halueval()
print("HaluEval sample before preprocessing:")
print(halueval_df.head())

# Preprocess
halueval_df = preprocess_dataframe(halueval_df)
print("HaluEval sample after preprocessing:")
print(halueval_df.head())

# Train/Test split
train_halu, test_halu = train_test_split_dataset(halueval_df)
print("HaluEval Train sample:")
print(train_halu.head())
print("HaluEval Test sample:")
print(test_halu.head())