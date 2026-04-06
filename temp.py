import pandas as pd

# Load your local file
df = pd.read_csv("data/halueval/halueval.csv")

print("--- HaluEval Dataset Statistics ---")
print(f"Total Rows: {len(df)}")

# Check if you have separate columns for right and hallucinated answers
if 'right_answer' in df.columns and 'hallucinated_answer' in df.columns:
    print(f"Faithful Samples: {len(df)} (from 'right_answer' column)")
    print(f"Hallucinated Samples: {len(df)} (from 'hallucinated_answer' column)")
    print("Total usable data points: ", len(df) * 2)

# If your CSV uses a 'label' column (0 = faithful, 1 = hallucinated)
elif 'label' in df.columns:
    counts = df['label'].value_counts()
    print(f"Faithful (0): {counts.get(0, 0)}")
    print(f"Hallucinated (1): {counts.get(1, 0)}")