import pandas as pd
import json
import os

# Updated paths based on your terminal output
CSV_PATH = "data/ragtruth/ragtruth.csv"
SOURCE_JSONL = "data/ragtruth/source_info.jsonl" 
OUTPUT_PATH = "data/ragtruth_final.csv"

def merge_data():
    print("Loading existing CSV (Responses and Labels)...")
    df = pd.read_csv(CSV_PATH)

    print("Loading source info (Context and Prompts)...")
    sources = []
    with open(SOURCE_JSONL, 'r') as f:
        for line in f:
            sources.append(json.loads(line))
    
    source_df = pd.DataFrame(sources)

    # --- FIX START: Force source_id to be the same type (string) ---
    df['source_id'] = df['source_id'].astype(str)
    source_df['source_id'] = source_df['source_id'].astype(str)
    # --- FIX END ---

    # Select only necessary columns
    source_df_subset = source_df[['source_id', 'source_info', 'prompt']]

    print("Merging data on 'source_id'...")
    merged_df = pd.merge(df, source_df_subset, on='source_id', how='left')

    # Rename for the assignment's terminology
    merged_df = merged_df.rename(columns={
        'source_info': 'context',
        'prompt': 'query'
    })

    # Final check: Drop any rows where context is missing (if some IDs didn't match)
    initial_count = len(merged_df)
    merged_df = merged_df.dropna(subset=['context'])
    final_count = len(merged_df)
    
    if initial_count > final_count:
        print(f"Warning: {initial_count - final_count} rows dropped due to missing source info.")

    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Final file saved to: {OUTPUT_PATH}")
    print(f"Final columns: {merged_df.columns.tolist()}")

if __name__ == "__main__":
    merge_data()