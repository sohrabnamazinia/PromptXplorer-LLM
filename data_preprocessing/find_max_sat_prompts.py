import pandas as pd
import re

chunk_size = 100000
max_cols = 0
total_rows_processed = 0

# Pass 1: Only determine how many split parts we need (no large in-memory DataFrame).
for chunk in pd.read_csv("notebooks/prompts.csv", chunksize=chunk_size):
    for prompt_text in chunk["prompt"]:
        total_rows_processed += 1
        if total_rows_processed % 100 == 0:
            print(f"Processing prompt number {total_rows_processed}...")
        parts = [x.strip() for x in re.split(r'[.,]', str(prompt_text)) if x.strip()]
        max_cols = max(max_cols, len(parts))

print(f"Found max split parts: {max_cols}")
