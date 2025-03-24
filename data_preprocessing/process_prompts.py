import csv
import pandas as pd
import re

max_cols = 128
chunk_size = 100000

output_cols = ["primary_prompt"] + [f"satellite_{i}" for i in range(1, max_cols)]

with open("processed_prompt.csv", "w", newline="", encoding="utf-8") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(output_cols)  # Header row

    total_rows_processed = 0
    for chunk in pd.read_csv("notebooks/prompts.csv", chunksize=chunk_size):
        for prompt_text in chunk["prompt"]:
            total_rows_processed += 1
            if total_rows_processed % 100 == 0:
                print(f"Processing prompt number {total_rows_processed}...")

            parts = [x.strip() for x in re.split(r'[.,]', str(prompt_text)) if x.strip()]
            if not parts:
                continue

            row_data = [parts[0]] + parts[1:]  # first is primary, rest are satellites
            # pad any missing columns up to max_cols
            row_data += [""] * (max_cols - len(parts))
            writer.writerow(row_data)

print("Done writing processed_prompt.csv.")
