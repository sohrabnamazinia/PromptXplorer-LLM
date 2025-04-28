import pandas as pd
import os

def clean_processed_prompt(csv_path, output_path):
    print("Loading original CSV...")
    satellite_cols = [f"satellite_{i}" for i in range(1, 128)]
    df = pd.read_csv(csv_path)

    print("Loading stopwords...")
    with open("stopwords.txt", "r") as f:
        stop_words = set(f.read().split())

    print("Cleaning satellite prompts...")
    for idx, row in df.iterrows():
        for col in satellite_cols:
            if col not in row:
                continue
            value = row[col]
            if pd.isna(value) or str(value).strip() == "":
                continue
            tokens = str(value).lower().split()
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
            if not filtered_tokens:
                df.at[idx, col] = pd.NA  # clean empty after processing

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} rows...")

    print(f"Saving cleaned CSV to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    input_csv = "data/processed_prompt.csv"
    output_csv = "data/processed_prompt_cleaned.csv"
    clean_processed_prompt(input_csv, output_csv)
