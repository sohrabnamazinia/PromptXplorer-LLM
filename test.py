import pandas as pd

def count_satellite_prompts(csv_path, max_rows=100):
    satellite_cols = [f"satellite_{i}" for i in range(1, 128)]
    df = pd.read_csv(csv_path, usecols=satellite_cols, nrows=max_rows)

    count = 0
    for _, row in df.iterrows():
        for satellite_prompt in row:
            if pd.notna(satellite_prompt) and str(satellite_prompt).strip() != "":
                count += 1

    print(f"Number of non-empty satellite prompts in first {max_rows} rows: {count}")

if __name__ == "__main__":
    count_satellite_prompts("data/processed_prompt.csv", max_rows=1000)
