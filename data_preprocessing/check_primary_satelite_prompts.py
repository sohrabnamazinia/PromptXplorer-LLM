def count_csv_rows(csv_path):
    # Subtract 1 for the header row
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1

primary_count = count_csv_rows("primary_prompts.csv")
satellite_count = count_csv_rows("satellite_prompts.csv")

print(f"primary_prompts.csv has {primary_count} rows (excluding header).")
print(f"satellite_prompts.csv has {satellite_count} rows (excluding header).")
