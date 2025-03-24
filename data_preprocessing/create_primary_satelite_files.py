import csv

input_path = "processed_prompt.csv"
primary_output_path = "primary_prompts.csv"
satellite_output_path = "satellite_prompts.csv"

with open(input_path, "r", encoding="utf-8", newline="") as fin, \
     open(primary_output_path, "w", encoding="utf-8", newline="") as fout_primary, \
     open(satellite_output_path, "w", encoding="utf-8", newline="") as fout_sat:

    reader = csv.reader(fin)
    primary_writer = csv.writer(fout_primary)
    satellite_writer = csv.writer(fout_sat)

    # Read header row
    header = next(reader, None)
    if not header:
        raise ValueError("processed_prompts.csv is empty or missing a header row.")

    # Find the primary_prompt column index
    try:
        primary_ix = header.index("primary_prompt")
    except ValueError:
        raise ValueError("Column 'primary_prompt' not found in processed_prompts.csv.")

    # Find all satellite_* column indices
    satellite_ixs = [i for i, col in enumerate(header) if col.startswith("satellite_")]

    # Write headers to output CSVs
    primary_writer.writerow(["primary_prompt"])
    satellite_writer.writerow(["satellite_prompt"])

    line_count = 0
    for row in reader:
        line_count += 1
        if line_count % 100 == 0:
            print(f"Processing line {line_count}...")

        # Safely get the primary prompt
        if primary_ix < len(row):
            primary_val = row[primary_ix].strip()
        else:
            primary_val = ""

        # Write primary prompt
        primary_writer.writerow([primary_val])

        # For each satellite column index, if available and non-empty, write it
        for ix in satellite_ixs:
            if ix < len(row):
                sat_val = row[ix].strip()
                if sat_val:  # non-empty
                    satellite_writer.writerow([sat_val])

print("Done. Created primary_prompts.csv and satellite_prompts.csv.")
