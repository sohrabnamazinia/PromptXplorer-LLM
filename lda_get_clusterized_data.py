import pandas as pd
import os

def get_clusterized_data(topics_count, max_rows):
    input_csv = "data/processed_prompt.csv"
    primary_csv = f"result_lda/top_topic_primary_{topics_count}_{max_rows}.csv"
    satellite_csv = f"result_lda/top_topic_satellite_{topics_count}_{max_rows}.csv"
    output_csv = f"result_lda/clusterized_data_{topics_count}_{max_rows}.csv"

    print("Loading original data...")
    df = pd.read_csv(input_csv, nrows=max_rows)

    print("Loading primary and satellite topic assignments...")
    df_primary = pd.read_csv(primary_csv)
    df_satellite = pd.read_csv(satellite_csv)

    print("Building mapping from prompt_id to topic_id...")
    primary_id_to_topic = dict(zip(df_primary["prompt_id"], df_primary["topic_id"]))
    satellite_id_to_topic = dict(zip(df_satellite["prompt_id"], df_satellite["topic_id"]))

    print("Replacing prompts with topic ids...")
    new_data = []

    satellite_counter = 0
    for idx, row in df.iterrows():
        new_row = []
        for col_idx, col_name in enumerate(row.index):
            value = row[col_name]
            if col_idx == 0:  # primary prompt
                topic_id = primary_id_to_topic.get(idx, -1)
                new_row.append(topic_id)
            else:  # satellite prompts
                if pd.isna(value) or str(value).strip() == "":
                    new_row.append("")
                else:
                    topic_id = satellite_id_to_topic.get(satellite_counter, "")
                    new_row.append(topic_id)
                    satellite_counter += 1
        new_data.append(new_row)

    clusterized_df = pd.DataFrame(new_data, columns=df.columns)

    print(f"Saving clusterized data to {output_csv}...")
    os.makedirs("result_lda", exist_ok=True)
    clusterized_df.to_csv(output_csv, index=False)

    print("Done.")

if __name__ == "__main__":
    topics_count = 2
    max_rows = 1000
    get_clusterized_data(topics_count, max_rows)
