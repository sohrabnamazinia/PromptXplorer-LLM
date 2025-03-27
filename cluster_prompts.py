import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

def main():
    parser = argparse.ArgumentParser(description="Text clustering for prompts")
    parser.add_argument("--prompt_type", choices=["primary", "satellite"], required=True)
    parser.add_argument("--num_rows", type=int, required=True)
    parser.add_argument("--num_clusters", type=int, required=True)
    args = parser.parse_args()

    if args.prompt_type == "primary":
        file_name = os.path.join("data", "primary_prompts.csv")
        col_name = "primary_prompt"
    else:
        file_name = os.path.join("data", "satellite_prompts.csv")
        col_name = "satellite_prompt"

    with open(file_name, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1

    num_rows = min(args.num_rows, total_rows)
    print(f"Processing {num_rows} rows out of {total_rows} available.")

    df = pd.read_csv(file_name, nrows=num_rows)
    texts = df[col_name].astype(str).tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    k = args.num_clusters
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
    kmeans.fit(X)
    labels = kmeans.labels_

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    rep_dict = {i: texts[idx] for i, idx in enumerate(closest)}

    output_rows = []
    for i, cluster in enumerate(labels):
        if i % 10 == 0:
            print(f"Processing row number {i}")
        output_rows.append({
            "row_index": i,
            "cluster_index": cluster,
            "cluster_representative": rep_dict[cluster]
        })

    output_df = pd.DataFrame(output_rows)
    output_folder = "results_cluster"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{args.prompt_type}_{num_rows}_{k}.csv")
    output_df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()
