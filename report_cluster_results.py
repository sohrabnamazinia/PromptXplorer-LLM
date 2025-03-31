import pandas as pd

n_clusters = 10
# Load the CSV file
df = pd.read_csv("results_cluster/primary_1999810_10.csv")

# Count rows where cluster_index is from 0 to 9
cluster_counts = df['cluster_index'].value_counts().sort_index()
cluster_counts = cluster_counts.loc[0:n_clusters]  # Keep only 0 to 9

# Print the results
for index, count in cluster_counts.items():
    print(f"cluster_index {index}: {count} rows")
