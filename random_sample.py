import pandas as pd

df = pd.read_csv("data/processed_prompt.csv")
df = df.sample(n=10000, random_state=42).reset_index(drop=True)
df.to_csv("data/processed_prompt_shuffled.csv", index=False)
