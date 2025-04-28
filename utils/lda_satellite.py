import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaModel
import os
import sys

def lda_satellite(topics_count, max_rows):
    csv_path = "data/processed_prompt.csv"
    output_path = "lda_topics_satellite.txt"
    model_folder = "model"
    result_folder = "result_lda"
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    model_path = f"{model_folder}/lda_model_{topics_count}_{max_rows}_satellite.model"
    id2word_path = f"{model_folder}/id2word_{topics_count}_{max_rows}_satellite.dict"
    result_path = f"{result_folder}/result_lda_{topics_count}_{max_rows}_satellite.txt"

    with open("stopwords.txt", "r") as f:
        stop_words = set(f.read().split())

    sys.stdout = open(result_path, "w")

    print("Loading CSV...")
    satellite_cols = [f"satellite_{i}" for i in range(1, 128)]
    df = pd.read_csv(csv_path, usecols=satellite_cols, nrows=max_rows)
    print(f"Loaded {len(df)} rows.")

    print("Extracting and cleaning satellite prompts...")
    data_words = []
    for _, row in df.iterrows():
        for satellite_prompt in row:
            if pd.isna(satellite_prompt):
                continue
            tokens = str(satellite_prompt).lower().split()
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
            if filtered_tokens:
                data_words.append(filtered_tokens)

    print(f"Extracted and cleaned {len(data_words)} satellite prompts.")

    print("Creating dictionary and corpus...")
    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]
    print(f"Dictionary size: {len(id2word)} unique tokens.")

    print("Training LDA model...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=topics_count,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    print("Saving model and dictionary...")
    lda_model.save(model_path)
    id2word.save(id2word_path)

    print("\nDiscovered Topics:")
    topics = lda_model.print_topics(num_words=10)
    for idx, topic in topics:
        print(f"Topic {idx}: {topic}")

    print(f"\nSaving topics to '{output_path}'...")
    with open(output_path, "w") as f:
        for idx, topic in topics:
            f.write(f"Topic {idx}: {topic}\n")

    print("Done.")
    sys.stdout.close()
