import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaModel
import os
import sys

def lda_satellite_inference(topics_count, max_rows, alpha, beta):
    csv_path = "data/processed_prompt.csv"
    model_folder = "model"
    result_folder = "result_lda"
    os.makedirs(result_folder, exist_ok=True)
    model_path = f"{model_folder}/lda_model_{topics_count}_{max_rows}_satellite.model"
    id2word_path = f"{model_folder}/id2word_{topics_count}_{max_rows}_satellite.dict"
    result_path = f"{result_folder}/result_lda_inference_{topics_count}_{max_rows}_satellite.txt"

    sys.stdout = open(result_path, "w")

    if not (os.path.exists(model_path) and os.path.exists(id2word_path)):
        print(f"Satellite model or dictionary for topics={topics_count}, max_rows={max_rows} does not exist.")
        sys.stdout.close()
        exit()

    with open("stopwords.txt", "r") as f:
        stop_words = set(f.read().split())

    print("Loading model and dictionary...")
    lda_model = LdaModel.load(model_path)
    id2word = corpora.Dictionary.load(id2word_path)

    print("Loading CSV...")
    satellite_cols = [f"satellite_{i}" for i in range(1, 128)]
    df = pd.read_csv(csv_path, usecols=satellite_cols, nrows=max_rows)

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

    corpus = [id2word.doc2bow(text) for text in data_words]

    print("\nSample satellite prompt topic distributions:")
    sample_indices = [i for i in range(alpha, beta)]
    for idx in sample_indices:
        bow = corpus[idx]
        prompt_text = " ".join(data_words[idx])
        topic_distribution = lda_model.get_document_topics(bow)

        print(f"\nSatellite Prompt {idx}: {prompt_text}")
        print("Topic distribution (topic_id, probability):")
        for topic_id, prob in topic_distribution:
            print(f"  Topic {topic_id}: {prob:.4f}")

    print("Done.")
    sys.stdout.close()
