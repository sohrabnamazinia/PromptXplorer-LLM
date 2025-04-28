import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaModel
import os
import sys

def lda_primary_inference(topics_count, max_rows, alpha, beta):
    csv_path = "data/processed_prompt.csv"
    model_folder = "model"
    result_folder = "result_lda"
    os.makedirs(result_folder, exist_ok=True)
    model_path = f"{model_folder}/lda_model_{topics_count}_{max_rows}.model"
    id2word_path = f"{model_folder}/id2word_{topics_count}_{max_rows}.dict"
    result_path = f"{result_folder}/result_lda_inference_{topics_count}_{max_rows}.txt"

    sys.stdout = open(result_path, "w")

    if not (os.path.exists(model_path) and os.path.exists(id2word_path)):
        print(f"Model or dictionary for topics={topics_count}, max_rows={max_rows} does not exist.")
        sys.stdout.close()
        exit()

    with open("stopwords.txt", "r") as f:
        stop_words = set(f.read().split())

    print("Loading model and dictionary...")
    lda_model = LdaModel.load(model_path)
    id2word = corpora.Dictionary.load(id2word_path)

    print("Loading CSV...")
    df = pd.read_csv(csv_path, usecols=[0], nrows=max_rows)
    df.columns = ["prompt"]

    print("Tokenizing and cleaning prompts...")
    data_words = []
    for prompt in df["prompt"]:
        tokens = str(prompt).lower().split()
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        data_words.append(filtered_tokens)

    corpus = [id2word.doc2bow(text) for text in data_words]

    print("\nSample prompt topic distributions:")
    sample_indices = [i for i in range(alpha, beta)]
    for idx in sample_indices:
        bow = corpus[idx]
        prompt_text = " ".join(data_words[idx])
        topic_distribution = lda_model.get_document_topics(bow)

        print(f"\nPrompt {idx}: {prompt_text}")
        print("Topic distribution (topic_id, probability):")
        for topic_id, prob in topic_distribution:
            print(f"  Topic {topic_id}: {prob:.4f}")

    print("Done.")
    sys.stdout.close()
