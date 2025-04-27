import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaModel

use_loaded_model = False
topics_count = 2
max_rows = 1000
# the prompts indices that result is gonna be printed for them
alpha, beta = 0, 5
csv_path = "data/processed_prompt.csv"
model_path = "lda_model.model"
id2word_path = "id2word.dict"
output_path = "lda_topics.txt"

with open("stopwords.txt", "r") as f:
    stop_words = set(f.read().split())

if use_loaded_model:
    print("Loading existing model and dictionary...")
    id2word = corpora.Dictionary.load(id2word_path)
    lda_model = LdaModel.load(model_path)

    print("Loading CSV for sampling prompts...")
    df = pd.read_csv(csv_path, usecols=[0], nrows=max_rows)
    df.columns = ["prompt"]

    print("Tokenizing and cleaning prompts...")
    data_words = []
    for prompt in df["prompt"]:
        tokens = str(prompt).lower().split()
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        data_words.append(filtered_tokens)

    corpus = [id2word.doc2bow(text) for text in data_words]

else:
    print("Loading CSV...")
    df = pd.read_csv(csv_path, usecols=[0], nrows=max_rows)
    df.columns = ["prompt"]
    print(f"Loaded {len(df)} rows.")

    print("Tokenizing and cleaning prompts...")
    data_words = []
    for prompt in df["prompt"]:
        tokens = str(prompt).lower().split()
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        data_words.append(filtered_tokens)

    print(f"Tokenized and cleaned {len(data_words)} prompts.")

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
