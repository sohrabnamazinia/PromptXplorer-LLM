import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaModel
import os
import sys
from langchain_openai import ChatOpenAI

def generate_topic_description(words_weights):
    # Initialize LLM inside the function
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.2,
        max_tokens=30,
    )

    prompt = (
        "Given the following words with their relative importance in a topic:\n\n"
    )
    for word, weight in words_weights:
        prompt += f"{word}: {weight:.4f}\n"
    prompt += (
        "\nWrite a very short (3-7 words) description summarizing the topic, "
        "ONLY the description itself, no explanations or extra text."
    )

    response = llm.invoke(prompt)
    return response.content.strip()

def save_cluster_description(lda_model, topics_count, max_rows, suffix=""):
    result_folder = "result_lda"
    os.makedirs(result_folder, exist_ok=True)

    descriptions = []
    for topic_id, topic in lda_model.show_topics(formatted=False, num_topics=topics_count, num_words=10):
        row = [topic_id]
        words_weights = []
        for word, weight in topic:
            row.extend([word, weight])
            words_weights.append((word, weight))
        
        topic_summary = generate_topic_description(words_weights)
        row.append(topic_summary)

        descriptions.append(row)

    columns = ["topic_id"]
    for i in range(1, 11):
        columns.extend([f"word_{i}", f"weight_word_{i}"])
    columns.append("topic_description")

    df_desc = pd.DataFrame(descriptions, columns=columns)
    filename = f"{result_folder}/cluster_description_{topics_count}_{max_rows}{suffix}.csv"
    df_desc.to_csv(filename, index=False)


def lda_primary(topics_count, max_rows):
    csv_path = "data/processed_prompt.csv"
    output_path = "lda_topics.txt"
    model_folder = "model"
    result_folder = "result_lda"
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    model_path = f"{model_folder}/lda_model_{topics_count}_{max_rows}.model"
    id2word_path = f"{model_folder}/id2word_{topics_count}_{max_rows}.dict"
    result_path = f"{result_folder}/result_lda_{topics_count}_{max_rows}.txt"

    with open("stopwords.txt", "r") as f:
        stop_words = set(f.read().split())

    sys.stdout = open(result_path, "w")

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

    save_cluster_description(lda_model, topics_count, max_rows)

    print("Done.")
    sys.stdout.close()

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

    save_cluster_description(lda_model, topics_count, max_rows, suffix="_satellite")

    print("Done.")
    sys.stdout.close()


def lda_primary_inference(topics_count, max_rows, alpha, beta):
    csv_path = "data/processed_prompt.csv"
    model_folder = "model"
    result_folder = "result_lda"
    os.makedirs(result_folder, exist_ok=True)
    model_path = f"{model_folder}/lda_model_{topics_count}_{max_rows}.model"
    id2word_path = f"{model_folder}/id2word_{topics_count}_{max_rows}.dict"
    result_path = f"{result_folder}/result_lda_inference_{topics_count}_{max_rows}.txt"
    top_topic_csv = f"{result_folder}/top_topic_primary_{topics_count}_{max_rows}.csv"
    full_dist_csv = f"{result_folder}/full_distribution_primary_{topics_count}_{max_rows}.csv"

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

    top_topics = []
    full_distributions = []

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

    print("\nCalculating and saving CSV files...")
    for idx, bow in enumerate(corpus):
        topic_distribution = lda_model.get_document_topics(bow)
        if topic_distribution:
            top_topic = max(topic_distribution, key=lambda x: x[1])[0]
            top_topics.append({"prompt_id": idx, "topic_id": top_topic})
            topic_probs = {f"topic_{i}": 0.0 for i in range(topics_count)}
            for topic_id, prob in topic_distribution:
                topic_probs[f"topic_{topic_id}"] = prob
            full_distributions.append({"prompt_id": idx, **topic_probs})

    pd.DataFrame(top_topics).to_csv(top_topic_csv, index=False)
    pd.DataFrame(full_distributions).to_csv(full_dist_csv, index=False)

    print("Done.")
    sys.stdout.close()

def lda_satellite_inference(topics_count, max_rows, alpha, beta):
    csv_path = "data/processed_prompt.csv"
    model_folder = "model"
    result_folder = "result_lda"
    os.makedirs(result_folder, exist_ok=True)
    model_path = f"{model_folder}/lda_model_{topics_count}_{max_rows}_satellite.model"
    id2word_path = f"{model_folder}/id2word_{topics_count}_{max_rows}_satellite.dict"
    result_path = f"{result_folder}/result_lda_inference_{topics_count}_{max_rows}_satellite.txt"
    top_topic_csv = f"{result_folder}/top_topic_satellite_{topics_count}_{max_rows}.csv"
    full_dist_csv = f"{result_folder}/full_distribution_satellite_{topics_count}_{max_rows}.csv"

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
    prompt_ids = []
    prompt_id_counter = 0

    for _, row in df.iterrows():
        for satellite_prompt in row:
            if pd.isna(satellite_prompt):
                continue
            tokens = str(satellite_prompt).lower().split()
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
            if filtered_tokens:
                data_words.append(filtered_tokens)
                prompt_ids.append(prompt_id_counter)
                prompt_id_counter += 1

    corpus = [id2word.doc2bow(text) for text in data_words]

    top_topics = []
    full_distributions = []

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

    print("\nCalculating and saving CSV files...")
    for idx, bow in enumerate(corpus):
        topic_distribution = lda_model.get_document_topics(bow)
        if topic_distribution:
            top_topic = max(topic_distribution, key=lambda x: x[1])[0]
            top_topics.append({"prompt_id": prompt_ids[idx], "topic_id": top_topic})
            topic_probs = {f"topic_{i}": 0.0 for i in range(topics_count)}
            for topic_id, prob in topic_distribution:
                topic_probs[f"topic_{topic_id}"] = prob
            full_distributions.append({"prompt_id": prompt_ids[idx], **topic_probs})

    pd.DataFrame(top_topics).to_csv(top_topic_csv, index=False)
    pd.DataFrame(full_distributions).to_csv(full_dist_csv, index=False)

    print("Done.")
    sys.stdout.close()
