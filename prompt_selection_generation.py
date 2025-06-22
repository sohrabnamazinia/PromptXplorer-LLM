import random
import os
import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaModel
from langchain_openai import ChatOpenAI
import difflib

# Load stopwords
with open("stopwords.txt", "r") as f:
    STOP_WORDS = set(f.read().split())

def preprocess_prompt(prompt):
    tokens = str(prompt).lower().split()
    return [word for word in tokens if word.isalpha() and word not in STOP_WORDS]

def lda_split(prompts, num_topics=2):
    data_words = [preprocess_prompt(p) for p in prompts]
    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]
    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    topic_assignments = []
    for bow in corpus:
        topic_distribution = lda_model.get_document_topics(bow)
        top_topic = max(topic_distribution, key=lambda x: x[1])[0]
        topic_assignments.append(top_topic)

    cluster1 = [p for p, t in zip(prompts, topic_assignments) if t == 0]
    cluster2 = [p for p, t in zip(prompts, topic_assignments) if t == 1]

    desc1 = generate_topic_description(lda_model.show_topic(0, topn=10))
    desc2 = generate_topic_description(lda_model.show_topic(1, topn=10))

    return (cluster1, desc1), (cluster2, desc2)

def generate_topic_description(words_weights):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0.2, max_tokens=30)
    prompt = "Given the following words with their relative importance in a topic:\n\n"
    for word, weight in words_weights:
        prompt += f"{word}: {weight:.4f}\n"
    prompt += "\nWrite a very short (3-7 words) description summarizing the topic, ONLY the description itself, no explanations or extra text."
    response = llm.invoke(prompt)
    return response.content.strip()

def select_prompt(user_query, candidate_satellite_prompts, threshold=10):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    candidates = candidate_satellite_prompts
    
    while len(candidates) > threshold:
        (cluster1, desc1), (cluster2, desc2) = lda_split(candidates)

        system_prompt = (
            f"User query: {user_query}\n"
            f"You are given two clusters of satellite prompts:\n"
            f"Cluster A: {desc1}\n"
            f"Cluster B: {desc2}\n"
            "Based on the user query, which cluster is more relevant? Reply with A or B only."
        )
        response = llm.invoke(system_prompt)
        choice = response.content.strip().upper()
        
        if choice == "A":
            candidates = cluster1
        elif choice == "B":
            candidates = cluster2
        else:
            candidates = random.choice([cluster1, cluster2])  # fallback safety

    # Final selection
    final_prompt = "Given the following candidate prompts:\n"
    for i, p in enumerate(candidates, 1):
        final_prompt += f"{i}. {p}\n"
    final_prompt += f"\nUser query: {user_query}\n"
    final_prompt += "Which one is the most appropriate to add to the user query? Reply with the full text of the selected prompt only."
    response = llm.invoke(final_prompt)

    return response.content.strip()

def generate_prompt(user_query, candidate_satellite_prompts):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    few_shot_examples = random.sample(candidate_satellite_prompts, min(len(candidate_satellite_prompts), random.randint(3, 4)))
    system_prompt = (
        f"You are given a user query and several example satellite prompts that have been previously added to similar queries.\n\n"
        f"User query: {user_query}\n"
        "Example satellite prompts:\n"
    )
    for i, p in enumerate(few_shot_examples, 1):
        system_prompt += f"{i}. {p}\n"
    system_prompt += ("\nGenerate one high-quality, representative, and cherry-picked satellite prompt that best captures the intent of these examples."
                      " Reply ONLY with the generated prompt.")
    
    response = llm.invoke(system_prompt)
    return response.content.strip()

def generate_satellite_prompts(user_query, num_prompts=100, output_file="sample_satellite_prompts.txt"):
    llm = ChatOpenAI(model="gpt-4.1", temperature=0.5, max_tokens=2000)
    
    system_prompt = (
        f"You are given a user query: \"{user_query}\".\n"
        f"Generate {num_prompts} diverse, short satellite prompts that can be added to this query.\n"
        "Output ONLY the list of prompts, separated by \n, with no numbering or extra text."
    )
    response = llm.invoke(system_prompt)
    
    with open(output_file, "w") as f:
        f.write(response.content.strip())

def fuzzy_remove(selected_prompt, available_prompts):
    # Normalize both selected and available prompts
    norm_selected = selected_prompt.strip().lower()
    norm_available = [p.strip().lower() for p in available_prompts]
    match = difflib.get_close_matches(norm_selected, norm_available, n=1, cutoff=0.7)
    if match:
        to_remove = match[0]
        index_to_remove = norm_available.index(to_remove)
        del available_prompts[index_to_remove]
    else:
        # fallback if no close match found
        pass

def evaluate_prompt_selection_vs_generation(user_query, n=5, num_prompts=100, file_path="sample_satellite_prompts.txt", log_file="log_prompt_selection_vs_generation.txt"):
    if not os.path.exists(file_path):
        generate_satellite_prompts(user_query, num_prompts, file_path)

    with open(file_path, "r") as f:
        satellite_prompts = [line.strip() for line in f if line.strip()]

    used_prompts = set()
    selector_scores = []
    generator_scores = []

    with open(log_file, "w") as log:
        log.write(f"User query: {user_query}\n\n")
        
        for round_num in range(1, n+1):
            available_prompts = [p for p in satellite_prompts if p not in used_prompts]
            if len(available_prompts) < 10:
                log.write("Not enough prompts left to continue.\n")
                break

            selected_prompt = select_prompt(user_query, available_prompts)
            fuzzy_remove(selected_prompt, available_prompts)
            generated_prompt = generate_prompt(user_query, available_prompts)
            
            llm = ChatOpenAI(model="gpt-4.1", temperature=0)
            eval_prompt = (
                f"User query: {user_query}\n\n"
                f"Prompt from selector: {selected_prompt}\n"
                f"Prompt from generator: {generated_prompt}\n\n"
                "For each prompt, estimate what percentage it is suitable for this user query."
                " Only output in the format: Selector: X% ; Generator: Y%"
            )
            response = llm.invoke(eval_prompt)
            result = response.content.strip()

            log.write(f"Round {round_num}:\n")
            log.write(f"Selected Prompt: {selected_prompt}\n")
            log.write(f"Generated Prompt: {generated_prompt}\n")
            log.write(f"Evaluation: {result}\n\n")

            try:
                selector_pct = int(result.split("Selector:")[1].split("%")[0].strip())
                generator_pct = int(result.split("Generator:")[1].split("%")[0].strip())
                selector_scores.append(selector_pct)
                generator_scores.append(generator_pct)
            except:
                log.write("Failed to parse evaluation percentages.\n\n")

            used_prompts.add(selected_prompt)

        if selector_scores and generator_scores:
            avg_selector = sum(selector_scores) / len(selector_scores)
            avg_generator = sum(generator_scores) / len(generator_scores)
            log.write(f"Average Selector Score: {avg_selector:.2f}%\n")
            log.write(f"Average Generator Score: {avg_generator:.2f}%\n")

# Example usage
if __name__ == "__main__":
    test_query = "generate an image of a human"
    evaluate_prompt_selection_vs_generation(test_query, n=5)
