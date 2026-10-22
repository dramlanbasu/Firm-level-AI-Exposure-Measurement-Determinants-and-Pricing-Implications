"""YAKE-based detection of the predefined AI terms, followed by Word2Vec labels.

Install: pip install yake pandas nltk gensim tqdm openpyxl
NLTK data: python -m nltk.downloader punkt punkt_tab
Provide the Google News Word2Vec binary separately.
"""
import os
import re
from collections import Counter

import yake
import pandas as pd
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
from tqdm import tqdm

# Update paths for your system.
w2v_model_path = "/home/gpu-server/Desktop/HaRdDrIvE/Amlan/GoogleNews-vectors-negative300.bin"
input_directory = "/home/gpu-server/Desktop/HaRdDrIvE/Amlan/Earning_Calls_Full"
output_directory = "/home/gpu-server/Desktop/HaRdDrIvE/Amlan/Keyword_Labels_Outputs/Summary"
batch_size = 288020

categories = {
    "Innovation": ["innovation", "invention", "discovery", "research"],
    "Productivity": ["productivity", "efficiency", "optimization", "automation"],
    "Opportunity": ["opportunity", "advantage", "scope"],
    "Risk": ["threat", "risk", "danger", "uncertainty"],
    "Positive": ["positive", "good", "optimistic"],
    "Business growth": ["revenue growth", "sales increase", "profit", "expansion", "business growth"],
    "Regulatory risk": ["regulation", "compliance", "legislation", "policy", "regulatory risk"],
    "Security risk": ["cyberattack", "weaponization", "fraud", "phishing", "security risk"],
    "Environment risk": ["Power consumption", "energy", "environment risk"],
    "Negative": ["negative", "bad", "pessimistic", "disappointment"],
}

tracked_keywords = [
    "AI", "ML", "Machine Learning", "Artificial Intelligence",
    "Deep learning", "Neural network", "Natural language processing",
    "NLP", "Large language model", "LLM", "Cognitive computing", "Data analytics",
]

# YAKE extracts candidate phrases; only candidates that match the supplied
# tracked_keywords are retained (no fuzzy matching or invented synonyms).
# n=3 supports the longest tracked terms. YAKE returns the lowest scores first.
kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=1.0, top=1000)


def normalize_phrase(phrase):
    return re.sub(r"\s+", " ", phrase.strip()).casefold()


canonical_by_normalized = {normalize_phrase(k): k for k in tracked_keywords}


def extract_tracked_keywords(sentence):
    """Return only specified terms YAKE selected as keywords in this sentence."""
    candidates = kw_extractor.extract_keywords(sentence)
    return {
        canonical_by_normalized[normalize_phrase(candidate)]
        for candidate, score in candidates
        if normalize_phrase(candidate) in canonical_by_normalized
    }


def load_model(path):
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    # Older gensim used init_sims; modern gensim uses fill_norms.
    model.fill_norms(force=True)
    return model


def classify_sentence(sentence, model):
    category_scores = {category: 0.0 for category in categories}
    words = re.findall(r"\b[\w]+\b", sentence.lower())
    for word in words:
        if word not in model.key_to_index:
            continue
        for category, seeds in categories.items():
            similarities = [
                model.similarity(word, seed)
                for seed in seeds if seed in model.key_to_index
            ]
            if similarities:
                category_scores[category] += max(similarities)
    if not words or all(score == 0 for score in category_scores.values()):
        return "Unclassified"
    return max(category_scores, key=category_scores.get)


def process_file(file_path, model):
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    keyword_counts = Counter({kw: 0 for kw in tracked_keywords})
    label_counts = Counter()
    labeled_statements = []

    for sentence in sent_tokenize(text):
        yake_matches = extract_tracked_keywords(sentence)
        if not yake_matches:
            continue
        # Count sentences where YAKE selected the term, rather than raw mentions.
        for term in yake_matches:
            keyword_counts[term] += 1
        label = classify_sentence(sentence, model)
        label_counts[label] += 1
        matches_text = ", ".join(k for k in tracked_keywords if k in yake_matches)
        labeled_statements.append(
            f"{file_name}: {sentence} -> {label} [YAKE: {matches_text}]"
        )

    return {
        "File Name": file_name,
        "Total Word Count": len(text.split()),
        **keyword_counts,
        **{category: label_counts[category] for category in [*categories, "Unclassified"]},
        "Labeled Statements": labeled_statements,
    }


def process_batch(batch_files, batch_number, out_dir, model):
    results = []
    all_statements = []
    for path in tqdm(batch_files, desc=f"Processing Batch {batch_number}"):
        result = process_file(path, model)
        all_statements.extend(result.pop("Labeled Statements"))
        results.append(result)

    statements_path = os.path.join(out_dir, f"batch_{batch_number}_statements.txt")
    with open(statements_path, "w", encoding="utf-8") as handle:
        for statement in all_statements:
            handle.write(statement + "\n")

    excel_path = os.path.join(out_dir, f"batch_{batch_number}_summary.xlsx")
    pd.DataFrame(results).to_excel(excel_path, index=False)
    print(f"Saved: {excel_path}\nSaved: {statements_path}")


def main():
    os.makedirs(output_directory, exist_ok=True)
    model = load_model(w2v_model_path)
    files = sorted(
        os.path.join(input_directory, name)
        for name in os.listdir(input_directory)
        if name.lower().endswith(".txt")
    )
    for batch_number, start in enumerate(range(0, len(files), batch_size), start=1):
        process_batch(files[start:start + batch_size], batch_number, output_directory, model)


if __name__ == "__main__":
    main()
