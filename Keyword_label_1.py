import os
import re
import yake
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
from multiprocessing import Pool
from tqdm import tqdm  # For progress bar

# Load a pretrained Word2Vec model
w2v_model_path = "/home/gpu-server/Desktop/HaRdDrIvE/Amlan/GoogleNews-vectors-negative300.bin"  # Update with your model path
w2v_model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
w2v_model.init_sims(replace=True)

# Define categories with their seed keywords
categories = {
    "Innovation": ["innovation","invention", "discovery", "research"],
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

# Function to classify a sentence based on Word2Vec similarity
def classify_sentence(sentence):
    category_scores = {category: 0 for category in categories}
    words = sentence.split()
    for word in words:
        if word in w2v_model.key_to_index:
            for category, seed_words in categories.items():
                similarity = max(
                    w2v_model.similarity(word, seed) 
                    for seed in seed_words if seed in w2v_model.key_to_index
                )
                category_scores[category] += similarity
    return max(category_scores, key=category_scores.get)

# Relevant keywords to track
tracked_keywords = ["AI", "ML", "Machine Learning", "Artificial Intelligence", 
                    "Deep learning", "Neural network", "Natural language processing", 
                    "NLP", "Large language model", "LLM", "Cognitive computing", "Data analytics"]

# Helper function to process a single file
def process_file(file_path):
    file_name = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    # Initialize counters
    keyword_counts = Counter({kw: 0 for kw in tracked_keywords})
    label_counts = Counter()
    labeled_statements = []

    # Count tracked keywords directly in the text
    for tracked_kw in tracked_keywords:
        matches = re.findall(rf"\b{re.escape(tracked_kw)}\b", text, re.IGNORECASE)
        keyword_counts[tracked_kw] += len(matches)

    # Sentence classification (only for sentences containing tracked keywords)
    sentences = sent_tokenize(text)
    for sentence in sentences:
        if any(re.search(rf"\b{re.escape(kw)}\b", sentence, re.IGNORECASE) for kw in tracked_keywords):
            label = classify_sentence(sentence)
            label_counts[label] += 1
            labeled_statements.append(f"{file_name}: {sentence} -> {label}")

    # Return summary and labeled statements
    return {
        "File Name": file_name,
        "Total Word Count": len(text.split()),
        **keyword_counts,
        **label_counts,
        "Labeled Statements": labeled_statements,
    }

# Batch processing function
def process_batch(batch_files, batch_number, output_directory):
    results = []
    all_labeled_statements = []

    for file_path in tqdm(batch_files, desc=f"Processing Batch {batch_number}"):
        result = process_file(file_path)
        results.append(result)
        all_labeled_statements.extend(result.pop("Labeled Statements"))

    # Save labeled statements to a text file
    statements_output_file = os.path.join(output_directory, f"batch_{batch_number}_statements.txt")
    with open(statements_output_file, "w", encoding="utf-8") as statements_output:
        for statement in all_labeled_statements:
            statements_output.write(statement + "\n")

    # Save batch results to an Excel file
    df_summary = pd.DataFrame(results)
    excel_output_path = os.path.join(output_directory, f"batch_{batch_number}_summary.xlsx")
    df_summary.to_excel(excel_output_path, index=False)

    print(f"Batch {batch_number} processing complete. Results saved to:")
    print(f"- {excel_output_path}")
    print(f"- {statements_output_file}")

# Main processing function with parallelization
def main(input_directory, output_directory, batch_size):
    os.makedirs(output_directory, exist_ok=True)

    # List all .txt files in the input directory
    file_list = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith(".txt")]

    # Process files in batches
    for batch_number, i in enumerate(range(0, len(file_list), batch_size), start=1):
        batch_files = file_list[i:i+batch_size]
        process_batch(batch_files, batch_number, output_directory)

# Input and output directories
input_directory = "/home/gpu-server/Desktop/HaRdDrIvE/Amlan/Earning_Calls_Full"  # Update with your directory path
output_directory = "/home/gpu-server/Desktop/HaRdDrIvE/Amlan/Keyword_Labels_Outputs/Summary"  # Update with desired output directory

# Run the main function with batch size
if __name__ == "__main__":
    batch_size = 288020  # Adjust batch size based on your system's memory capacity
    main(input_directory, output_directory, batch_size)

