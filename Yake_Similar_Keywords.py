# Author: gvb17226 (Amlan Basu)
import os
import glob
import yake
from difflib import get_close_matches
from collections import defaultdict

# Define predefined AI-related keywords
predefined_keywords = [
    "AI", "ML", "Machine Learning", "Artificial Intelligence", 
    "Deep Learning", "Neural Network", "Natural Language Processing", 
    "NLP", "Large Language Model", "LLM", "Cognitive Computing", "Data Analytics"
]

# YAKE keyword extractor configuration
kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=20)  # Extract top 20 keywords

def extract_keywords(text):
    """Extract keywords from a given text using YAKE."""
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, score in keywords]

def find_closest_keywords(all_extracted_keywords):
    """Find the closest matches between extracted keywords and predefined AI keywords."""
    matches = defaultdict(set)  # Use a set to avoid duplicate matches
    for keyword in predefined_keywords:
        closest = get_close_matches(keyword, all_extracted_keywords, n=5, cutoff=0.5)  # Get top 5 close matches
        matches[keyword].update(closest)
    return matches

def process_files(directory, file_extension="txt", max_files=1000, output_file="keyword_results.txt"):
    """Process text files in a directory, extract keywords, and find matches. Save results to a .txt file."""
    all_extracted_keywords = set()  # Store all extracted keywords globally

    file_list = glob.glob(os.path.join(directory, f"*.{file_extension}"))[:max_files]  # Limit to max_files for performance

    for file_path in file_list:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            extracted_keywords = extract_keywords(text)
            all_extracted_keywords.update(extracted_keywords)  # Add to global keyword set

            print(f"Processed: {file_path}")  # Progress tracking

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Find closest matches from all extracted keywords
    closest_matches = find_closest_keywords(all_extracted_keywords)

    # Format and save results to a .txt file
    with open(output_file, "w", encoding="utf-8") as f:
        for keyword, matches in closest_matches.items():
            f.write(f"{keyword}: {', '.join(matches) if matches else 'No close matches'}\n")

    print(f"Keyword extraction completed. Results saved to '{output_file}'.")

# Example Usage: Replace 'your_text_files_directory' with the actual directory path
directory_path = "D:\Earnings calls - all countries transcripts"
process_files(directory_path, max_files=5000, output_file="keyword_results.txt")  # Adjust max_files as needed
