import os
import json
import pandas as pd
from phrase_detection import PhraseDetection

# Load phrases and alternatives
with open("ableist_phrases.txt") as f:
    phrases = [line.strip() for line in f if line.strip()]

with open("suggested_replacements.txt") as f:
    alternatives = [line.strip() for line in f if line.strip()]

# Initialize detector
detector = PhraseDetection(phrases, alternatives=alternatives, window_size=7, threshold=0.85)

# Directory of text files
with open("rewritten_sentences.json", "r") as file:
    data = json.load(file)

df = pd.DataFrame(data)
sentences = df["replacement_sentence"].tolist()

all_flagged_sentences = []

counter = 0
# Process each .txt file in directory
for sent in sentences:
    flagged = detector.process_text(sent)
    for entry in flagged:
        entry["Source"] = "replacements" 
    all_flagged_sentences.extend(flagged)
    counter += 1
    print(f"Processed {counter} files...")
    print(f"Found {len(all_flagged_sentences)} so far...")

# Save to a JSON file
with open("flagged_replacements.json", "w", encoding="utf-8") as f:
    json.dump(all_flagged_sentences, f, indent=2, ensure_ascii=False)
