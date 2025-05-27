import os
import json
from phrase_detection import PhraseDetection

# Load phrases and alternatives
with open("ableist_phrases.txt") as f:
    phrases = [line.strip() for line in f if line.strip()]

with open("suggested_replacements.txt") as f:
    alternatives = [line.strip() for line in f if line.strip()]

# Initialize detector
detector = PhraseDetection(phrases, alternatives=alternatives, window_size=5, threshold=0.8)

# Directory of text files
input_dir = "ACM_Papers_Mistral"
all_flagged_sentences = []

# Process each .txt file in directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            paper_text = f.read()

        flagged = detector.process_text(paper_text)
        for entry in flagged:
            entry["Source"] = filename  # Track which file it came from
        all_flagged_sentences.extend(flagged)
        print(len(all_flagged_sentences))

# Save to a JSON file
with open("flagged_sentences.json", "w", encoding="utf-8") as f:
    json.dump(all_flagged_sentences, f, indent=2, ensure_ascii=False)
