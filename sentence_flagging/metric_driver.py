import os
import json
import pandas as pd
from phrase_detection import PhraseDetection
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

model = SentenceTransformer('all-MiniLM-L6-v2')

# Load phrases and alternatives
with open("rewritten_sentences_gem.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)
sentences = df["original_sentence"].tolist()
replacement_sentences = df["replacement_sentence"].tolist()

similarity_scores = []


sent_embed = normalize(model.encode(sentences))
replacement_embed = normalize(model.encode(replacement_sentences))
similarity_scores = np.sum(sent_embed * replacement_embed, axis=1)

min_idex = np.argmin(similarity_scores)

print(f"Original sentence: {sentences[min_idex]} \n Replacement Sentence: {replacement_sentences[min_idex]} \n Similarity: {similarity_scores[min_idex]}")


'''
ranges = {"50-60": 0, "60-70": 0, "70-80": 0, "80-90": 0, "90-100": 0}
for score in similarity_scores:
    score = int(score*100)
    if score > 50 and score <=60:
        ranges["50-60"] += 1
    elif score > 60 and score <=70:
        ranges["60-70"] += 1
    elif score > 70 and score <= 80:
        ranges["70-80"] += 1
    elif score > 80 and score <= 90:
        ranges["80-90"] += 1
    elif score > 90 and score <= 100:
        ranges["90-100"] += 1

print(ranges)

labels = list(ranges.keys())
values = list(ranges.values())

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(labels, values, edgecolor='black')
plt.xlabel("Similarity Range")
plt.ylabel("Count")
plt.title("Similarity Score Distribution")
plt.tight_layout()
plt.show()
'''