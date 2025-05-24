from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import nltk
from nltk import sent_tokenize
nltk.download('punkt')
import re
from bs4 import BeautifulSoup
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

def clean_text(raw_text):
    """Remove HTML tags, fix line breaks, and clean up text formatting.
    
    Args:
        raw_text (str): Raw text with HTML and formatting issues.
        
    Returns:
        str: Clean text ready for processing.
    """
    text = BeautifulSoup(raw_text, "html.parser").get_text()
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n{2,}', '<<<PARA>>>', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\s*#{1,6}\s*', '\n\n', text)
    text = re.sub(r'\[\^?\d+\^?\]', '', text)
    text = re.sub(r'\[\d+(,\s*\d+)*(-\d+)?\]', '', text)
    text = re.sub(r'\([A-Za-z]+ et al\., \d{4}\)', '', text)
    text = re.sub(r'\b(Fig|Table|Eq|Eqn|Section|Sec)\.?\s*\d+(\.\d+)?', '', text)
    text = re.sub(r'\b\d+\s*[-–]\s*\d+\b', '', text)
    text = text.replace('<<<PARA>>>', '\n\n')
    return text

def split_on_window(sequence, limit=5):
    """Split sentence into overlapping word windows.
    
    Args:
        sequence (str): Sentence to split into windows.
        limit (int): Number of words per window.
        
    Returns:
        list: List of overlapping word windows.
    """
    split_sequence = sequence.split()
    iterators = [iter(split_sequence[index:]) for index in range(limit)]
    return [' '.join(window) for window in zip(*iterators)]

def process_text(text):
    """Clean text, split into sentences and windows, then encode all windows.
    
    Args:
        text (str): Raw input text to process.
        
    Returns:
        tuple: (window_embeddings, window_to_sentence_map, cleaned_sentences)
    """
    document = clean_text(text)
    sents_cleaneded = sent_tokenize(document)
    
    # Collect all windows with metadata
    all_windows = []
    window_to_sentence = []
    
    for sent_idx, sentence in enumerate(sents_cleaneded):
        windows = split_on_window(sentence, 5)
        for window in windows:
            all_windows.append(window)
            window_to_sentence.append(sent_idx)
    
    # Encode all windows at once
    all_embeds = model.encode(all_windows)
    
    return all_embeds, window_to_sentence, sents_cleaneded

# Load ableist phrases and embed
with open("ableist_phrases.txt", "r") as f:
    phrases_lst = [line.strip() for line in f if line.strip()]
phrase_embed = model.encode(phrases_lst)

# Open test file, clean and embed
with open("ACM_Papers_Mistral/3611659.3615713.txt", "r") as f:
    raw_text = f.read()

all_window_embeds, window_to_sentence, sents_cleaneded = process_text(raw_text)

threshold = 0.85

# Compute all similarities at once - this is the key improvement
similarities = cos_sim(phrase_embed, all_window_embeds)

# Find matches
for phrase_idx in range(len(phrases_lst)):
    for window_idx in range(len(all_window_embeds)):
        score = similarities[phrase_idx][window_idx]
        if score >= threshold:
            sentence_idx = window_to_sentence[window_idx]
            print(f"[Phrase: '{phrases_lst[phrase_idx]}' | Sentence: {sents_cleaneded[sentence_idx]} | Score: {score}]")