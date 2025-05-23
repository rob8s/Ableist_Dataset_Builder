from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import nltk
from nltk import sent_tokenize
nltk.download('punkt')
import re
from bs4 import BeautifulSoup

model = SentenceTransformer('all-mpnet-base-v2')

def clean_text(raw_text):
    """
    Cleans input text by removing HTML tags, line-break hyphenation, extra whitespace,
    and certain markdown/footnote elements. Paragraph breaks are preserved.

    Args:
        raw_text (str): Raw text input.

    Returns:
        str: Cleaned text.
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
    """
    Splits a string into overlapping word windows of a given size.

    Args:
        sequence (str): Sentence to split.
        limit (int): Window size.

    Returns:
        list of str: List of overlapping word windows.
    """
    split_sequence = sequence.split()
    iterators = [iter(split_sequence[index:]) for index in range(limit)]
    return [' '.join(window) for window in zip(*iterators)]

def process_text(text):
    """
    Cleans and tokenizes text, splits into word windows per sentence,
    and encodes each window with the model.

    Args:
        text (str): Raw input text.

    Returns:
        tuple:
            - List[List[str]]: List of word windows per sentence.
            - List[List[np.ndarray]]: Embeddings per window per sentence.
            - List[str]: List of original sentences.
    """
    document = clean_text(text)
    sents_cleaneded = sent_tokenize(document)
    all_embeds = []
    for sentence in sents_cleaneded:
        windows = split_on_window(sentence, 5)
        embeds = model.encode(windows)
        all_embeds.append(embeds)
    return all_embeds, sents_cleaneded

# Load ableist phrases and embed
with open("ableist_phrases.csv", "r") as f:
    phrases_lst = [line.strip() for line in f.readlines()[1:]]
phrase_embed = model.encode(phrases_lst)

# Open test file, clean and embed
with open("ACM_Papers_Mistral/3611659.3615713.txt", "r") as f:
    raw_text = f.read()

doc_split, sents_cleaneded = process_text(raw_text)

threshold = 0.85
for i, sentence in enumerate(doc_split):
    for j, phrase in enumerate(phrase_embed):
        for k, window in enumerate(sentence):
            score = cos_sim(phrase, window)
            if score >= threshold:
                print(f"[Phrase: '{phrases_lst[j]}' | Sentence: {sents_cleaneded[i]} | Score: {score}]")
                break

