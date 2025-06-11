from sentence_transformers import SentenceTransformer
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import normalize

nltk.download('punkt')
nltk.download('wordnet')


class PhraseDetection:
    """
    A class to detect semantically similar phrases in a given text using sentence embeddings.

    Attributes:
        model (SentenceTransformer): The sentence transformer model used to embed phrases and text.
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer for preprocessing.
        window_size (int): Number of words per sliding window in a sentence.
        threshold (float): Cosine similarity threshold to flag a sentence.
        raw_phrases (List[str]): Original phrases to detect.
        phrase_embeddings (np.ndarray): Normalized embeddings of lemmatized phrases.
    """

    def __init__(self, phrases, window_size=10, threshold=0.75):
        """
        Initializes the PhraseDetection object with target phrases and configuration.

        Args:
            phrases (List[str]): List of target phrases to detect.
            window_size (int, optional): Size of the sliding window. Defaults to 10.
            threshold (float, optional): Cosine similarity threshold for detection. Defaults to 0.75.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.lemmatizer = WordNetLemmatizer()
        self.window_size = window_size
        self.threshold = threshold

        self.raw_phrases = phrases
        lemmatized_phrases = [self.lemmatize_text(p) for p in phrases]
        self.phrase_embeddings = normalize(self.model.encode(lemmatized_phrases))

    def clean_text(self, text):
        """
        Cleans input text by removing formatting, HTML, and special characters.

        Args:
            text (str): Raw input text.

        Returns:
            str: Cleaned and formatted plain text.
        """
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'(\n\|.*\|\n)+', '\n', text)
        text = re.sub(r'\n\|[-:| ]+\|\n', '\n', text)
        text = re.sub(r'\|[^|]+\|', '', text)
        text = text.replace('|', '')
        text = re.sub(r'\n{2,}', '\n\n', text)
        text = re.sub(r'\[\^?\d+\^?\]|\[\d+(,\s*\d+)*(-\d+)?\]', '', text)
        text = re.sub(r'\(([^()]+ et al\., \d{4})\)', '', text)
        text = re.sub(r'\b(Fig|Table|Eqn?|Sec(tion)?)\.?\s*\d+(\.\d+)?', '', text)
        text = re.sub(r'\b\d+\s*[-–]\s*\d+\b', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'\$(.*?)\$', '', text)
        return text.strip()

    def lemmatize_text(self, text):
        """
        Tokenizes and lemmatizes the input text.

        Args:
            text (str): Text to lemmatize.

        Returns:
            str: Space-separated lemmatized words.
        """
        tokens = word_tokenize(text.lower())
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized)

    def create_windows(self, sentence):
        """
        Generates overlapping windows of words from a sentence for local phrase comparison.

        Args:
            sentence (str): Sentence to create windows from.

        Returns:
            List[str]: List of text windows.
        """
        lemmatized = self.lemmatize_text(sentence)
        words = lemmatized.split()

        if len(words) <= self.window_size:
            return [lemmatized]

        windows = []
        for i in range(len(words) - self.window_size + 1):
            window = ' '.join(words[i:i + self.window_size])
            windows.append(window)

        return windows

    def process_text(self, text):
        """
        Detects semantically similar phrases in the input text.

        Args:
            text (str): Input document text to analyze.

        Returns:
            List[Dict]: List of flagged sentences with the detected phrase and similarity score.
        """
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)

        flagged_sentences = []

        for sentence in sentences:
            windows = self.create_windows(sentence)

            window_embeddings = normalize(self.model.encode(windows))
            similarity_matrix = np.dot(self.phrase_embeddings, window_embeddings.T)

            max_similarities = similarity_matrix.max(axis=1)
            best_phrase_idx = max_similarities.argmax()
            max_score = max_similarities[best_phrase_idx].item()

            if max_score >= self.threshold:
                best_phrase = self.raw_phrases[best_phrase_idx]

                print(f"Detected phrase: '{best_phrase}' | Sentence: '{sentence}' | Score: {max_score:.4f}")

                flagged_sentences.append({
                    "sentence": sentence,
                    "detected_phrase": best_phrase,
                    "similarity_score": max_score
                })

        return flagged_sentences
