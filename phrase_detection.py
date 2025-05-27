from sentence_transformers import SentenceTransformer
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import normalize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class PhraseDetection:
    """
    Detects phrases within a text using semantic similarity of phrase embeddings
    and lemmatized windowed sentence segments.

    Attributes:
        model (SentenceTransformer): Pretrained sentence transformer model for embeddings.
        lemmatizer (WordNetLemmatizer): WordNet lemmatizer instance.
        window_size (int): Number of words per sliding window on sentences.
        threshold (float): Similarity threshold to consider a phrase detected.
        raw_phrases (list[str]): Original list of input phrases.
        phrase_embeddings (np.ndarray): Normalized embeddings of lemmatized phrases.
        alternatives (list[str]): List of alternative phrases for suggestions.
    """

    def __init__(self, phrases, alternatives, window_size=10, threshold=0.75):
        """
        Initializes the PhraseDetection instance.

        Args:
            phrases (list[str]): List of phrases to detect.
            alternatives (list[str]): Corresponding list of alternative phrases.
            window_size (int, optional): Number of words per window. Defaults to 10.
            threshold (float, optional): Cosine similarity threshold for detection. Defaults to 0.75.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.lemmatizer = WordNetLemmatizer()
        self.window_size = window_size
        self.threshold = threshold

        self.alternatives = alternatives
        self.raw_phrases = phrases
        lemmatized_phrases = [self.lemmatize_text(p) for p in phrases]
        self.phrase_embeddings = self.model.encode(lemmatized_phrases)

    def get_wordnet_pos(self, tag):
        """
        Maps POS tag from nltk.pos_tag to WordNet POS tag for lemmatization.

        Args:
            tag (str): POS tag from nltk.pos_tag.

        Returns:
            str: Corresponding WordNet POS tag.
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def clean_text(self, text):
        """
        Cleans the input text by removing HTML tags, markdown, citations, tables,
        extra whitespace, and other noise.

        Args:
            text (str): Raw input text.

        Returns:
            str: Cleaned text.
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
        return text.strip()

    def lemmatize_text(self, text):
        """
        Tokenizes and lemmatizes the input text using POS tagging.

        Args:
            text (str): Input string.

        Returns:
            str: Lemmatized version of input.
        """
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        lemmatized = [
            self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos))
            for word, pos in tagged
        ]
        return ' '.join(lemmatized)

    def window_sentences(self, sentence):
        """
        Splits sentence into lemmatized sliding word windows.

        Args:
            sentence (str): Sentence to process.

        Returns:
            list[str]: List of sliding word windows.
        """
        lemmatized = self.lemmatize_text(sentence)
        words = lemmatized.split()
        if len(words) <= self.window_size:
            return [lemmatized]
        return [' '.join(words[i:i+self.window_size]) for i in range(len(words) - self.window_size + 1)]

    def process_text(self, text):
        """
        Processes text, detects phrases above threshold, and recommends replacements.

        Args:
            text (str): Input text to analyze.

        Returns:
            list[dict]: List of detected phrases with sentence and alternatives.
        """
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)

        flagged_sentences = []
        for sentence in sentences:
            windows = self.window_sentences(sentence)
            window_embeddings = self.model.encode(windows)
            similarity_matrix = np.dot(self.phrase_embeddings, window_embeddings.T)
            max_per_phrase = similarity_matrix.max(axis=1)
            phrase_idx = max_per_phrase.argmax()
            max_score = max_per_phrase[phrase_idx]

            if max_score >= self.threshold:
                best_phrase = self.raw_phrases[phrase_idx]
                alternative = self.alternatives[phrase_idx]

                print(f"[Best Phrase: '{best_phrase}' | Sentence: '{sentence}' | Score: {max_score:.4f}]")

                flagged_sentences.append({
                    "Sentence": sentence,
                    "Phrase": best_phrase,
                    "Alternative": alternative
                })

        return flagged_sentences