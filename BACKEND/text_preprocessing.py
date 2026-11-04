"""Effective training transformation, shared with inference.

Training previously calculated stems but returned lemmas of the original tokens.
That behavior is preserved. Install NLTK data explicitly; never download on import.
"""
import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


@lru_cache(maxsize=1)
def _nlp_tools():
    try:
        words = set(stopwords.words("english"))
        words -= {"not", "no", "never", "nor", "don't", "can't", "won't"}
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("resources")
        word_tokenize("resource check")
    except LookupError as error:
        raise RuntimeError(
            "Missing NLTK data. Run: python -m nltk.downloader "
            "punkt_tab stopwords wordnet"
        ) from error
    return words, lemmatizer


def clean_text(text):
    if not isinstance(text, str):
        return ""
    stop_words, lemmatizer = _nlp_tools()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    return " ".join(lemmatizer.lemmatize(word) for word in tokens)
