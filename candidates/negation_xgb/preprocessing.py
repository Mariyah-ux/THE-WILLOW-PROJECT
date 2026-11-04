"""Corrected candidate transformation, shared by training and inference.

Training previously calculated stems but returned lemmas of the original tokens.
That behavior is preserved. Install NLTK data explicitly; never download on import.
"""
import re
import unicodedata
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


@lru_cache(maxsize=1)
def _nlp_tools():
    try:
        words = set(stopwords.words("english"))
        words -= {"not", "no", "never", "nor", "neither", "but", "although",
                  "however", "unless", "if", "against"}
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
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'").replace("‘", "'")
    # Expand before punctuation removal; retain the historical noun lemmatizer.
    for pattern, replacement in (
        (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"),
        (r"\bshan't\b", "shall not"), (r"\bcannot\b", "can not"),
        (r"\bain't\b", "is not"), (r"n't\b", " not"),
        (r"\bi'm\b", "i am"), (r"'re\b", " are"),
        (r"'ve\b", " have"), (r"'ll\b", " will"),
        (r"'d\b", " would"),
        (r"\b(it|he|she|that|there|what|who)'s\b", r"\1 is"),
    ):
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    return " ".join(lemmatizer.lemmatize(word) for word in tokens)
