import re
import shlex
from typing import List

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("wordnet")


class Tokenizer:
    def __init__(self, n_gram: int = 1, ignore_quotes: bool = True):
        self._n_gram = n_gram
        self._ignore_quotes = ignore_quotes
        self._stopwords = stopwords.words("english")
        self._lemmatizer = WordNetLemmatizer()

    @staticmethod
    def tokenize_without_quotes(text: str) -> List[str]:
        return text.split(" ")

    @staticmethod
    def tokenize_with_quotes(text: str) -> List[str]:
        try:
            split = shlex.split(text)
        except:
            split = Tokenizer.tokenize_without_quotes(text)
        return split

    def preprocess(self, text: str) -> str:
        text = re.sub("[']", "", text).lower()
        return text

    def postprocess(self, tokens: List[str]) -> List[str]:
        tokens = [re.sub(r"[\"`!?\[|\]\(\)\-=\\>._%$#*+,@<&“…{}—:/]", "", token) for token in tokens]
        tokens = [token for token in tokens if token not in self._stopwords]
        tokens = [self._lemmatizer.lemmatize(token) for token in tokens if len(token) > 0]
        return tokens

    def _make_n_grams(self, tokens: List[str]) -> List[str]:
        if self._n_gram == 1:
            return tokens

        n_grams = []
        for i in range(1, len(tokens)):
            n_grams.append(" ".join([tokens[i - 1], tokens[i]]))

        return n_grams

    def tokenize(self, text: str) -> List[str]:
        text = self.preprocess(text)

        if self._ignore_quotes:
            tokens = Tokenizer.tokenize_without_quotes(text)
        else:
            tokens = Tokenizer.tokenize_with_quotes(text)

        tokens = self._make_n_grams(self.postprocess(tokens))
        return tokens
