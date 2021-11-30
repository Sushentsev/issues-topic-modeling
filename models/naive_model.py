from collections import Counter
from typing import List, Dict

import numpy as np

from tokenizer import Tokenizer


class FreqController:
    def __init__(self, max_freq: float = 1., min_freq: float = 0.):
        self._max_freq = max_freq
        self._min_freq = min_freq

    def filter(self, texts: List[List[str]]) -> List[List[str]]:
        freqs = Counter()

        for text in texts:
            freqs += Counter(set(text))

        freqs = {key: value / len(texts) for key, value in freqs.items()}

        return [[token for token in text if self._min_freq <= freqs[token] <= self._max_freq]
                for text in texts]


class NaiveModel:
    def __init__(self, n_words: int, tokenizer: Tokenizer, max_freq: float = 1., min_freq: float = 0.):
        self._n_words = n_words
        self._tokenizer = tokenizer
        self._freq_controller = FreqController(max_freq, min_freq)

    def _tfs(self, texts: List[List[str]]) -> Dict[str, float]:
        token_counter = Counter()
        for text in texts:
            token_counter.update(text)

        return {token: counts / sum(map(len, texts)) for token, counts in token_counter.items()}

    def _idfs(self, texts: List[List[str]]) -> Dict[str, float]:
        token_counter = Counter()
        for text in texts:
            token_counter.update(set(text))

        return {token: np.log(len(texts) / counts) for token, counts in token_counter.items()}

    def most_tf_words(self, texts: List[str]) -> List[str]:
        tokens = [self._tokenizer.tokenize(text) for text in texts]
        tokens = self._freq_controller.filter(tokens)
        tfs = self._tfs(tokens)
        tfs_sorted = dict(sorted(tfs.items(), key=lambda item: -item[1]))
        words = list(tfs_sorted.keys())
        return words[:self._n_words]

    def most_tf_idf_words(self, texts: List[str]) -> List[str]:
        tokens = [self._tokenizer.tokenize(text) for text in texts]
        tokens = self._freq_controller.filter(tokens)
        tfs = self._tfs(tokens)
        idfs = self._idfs(tokens)
        tfs_idfs = {token: tfs[token] * idfs[token] for token in tfs}
        tfs_idfs_sorted = dict(sorted(tfs_idfs.items(), key=lambda item: -item[1]))
        words = list(tfs_idfs_sorted.keys())
        return words[:self._n_words]
