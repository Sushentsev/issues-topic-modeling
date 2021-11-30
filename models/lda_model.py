from typing import List, Optional

from gensim import matutils
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from gensim import corpora
from tokenizer import Tokenizer

import pyLDAvis.gensim_models


def coherence_scores(tokenizer: Tokenizer, texts: List[str],
                     components: List[int], random_state: Optional[int] = None) -> List[float]:
    scores = []

    vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize)
    X = vectorizer.fit_transform(texts)
    for n_components in tqdm(components):
        lda = LdaModel(corpus=matutils.Sparse2Corpus(X, documents_columns=False),
                       num_topics=n_components,
                       id2word={i: name for i, name in enumerate(vectorizer.get_feature_names())},
                       random_state=random_state)
        coherence_model = CoherenceModel(model=lda,
                                         corpus=matutils.Sparse2Corpus(X, documents_columns=False),
                                         coherence="u_mass")
        scores.append(coherence_model.get_coherence())

    return scores


class LDAModel:
    def __init__(self, n_components: int, tokenizer: Tokenizer, random_state: Optional[int] = None):
        self._n_components = n_components
        self._random_state = random_state
        self._lda = None
        self._vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize)

    def fit(self, texts: List[str]) -> "LDAModel":
        X = self._vectorizer.fit_transform(texts)
        self._lda = LdaModel(corpus=matutils.Sparse2Corpus(X, documents_columns=False),
                             num_topics=self._n_components,
                             id2word={i: name for i, name in enumerate(self._vectorizer.get_feature_names_out())},
                             random_state=self._random_state)

        return self

    def print_topic(self, topic_no: int) -> str:
        return self._lda.print_topic(topic_no)

    def get_topic_terms(self, topic_no: int, top_n: int = 10) -> List[str]:
        terms = self._lda.get_topic_terms(topic_no, top_n)
        return self._vectorizer.inverse_transform([terms])[0]

    def get_topics(self, texts: List[str]) -> List[int]:
        topics = []
        X = self._vectorizer.transform(texts)

        for i, x in enumerate(X):
            probs = self._lda.get_document_topics([(i, count) for i, count in enumerate(x.toarray()[0]) if count > 0])
            topics_sorted = sorted(probs, key=lambda pair: -pair[1])
            if len(topics_sorted) > 0:
                topics.append(topics_sorted[0][0])
            else:
                topics.append(-1)

        return topics

    def visualize(self, texts: List[str]):
        pyLDAvis.enable_notebook()
        X = self._vectorizer.transform(texts)

        dictionary = corpora.Dictionary()
        dictionary.token2id = self._vectorizer.vocabulary_
        dictionary.id2token = {i: name for i, name in enumerate(self._vectorizer.get_feature_names())}

        prepared = pyLDAvis.gensim_models.prepare(self._lda, matutils.Sparse2Corpus(X, documents_columns=False),
                                           dictionary)
        return prepared
