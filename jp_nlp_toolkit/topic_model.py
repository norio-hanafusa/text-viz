"""トピックモデル: LDA (gensim) / NMF (sklearn)。"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .utils import optional_import


class LDATopicModel:
    """gensim LDA。"""

    def __init__(self, n_topics: int = 10, passes: int = 5, random_state: int = 42):
        self.n_topics = n_topics
        self.passes = passes
        self.random_state = random_state
        self.model = None
        self.dictionary = None
        self.corpus = None
        self._docs: Optional[list[list[str]]] = None

    def fit(self, docs: list[list[str]]) -> "LDATopicModel":
        gensim = optional_import("gensim", "LDA")
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel
        self._docs = docs
        self.dictionary = Dictionary(docs)
        self.dictionary.filter_extremes(no_below=2, no_above=0.95)
        self.corpus = [self.dictionary.doc2bow(d) for d in docs]
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            passes=self.passes,
            random_state=self.random_state,
        )
        return self

    def get_topics(self, n_words: int = 10) -> list[list[tuple[str, float]]]:
        assert self.model is not None, "fit() を先に実行してください"
        return [
            self.model.show_topic(t, topn=n_words)
            for t in range(self.n_topics)
        ]

    def topics_dataframe(self, n_words: int = 10) -> pd.DataFrame:
        rows = []
        for i, topic in enumerate(self.get_topics(n_words)):
            for w, p in topic:
                rows.append({"topic": i, "word": w, "prob": p})
        return pd.DataFrame(rows)

    def predict(self, doc: list[str]) -> np.ndarray:
        assert self.model is not None
        bow = self.dictionary.doc2bow(doc)
        dist = self.model.get_document_topics(bow, minimum_probability=0.0)
        vec = np.zeros(self.n_topics)
        for t, p in dist:
            vec[t] = p
        return vec

    def document_topic_matrix(self) -> np.ndarray:
        assert self.model is not None
        return np.vstack([self.predict(d) for d in self._docs])

    def visualize(self, output: str = "lda.html") -> str:
        assert self.model is not None
        pyLDAvis = optional_import("pyLDAvis", "LDA 可視化")
        import pyLDAvis.gensim_models as gm
        vis = gm.prepare(self.model, self.corpus, self.dictionary)
        pyLDAvis.save_html(vis, output)
        return output

    def coherence_score(self, coherence: str = "c_v") -> float:
        from gensim.models import CoherenceModel
        cm = CoherenceModel(model=self.model, texts=self._docs, dictionary=self.dictionary, coherence=coherence)
        return float(cm.get_coherence())

    def optimal_n_topics(self, range_n: tuple = (2, 20), step: int = 2) -> pd.DataFrame:
        from gensim.models import LdaModel, CoherenceModel
        rows = []
        for n in range(range_n[0], range_n[1] + 1, step):
            model = LdaModel(
                corpus=self.corpus, id2word=self.dictionary,
                num_topics=n, passes=self.passes, random_state=self.random_state,
            )
            cm = CoherenceModel(model=model, texts=self._docs, dictionary=self.dictionary, coherence="c_v")
            rows.append({"n_topics": n, "coherence": float(cm.get_coherence())})
        return pd.DataFrame(rows)


class NMFTopicModel:
    """NMF (sklearn)。"""

    def __init__(self, n_topics: int = 10, random_state: int = 42):
        self.n_topics = n_topics
        self.random_state = random_state
        self.model = None
        self.feature_names_: Optional[list[str]] = None
        self.doc_topic_: Optional[np.ndarray] = None

    def fit(self, docs: list[list[str]]) -> "NMFTopicModel":
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer
        joined = [" ".join(d) for d in docs]
        vec = TfidfVectorizer(token_pattern=r"(?u)\S+")
        X = vec.fit_transform(joined)
        self.feature_names_ = list(vec.get_feature_names_out())
        self.model = NMF(n_components=self.n_topics, random_state=self.random_state, init="nndsvd")
        self.doc_topic_ = self.model.fit_transform(X)
        return self

    def get_topics(self, n_words: int = 10) -> list[list[str]]:
        assert self.model is not None
        feats = np.array(self.feature_names_)
        return [
            list(feats[np.argsort(comp)[::-1][:n_words]])
            for comp in self.model.components_
        ]

    def topics_dataframe(self, n_words: int = 10) -> pd.DataFrame:
        rows = []
        for i, topic in enumerate(self.get_topics(n_words)):
            for rank, w in enumerate(topic):
                rows.append({"topic": i, "rank": rank, "word": w})
        return pd.DataFrame(rows)
