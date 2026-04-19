"""分散表現: Word2Vec / Doc2Vec / SBERT。"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import optional_import


class Word2VecTrainer:
    """gensim Word2Vec ラッパー。"""

    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 5,
        epochs: int = 10,
        sg: int = 1,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.model = None

    def fit(self, docs: list[list[str]]) -> "Word2VecTrainer":
        gensim = optional_import("gensim", "Word2Vec")
        from gensim.models import Word2Vec
        self.model = Word2Vec(
            sentences=docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
            workers=4,
        )
        return self

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        assert self.model is not None
        if word not in self.model.wv:
            return []
        return self.model.wv.most_similar(word, topn=topn)

    def analogy(self, positive: list[str], negative: list[str], topn: int = 5) -> list[tuple[str, float]]:
        assert self.model is not None
        positive = [w for w in positive if w in self.model.wv]
        negative = [w for w in negative if w in self.model.wv]
        if not positive:
            return []
        return self.model.wv.most_similar(positive=positive, negative=negative, topn=topn)

    def vector(self, word: str) -> Optional[np.ndarray]:
        if self.model is None or word not in self.model.wv:
            return None
        return self.model.wv[word]

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str) -> "Word2VecTrainer":
        from gensim.models import Word2Vec
        self.model = Word2Vec.load(path)
        return self


class Doc2VecTrainer:
    """gensim Doc2Vec ラッパー (PV-DM)。"""

    def __init__(self, vector_size: int = 200, window: int = 5, min_count: int = 5, epochs: int = 20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None

    def fit(self, docs: list[list[str]]) -> "Doc2VecTrainer":
        gensim = optional_import("gensim", "Doc2Vec")
        from gensim.models import Doc2Vec
        from gensim.models.doc2vec import TaggedDocument
        tagged = [TaggedDocument(words=d, tags=[i]) for i, d in enumerate(docs)]
        self.model = Doc2Vec(
            documents=tagged,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4,
        )
        return self

    def infer_vector(self, doc: list[str]) -> np.ndarray:
        assert self.model is not None
        return self.model.infer_vector(doc)

    def document_vectors(self) -> np.ndarray:
        assert self.model is not None
        return np.vstack([self.model.dv[i] for i in range(len(self.model.dv))])

    def save(self, path: str):
        self.model.save(path)


class SBERTEncoder:
    """Sentence-BERT エンコーダ (多言語・日本語対応モデルがデフォルト)。"""

    DEFAULT_MODELS = {
        "ja": "pkshatech/GLuCoSE-base-ja",
        "multi": "intfloat/multilingual-e5-base",
        "en": "sentence-transformers/all-MiniLM-L6-v2",
    }

    def __init__(self, model_name: Optional[str] = None, language: str = "multi"):
        st = optional_import("sentence_transformers", "SBERT")
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or self.DEFAULT_MODELS.get(language, self.DEFAULT_MODELS["multi"])
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
