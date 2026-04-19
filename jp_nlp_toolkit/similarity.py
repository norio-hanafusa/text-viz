"""FAISS による類似度検索。"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .embedding import SBERTEncoder
from .utils import optional_import


class SimilaritySearch:
    """SBERT + FAISS で文書類似検索。"""

    def __init__(
        self,
        encoder: Optional[SBERTEncoder] = None,
        index_type: str = "flat",
    ):
        self.encoder = encoder or SBERTEncoder(language="multi")
        self.index_type = index_type
        self.index = None
        self._docs: list[str] = []
        self._vectors: Optional[np.ndarray] = None

    def build_index(self, docs: list[str]) -> "SimilaritySearch":
        faiss = optional_import("faiss", "FAISS")
        self._docs = list(docs)
        vecs = self.encoder.encode(self._docs, normalize=True).astype("float32")
        self._vectors = vecs
        dim = vecs.shape[1]

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = max(4, int(len(docs) ** 0.5))
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(vecs)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"unknown index_type: {self.index_type}")

        self.index.add(vecs)
        return self

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        assert self.index is not None, "build_index() を先に呼んでください"
        qv = self.encoder.encode([query], normalize=True).astype("float32")
        scores, idxs = self.index.search(qv, top_k)
        rows = []
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0])):
            if i < 0:
                continue
            rows.append({"rank": rank, "doc_id": int(i), "score": float(s), "text": self._docs[i]})
        return pd.DataFrame(rows)

    def save(self, path: str) -> str:
        faiss = optional_import("faiss", "FAISS")
        p = Path(path)
        faiss.write_index(self.index, str(p.with_suffix(".faiss")))
        with p.with_suffix(".pkl").open("wb") as f:
            pickle.dump({"docs": self._docs, "model_name": self.encoder.model_name}, f)
        return str(p)

    def load(self, path: str) -> "SimilaritySearch":
        faiss = optional_import("faiss", "FAISS")
        p = Path(path)
        self.index = faiss.read_index(str(p.with_suffix(".faiss")))
        with p.with_suffix(".pkl").open("rb") as f:
            data = pickle.load(f)
        self._docs = data["docs"]
        return self


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1
    v = vectors / norm
    return v @ v.T
