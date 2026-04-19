"""クラスタリング・次元削減・自己組織化マップ。"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import silhouette_score

from .utils import optional_import

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


def _vectorize(docs: list[list[str]], vectorizer: str = "tfidf") -> np.ndarray:
    joined = [" ".join(d) for d in docs]
    if vectorizer == "tfidf":
        X = TfidfVectorizer(token_pattern=r"(?u)\S+").fit_transform(joined).toarray()
    elif vectorizer == "count":
        from sklearn.feature_extraction.text import CountVectorizer
        X = CountVectorizer(token_pattern=r"(?u)\S+").fit_transform(joined).toarray()
    else:
        raise ValueError(f"unknown vectorizer: {vectorizer}")
    return X


class DocumentClustering:
    """文書クラスタリング。"""

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: int = 5,
        vectorizer: str = "tfidf",
        random_state: int = 42,
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.vectorizer = vectorizer
        self.random_state = random_state
        self.X_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.linkage_: Optional[np.ndarray] = None

    def fit_predict(self, docs: list[list[str]]) -> np.ndarray:
        self.X_ = _vectorize(docs, self.vectorizer)
        if self.method == "kmeans":
            model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            self.labels_ = model.fit_predict(self.X_)
        elif self.method == "hierarchical":
            self.linkage_ = linkage(self.X_, method="ward")
            self.labels_ = fcluster(self.linkage_, t=self.n_clusters, criterion="maxclust") - 1
        elif self.method == "dbscan":
            self.labels_ = DBSCAN(eps=0.5, min_samples=3).fit_predict(self.X_)
        elif self.method == "hdbscan":
            hdb = optional_import("hdbscan", "HDBSCAN")
            self.labels_ = hdb.HDBSCAN(min_cluster_size=3).fit_predict(self.X_)
        else:
            raise ValueError(f"unknown method: {self.method}")
        return self.labels_

    def plot_dendrogram(
        self,
        figsize: tuple = (15, 8),
        labels: Optional[list] = None,
    ) -> plt.Figure:
        if self.linkage_ is None:
            raise RuntimeError("hierarchical で fit_predict を先に実行してください")
        from scipy.cluster.hierarchy import dendrogram
        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(self.linkage_, labels=labels, ax=ax, leaf_rotation=90)
        return fig

    def silhouette_score(self) -> float:
        if self.X_ is None or self.labels_ is None:
            raise RuntimeError("fit_predict を先に実行してください")
        valid = self.labels_ != -1
        if valid.sum() < 2 or len(set(self.labels_[valid])) < 2:
            return 0.0
        return float(silhouette_score(self.X_[valid], self.labels_[valid]))


class DimensionReducer:
    """次元削減 (PCA / t-SNE / UMAP / MDS)。"""

    def __init__(
        self,
        method: str = "umap",
        n_components: int = 2,
        random_state: int = 42,
    ):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.coords_: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "pca":
            self.coords_ = PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        elif self.method == "tsne":
            self.coords_ = TSNE(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        elif self.method == "umap":
            umap = optional_import("umap", "UMAP")
            self.coords_ = umap.UMAP(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        elif self.method == "mds":
            self.coords_ = MDS(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        else:
            raise ValueError(f"unknown method: {self.method}")
        return self.coords_

    def plot_2d(
        self,
        labels: Optional[list] = None,
        texts: Optional[list] = None,
        figsize: tuple = (10, 8),
        title: str = "",
    ) -> plt.Figure:
        assert self.coords_ is not None, "fit_transform を先に実行してください"
        fig, ax = plt.subplots(figsize=figsize)
        if labels is not None:
            labs = np.asarray(labels)
            for lab in np.unique(labs):
                mask = labs == lab
                ax.scatter(self.coords_[mask, 0], self.coords_[mask, 1], label=str(lab), alpha=0.7)
            ax.legend()
        else:
            ax.scatter(self.coords_[:, 0], self.coords_[:, 1], alpha=0.7)
        if texts is not None:
            for i, t in enumerate(texts):
                ax.annotate(str(t), (self.coords_[i, 0], self.coords_[i, 1]), fontsize=8, alpha=0.6)
        ax.set_title(title or f"{self.method.upper()}")
        return fig


class SOM:
    """自己組織化マップ (MiniSom ラッパー)。"""

    def __init__(self, x: int = 10, y: int = 10, sigma: float = 1.0, learning_rate: float = 0.5):
        minisom = optional_import("minisom", "SOM")
        self.x = x
        self.y = y
        self._som = None
        self._sigma = sigma
        self._lr = learning_rate
        self._MiniSom = minisom.MiniSom

    def fit(self, X: np.ndarray, iterations: int = 500) -> "SOM":
        self._som = self._MiniSom(self.x, self.y, X.shape[1], sigma=self._sigma, learning_rate=self._lr, random_seed=42)
        self._som.random_weights_init(X)
        self._som.train_random(X, iterations)
        return self

    def plot(self, figsize=(10, 10)) -> plt.Figure:
        assert self._som is not None, "fit を先に実行してください"
        fig, ax = plt.subplots(figsize=figsize)
        ax.pcolor(self._som.distance_map().T, cmap="bone_r")
        ax.set_title("SOM U-Matrix")
        return fig
