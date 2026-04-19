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
    """自己組織化マップ (MiniSom ラッパー)。

    学習後は以下の多彩な可視化で「各セルが何を表すか」を評価できる:
      - plot()               : U-Matrix (クラスタ境界)
      - hit_map()            : 文書度数マップ
      - component_plane()    : 指定した特徴 (単語) の重みマップ
      - top_words_per_cell() : 各セルの代表語 (テキスト)
      - top_words_fig()      : 各セルの代表語を U-Matrix 上に重ね描き
      - representative_docs_df(): 各セルの代表文書 (DataFrame)
      - label_overlay()      : カテゴリ変数の支配色マップ
    """

    def __init__(self, x: int = 10, y: int = 10, sigma: float = 1.0, learning_rate: float = 0.5):
        minisom = optional_import("minisom", "SOM")
        self.x = x
        self.y = y
        self._som = None
        self._sigma = sigma
        self._lr = learning_rate
        self._MiniSom = minisom.MiniSom
        self.X_: Optional[np.ndarray] = None
        self._bmus_cache: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, iterations: int = 500) -> "SOM":
        self._som = self._MiniSom(
            self.x, self.y, X.shape[1],
            sigma=self._sigma, learning_rate=self._lr, random_seed=42,
        )
        self._som.random_weights_init(X)
        self._som.train_random(X, iterations)
        self.X_ = X
        self._bmus_cache = None
        return self

    def _require_fit(self):
        assert self._som is not None and self.X_ is not None, "fit() を先に実行してください"

    def _bmus(self) -> np.ndarray:
        """全サンプルの BMU (shape: N×2) をキャッシュ付きで返す。"""
        self._require_fit()
        if self._bmus_cache is None:
            self._bmus_cache = np.array([self._som.winner(x) for x in self.X_])
        return self._bmus_cache

    # -----------------------------------------------------------------
    # 1. U-Matrix
    # -----------------------------------------------------------------
    def plot(self, figsize: tuple = (10, 10)) -> plt.Figure:
        """U-Matrix: 近傍セル間距離のヒートマップ (濃色 = クラスタ境界)。"""
        assert self._som is not None, "fit() を先に実行してください"
        fig, ax = plt.subplots(figsize=figsize)
        pc = ax.pcolor(self._som.distance_map().T, cmap="bone_r")
        ax.set_title("SOM U-Matrix (濃色 = クラスタ境界)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(pc, ax=ax, label="近傍距離")
        return fig

    # -----------------------------------------------------------------
    # 2. Hit Map
    # -----------------------------------------------------------------
    def hit_map(self, figsize: tuple = (10, 10)) -> plt.Figure:
        """各セルにマップされた文書数のヒートマップ。"""
        self._require_fit()
        bmus = self._bmus()
        counts = np.zeros((self.x, self.y), dtype=int)
        for i, j in bmus:
            counts[i, j] += 1

        fig, ax = plt.subplots(figsize=figsize)
        pc = ax.pcolor(counts.T, cmap="YlGnBu")
        for i in range(self.x):
            for j in range(self.y):
                if counts[i, j] > 0:
                    ax.text(i + 0.5, j + 0.5, str(counts[i, j]),
                            ha="center", va="center", fontsize=8,
                            color="white" if counts[i, j] > counts.max() / 2 else "black")
        ax.set_title(f"Hit Map (総文書数 {counts.sum()} / 使用セル数 {(counts > 0).sum()})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(pc, ax=ax, label="文書数")
        return fig

    # -----------------------------------------------------------------
    # 3. Component Plane
    # -----------------------------------------------------------------
    def component_plane(
        self,
        feature_idx: int,
        feature_name: str = "",
        figsize: tuple = (10, 10),
    ) -> plt.Figure:
        """指定した特徴 (単語) の重みを格子上にヒートマップ化。"""
        self._require_fit()
        weights = self._som.get_weights()  # (x, y, dim)
        if feature_idx < 0 or feature_idx >= weights.shape[2]:
            raise IndexError(f"feature_idx {feature_idx} out of range (0..{weights.shape[2] - 1})")
        plane = weights[:, :, feature_idx]

        fig, ax = plt.subplots(figsize=figsize)
        pc = ax.pcolor(plane.T, cmap="viridis")
        title = f"Component Plane: {feature_name}" if feature_name else f"Component Plane (dim {feature_idx})"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(pc, ax=ax, label="重み")
        return fig

    # -----------------------------------------------------------------
    # 4. Top-N words per cell
    # -----------------------------------------------------------------
    def top_words_per_cell(
        self,
        feature_names: list,
        top_n: int = 3,
    ) -> list:
        """各セルの重みが大きい上位 N 語を返す。shape: x × y × top_n の入れ子リスト。"""
        self._require_fit()
        weights = self._som.get_weights()  # (x, y, dim)
        features = np.asarray(feature_names)
        result = []
        for i in range(self.x):
            row = []
            for j in range(self.y):
                w = weights[i, j]
                idx = np.argsort(w)[::-1][:top_n]
                row.append(list(features[idx]))
            result.append(row)
        return result

    def top_words_fig(
        self,
        feature_names: list,
        top_n: int = 3,
        figsize: tuple = (12, 12),
        fontsize: int = 7,
    ) -> plt.Figure:
        """U-Matrix の上に各セルの top-N 語を重ね描き。"""
        self._require_fit()
        words = self.top_words_per_cell(feature_names, top_n=top_n)

        fig, ax = plt.subplots(figsize=figsize)
        ax.pcolor(self._som.distance_map().T, cmap="bone_r", alpha=0.6)
        for i in range(self.x):
            for j in range(self.y):
                text = "\n".join(words[i][j])
                ax.text(i + 0.5, j + 0.5, text, ha="center", va="center",
                        fontsize=fontsize, color="#2c5282", weight="bold")
        ax.set_title(f"Top-{top_n} words per cell")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return fig

    # -----------------------------------------------------------------
    # 5. Representative documents
    # -----------------------------------------------------------------
    def representative_docs_df(
        self,
        docs: list,
        top_per_cell: int = 1,
        doc_preview_len: int = 200,
    ) -> "pd.DataFrame":
        """各セルに属する文書のうち重みに最も近い top_per_cell 件を返す。"""
        self._require_fit()
        bmus = self._bmus()
        weights = self._som.get_weights()
        rows = []
        for i in range(self.x):
            for j in range(self.y):
                mask = (bmus[:, 0] == i) & (bmus[:, 1] == j)
                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    continue
                dists = np.linalg.norm(self.X_[idxs] - weights[i, j], axis=1)
                sort_ix = np.argsort(dists)[:top_per_cell]
                for k in sort_ix:
                    doc_idx = int(idxs[k])
                    doc_text = str(docs[doc_idx]) if doc_idx < len(docs) else ""
                    preview = (doc_text[:doc_preview_len] +
                               ("…" if len(doc_text) > doc_preview_len else ""))
                    rows.append({
                        "cell_x": int(i), "cell_y": int(j),
                        "n_in_cell": int(len(idxs)),
                        "doc_idx": doc_idx,
                        "distance": float(dists[k]),
                        "doc": preview,
                    })
        return pd.DataFrame(rows).sort_values(["cell_x", "cell_y"]).reset_index(drop=True)

    # -----------------------------------------------------------------
    # 6. Label overlay
    # -----------------------------------------------------------------
    def label_overlay(
        self,
        labels: list,
        figsize: tuple = (12, 10),
        annotate: bool = True,
    ) -> plt.Figure:
        """各セルに属する文書の支配的カテゴリで色分け。"""
        from collections import Counter
        self._require_fit()
        assert len(labels) == len(self.X_), "labels は学習データと同じ長さである必要があります"

        labels_arr = np.asarray([str(l) for l in labels])
        bmus = self._bmus()
        unique_labels = sorted(set(labels_arr))
        color_map = {lab: idx for idx, lab in enumerate(unique_labels)}

        # 支配ラベルと文書数を計算
        dominant = [["" for _ in range(self.y)] for _ in range(self.x)]
        counts = np.zeros((self.x, self.y), dtype=int)
        color_grid = np.full((self.x, self.y), -1, dtype=float)
        for i in range(self.x):
            for j in range(self.y):
                mask = (bmus[:, 0] == i) & (bmus[:, 1] == j)
                n = int(mask.sum())
                if n == 0:
                    continue
                counts[i, j] = n
                c = Counter(labels_arr[mask])
                top_lab, top_cnt = c.most_common(1)[0]
                dominant[i][j] = f"{top_lab}\n({top_cnt}/{n})"
                color_grid[i, j] = color_map[top_lab]

        # マスク: 空セルは灰色に
        masked = np.ma.masked_where(color_grid < 0, color_grid)
        cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 2))

        fig, ax = plt.subplots(figsize=figsize)
        pc = ax.pcolor(masked.T, cmap=cmap, vmin=0, vmax=max(len(unique_labels) - 1, 1))
        if annotate:
            for i in range(self.x):
                for j in range(self.y):
                    if dominant[i][j]:
                        ax.text(i + 0.5, j + 0.5, dominant[i][j],
                                ha="center", va="center", fontsize=7, color="white")
        ax.set_title(f"Dominant Category per Cell ({len(unique_labels)} categories)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # 凡例
        import matplotlib.patches as mpatches
        denom = max(len(unique_labels) - 1, 1)
        patches = [
            mpatches.Patch(color=cmap(color_map[l] / denom), label=l)
            for l in unique_labels
        ]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1),
                  loc="upper left", fontsize=8, title="カテゴリ")
        fig.tight_layout()
        return fig
