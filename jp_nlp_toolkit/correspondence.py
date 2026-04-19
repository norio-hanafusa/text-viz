"""対応分析 (CA / MCA)。計量テキスト分析で古くから用いられる手法 (Benzécri, 1973)。"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import optional_import

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


class CorrespondenceAnalysis:
    """語 × 外部変数 (カテゴリ) の対応分析。"""

    def __init__(self, n_components: int = 2):
        prince = optional_import("prince", "対応分析")
        self.n_components = n_components
        self.model = prince.CA(n_components=n_components)
        self._contingency: Optional[pd.DataFrame] = None
        self._word_coords: Optional[pd.DataFrame] = None
        self._label_coords: Optional[pd.DataFrame] = None

    def fit(
        self,
        docs: list[list[str]],
        labels: list[str],
        top_n_words: int = 50,
    ) -> "CorrespondenceAnalysis":
        """docs と同じ長さの labels (グループ名) で、グループ × 頻出語のクロス表を構築し CA。"""
        assert len(docs) == len(labels), "docs と labels の長さが一致しません"

        # 全体頻出語 top_n_words に絞る
        freq: Counter = Counter()
        for d in docs:
            freq.update(d)
        top_words = [w for w, _ in freq.most_common(top_n_words)]

        # 分割表 (行: ラベル, 列: 語)
        groups = sorted(set(labels))
        mat = np.zeros((len(groups), len(top_words)), dtype=int)
        g_idx = {g: i for i, g in enumerate(groups)}
        w_idx = {w: i for i, w in enumerate(top_words)}
        for doc, lab in zip(docs, labels):
            for w in doc:
                if w in w_idx:
                    mat[g_idx[lab], w_idx[w]] += 1
        self._contingency = pd.DataFrame(mat, index=groups, columns=top_words)

        self.model.fit(self._contingency)
        self._label_coords = self.model.row_coordinates(self._contingency)
        self._word_coords = self.model.column_coordinates(self._contingency)
        return self

    def get_coordinates(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(語の座標, ラベルの座標) を返す。"""
        assert self._word_coords is not None, "fit() を先に実行してください"
        return self._word_coords, self._label_coords

    def plot(
        self,
        figsize: tuple = (12, 10),
        annotate: bool = True,
        color_words: str = "tab:blue",
        color_labels: str = "tab:red",
        title: str = "対応分析",
    ) -> plt.Figure:
        assert self._word_coords is not None, "fit() を先に実行してください"
        fig, ax = plt.subplots(figsize=figsize)
        wc = self._word_coords.iloc[:, :2].values
        lc = self._label_coords.iloc[:, :2].values
        ax.scatter(wc[:, 0], wc[:, 1], c=color_words, alpha=0.6, s=40, label="語")
        ax.scatter(lc[:, 0], lc[:, 1], c=color_labels, alpha=0.9, s=120, marker="^", label="カテゴリ")
        if annotate:
            for i, w in enumerate(self._word_coords.index):
                ax.annotate(str(w), (wc[i, 0], wc[i, 1]), fontsize=9, alpha=0.7)
            for i, l in enumerate(self._label_coords.index):
                ax.annotate(str(l), (lc[i, 0], lc[i, 1]), fontsize=12, weight="bold", color=color_labels)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(title)
        ax.legend()
        return fig


class MultipleCorrespondenceAnalysis:
    """3 つ以上のカテゴリ変数を扱う MCA。"""

    def __init__(self, n_components: int = 2):
        prince = optional_import("prince", "MCA")
        self.n_components = n_components
        self.model = prince.MCA(n_components=n_components)

    def fit(self, df: pd.DataFrame) -> "MultipleCorrespondenceAnalysis":
        self.model.fit(df)
        self._df = df
        return self

    def plot(self, figsize=(12, 10), title: str = "MCA") -> plt.Figure:
        coords = self.model.column_coordinates(self._df)
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], alpha=0.7)
        for i, lab in enumerate(coords.index):
            ax.annotate(str(lab), (coords.iloc[i, 0], coords.iloc[i, 1]), fontsize=9)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(title)
        return fig
