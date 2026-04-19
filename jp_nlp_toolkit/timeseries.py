"""時系列テキスト分析 — 語の推移・急増語・期間別ネットワーク。"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from .cooccurrence import CooccurrenceNetwork

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


class TemporalAnalyzer:
    """日付列と分かち書き済みトークン列を持つ DataFrame を時系列解析。

    df: pd.DataFrame
    text_col: トークン列名 (list[str] の値を持つ) または文字列 (空白区切り)
    date_col: pd.Timestamp に変換可能な列名
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        date_col: str,
    ):
        self.df = df.copy()
        self.text_col = text_col
        self.date_col = date_col
        self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
        self.df = self.df.dropna(subset=[date_col])
        # トークン列を標準化
        first = self.df[text_col].iloc[0] if len(self.df) else None
        if isinstance(first, str):
            self.df["_tokens"] = self.df[text_col].fillna("").map(str.split)
        else:
            self.df["_tokens"] = self.df[text_col].apply(lambda x: list(x) if x is not None else [])

    def _period_key(self, freq: str) -> pd.Series:
        return self.df[self.date_col].dt.to_period(freq).astype(str)

    def word_trend(self, words: list[str], freq: str = "M") -> pd.DataFrame:
        """freq: 'D' | 'W' | 'M' | 'Q' | 'Y'"""
        keys = self._period_key(freq)
        rows = []
        for period in sorted(keys.unique()):
            sub_tokens: list[str] = []
            for toks in self.df.loc[keys == period, "_tokens"]:
                sub_tokens.extend(toks)
            counts = Counter(sub_tokens)
            total = sum(counts.values())
            for w in words:
                rows.append({
                    "period": period,
                    "word": w,
                    "count": counts.get(w, 0),
                    "freq": counts.get(w, 0) / total if total else 0.0,
                })
        return pd.DataFrame(rows)

    def plot_trend(
        self,
        words: list[str],
        freq: str = "M",
        metric: str = "count",
        figsize=(12, 6),
    ) -> plt.Figure:
        trend = self.word_trend(words, freq=freq)
        fig, ax = plt.subplots(figsize=figsize)
        for w in words:
            sub = trend[trend["word"] == w]
            ax.plot(sub["period"], sub[metric], marker="o", label=w)
        ax.set_xlabel("期間")
        ax.set_ylabel(metric)
        ax.legend()
        plt.xticks(rotation=45)
        fig.tight_layout()
        return fig

    def emerging_words(
        self,
        window: str,
        baseline: str,
        top_n: int = 20,
        min_count: int = 3,
    ) -> pd.DataFrame:
        """window (比較対象期間) で baseline (基準期間) より急増した語。

        window/baseline は pd.Period 互換文字列 ('2024', '2024-03' 等)。
        """
        dates = self.df[self.date_col]
        base_mask = dates.astype(str).str.startswith(baseline)
        win_mask = dates.astype(str).str.startswith(window)

        base_counts: Counter = Counter()
        for toks in self.df.loc[base_mask, "_tokens"]:
            base_counts.update(toks)
        win_counts: Counter = Counter()
        for toks in self.df.loc[win_mask, "_tokens"]:
            win_counts.update(toks)

        rows = []
        vocab = set(base_counts) | set(win_counts)
        for w in vocab:
            b = base_counts.get(w, 0)
            v = win_counts.get(w, 0)
            if v < min_count:
                continue
            ratio = (v + 1) / (b + 1)  # Laplace 平滑化
            rows.append({
                "word": w,
                "baseline_count": b,
                "window_count": v,
                "ratio": ratio,
                "delta": v - b,
            })
        return (
            pd.DataFrame(rows)
            .sort_values("ratio", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def temporal_cooccurrence(
        self,
        freq: str = "Y",
        min_edge_weight: int = 2,
        top_n_nodes: int = 50,
    ) -> dict[str, nx.Graph]:
        """期間ごとの共起ネットワークを構築。"""
        keys = self._period_key(freq)
        out: dict[str, nx.Graph] = {}
        for period in sorted(keys.unique()):
            docs = self.df.loc[keys == period, "_tokens"].tolist()
            if not docs:
                continue
            net = CooccurrenceNetwork(docs, scope="document")
            out[period] = net.build(min_edge_weight=min_edge_weight, top_n_nodes=top_n_nodes)
        return out
