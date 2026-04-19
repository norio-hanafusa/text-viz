"""頻度分析・KWIC・共起統計量。"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def word_frequency(
    docs: list[list[str]],
    top_n: Optional[int] = 100,
) -> pd.DataFrame:
    """単語頻度 (TF) と文書頻度 (DF)。"""
    tf: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for doc in docs:
        tf.update(doc)
        df.update(set(doc))

    rows = [{"word": w, "tf": c, "df": df[w]} for w, c in tf.items()]
    if not rows:
        return pd.DataFrame(columns=["word", "tf", "df"])
    out = pd.DataFrame(rows).sort_values("tf", ascending=False).reset_index(drop=True)
    return out.head(top_n) if top_n else out


def ngram_frequency(
    docs: list[list[str]],
    n: int = 2,
    top_n: Optional[int] = 100,
) -> pd.DataFrame:
    c: Counter[tuple] = Counter()
    for doc in docs:
        c.update(tuple(doc[i : i + n]) for i in range(len(doc) - n + 1))
    rows = [{"ngram": " ".join(k), "count": v} for k, v in c.items()]
    if not rows:
        return pd.DataFrame(columns=["ngram", "count"])
    out = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return out.head(top_n) if top_n else out


def tfidf(
    docs: list[list[str]],
    max_features: int = 5000,
) -> tuple[np.ndarray, list[str]]:
    """TF-IDF 行列 (shape: n_docs × n_features) と特徴語リストを返す。"""
    joined = [" ".join(d) for d in docs]
    vec = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\S+")
    X = vec.fit_transform(joined).toarray()
    return X, list(vec.get_feature_names_out())


class KWIC:
    """KWIC コンコーダンス (Keyword-in-Context) 検索。"""

    def __init__(self, docs: list[str], tokenizer=None):
        """docs は原文 (文字列) のリスト。tokenizer があれば分かち書きに使う。"""
        self.docs = docs
        self.tokenizer = tokenizer
        self._tokenized: list[list[str]] = [
            self._tokenize(d) for d in docs
        ]

    def _tokenize(self, text: str) -> list[str]:
        if self.tokenizer is None:
            return text.split()
        return self.tokenizer.tokenize_to_surface(text)

    def search(
        self,
        keyword: str,
        window: int = 5,
        max_results: int = 100,
    ) -> pd.DataFrame:
        rows = []
        for doc_id, tokens in enumerate(self._tokenized):
            for i, t in enumerate(tokens):
                if t == keyword:
                    left = " ".join(tokens[max(0, i - window) : i])
                    right = " ".join(tokens[i + 1 : i + 1 + window])
                    rows.append({
                        "doc_id": doc_id,
                        "left_context": left,
                        "keyword": t,
                        "right_context": right,
                    })
                    if len(rows) >= max_results:
                        return pd.DataFrame(rows)
        return pd.DataFrame(rows)

    def search_regex(
        self,
        pattern: str,
        window: int = 5,
        max_results: int = 100,
    ) -> pd.DataFrame:
        rgx = re.compile(pattern)
        rows = []
        for doc_id, tokens in enumerate(self._tokenized):
            for i, t in enumerate(tokens):
                if rgx.search(t):
                    left = " ".join(tokens[max(0, i - window) : i])
                    right = " ".join(tokens[i + 1 : i + 1 + window])
                    rows.append({
                        "doc_id": doc_id,
                        "left_context": left,
                        "keyword": t,
                        "right_context": right,
                    })
                    if len(rows) >= max_results:
                        return pd.DataFrame(rows)
        return pd.DataFrame(rows)


def _cooccur_counts(
    docs: list[list[str]],
    target_word: str,
    window: Optional[int] = None,
) -> tuple[int, int, Counter, int]:
    """(n_docs_with_target, total_windows_or_docs, co_counts, N) を返す。"""
    if window is None:
        # 文書単位の共起
        N = len(docs)
        target_docs = [set(d) for d in docs if target_word in d]
        co: Counter = Counter()
        for s in target_docs:
            for w in s:
                if w != target_word:
                    co[w] += 1
        return len(target_docs), N, co, N

    # 窓単位
    N = 0
    target_windows = 0
    co: Counter = Counter()
    for doc in docs:
        for i, t in enumerate(doc):
            N += 1
            if t == target_word:
                target_windows += 1
                ctx = doc[max(0, i - window) : i] + doc[i + 1 : i + 1 + window]
                for w in ctx:
                    co[w] += 1
    return target_windows, N, co, N


def cooccurrence_stats(
    docs: list[list[str]],
    target_word: str,
    measure: str = "jaccard",
    window: Optional[int] = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """measure: 'jaccard' | 'dice' | 'pmi' | 't_score' | 'log_likelihood'"""
    df_counts: Counter = Counter()
    for d in docs:
        df_counts.update(set(d))
    n_target, total, co, N = _cooccur_counts(docs, target_word, window)
    if n_target == 0:
        return pd.DataFrame(columns=["word", "cooccur", "score"])

    rows = []
    for w, c in co.items():
        n_w = df_counts[w]
        score = _score(measure, c, n_target, n_w, N)
        rows.append({"word": w, "cooccur": c, "score": score})
    out = (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return out.head(top_n)


def _score(measure: str, c: int, a: int, b: int, N: int) -> float:
    """a = target 出現数、b = w 出現数、c = 共起数、N = 全文書数。"""
    if measure == "jaccard":
        denom = a + b - c
        return c / denom if denom else 0.0
    if measure == "dice":
        denom = a + b
        return (2 * c) / denom if denom else 0.0
    if measure == "pmi":
        if c == 0 or a == 0 or b == 0:
            return 0.0
        return math.log2((c * N) / (a * b))
    if measure == "t_score":
        expected = (a * b) / N if N else 0
        return (c - expected) / math.sqrt(c) if c else 0.0
    if measure == "log_likelihood":
        # Dunning log-likelihood
        def ll(k, n, p):
            if p <= 0 or p >= 1 or n == 0:
                return 0
            return k * math.log(p) + (n - k) * math.log(1 - p)

        p = b / N if N else 0
        p1 = c / a if a else 0
        p2 = (b - c) / (N - a) if (N - a) else 0
        if p in (0, 1) or a in (0, N):
            return 0.0
        try:
            score = -2 * (
                ll(c, a, p) + ll(b - c, N - a, p)
                - ll(c, a, p1) - ll(b - c, N - a, p2)
            )
            return score
        except ValueError:
            return 0.0
    raise ValueError(f"unknown measure: {measure}")
