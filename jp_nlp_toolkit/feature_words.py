"""グループ間の特徴語抽出 (χ² / 対数尤度 / Jaccard)。"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def _group_counts(docs: list[list[str]], labels: list) -> tuple[dict, dict, set]:
    """各グループの単語頻度・総頻度と全単語集合を返す。"""
    assert len(docs) == len(labels)
    groups: dict[str, Counter] = {}
    total: Counter = Counter()
    for d, lab in zip(docs, labels):
        groups.setdefault(lab, Counter()).update(d)
        total.update(d)
    vocab = set(total)
    return groups, total, vocab


def chi2_feature_words(
    docs: list[list[str]],
    labels: list,
    top_n: int = 30,
    min_count: int = 3,
) -> pd.DataFrame:
    """各グループの特徴語を χ² で抽出。"""
    groups, total, vocab = _group_counts(docs, labels)
    group_sums = {g: sum(c.values()) for g, c in groups.items()}
    grand = sum(group_sums.values())

    rows = []
    for g, gc in groups.items():
        for w in vocab:
            if total[w] < min_count:
                continue
            a = gc[w]
            b = group_sums[g] - a
            c = total[w] - a
            d = (grand - group_sums[g]) - c
            if a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
                continue
            table = np.array([[a, b], [c, d]])
            try:
                chi2, p, _, _ = chi2_contingency(table, correction=False)
            except ValueError:
                continue
            # 「グループ g で特徴的」= 期待値以上であること
            expected = (a + b) * (a + c) / grand
            if a < expected:
                continue
            rows.append({"group": g, "word": w, "count": a, "chi2": chi2, "p": p})
    out = (
        pd.DataFrame(rows)
        .sort_values(["group", "chi2"], ascending=[True, False])
        .groupby("group", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def log_likelihood_feature_words(
    docs: list[list[str]],
    labels: list,
    top_n: int = 30,
    min_count: int = 3,
) -> pd.DataFrame:
    """Dunning の対数尤度比で特徴語抽出。"""
    groups, total, vocab = _group_counts(docs, labels)
    group_sums = {g: sum(c.values()) for g, c in groups.items()}
    grand = sum(group_sums.values())

    def _ll(k, n, p):
        if p <= 0 or p >= 1 or n == 0:
            return 0
        return k * math.log(p) + (n - k) * math.log(1 - p)

    rows = []
    for g, gc in groups.items():
        c1 = group_sums[g]
        c2 = grand - c1
        for w in vocab:
            if total[w] < min_count:
                continue
            a = gc[w]
            b = total[w] - a
            if c1 == 0 or c2 == 0:
                continue
            p = total[w] / grand
            p1 = a / c1 if c1 else 0
            p2 = b / c2 if c2 else 0
            if p1 == 0 or p2 == 0:
                continue
            ll = -2 * (_ll(a, c1, p) + _ll(b, c2, p) - _ll(a, c1, p1) - _ll(b, c2, p2))
            if p1 < p:
                ll = -ll  # 期待より少ない
            rows.append({"group": g, "word": w, "count": a, "log_likelihood": ll})
    out = (
        pd.DataFrame(rows)
        .sort_values(["group", "log_likelihood"], ascending=[True, False])
        .groupby("group", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def jaccard_feature_words(
    docs: list[list[str]],
    labels: list,
    top_n: int = 30,
    min_count: int = 3,
) -> pd.DataFrame:
    """グループ内出現文書割合 / 全体出現文書割合 の比率 (文書レベル Jaccard 近似)。"""
    groups_docs: dict[object, list[set]] = {}
    all_docs: list[set] = []
    for d, lab in zip(docs, labels):
        s = set(d)
        groups_docs.setdefault(lab, []).append(s)
        all_docs.append(s)

    df_all: Counter = Counter()
    for s in all_docs:
        df_all.update(s)

    rows = []
    for g, group_sets in groups_docs.items():
        df_g: Counter = Counter()
        for s in group_sets:
            df_g.update(s)
        for w, c in df_g.items():
            if df_all[w] < min_count:
                continue
            denom = (len(group_sets) + df_all[w] - c)
            score = c / denom if denom else 0.0
            rows.append({"group": g, "word": w, "count": c, "jaccard": score})
    out = (
        pd.DataFrame(rows)
        .sort_values(["group", "jaccard"], ascending=[True, False])
        .groupby("group", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out


def compare_groups(
    docs: list[list[str]],
    labels: list,
    measure: str = "chi2",
    top_n: int = 30,
) -> pd.DataFrame:
    if measure == "chi2":
        return chi2_feature_words(docs, labels, top_n=top_n)
    if measure == "log_likelihood":
        return log_likelihood_feature_words(docs, labels, top_n=top_n)
    if measure == "jaccard":
        return jaccard_feature_words(docs, labels, top_n=top_n)
    raise ValueError(f"unknown measure: {measure}")
