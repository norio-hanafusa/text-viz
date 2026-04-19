"""コーディングルール機能の独自実装。KH Coder で提供される同種機能を参考にしている。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import networkx as nx
import pandas as pd


def load_rules_yaml(path: Union[str, Path]) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CodingRule:
    """辞書型ルール { コード名: [マッチ語...] } を文書に適用。"""

    def __init__(self, rules: Union[dict[str, list[str]], str, Path]):
        if isinstance(rules, (str, Path)):
            self.rules = load_rules_yaml(rules)
        else:
            self.rules = dict(rules)
        # 事前コンパイル
        self._patterns = {
            code: re.compile("|".join(re.escape(w) for w in words))
            for code, words in self.rules.items()
            if words
        }

    def _apply_doc(self, doc: str) -> dict[str, int]:
        return {code: len(p.findall(doc)) for code, p in self._patterns.items()}

    def apply(self, docs: list[str]) -> pd.DataFrame:
        """各文書にコード別ヒット数を付与した DataFrame を返す。"""
        rows = [self._apply_doc(d) for d in docs]
        return pd.DataFrame(rows).fillna(0).astype(int)

    def frequency(self, docs: list[str]) -> pd.Series:
        """コード別出現頻度 (合計ヒット数)。"""
        return self.apply(docs).sum().sort_values(ascending=False)

    def document_frequency(self, docs: list[str]) -> pd.Series:
        """コードが出現した文書数。"""
        return (self.apply(docs) > 0).sum().sort_values(ascending=False)

    def cooccurrence(self, docs: list[str]) -> nx.Graph:
        """コード間の共起 (同一文書内で両コードがヒット) をネットワーク化。"""
        df = self.apply(docs)
        codes = list(df.columns)
        G = nx.Graph()
        for c in codes:
            G.add_node(c, frequency=int(df[c].sum()))
        binary = (df > 0).astype(int)
        for i, a in enumerate(codes):
            for b in codes[i + 1:]:
                co = int(((binary[a] > 0) & (binary[b] > 0)).sum())
                if co > 0:
                    G.add_edge(a, b, weight=co)
        return G

    def cross_tab(
        self,
        docs: list[str],
        external_var: list,
    ) -> pd.DataFrame:
        """コード × 外部変数のクロス集計。"""
        df = self.apply(docs)
        df["__ext__"] = external_var
        return df.groupby("__ext__").sum()
