"""共起ネットワーク構築・コミュニティ検出・中心性解析。"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Optional

import networkx as nx
import pandas as pd

from .frequency import _score
from .utils import optional_import


class CooccurrenceNetwork:
    """共起ネットワーク。

    scope:
        'document'  — 文書単位の共起
        'sentence'  — 文単位 (. 。! ? でsplit)
        'window'    — 窓幅 window_size 内の共起
    """

    def __init__(
        self,
        docs: list[list[str]],
        window_size: int = 5,
        scope: str = "document",
    ):
        self.docs = docs
        self.window_size = window_size
        self.scope = scope
        self.graph: Optional[nx.Graph] = None
        self._word_freq: Counter = Counter()
        for d in docs:
            self._word_freq.update(d)

    def _iter_contexts(self):
        if self.scope == "document":
            for doc in self.docs:
                yield list(set(doc))
        elif self.scope == "sentence":
            for doc in self.docs:
                joined = " ".join(doc)
                for sent in _split_sentences(joined):
                    tokens = sent.split()
                    if tokens:
                        yield list(set(tokens))
        elif self.scope == "window":
            for doc in self.docs:
                for i in range(len(doc)):
                    ctx = doc[max(0, i - self.window_size) : i + self.window_size + 1]
                    yield list(set(ctx))
        else:
            raise ValueError(f"unknown scope: {self.scope}")

    def build(
        self,
        min_edge_weight: int = 3,
        measure: str = "jaccard",
        top_n_nodes: int = 100,
    ) -> nx.Graph:
        # 共起カウント
        pair_counts: Counter = Counter()
        for ctx in self._iter_contexts():
            for a, b in itertools.combinations(sorted(ctx), 2):
                pair_counts[(a, b)] += 1

        # 上位ノード限定
        top_words = {w for w, _ in self._word_freq.most_common(top_n_nodes)}
        N = len(self.docs)

        G = nx.Graph()
        for w, freq in self._word_freq.items():
            if w in top_words:
                G.add_node(w, frequency=freq)

        for (a, b), c in pair_counts.items():
            if c < min_edge_weight:
                continue
            if a not in top_words or b not in top_words:
                continue
            score = _score(
                measure, c,
                self._word_freq[a], self._word_freq[b], max(N, 1),
            )
            G.add_edge(a, b, weight=score, cooccur=c)

        # 孤立ノード削除
        G.remove_nodes_from(list(nx.isolates(G)))
        self.graph = G
        return G

    def detect_communities(self, method: str = "louvain") -> dict[str, int]:
        self._require_graph()
        if method == "louvain":
            community = optional_import("community", "python-louvain")
            return community.best_partition(self.graph)
        if method == "greedy":
            comms = nx.algorithms.community.greedy_modularity_communities(self.graph)
            return {n: i for i, c in enumerate(comms) for n in c}
        if method == "girvan_newman":
            comp = next(nx.algorithms.community.girvan_newman(self.graph))
            return {n: i for i, c in enumerate(comp) for n in c}
        raise ValueError(f"unknown method: {method}")

    def centrality(self, measure: str = "betweenness") -> dict[str, float]:
        self._require_graph()
        if measure == "betweenness":
            return nx.betweenness_centrality(self.graph)
        if measure == "closeness":
            return nx.closeness_centrality(self.graph)
        if measure == "eigenvector":
            try:
                return nx.eigenvector_centrality(self.graph, max_iter=500)
            except nx.PowerIterationFailedConvergence:
                return {n: 0.0 for n in self.graph.nodes}
        if measure == "pagerank":
            return nx.pagerank(self.graph)
        if measure == "degree":
            return dict(nx.degree_centrality(self.graph))
        raise ValueError(f"unknown measure: {measure}")

    def node_dataframe(self) -> pd.DataFrame:
        self._require_graph()
        rows = []
        for n, data in self.graph.nodes(data=True):
            rows.append({"node": n, **data})
        return pd.DataFrame(rows)

    def edge_dataframe(self) -> pd.DataFrame:
        self._require_graph()
        rows = []
        for a, b, data in self.graph.edges(data=True):
            rows.append({"source": a, "target": b, **data})
        return pd.DataFrame(rows)

    def visualize(
        self,
        backend: str = "pyvis",
        output: str = "network.html",
        color_by: str = "community",
        size_by: str = "frequency",
        communities: Optional[dict[str, int]] = None,
    ):
        self._require_graph()
        if backend == "pyvis":
            return self._visualize_pyvis(output, color_by, size_by, communities)
        if backend == "matplotlib":
            return self._visualize_matplotlib(color_by, size_by, communities)
        raise ValueError(f"unknown backend: {backend}")

    def _visualize_pyvis(self, output, color_by, size_by, communities):
        pyvis = optional_import("pyvis", "pyvis")
        from pyvis.network import Network
        if color_by == "community" and communities is None:
            try:
                communities = self.detect_communities("louvain")
            except ImportError:
                communities = {}
        net = Network(height="700px", width="100%", notebook=False)
        for n, data in self.graph.nodes(data=True):
            size = 10 + (data.get(size_by, 1) ** 0.5) * 3
            group = communities.get(n, 0) if communities else 0
            net.add_node(n, label=n, size=size, group=group, title=f"freq: {data.get('frequency', 0)}")
        for a, b, data in self.graph.edges(data=True):
            net.add_edge(a, b, value=float(data.get("weight", 1)))
        net.write_html(output, notebook=False)
        return output

    def _visualize_matplotlib(self, color_by, size_by, communities):
        import matplotlib.pyplot as plt
        try:
            import japanize_matplotlib  # noqa: F401
        except ImportError:
            pass
        if color_by == "community" and communities is None:
            try:
                communities = self.detect_communities("louvain")
            except ImportError:
                communities = {n: 0 for n in self.graph.nodes}

        pos = nx.spring_layout(self.graph, seed=42)
        fig, ax = plt.subplots(figsize=(12, 9))
        sizes = [300 + self.graph.nodes[n].get(size_by, 1) * 30 for n in self.graph.nodes]
        colors = [communities.get(n, 0) for n in self.graph.nodes] if communities else "lightblue"
        nx.draw_networkx_nodes(self.graph, pos, node_size=sizes, node_color=colors, cmap="tab20", alpha=0.8, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=10, ax=ax)
        ax.set_axis_off()
        return fig

    def to_gephi(self, path: str) -> str:
        self._require_graph()
        nx.write_gexf(self.graph, path)
        return path

    def _require_graph(self):
        if self.graph is None:
            raise RuntimeError("build() を先に呼び出してください。")


def _split_sentences(text: str) -> list[str]:
    import re
    return [s.strip() for s in re.split(r"[。.!?！？\n]+", text) if s.strip()]
