"""可視化統合: wordcloud / bar / sunburst / treemap / heatmap / scatter / network。"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import ensure_font, optional_import

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass


def wordcloud(
    docs: list[list[str]],
    font_path: Optional[str] = None,
    max_words: int = 200,
    width: int = 1200,
    height: int = 600,
    background_color: str = "white",
    colormap: str = "tab20_r",
) -> plt.Figure:
    wc_mod = optional_import("wordcloud", "wordcloud")
    text = " ".join(" ".join(d) for d in docs)
    font = ensure_font(font_path)
    wc = wc_mod.WordCloud(
        font_path=font,
        max_words=max_words,
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def bar_frequency(
    freq_df: pd.DataFrame,
    top_n: int = 30,
    x_col: str = "tf",
    y_col: str = "word",
    title: str = "単語頻度",
) -> go.Figure:
    d = freq_df.head(top_n).iloc[::-1]
    fig = px.bar(d, x=x_col, y=y_col, orientation="h", title=title)
    fig.update_layout(height=max(400, top_n * 18))
    return fig


def sunburst_chart(
    df: pd.DataFrame,
    path_cols: list[str],
    value_col: Optional[str] = None,
    title: str = "Sunburst",
) -> go.Figure:
    return px.sunburst(df, path=path_cols, values=value_col, title=title)


def treemap(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = "Treemap",
) -> go.Figure:
    return px.treemap(df, path=[group_col], values=value_col, title=title)


def heatmap_cooccurrence(
    matrix: np.ndarray,
    labels: list[str],
    title: str = "共起ヒートマップ",
) -> go.Figure:
    return px.imshow(
        matrix, x=labels, y=labels,
        color_continuous_scale="Blues",
        title=title,
    )


def scatter_2d(
    coords: np.ndarray,
    labels: Optional[list] = None,
    texts: Optional[list] = None,
    interactive: bool = True,
    title: str = "2D scatter",
) -> go.Figure:
    df = pd.DataFrame(coords, columns=["x", "y"])
    if labels is not None:
        df["label"] = [str(l) for l in labels]
    if texts is not None:
        df["text"] = texts
    color = "label" if labels is not None else None
    hover = "text" if texts is not None else None
    fig = px.scatter(df, x="x", y="y", color=color, hover_name=hover, title=title)
    fig.update_traces(marker=dict(size=8))
    return fig


def network_interactive(
    graph: nx.Graph,
    output: str = "net.html",
    communities: Optional[dict] = None,
) -> str:
    pyvis = optional_import("pyvis", "pyvis")
    from pyvis.network import Network
    net = Network(height="700px", width="100%", notebook=False)
    for n, data in graph.nodes(data=True):
        size = 10 + (data.get("frequency", 1) ** 0.5) * 3
        group = communities.get(n, 0) if communities else 0
        net.add_node(n, label=str(n), size=size, group=group)
    for a, b, data in graph.edges(data=True):
        net.add_edge(a, b, value=float(data.get("weight", 1)))
    net.write_html(output, notebook=False)
    return output


def dendrogram_figure(
    linkage_matrix: np.ndarray,
    labels: Optional[list] = None,
    figsize=(15, 8),
) -> plt.Figure:
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90)
    return fig
